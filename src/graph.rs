//! Sparse topology in CSR (Compressed Sparse Row) form, indexed by source
//! neuron. The out-edges of neuron `i` are the slice
//! `col_idx[row_ptr[i]..row_ptr[i+1]]`, with matching weights in `weights`.
//!
//! Propagation reads a source's out-edges as a contiguous slice (cache-friendly).
//! Topology changes (RigL) rebuild the arrays rather than editing them in place.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::config::Config;

pub struct Topology {
    pub n: usize,
    // --- forward (out-edge) CSR, indexed by source ---
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<u32>,
    pub weights: Vec<f32>,
    // --- auxiliary structures for parallel execution (README §8) ---
    /// Source neuron of each edge (parallel to `col_idx`/`weights`).
    pub edge_src: Vec<u32>,
    /// Transpose (in-edge) CSR indexed by *target*: the in-edges of neuron `j`
    /// are `in_edge[in_row_ptr[j]..in_row_ptr[j+1]]`, each an index `e` into the
    /// forward arrays. Lets the forward gather run in parallel over targets with
    /// no write collisions (weights are read live, never duplicated here).
    pub in_row_ptr: Vec<u32>,
    pub in_edge: Vec<u32>,
}

impl Topology {
    /// Build a topology from forward-CSR arrays, computing the auxiliary
    /// transpose structures. Use this everywhere a topology is (re)constructed.
    pub fn from_csr(n: usize, row_ptr: Vec<u32>, col_idx: Vec<u32>, weights: Vec<f32>) -> Topology {
        let e = col_idx.len();
        // source of each edge
        let mut edge_src = vec![0u32; e];
        for i in 0..n {
            for idx in row_ptr[i] as usize..row_ptr[i + 1] as usize {
                edge_src[idx] = i as u32;
            }
        }
        // transpose: count in-degrees, prefix-sum, then scatter edge indices
        let mut in_row_ptr = vec![0u32; n + 1];
        for &j in &col_idx {
            in_row_ptr[j as usize + 1] += 1;
        }
        for j in 0..n {
            in_row_ptr[j + 1] += in_row_ptr[j];
        }
        let mut in_edge = vec![0u32; e];
        let mut cursor = in_row_ptr.clone();
        for edge in 0..e {
            let j = col_idx[edge] as usize;
            in_edge[cursor[j] as usize] = edge as u32;
            cursor[j] += 1;
        }
        Topology { n, row_ptr, col_idx, weights, edge_src, in_row_ptr, in_edge }
    }

    pub fn num_edges(&self) -> usize {
        self.col_idx.len()
    }

    /// Out-edge slice range for neuron `i`.
    #[inline]
    pub fn out_range(&self, i: usize) -> std::ops::Range<usize> {
        self.row_ptr[i] as usize..self.row_ptr[i + 1] as usize
    }

    /// In-edge slice range for neuron `j` (indices into the forward arrays).
    #[inline]
    pub fn in_range(&self, j: usize) -> std::ops::Range<usize> {
        self.in_row_ptr[j] as usize..self.in_row_ptr[j + 1] as usize
    }

    /// Initialize a random sparse graph: each neuron gets `k` distinct out-edges
    /// (no self-loops), weights ~ N(0, 1/k), then globally rescaled so the
    /// spectral radius matches the configured target.
    pub fn init_random(cfg: &Config, rng: &mut StdRng) -> Topology {
        let n = cfg.n;
        let k = cfg.k.min(n - 1);
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::with_capacity(n * k);
        let mut weights = Vec::with_capacity(n * k);

        let std = (1.0 / k as f32).sqrt();
        let mut candidates: Vec<u32> = (0..n as u32).collect();
        row_ptr.push(0);
        for i in 0..n {
            // Sample k distinct targets != i via a partial shuffle.
            candidates.shuffle(rng);
            let mut chosen = 0;
            let mut idx = 0;
            while chosen < k {
                let t = candidates[idx];
                idx += 1;
                if t as usize != i {
                    col_idx.push(t);
                    weights.push(gaussian(rng) * std);
                    chosen += 1;
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }

        let mut topo = Topology::from_csr(n, row_ptr, col_idx, weights);
        topo.rescale_spectral_radius(cfg.spectral_radius);
        topo
    }

    /// Estimate the spectral radius (largest |eigenvalue|) via power iteration on
    /// the sparse weight matrix `A` where `A[j,i] = w(i->j)` (i.e. `y = A x`
    /// aggregates each source's contribution into its targets).
    pub fn spectral_radius(&self, iters: usize, rng: &mut StdRng) -> f32 {
        let n = self.n;
        let mut x: Vec<f32> = (0..n).map(|_| gaussian(rng)).collect();
        normalize(&mut x);
        let mut y = vec![0.0f32; n];
        let mut lambda = 0.0f32;
        for _ in 0..iters {
            for v in y.iter_mut() {
                *v = 0.0;
            }
            for i in 0..n {
                let xi = x[i];
                for e in self.out_range(i) {
                    let j = self.col_idx[e] as usize;
                    y[j] += self.weights[e] * xi;
                }
            }
            lambda = norm(&y);
            if lambda < 1e-12 {
                break;
            }
            for (xv, yv) in x.iter_mut().zip(&y) {
                *xv = yv / lambda;
            }
        }
        lambda
    }

    /// Rescale all weights so the estimated spectral radius equals `target`.
    pub fn rescale_spectral_radius(&mut self, target: f32) {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let rho = self.spectral_radius(50, &mut rng);
        if rho > 1e-9 {
            let s = target / rho;
            for w in self.weights.iter_mut() {
                *w *= s;
            }
        }
    }
}

/// Standard-normal sample (Box–Muller); avoids pulling in `rand_distr`.
pub fn gaussian(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.gen_range(1e-7..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_invariants() {
        let cfg = Config::default();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let topo = Topology::init_random(&cfg, &mut rng);

        assert_eq!(topo.row_ptr.len(), cfg.n + 1);
        assert_eq!(topo.num_edges(), cfg.n * cfg.k);

        // No self-loops, no duplicate targets within a neuron.
        for i in 0..cfg.n {
            let mut seen = std::collections::HashSet::new();
            for e in topo.out_range(i) {
                let j = topo.col_idx[e] as usize;
                assert_ne!(j, i, "self-loop at {i}");
                assert!(seen.insert(j), "duplicate edge {i}->{j}");
            }
        }
    }

    #[test]
    fn spectral_radius_matches_target() {
        let cfg = Config::default();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let topo = Topology::init_random(&cfg, &mut rng);
        let mut r2 = StdRng::seed_from_u64(7);
        let rho = topo.spectral_radius(80, &mut r2);
        let rel = (rho - cfg.spectral_radius).abs() / cfg.spectral_radius;
        assert!(rel < 0.05, "rho = {rho}, target = {}", cfg.spectral_radius);
    }
}
