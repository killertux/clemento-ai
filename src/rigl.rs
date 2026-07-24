//! RigL-style dynamic sparse training (README §7).
//!
//! Every `rigl_interval` optimizer steps, for each neuron we **prune** the `k`
//! out-edges with the smallest weight magnitude and **grow** `k` new edges
//! toward the candidate targets whose (currently-nonexistent) connection has the
//! largest gradient magnitude. Total edge count is held constant; new edges
//! start at weight 0 (so the function is unchanged at the instant of rewiring)
//! and inherit zeroed Adam moments. `k` cosine-decays to 0 over training.

use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::backprop::Grads;
use crate::graph::Topology;
use crate::network::{Network, Tape};
use crate::optim::Moment;

/// Fraction of edges to rewire at `opt_step`, cosine-decayed from
/// `cfg.rigl_fraction` to 0 over `cfg.rigl_decay_steps`.
pub fn churn_fraction(cfg: &crate::config::Config, opt_step: u64) -> f32 {
    let frac = (opt_step as f32 / cfg.rigl_decay_steps.max(1) as f32).min(1.0);
    cfg.rigl_fraction * 0.5 * (1.0 + (std::f32::consts::PI * frac).cos())
}

/// Perform one prune/grow update in place. Returns the actual churn fraction
/// (edges rewired / total edges) for logging. Requires the most recent forward
/// tape and its gradients (for scoring candidate edges).
pub fn rigl_step(
    net: &mut Network,
    wmom: &mut Moment,
    tape: &Tape,
    grads: &Grads,
    opt_step: u64,
) -> f32 {
    let cfg = net.cfg.clone();
    let n = cfg.n;
    let frac = churn_fraction(&cfg, opt_step);

    let total_edges = net.topo.num_edges();
    let mut new_row: Vec<u32> = Vec::with_capacity(n + 1);
    let mut new_col: Vec<u32> = Vec::with_capacity(total_edges);
    let mut new_w: Vec<f32> = Vec::with_capacity(total_edges);
    let mut new_m: Vec<f32> = Vec::with_capacity(total_edges);
    let mut new_v: Vec<f32> = Vec::with_capacity(total_edges);
    new_row.push(0);

    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x5164_1234 ^ opt_step);
    let mut rewired = 0usize;

    for i in 0..n {
        let range = net.topo.out_range(i);
        let deg = range.len();
        // Number of edges to rewire for this neuron.
        let kk = ((frac * deg as f32).round() as usize).min(deg);

        // Current out-edges as (target, weight, edge-index), sorted by |w| desc.
        let mut edges: Vec<(u32, f32, usize)> =
            range.map(|e| (net.topo.col_idx[e], net.topo.weights[e], e)).collect();
        edges.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        let keep = deg - kk; // survivors (largest |w|)
        // Exclude every currently-connected target (survivors *and* pruned) plus
        // self, so grown edges are genuinely new and can never collide with a
        // pruned edge that the refill fallback might re-add.
        let mut connected: HashSet<u32> = edges.iter().map(|e| e.0).collect();
        connected.insert(i as u32);

        // Score sampled candidate targets by |candidate-edge gradient|:
        //   grad(i -> j) = Σ_t s_prev[t][i] · dpre[t][j]
        let mut cand: Vec<(u32, f32)> = Vec::new();
        let mut tried: HashSet<u32> = HashSet::new();
        for _ in 0..cfg.rigl_candidates {
            let j = rng.gen_range(0..n as u32);
            if connected.contains(&j) || !tried.insert(j) {
                continue;
            }
            let mut g = 0.0f32;
            for t in 0..cfg.window {
                let sp = if t == 0 { tape.state_init[i] } else { tape.state[t - 1][i] };
                if sp != 0.0 {
                    g += sp * grads.dpre[t][j as usize];
                }
            }
            cand.push((j, g.abs()));
        }
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let grow_n = kk.min(cand.len());

        // Emit survivors (carry weight + Adam moments).
        for &(c, w, e) in &edges[..keep] {
            new_col.push(c);
            new_w.push(w);
            new_m.push(wmom.m[e]);
            new_v.push(wmom.v[e]);
        }
        // Emit grown edges (weight 0, zeroed moments).
        for &(j, _) in cand.iter().take(grow_n) {
            new_col.push(j);
            new_w.push(0.0);
            new_m.push(0.0);
            new_v.push(0.0);
        }
        // If too few unique candidates were found, re-add the largest pruned
        // edges so the per-neuron degree (and total budget) stays exactly fixed.
        let mut filled = keep + grow_n;
        let mut pruned = edges[keep..].iter(); // smallest |w| first? no: sorted desc, so [keep..] are smallest
        // edges[keep..] are the pruned (smallest |w|); iterate largest-of-pruned first.
        while filled < deg {
            if let Some(&(c, w, e)) = pruned.next() {
                new_col.push(c);
                new_w.push(w);
                new_m.push(wmom.m[e]);
                new_v.push(wmom.v[e]);
                filled += 1;
            } else {
                break;
            }
        }

        rewired += grow_n;
        new_row.push(new_col.len() as u32);
    }

    net.topo = Topology::from_csr(n, new_row, new_col, new_w);
    wmom.m = new_m;
    wmom.v = new_v;

    rewired as f32 / total_edges as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn edge_budget_is_constant_and_new_edges_are_zero() {
        let mut cfg = Config::tiny();
        cfg.rigl_candidates = 12;
        cfg.rigl_fraction = 0.5;
        cfg.rigl_decay_steps = 1000;
        let vocab = 5;
        let mut net = Network::new(cfg.clone(), vocab);
        let mut wmom = Moment::zeros(net.topo.num_edges());

        let before_edges = net.topo.num_edges();
        // per-neuron degree before
        let deg_before: Vec<usize> = (0..cfg.n).map(|i| net.topo.out_range(i).len()).collect();

        let seq: Vec<u32> = (0..cfg.window + 1).map(|t| (t % vocab) as u32).collect();
        let init = vec![0.0f32; cfg.n];
        let tape = net.run_window(&seq, &init);
        let grads = net.backward(&tape);

        let churn = rigl_step(&mut net, &mut wmom, &tape, &grads, 10);

        assert_eq!(net.topo.num_edges(), before_edges, "total edge budget changed");
        assert_eq!(wmom.m.len(), before_edges);
        for i in 0..cfg.n {
            assert_eq!(net.topo.out_range(i).len(), deg_before[i], "degree of {i} changed");
            // no self-loops, no duplicate targets
            let mut seen = HashSet::new();
            for e in net.topo.out_range(i) {
                let j = net.topo.col_idx[e];
                assert_ne!(j as usize, i);
                assert!(seen.insert(j), "dup edge {i}->{j}");
            }
        }
        assert!(churn > 0.0, "nothing was rewired");
    }

    #[test]
    fn churn_decays_to_zero() {
        let cfg = Config::default();
        let early = churn_fraction(&cfg, 0);
        let mid = churn_fraction(&cfg, cfg.rigl_decay_steps as u64 / 2);
        let late = churn_fraction(&cfg, cfg.rigl_decay_steps as u64);
        assert!((early - cfg.rigl_fraction).abs() < 1e-6);
        assert!(late < 1e-6);
        assert!(mid < early && mid > late);
    }
}
