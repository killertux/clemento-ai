//! Truncated backpropagation through time over a recorded `Tape`.
//!
//! The cyclic graph unrolled over `window` steps is a DAG (README §4): an edge
//! delivers its signal on the *next* step, so gradients flow both upstream
//! (through edges within a step) and backwards in time (into the previous
//! step's state). The gate is handled with a surrogate gradient (README §6):
//! `activate_grad` uses the soft-gate derivative regardless of gate mode.

use rayon::prelude::*;

use crate::network::{activate_grad, Network, Tape};

pub struct Grads {
    /// Gradient per edge (parallel to `Topology::weights`).
    pub g_w: Vec<f32>,
    /// Gradient of the embedding, `vocab * n_in`.
    pub g_embed: Vec<f32>,
    /// Gradient of the readout, `n_out * vocab`.
    pub g_readout: Vec<f32>,
    /// Gradient of the output bias, `vocab`.
    pub g_bias: Vec<f32>,
    /// `dL/dpre` per step (kept so RigL can score candidate edges). `window`
    /// entries each of length `n`.
    pub dpre: Vec<Vec<f32>>,
}

impl Network {
    /// Backpropagate one window. Returns gradients for every trainable
    /// parameter. Gradients into `state_init` are dropped (truncation).
    pub fn backward(&self, tape: &Tape) -> Grads {
        let cfg = &self.cfg;
        let n = cfg.n;
        let w = cfg.window;
        let vocab = self.vocab;
        let base = cfg.out_start();
        let inv = 1.0 / w as f32;

        let mut g_w = vec![0.0f32; self.topo.num_edges()];
        let mut g_embed = vec![0.0f32; vocab * cfg.n_in];
        let mut g_readout = vec![0.0f32; cfg.n_out * vocab];
        let mut g_bias = vec![0.0f32; vocab];
        let mut dpre_tape: Vec<Vec<f32>> = vec![Vec::new(); w];

        // grad into the current step's state arriving from the future (step t+1)
        let mut ds_future = vec![0.0f32; n];

        for t in (0..w).rev() {
            let s_t = &tape.state[t];
            let p_t = &tape.pre[t];

            // dL/ds[t] = future recurrence + readout path
            let mut ds = ds_future; // moved; reallocated below as ds_prev
            let probs = &tape.probs[t];
            let target = tape.targets[t] as usize;

            // cross-entropy: dL/dlogit[c] = (p[c] - 1{c==target}) / window
            let mut dlog = probs.clone();
            dlog[target] -= 1.0;
            for c in 0..vocab {
                dlog[c] *= inv;
                g_bias[c] += dlog[c];
            }
            for m in 0..cfg.n_out {
                let sv = s_t[base + m];
                let row = m * vocab;
                let mut acc = 0.0f32;
                for c in 0..vocab {
                    g_readout[row + c] += sv * dlog[c];
                    acc += self.readout[row + c] * dlog[c];
                }
                ds[base + m] += acc;
            }

            // dL/dpre = dL/ds * dy/dp  (surrogate gate derivative inside)
            let dpre: Vec<f32> =
                (0..n).into_par_iter().map(|j| ds[j] * activate_grad(p_t[j], cfg)).collect();

            // embedding grad (token injected into input neurons at this step)
            let erow = tape.inputs[t] as usize * cfg.n_in;
            for i in 0..cfg.n_in {
                g_embed[erow + i] += dpre[i];
            }

            // Backprop into the previous step's state — parallel over sources
            // (each source owns a disjoint out-edge range, so no collisions).
            let s_prev: &[f32] = if t == 0 { &tape.state_init } else { &tape.state[t - 1] };
            let topo = &self.topo;
            let ds_prev: Vec<f32> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut acc = 0.0f32;
                    for e in topo.out_range(i) {
                        acc += topo.weights[e] * dpre[topo.col_idx[e] as usize];
                    }
                    acc
                })
                .collect();
            // Edge-weight grads — parallel over edges (each edge is written once
            // this step; accumulation across steps stays sequential in `t`).
            g_w.par_iter_mut().enumerate().for_each(|(e, g)| {
                let src = topo.edge_src[e] as usize;
                let j = topo.col_idx[e] as usize;
                *g += s_prev[src] * dpre[j];
            });

            dpre_tape[t] = dpre;
            ds_future = ds_prev; // becomes dL/ds[t-1] for the next iteration
        }
        // ds_future here is dL/dstate_init — dropped (truncated BPTT).

        Grads { g_w, g_embed, g_readout, g_bias, dpre: dpre_tape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, GateMode};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Finite-difference gradient check on a tiny, fully-differentiable
    /// (soft-gate) network. Validates the entire chain-rule wiring: readout,
    /// bias, embedding, per-edge weights, tanh, gate, and recurrence through
    /// time. With the soft gate, forward and backward are exactly consistent.
    #[test]
    fn gradient_check() {
        let mut cfg = Config::tiny();
        cfg.gate = GateMode::Soft;
        let vocab = 5;
        let mut net = Network::new(cfg.clone(), vocab);

        let mut rng = StdRng::seed_from_u64(123);
        let seq: Vec<u32> = (0..cfg.window + 1).map(|_| rng.gen_range(0..vocab as u32)).collect();
        let init = vec![0.0f32; cfg.n];

        let tape = net.run_window(&seq, &init);
        let grads = net.backward(&tape);

        let eps = 3e-3f32;
        let mut checked = 0usize;

        // Central-difference one parameter (already set to `orig`) and compare to
        // the analytic gradient. Returns whether it was checked (skips ~0 grads).
        let mut assert_grad = |num: f32, ana: f32, label: &str| {
            if ana.abs() < 1e-4 {
                return; // finite diff is unreliable where the gradient is ~0
            }
            let rel = (num - ana).abs() / (ana.abs() + num.abs() + 1e-4);
            assert!(rel < 3e-2, "{label}: analytic={ana} numeric={num} rel={rel}");
            checked += 1;
        };

        macro_rules! fd {
            ($ana:expr, $slot:expr, $label:expr) => {{
                let ana = $ana;
                if ana.abs() >= 1e-4 {
                    let orig = $slot;
                    $slot = orig + eps;
                    let lp = net.run_window(&seq, &init).loss;
                    $slot = orig - eps;
                    let lm = net.run_window(&seq, &init).loss;
                    $slot = orig;
                    let num = (lp - lm) / (2.0 * eps);
                    assert_grad(num, ana, $label);
                }
            }};
        }

        for idx in (0..net.embed.len()).step_by(3) {
            fd!(grads.g_embed[idx], net.embed[idx], "embed");
        }
        for idx in (0..net.readout.len()).step_by(3) {
            fd!(grads.g_readout[idx], net.readout[idx], "readout");
        }
        for idx in 0..net.bias_out.len() {
            fd!(grads.g_bias[idx], net.bias_out[idx], "bias_out");
        }
        for e in (0..net.topo.num_edges()).step_by(2) {
            fd!(grads.g_w[e], net.topo.weights[e], "g_w");
        }

        assert!(checked > 20, "too few parameters exercised: {checked}");
    }
}
