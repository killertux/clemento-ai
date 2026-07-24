//! The network: topology + trainable dense in/out projections, plus the
//! forward pass (`run_window`) that executes one BFS wave per time step and
//! records a `Tape` for backpropagation.
//!
//! Neuron index layout: `[0, n_in)` are input neurons (receive the token
//! embedding); `[n - n_out, n)` are output neurons (read out as logits). These
//! ranges may overlap the hidden pool — an input neuron is still a normal neuron
//! with incoming recurrent edges.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::config::{Config, GateMode};
use crate::graph::{gaussian, Topology};

pub struct Network {
    pub cfg: Config,
    pub topo: Topology,
    /// Token embedding, `vocab * n_in`, row-major by token id.
    pub embed: Vec<f32>,
    /// Readout, `n_out * vocab`, row-major by output-neuron offset.
    pub readout: Vec<f32>,
    /// Output bias, `vocab`.
    pub bias_out: Vec<f32>,
    pub vocab: usize,
}

/// Everything recorded during the forward pass needed to backpropagate.
pub struct Tape {
    /// Post-activation state per step, `window` entries each of length `n`.
    pub state: Vec<Vec<f32>>,
    /// Pre-activation per step (input to gate/tanh), `window` entries.
    pub pre: Vec<Vec<f32>>,
    /// State carried in from before the window (detached; grads stop here).
    pub state_init: Vec<f32>,
    /// Softmax probabilities per step, `window` entries of length `vocab`.
    pub probs: Vec<Vec<f32>>,
    /// Prediction target (next token id) per step.
    pub targets: Vec<u32>,
    /// Input token id injected per step.
    pub inputs: Vec<u32>,
    /// Mean cross-entropy loss over the window (nats).
    pub loss: f32,
}

impl Network {
    pub fn new(cfg: Config, vocab: usize) -> Network {
        let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0xA11CE);
        let embed_std = (1.0 / cfg.n_in as f32).sqrt();
        let read_std = (1.0 / cfg.n_out as f32).sqrt();
        let embed = (0..vocab * cfg.n_in).map(|_| gaussian(&mut rng) * embed_std).collect();
        let readout = (0..cfg.n_out * vocab).map(|_| gaussian(&mut rng) * read_std).collect();
        let bias_out = vec![0.0; vocab];
        let mut topo_rng = StdRng::seed_from_u64(cfg.seed);
        let topo = Topology::init_random(&cfg, &mut topo_rng);
        Network { cfg, topo, embed, readout, bias_out, vocab }
    }

    /// Run one truncated-BPTT window. `seq` has length `window + 1`; token
    /// `seq[t]` is injected at step `t` and `seq[t + 1]` is its target.
    /// `state_init` is carried recurrent state (use zeros to reset).
    pub fn run_window(&self, seq: &[u32], state_init: &[f32]) -> Tape {
        let cfg = &self.cfg;
        let w = cfg.window;
        assert!(seq.len() >= w + 1, "sequence shorter than window + 1");

        let mut state = Vec::with_capacity(w);
        let mut pre = Vec::with_capacity(w);
        let mut probs = Vec::with_capacity(w);
        let mut targets = Vec::with_capacity(w);
        let mut inputs = Vec::with_capacity(w);
        let mut loss = 0.0f32;

        let mut prev = state_init.to_vec();
        for t in 0..w {
            let tok = seq[t];
            // pre-activation: token injection into input neurons ...
            let mut p = vec![0.0f32; cfg.n];
            let erow = tok as usize * cfg.n_in;
            for i in 0..cfg.n_in {
                p[i] += self.embed[erow + i];
            }
            // ... plus recurrent contributions, gathered per target in parallel.
            self.gather_recurrent(&mut p, &prev);
            // activation (parallel, element-wise)
            let s: Vec<f32> = p.par_iter().map(|&pj| activate(pj, cfg)).collect();
            // readout -> logits -> softmax -> loss
            let logits = self.logits(&s);
            let (pr, l) = softmax_ce(&logits, seq[t + 1]);
            loss += l;

            probs.push(pr);
            targets.push(seq[t + 1]);
            inputs.push(tok);
            pre.push(p);
            state.push(s.clone());
            prev = s;
        }

        Tape {
            state,
            pre,
            state_init: state_init.to_vec(),
            probs,
            targets,
            inputs,
            loss: loss / w as f32,
        }
    }

    /// Add recurrent contributions to the pre-activation `p`, gathering each
    /// target neuron's in-edges independently (parallel, no write collisions).
    /// `p` must already contain the token injection for input neurons.
    fn gather_recurrent(&self, p: &mut [f32], prev: &[f32]) {
        let topo = &self.topo;
        p.par_iter_mut().enumerate().for_each(|(j, pj)| {
            let mut acc = *pj;
            for idx in topo.in_range(j) {
                let e = topo.in_edge[idx] as usize;
                let src = topo.edge_src[e] as usize;
                let sv = prev[src];
                if sv != 0.0 {
                    acc += topo.weights[e] * sv;
                }
            }
            *pj = acc;
        });
    }

    /// Logits from a state vector: `bias + readout^T · state[out neurons]`.
    pub fn logits(&self, state: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let mut logits = self.bias_out.clone();
        let base = cfg.out_start();
        for m in 0..cfg.n_out {
            let sv = state[base + m];
            if sv == 0.0 {
                continue;
            }
            let row = m * self.vocab;
            for c in 0..self.vocab {
                logits[c] += self.readout[row + c] * sv;
            }
        }
        logits
    }

    /// Advance the recurrent state by one step for a single injected token
    /// (used by generation). Returns `(new_state, logits)`.
    pub fn step_once(&self, tok: u32, prev: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.cfg;
        let mut p = vec![0.0f32; cfg.n];
        let erow = tok as usize * cfg.n_in;
        for i in 0..cfg.n_in {
            p[i] += self.embed[erow + i];
        }
        self.gather_recurrent(&mut p, prev);
        let s: Vec<f32> = p.par_iter().map(|&pj| activate(pj, cfg)).collect();
        let logits = self.logits(&s);
        (s, logits)
    }
}

// --- gate / activation math -------------------------------------------------

/// Sigmoid.
#[inline]
pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Gate value used in the forward pass.
///
/// `Soft`: `σ(β(p−θ)) + σ(β(−p−θ))` — differentiable.
/// `Hard`: `1` if `|p| > θ` else `0` — event-driven.
#[inline]
pub fn gate(p: f32, cfg: &Config) -> f32 {
    match cfg.gate {
        GateMode::Soft => sigmoid(cfg.beta * (p - cfg.theta)) + sigmoid(cfg.beta * (-p - cfg.theta)),
        GateMode::Hard => {
            if p.abs() > cfg.theta {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Derivative `dg/dp` of the *soft* gate. Used for both `Soft` (exact) and
/// `Hard` (surrogate) modes in the backward pass.
#[inline]
pub fn gate_grad(p: f32, cfg: &Config) -> f32 {
    let a = sigmoid(cfg.beta * (p - cfg.theta));
    let b = sigmoid(cfg.beta * (-p - cfg.theta));
    cfg.beta * a * (1.0 - a) - cfg.beta * b * (1.0 - b)
}

/// Neuron output `y = gate(p) · tanh(p)`.
#[inline]
pub fn activate(p: f32, cfg: &Config) -> f32 {
    gate(p, cfg) * p.tanh()
}

/// `dy/dp` — the local gradient of `activate`. In `Hard` mode the gate value is
/// the hard step (matching the forward pass) but its derivative is the soft
/// surrogate `gate_grad`.
#[inline]
pub fn activate_grad(p: f32, cfg: &Config) -> f32 {
    let t = p.tanh();
    let g = gate(p, cfg);
    let dg = gate_grad(p, cfg);
    dg * t + g * (1.0 - t * t)
}

/// Softmax + cross-entropy for one step. Returns `(probs, loss_nats)`.
pub fn softmax_ce(logits: &[f32], target: u32) -> (Vec<f32>, f32) {
    let maxv = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&z| (z - maxv).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let loss = -(probs[target as usize].max(1e-30)).ln();
    (probs, loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn activity_stays_in_band() {
        // Free-run the network (no meaningful input) and confirm activity neither
        // saturates nor dies — validates theta + spectral radius defaults together.
        let mut cfg = Config::default();
        cfg.n = 1024;
        let net = Network::new(cfg.clone(), 8);
        let mut prev = vec![0.0f32; cfg.n];
        // kick-start a few input neurons
        for i in 0..cfg.n_in {
            prev[i] = 0.5;
        }
        let mut ok_steps = 0;
        for _ in 0..200 {
            let (s, _) = net.step_once(0, &prev);
            let active = s.iter().filter(|&&v| v != 0.0).count();
            let frac = active as f32 / cfg.n as f32;
            assert!(frac < 0.9, "saturated: {frac}");
            if frac > 0.0 {
                ok_steps += 1;
            }
            prev = s;
        }
        assert!(ok_steps > 100, "activity died too early: {ok_steps} live steps");
    }

    #[test]
    fn gate_grad_matches_finite_difference() {
        let mut cfg = Config::tiny();
        cfg.gate = GateMode::Soft;
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let p: f32 = rng.gen_range(-2.0..2.0);
            let eps = 1e-3;
            let num = (activate(p + eps, &cfg) - activate(p - eps, &cfg)) / (2.0 * eps);
            let ana = activate_grad(p, &cfg);
            assert!((num - ana).abs() < 1e-2, "p={p} num={num} ana={ana}");
        }
    }
}
