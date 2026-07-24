//! Hyperparameters for the free-topology recurrent network.
//!
//! See `README.md` for the meaning of each field. Defaults target a small,
//! CPU-trainable char-level language model.

/// Which gate the activation uses.
///
/// - `Soft`: fully differentiable `g(p) = σ(β(p−θ)) + σ(β(−p−θ))`, used in the
///   forward pass too. This makes forward and backward exactly consistent, which
///   is what the finite-difference gradient check relies on.
/// - `Hard`: forward uses the hard double-threshold `|p| > θ` (event-driven,
///   sparse); backward uses the *same* soft derivative as `Soft` — this is the
///   surrogate gradient (README §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateMode {
    Soft,
    Hard,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Total number of neurons in the pool.
    pub n: usize,
    /// Out-edges created per neuron at initialization.
    pub k: usize,
    /// Number of neurons that receive the injected token embedding.
    pub n_in: usize,
    /// Number of neurons whose state is read out as logits.
    pub n_out: usize,

    /// Gate threshold θ.
    pub theta: f32,
    /// Gate steepness β for the (surrogate) sigmoid.
    pub beta: f32,
    /// Forward/backward gate behaviour.
    pub gate: GateMode,

    /// Target spectral radius after rescaling recurrent weights.
    pub spectral_radius: f32,

    /// Truncated-BPTT window length (steps unrolled before an update).
    pub window: usize,

    /// Adam learning rate.
    pub lr: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    /// Global gradient-norm clip.
    pub grad_clip: f32,

    /// RigL: apply a prune/grow update every this many optimizer steps.
    pub rigl_interval: usize,
    /// RigL: initial fraction of each neuron's out-edges rewired per update.
    pub rigl_fraction: f32,
    /// RigL: candidate targets sampled per neuron when growing.
    pub rigl_candidates: usize,
    /// RigL: total optimizer steps over which `rigl_fraction` cosine-decays to 0.
    pub rigl_decay_steps: usize,
    /// Enable topology learning (v1). When false, topology is frozen (v0).
    pub rigl_enabled: bool,

    /// PRNG seed for reproducible runs.
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n: 4096,
            k: 32,
            n_in: 64,
            n_out: 64,
            theta: 0.1,
            beta: 4.0,
            gate: GateMode::Hard,
            spectral_radius: 0.95,
            window: 64,
            lr: 1e-3,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            grad_clip: 1.0,
            rigl_interval: 100,
            rigl_fraction: 0.3,
            rigl_candidates: 256,
            rigl_decay_steps: 20_000,
            rigl_enabled: false,
            seed: 42,
        }
    }
}

impl Config {
    /// A tiny configuration used by tests (fast, fully differentiable).
    #[allow(dead_code)]
    pub fn tiny() -> Self {
        Config {
            n: 16,
            k: 4,
            n_in: 4,
            n_out: 4,
            theta: 0.1,
            beta: 4.0,
            gate: GateMode::Soft,
            spectral_radius: 0.9,
            window: 5,
            rigl_enabled: false,
            seed: 1,
            ..Config::default()
        }
    }

    /// Index range of output neurons (the last `n_out` neurons).
    pub fn out_start(&self) -> usize {
        self.n - self.n_out
    }
}
