//! Adam optimizer over flat parameter arrays, plus global-norm gradient
//! clipping. Parameters and their gradients are kept as parallel `Vec<f32>`s;
//! Adam maintains first/second moment buffers of the same shape.

use crate::config::Config;

/// First and second moment buffers for one parameter array.
pub struct Moment {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl Moment {
    pub fn zeros(len: usize) -> Moment {
        Moment { m: vec![0.0; len], v: vec![0.0; len] }
    }
}

pub struct Adam {
    b1: f32,
    b2: f32,
    eps: f32,
    lr: f32,
    /// Global step count (shared bias correction across all arrays).
    pub t: u64,
}

impl Adam {
    pub fn new(cfg: &Config) -> Adam {
        Adam { b1: cfg.adam_beta1, b2: cfg.adam_beta2, eps: cfg.adam_eps, lr: cfg.lr, t: 0 }
    }

    /// Update one parameter array in place. Call `begin_step` once per optimizer
    /// step *before* updating the arrays so bias correction uses a consistent `t`.
    pub fn update(&self, p: &mut [f32], g: &[f32], mom: &mut Moment) {
        let bc1 = 1.0 - self.b1.powi(self.t as i32);
        let bc2 = 1.0 - self.b2.powi(self.t as i32);
        for i in 0..p.len() {
            let gi = g[i];
            mom.m[i] = self.b1 * mom.m[i] + (1.0 - self.b1) * gi;
            mom.v[i] = self.b2 * mom.v[i] + (1.0 - self.b2) * gi * gi;
            let mhat = mom.m[i] / bc1;
            let vhat = mom.v[i] / bc2;
            p[i] -= self.lr * mhat / (vhat.sqrt() + self.eps);
        }
    }

    pub fn begin_step(&mut self) {
        self.t += 1;
    }
}

/// L2 norm across several gradient arrays treated as one long vector.
pub fn global_norm(arrays: &[&[f32]]) -> f32 {
    let mut sum = 0.0f64;
    for a in arrays {
        for &x in *a {
            sum += (x as f64) * (x as f64);
        }
    }
    sum.sqrt() as f32
}

/// Scale all gradient arrays by `max_norm / norm` if the global norm exceeds
/// `max_norm`. Returns the pre-clip norm (useful for logging).
pub fn clip_in_place(arrays: &mut [&mut [f32]], max_norm: f32) -> f32 {
    let norm = {
        let views: Vec<&[f32]> = arrays.iter().map(|a| &**a).collect();
        global_norm(&views)
    };
    if norm > max_norm && norm > 0.0 {
        let s = max_norm / norm;
        for a in arrays.iter_mut() {
            for x in a.iter_mut() {
                *x *= s;
            }
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_scales_down_large_grads() {
        let mut a = vec![3.0f32, 4.0]; // norm 5
        let mut refs: Vec<&mut [f32]> = vec![&mut a[..]];
        let n = clip_in_place(&mut refs, 1.0);
        assert!((n - 5.0).abs() < 1e-5);
        let new_norm = (a[0] * a[0] + a[1] * a[1]).sqrt();
        assert!((new_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn adam_decreases_a_quadratic() {
        // Minimize f(x) = x^2 from x0 = 5; gradient 2x.
        let mut cfg = Config::default();
        cfg.lr = 0.1;
        let mut adam = Adam::new(&cfg);
        let mut x = vec![5.0f32];
        let mut mom = Moment::zeros(1);
        for _ in 0..2000 {
            let g = vec![2.0 * x[0]];
            adam.begin_step();
            adam.update(&mut x, &g, &mut mom);
        }
        assert!(x[0].abs() < 1e-2, "x = {}", x[0]);
    }
}
