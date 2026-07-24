//! Minimal binary (de)serialization of a trained model: config, vocabulary,
//! topology, and the dense in/out parameters. Little-endian, hand-rolled — no
//! serde dependency.

use std::io::{self, Read, Write};

use crate::config::{Config, GateMode};
use crate::graph::Topology;
use crate::network::Network;

const MAGIC: &[u8; 4] = b"CLAI";
const VERSION: u32 = 1;

pub fn save(path: &str, net: &Network, id_to_char: &[char]) -> io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);
    put_u32(&mut buf, VERSION);

    // config (scalars we need to reconstruct + resume)
    let c = &net.cfg;
    for v in [c.n, c.k, c.n_in, c.n_out, c.window, c.rigl_interval, c.rigl_candidates, c.rigl_decay_steps] {
        put_u32(&mut buf, v as u32);
    }
    for v in [c.theta, c.beta, c.spectral_radius, c.lr, c.adam_beta1, c.adam_beta2, c.adam_eps, c.grad_clip, c.rigl_fraction] {
        put_f32(&mut buf, v);
    }
    put_u32(&mut buf, matches!(c.gate, GateMode::Hard) as u32);
    put_u32(&mut buf, c.rigl_enabled as u32);
    put_u64(&mut buf, c.seed);

    // vocabulary
    put_u32(&mut buf, id_to_char.len() as u32);
    for &ch in id_to_char {
        put_u32(&mut buf, ch as u32);
    }

    // topology
    put_u32(&mut buf, net.topo.num_edges() as u32);
    put_u32_slice(&mut buf, &net.topo.row_ptr);
    put_u32_slice(&mut buf, &net.topo.col_idx);
    put_f32_slice(&mut buf, &net.topo.weights);

    // dense params
    put_f32_slice(&mut buf, &net.embed);
    put_f32_slice(&mut buf, &net.readout);
    put_f32_slice(&mut buf, &net.bias_out);

    let mut f = std::fs::File::create(path)?;
    f.write_all(&buf)
}

pub fn load(path: &str) -> io::Result<(Network, Vec<char>)> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let mut r = Reader { buf: &buf, pos: 0 };

    let magic = r.take(4);
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let _version = r.u32();

    let n = r.u32() as usize;
    let k = r.u32() as usize;
    let n_in = r.u32() as usize;
    let n_out = r.u32() as usize;
    let window = r.u32() as usize;
    let rigl_interval = r.u32() as usize;
    let rigl_candidates = r.u32() as usize;
    let rigl_decay_steps = r.u32() as usize;

    let theta = r.f32();
    let beta = r.f32();
    let spectral_radius = r.f32();
    let lr = r.f32();
    let adam_beta1 = r.f32();
    let adam_beta2 = r.f32();
    let adam_eps = r.f32();
    let grad_clip = r.f32();
    let rigl_fraction = r.f32();

    let gate = if r.u32() == 1 { GateMode::Hard } else { GateMode::Soft };
    let rigl_enabled = r.u32() == 1;
    let seed = r.u64();

    let vocab = r.u32() as usize;
    let id_to_char: Vec<char> =
        (0..vocab).map(|_| char::from_u32(r.u32()).unwrap_or('\u{FFFD}')).collect();

    let n_edges = r.u32() as usize;
    let row_ptr = r.u32_vec(n + 1);
    let col_idx = r.u32_vec(n_edges);
    let weights = r.f32_vec(n_edges);

    let embed = r.f32_vec(vocab * n_in);
    let readout = r.f32_vec(n_out * vocab);
    let bias_out = r.f32_vec(vocab);

    let cfg = Config {
        n, k, n_in, n_out, window, theta, beta, gate, spectral_radius, lr,
        adam_beta1, adam_beta2, adam_eps, grad_clip,
        rigl_interval, rigl_fraction, rigl_candidates, rigl_decay_steps, rigl_enabled,
        seed,
    };
    let topo = Topology::from_csr(n, row_ptr, col_idx, weights);
    let net = Network { cfg, topo, embed, readout, bias_out, vocab };
    Ok((net, id_to_char))
}

// --- little-endian helpers --------------------------------------------------

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u32_slice(b: &mut Vec<u8>, s: &[u32]) {
    for &v in s {
        put_u32(b, v);
    }
}
fn put_f32_slice(b: &mut Vec<u8>, s: &[f32]) {
    for &v in s {
        put_f32(b, v);
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u32_vec(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.u32()).collect()
    }
    fn f32_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_roundtrip() {
        let cfg = Config::tiny();
        let net = Network::new(cfg, 6);
        let vocab: Vec<char> = "abcdef".chars().collect();
        let dir = std::env::temp_dir();
        let path = dir.join("clai_ckpt_test.bin");
        let path = path.to_str().unwrap();
        save(path, &net, &vocab).unwrap();
        let (net2, v2) = load(path).unwrap();
        assert_eq!(net2.cfg.n, net.cfg.n);
        assert_eq!(v2, vocab);
        assert_eq!(net2.topo.weights, net.topo.weights);
        assert_eq!(net2.embed, net.embed);
        let _ = std::fs::remove_file(path);
    }
}
