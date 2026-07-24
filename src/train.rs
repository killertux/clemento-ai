//! Training loop (truncated BPTT + Adam, optional RigL) and text generation.

use crate::backprop::Grads;
use crate::checkpoint;
use crate::config::Config;
use crate::data::Corpus;
use crate::network::{softmax_ce, Network};
use crate::optim::{clip_in_place, Adam, Moment};
use crate::rigl;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Adam moment buffers for every trainable parameter array.
struct Moments {
    w: Moment,
    embed: Moment,
    readout: Moment,
    bias: Moment,
}

impl Moments {
    fn new(net: &Network) -> Moments {
        Moments {
            w: Moment::zeros(net.topo.num_edges()),
            embed: Moment::zeros(net.embed.len()),
            readout: Moment::zeros(net.readout.len()),
            bias: Moment::zeros(net.bias_out.len()),
        }
    }
}

pub struct TrainOpts {
    pub corpus_path: String,
    pub out_path: String,
    pub steps: usize,
    pub log_every: usize,
    pub checkpoint_every: usize,
    /// Fraction of the corpus held out (from the end) for validation.
    pub val_frac: f32,
}

/// Mean cross-entropy over held-out windows (forward only, no gradients). State
/// is carried across contiguous windows and reset at the start. Capped at
/// `max_windows` windows to keep evaluation cheap on large validation sets.
fn eval_loss(net: &Network, windows: &[&[u32]], max_windows: usize) -> Option<f32> {
    if windows.is_empty() {
        return None;
    }
    let count = windows.len().min(max_windows);
    let mut state = vec![0.0f32; net.cfg.n];
    let mut acc = 0.0f32;
    for w in windows.iter().take(count) {
        let tape = net.run_window(w, &state);
        acc += tape.loss;
        state = tape.state.last().unwrap().clone();
    }
    Some(acc / count as f32)
}

pub fn train(mut cfg: Config, opts: &TrainOpts) -> std::io::Result<()> {
    let corpus = Corpus::load(&opts.corpus_path)?;
    let vocab = corpus.vocab_size();
    let baseline = corpus.unigram_cross_entropy();
    println!(
        "corpus: {} tokens, vocab {}, unigram baseline {:.4} nats ({:.4} bits/char)",
        corpus.tokens.len(),
        vocab,
        baseline,
        baseline / std::f32::consts::LN_2
    );

    // Ensure the graph can actually be exercised by the data.
    if cfg.n_out < 1 || cfg.n_in < 1 {
        panic!("n_in and n_out must be >= 1");
    }
    let mut net = Network::new(cfg.clone(), vocab);
    let mut mom = Moments::new(&net);
    let mut adam = Adam::new(&cfg);

    // Split off the tail of the corpus for validation. The train/val boundary
    // is a single contiguous cut, so no window straddles both sets.
    let val_frac = opts.val_frac.clamp(0.0, 0.9);
    let cut = ((corpus.tokens.len() as f32) * (1.0 - val_frac)) as usize;
    let (train_tokens, val_tokens) = corpus.tokens.split_at(cut);
    let windows: Vec<&[u32]> = train_tokens.chunks_exact(cfg.window + 1).collect();
    let val_windows: Vec<&[u32]> = val_tokens.chunks_exact(cfg.window + 1).collect();
    if windows.is_empty() {
        panic!("corpus too small for window {}", cfg.window);
    }
    println!(
        "{} train / {} val windows of length {}",
        windows.len(),
        val_windows.len(),
        cfg.window + 1
    );
    if val_windows.is_empty() && val_frac > 0.0 {
        println!("  (warning: validation split too small for one window; val loss disabled)");
    }

    let mut state = vec![0.0f32; cfg.n];
    let mut loss_acc = 0.0f32;
    let mut active_acc = 0.0f32;
    let mut logged = 0usize;

    for step in 0..opts.steps {
        // Contiguous walk through the corpus; reset carried state at each wrap.
        let wi = step % windows.len();
        if wi == 0 {
            state.iter_mut().for_each(|s| *s = 0.0);
        }
        let seq = windows[wi];

        let tape = net.run_window(seq, &state);
        let grads = net.backward(&tape);

        // metrics
        loss_acc += tape.loss;
        let active: usize = tape.state.iter().map(|s| s.iter().filter(|&&v| v != 0.0).count()).sum();
        active_acc += active as f32 / (cfg.n as f32 * cfg.window as f32);

        apply_update(&mut net, &mut adam, &mut mom, &grads, &cfg);

        // RigL topology update
        if cfg.rigl_enabled && cfg.rigl_interval > 0 && step > 0 && step % cfg.rigl_interval == 0 {
            let churn = rigl::rigl_step(&mut net, &mut mom.w, &tape, &grads, adam.t);
            if step % opts.log_every == 0 {
                println!("  [rigl] step {step} churn {:.4}", churn);
            }
        }

        // carry final state to the next (contiguous) window
        state = tape.state.last().unwrap().clone();

        logged += 1;
        if step % opts.log_every == 0 && step > 0 {
            let avg_loss = loss_acc / logged as f32;
            let val = eval_loss(&net, &val_windows, 128);
            let val_str = match val {
                Some(v) => format!("val {v:.4}"),
                None => "val n/a".to_string(),
            };
            println!(
                "step {step:>7}  train {avg_loss:.4}  {val_str}  (baseline {baseline:.4})  active {:.1}%",
                100.0 * active_acc / logged as f32
            );
            loss_acc = 0.0;
            active_acc = 0.0;
            logged = 0;
        }

        if opts.checkpoint_every > 0 && step > 0 && step % opts.checkpoint_every == 0 {
            checkpoint::save(&opts.out_path, &net, &corpus.id_to_char)?;
        }
    }

    checkpoint::save(&opts.out_path, &net, &corpus.id_to_char)?;
    println!("saved checkpoint to {}", opts.out_path);
    let _ = &mut cfg; // cfg is owned; kept for clarity
    Ok(())
}

/// Clip gradients globally, then take one Adam step over every parameter array.
fn apply_update(net: &mut Network, adam: &mut Adam, mom: &mut Moments, grads: &Grads, cfg: &Config) {
    // Copy gradients into mutable buffers so we can clip them together.
    let mut g_w = grads.g_w.clone();
    let mut g_embed = grads.g_embed.clone();
    let mut g_readout = grads.g_readout.clone();
    let mut g_bias = grads.g_bias.clone();
    {
        let mut views: Vec<&mut [f32]> =
            vec![&mut g_w, &mut g_embed, &mut g_readout, &mut g_bias];
        clip_in_place(&mut views, cfg.grad_clip);
    }
    adam.begin_step();
    adam.update(&mut net.topo.weights, &g_w, &mut mom.w);
    adam.update(&mut net.embed, &g_embed, &mut mom.embed);
    adam.update(&mut net.readout, &g_readout, &mut mom.readout);
    adam.update(&mut net.bias_out, &g_bias, &mut mom.bias);
}

pub struct GenOpts {
    pub ckpt_path: String,
    pub prompt: String,
    pub length: usize,
    pub temperature: f32,
    pub seed: u64,
}

pub fn generate(opts: &GenOpts) -> std::io::Result<String> {
    let (net, id_to_char) = checkpoint::load(&opts.ckpt_path)?;
    let char_to_id: std::collections::HashMap<char, u32> =
        id_to_char.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();

    let mut rng = StdRng::seed_from_u64(opts.seed);
    let mut state = vec![0.0f32; net.cfg.n];
    let mut out = String::new();

    // Warm up the state on the prompt (feed each prompt token, keep its char).
    let mut last_tok = 0u32;
    for ch in opts.prompt.chars() {
        out.push(ch);
        if let Some(&tok) = char_to_id.get(&ch) {
            let (s, _) = net.step_once(tok, &state);
            state = s;
            last_tok = tok;
        }
    }

    // Autoregressive sampling.
    for _ in 0..opts.length {
        let (s, logits) = net.step_once(last_tok, &state);
        state = s;
        let tok = sample(&logits, opts.temperature, &mut rng);
        out.push(id_to_char[tok as usize]);
        last_tok = tok;
    }
    Ok(out)
}

/// Temperature sampling from logits.
fn sample(logits: &[f32], temperature: f32, rng: &mut StdRng) -> u32 {
    if temperature <= 1e-6 {
        // greedy
        let mut best = 0usize;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return best as u32;
    }
    let scaled: Vec<f32> = logits.iter().map(|&z| z / temperature).collect();
    let (probs, _) = softmax_ce(&scaled, 0);
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}
