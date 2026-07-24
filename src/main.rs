//! clemento-ai — a free-topology recurrent neural network (see README.md).

mod backprop;
mod checkpoint;
mod config;
mod data;
mod graph;
mod network;
mod optim;
mod rigl;
mod train;

use clap::{Parser, Subcommand};

use config::Config;
use data::Corpus;
use train::{GenOpts, TrainOpts};

#[derive(Parser)]
#[command(
    name = "clemento-ai",
    about = "Free-topology recurrent neural network",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect a corpus and print the unigram baseline loss.
    Stats {
        /// Path to the text corpus.
        #[arg(long, default_value = "data/input.txt")]
        data: String,
    },
    /// Train the network on a corpus.
    Train {
        /// Path to the text corpus.
        #[arg(long, default_value = "data/input.txt")]
        data: String,
        /// Where to write the trained model.
        #[arg(long, default_value = "model.bin")]
        out: String,
        /// Number of training steps (windows).
        #[arg(long, default_value_t = 10_000)]
        steps: usize,
        /// Log metrics every N steps.
        #[arg(long = "log-every", default_value_t = 100)]
        log_every: usize,
        /// Write a checkpoint every N steps (0 disables periodic checkpoints).
        #[arg(long = "ckpt-every", default_value_t = 1000)]
        ckpt_every: usize,

        /// Number of neurons.
        #[arg(long)]
        n: Option<usize>,
        /// Out-edges per neuron.
        #[arg(long)]
        k: Option<usize>,
        /// Truncated-BPTT window length.
        #[arg(long)]
        window: Option<usize>,
        /// Adam learning rate.
        #[arg(long)]
        lr: Option<f32>,
        /// PRNG seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Enable RigL topology learning (prune/grow).
        #[arg(long)]
        rigl: bool,
    },
    /// Generate text from a trained model.
    Generate {
        /// Path to the trained model.
        #[arg(long, default_value = "model.bin")]
        model: String,
        /// Seed text to prime the network.
        #[arg(long, default_value = "")]
        prompt: String,
        /// Number of characters to generate.
        #[arg(long = "len", default_value_t = 500)]
        length: usize,
        /// Sampling temperature (0 = greedy).
        #[arg(long = "temp", default_value_t = 0.8)]
        temperature: f32,
        /// PRNG seed for sampling.
        #[arg(long, default_value_t = 1234)]
        seed: u64,
    },
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Stats { data } => cmd_stats(&data),
        Command::Train {
            data,
            out,
            steps,
            log_every,
            ckpt_every,
            n,
            k,
            window,
            lr,
            seed,
            rigl,
        } => {
            let mut cfg = Config::default();
            if let Some(v) = n {
                cfg.n = v;
            }
            if let Some(v) = k {
                cfg.k = v;
            }
            if let Some(v) = window {
                cfg.window = v;
            }
            if let Some(v) = lr {
                cfg.lr = v;
            }
            if let Some(v) = seed {
                cfg.seed = v;
            }
            cfg.rigl_enabled = rigl;

            println!(
                "config: n={} k={} window={} lr={} rigl={}",
                cfg.n, cfg.k, cfg.window, cfg.lr, cfg.rigl_enabled
            );
            let opts = TrainOpts {
                corpus_path: data,
                out_path: out,
                steps,
                log_every,
                checkpoint_every: ckpt_every,
            };
            train::train(cfg, &opts)
        }
        Command::Generate { model, prompt, length, temperature, seed } => {
            let opts = GenOpts { ckpt_path: model, prompt, length, temperature, seed };
            train::generate(&opts).map(|text| println!("{text}"))
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn cmd_stats(data: &str) -> std::io::Result<()> {
    let corpus = Corpus::load(data)?;
    let h = corpus.unigram_cross_entropy();
    println!("file: {data}");
    println!("tokens: {}", corpus.tokens.len());
    println!("vocab size: {}", corpus.vocab_size());
    println!(
        "unigram cross-entropy: {:.4} nats ({:.4} bits/char)",
        h,
        h / std::f32::consts::LN_2
    );
    Ok(())
}
