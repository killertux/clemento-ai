//! Corpus loading and the character-level vocabulary.
//!
//! A "token" is a single Unicode scalar (`char`). The vocabulary is built from
//! the unique characters present in the corpus. Training consumes the corpus as
//! one long stream of token ids, sliced into overlapping windows of `window + 1`
//! ids (the extra id is the final prediction target).

use std::collections::BTreeMap;

pub struct Corpus {
    /// The whole corpus as token ids.
    pub tokens: Vec<u32>,
    /// id -> char, indexed by token id.
    pub id_to_char: Vec<char>,
    /// char -> id.
    #[allow(dead_code)] // used by encode() and the test suite
    char_to_id: BTreeMap<char, u32>,
}

impl Corpus {
    /// Build a corpus (and its vocabulary) from raw text.
    pub fn from_text(text: &str) -> Corpus {
        let mut id_to_char: Vec<char> = text.chars().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        id_to_char.sort_unstable();
        let char_to_id: BTreeMap<char, u32> =
            id_to_char.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
        let tokens = text.chars().map(|c| char_to_id[&c]).collect();
        Corpus { tokens, id_to_char, char_to_id }
    }

    pub fn load(path: &str) -> std::io::Result<Corpus> {
        let text = std::fs::read_to_string(path)?;
        Ok(Corpus::from_text(&text))
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }

    #[allow(dead_code)] // public API; exercised by tests
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars().filter_map(|c| self.char_to_id.get(&c).copied()).collect()
    }

    #[allow(dead_code)] // public API; exercised by tests
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.id_to_char[i as usize]).collect()
    }

    /// Cross-entropy (nats) of the unigram model — the baseline that a trained
    /// model must beat. Equals the entropy of the empirical char distribution.
    pub fn unigram_cross_entropy(&self) -> f32 {
        let v = self.vocab_size();
        let mut counts = vec![0u64; v];
        for &t in &self.tokens {
            counts[t as usize] += 1;
        }
        let total = self.tokens.len() as f64;
        let mut h = 0.0f64;
        for &c in &counts {
            if c > 0 {
                let p = c as f64 / total;
                h -= p * p.ln();
            }
        }
        h as f32
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let c = Corpus::from_text("hello world\nhello");
        let ids = c.encode("hello");
        assert_eq!(c.decode(&ids), "hello");
    }

    #[test]
    fn vocab_is_unique_chars() {
        let c = Corpus::from_text("aabbc");
        assert_eq!(c.vocab_size(), 3); // a, b, c
    }

    #[test]
    fn unigram_entropy_of_uniform_is_ln_v() {
        // Each of 4 chars appears equally -> entropy = ln(4).
        let c = Corpus::from_text("abcd");
        let h = c.unigram_cross_entropy();
        assert!((h - (4f32).ln()).abs() < 1e-5, "h = {h}");
    }
}
