use anyhow::{Context, Result};
use std::collections::HashMap;

/// Simple character-level tokenizer compatible with existing ONNX models
#[derive(Clone, Debug)]
pub struct CharTokenizer {
    char2idx: HashMap<char, i64>,
    idx2char: HashMap<i64, char>,
    pub pad_id: i64,
    pub sos_id: i64,
    pub eos_id: i64,
    pub unk_id: i64,
}

impl CharTokenizer {
    pub fn from_maps(char2idx: HashMap<char, i64>, idx2char: HashMap<i64, char>) -> Self {
        Self {
            char2idx,
            idx2char,
            pad_id: 0,
            sos_id: 1,
            eos_id: 2,
            unk_id: 3,
        }
    }

    /// Load from a JSON vocab file like models/local/char2idx.json
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path).context("Failed to read vocab file")?;
        let raw: HashMap<String, i64> =
            serde_json::from_str(&content).context("Failed to parse vocab JSON")?;
        let char2idx: HashMap<char, i64> = raw
            .iter()
            .filter_map(|(k, v)| k.chars().next().map(|c| (c, *v)))
            .collect();
        let idx2char: HashMap<i64, char> = char2idx.iter().map(|(c, i)| (*i, *c)).collect();
        Ok(Self::from_maps(char2idx, idx2char))
    }

    /// Text -> token IDs with padding up to max_len
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<i64> {
        let mut ids: Vec<i64> = text
            .chars()
            .take(max_len)
            .map(|c| *self.char2idx.get(&c).unwrap_or(&self.unk_id))
            .collect();
        while ids.len() < max_len {
            ids.push(self.pad_id);
        }
        ids
    }

    /// Token IDs -> text, stop at EOS and skip PAD/SOS
    pub fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .take_while(|&&id| id != self.eos_id)
            .filter(|&&id| id > self.sos_id)
            .filter_map(|id| self.idx2char.get(id))
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.idx2char.len()
    }
}
