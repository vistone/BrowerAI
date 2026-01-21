#[cfg(feature = "ai")]
use anyhow::Result;

#[cfg(feature = "ai")]
use ndarray::Array2;

#[cfg(feature = "ai")]
use ort::{
    session::{input::SessionInputValue, Session},
    value::Value,
};

#[cfg(feature = "ai")]
use std::cmp::Ordering;

#[cfg(feature = "ai")]
use crate::tokenizer::CharTokenizer;

/// Parameters controlling beam search generation
#[derive(Clone, Debug)]
pub struct BeamSearchParams {
    pub max_len: usize,
    pub beam_size: usize,
    pub len_penalty: f32,
    pub no_repeat_ngram: usize,
    pub min_len: usize,
    pub top_expansion: usize,
}

impl Default for BeamSearchParams {
    fn default() -> Self {
        Self {
            max_len: 256,
            beam_size: 5,
            len_penalty: 1.0,
            no_repeat_ngram: 3,
            min_len: 80,
            top_expansion: 8,
        }
    }
}

#[cfg(feature = "ai")]
fn has_ngram_repeat(seq: &[i64], n: usize) -> bool {
    if n == 0 || seq.len() < n {
        return false;
    }
    let last = &seq[seq.len() - n..];
    for window in seq[..seq.len() - n].windows(n) {
        if window == last {
            return true;
        }
    }
    false
}

/// Run ONNX sequence generation with beam search over logits emitted per step.
#[cfg(feature = "ai")]
pub fn generate_with_beam_search(
    session: &mut Session,
    tokenizer: &CharTokenizer,
    source_code: &str,
    params: &BeamSearchParams,
) -> Result<String> {
    let max_len = params.max_len;

    // Encode source once
    let src_ids = tokenizer.encode(source_code, max_len);
    let src_ids_buf = src_ids.clone();

    // (generated_ids, score)
    let mut beams: Vec<(Vec<i64>, f32)> = vec![(Vec::new(), 0.0)];

    for step in 0..max_len {
        let mut candidates: Vec<(Vec<i64>, f32)> = Vec::new();

        for (seq, score) in &beams {
            if seq.last().copied() == Some(tokenizer.eos_id) {
                candidates.push((seq.clone(), *score));
                continue;
            }

            // tgt = SOS + seq + PAD
            let mut tgt_ids = vec![tokenizer.sos_id];
            tgt_ids.extend(seq.iter().copied());
            if tgt_ids.len() < max_len {
                tgt_ids.extend(std::iter::repeat_n(
                    tokenizer.pad_id,
                    max_len - tgt_ids.len(),
                ));
            }

            let tgt_array = Array2::from_shape_vec((1, max_len), tgt_ids)
                .map_err(|e| anyhow::anyhow!("ndarray shape error: {}", e))?;
            let (tgt_vec, _offset) = tgt_array.into_raw_vec_and_offset();
            let tgt_val: Value = Value::from_array((vec![1, max_len as i64], tgt_vec))?.into();
            let src_val: Value =
                Value::from_array((vec![1, max_len as i64], src_ids_buf.clone()))?.into();
            let inputs = [
                SessionInputValue::from(src_val),
                SessionInputValue::from(tgt_val),
            ];
            let outputs = session.run(inputs)?;
            let (_name, value) = outputs
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No outputs"))?;

            // step logits
            let (_shape, data) = value.try_extract_tensor::<f32>()?;
            if data.len() % max_len != 0 {
                return Err(anyhow::anyhow!(format!(
                    "Unexpected logits len: {}",
                    data.len()
                )));
            }
            let vocab_size = data.len() / max_len;
            let base = step * vocab_size;
            let slice = &data[base..base + vocab_size];

            // log-softmax
            let mut maxv = f32::MIN;
            for &v in slice.iter() {
                if v > maxv {
                    maxv = v;
                }
            }
            let mut sum_exp = 0.0f32;
            for &v in slice.iter() {
                sum_exp += (v - maxv).exp();
            }
            let lse = maxv + sum_exp.ln();

            // top-k candidates at this step (local)
            let mut top: Vec<(i64, f32)> = Vec::new();
            for (i, &logit) in slice.iter().enumerate() {
                if i >= vocab_size {
                    break;
                }
                let lp = logit - lse; // logprob
                                      // forbid early EOS
                if i as i64 == tokenizer.eos_id && step + 1 < params.min_len {
                    continue;
                }
                // skip PAD/SOS
                if i as i64 == tokenizer.pad_id || i as i64 == tokenizer.sos_id {
                    continue;
                }
                // simple repetition penalty
                let rp = if seq.len() >= 16 && seq[seq.len() - 16..].contains(&(i as i64)) {
                    0.9
                } else {
                    1.0
                };
                let adj = lp * rp;
                if top.len() < params.top_expansion {
                    top.push((i as i64, adj));
                } else {
                    let mut min_idx = 0usize;
                    for (ti, &(_, sc)) in top.iter().enumerate() {
                        if sc < top[min_idx].1 {
                            min_idx = ti;
                        }
                    }
                    if adj > top[min_idx].1 {
                        top[min_idx] = (i as i64, adj);
                    }
                }
            }

            for (tok, lp) in top.into_iter() {
                let mut new_seq = seq.clone();
                new_seq.push(tok);
                if params.no_repeat_ngram > 0 && has_ngram_repeat(&new_seq, params.no_repeat_ngram)
                {
                    continue;
                }
                let new_len = (new_seq.len() as f32).max(1.0);
                let len_norm = new_len.powf(params.len_penalty);
                let new_score = (*score + lp) / len_norm;
                candidates.push((new_seq, new_score));
            }
        }

        if !candidates.is_empty() {
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            beams = candidates.into_iter().take(params.beam_size).collect();
        }

        if beams
            .iter()
            .all(|(s, _)| s.last().copied() == Some(tokenizer.eos_id))
        {
            break;
        }
    }

    let best = beams
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .unwrap();
    Ok(tokenizer.decode(&best.0))
}
