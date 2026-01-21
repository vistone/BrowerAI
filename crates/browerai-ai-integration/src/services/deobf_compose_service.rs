use anyhow::Result;
use std::path::PathBuf;
#[cfg(feature = "ai")]
use std::time::Instant;

#[cfg(feature = "ai")]
use browerai_ai_core::tech_model_library::{
    GenerateTask, LayoutRegeneration, ModelRegistry, TaskKind,
};

#[cfg(feature = "ai")]
use crate::decoder::beam_search::{generate_with_beam_search, BeamSearchParams};
#[cfg(feature = "ai")]
use crate::tokenizer::CharTokenizer;

/// Configuration for deobfuscation composition
#[derive(Clone, Debug)]
pub struct DeobfComposeConfig {
    /// Minimum acceptable generated length; otherwise fallback to original
    pub min_generated_len: usize,
    /// If ratio of control characters exceeds this, fallback
    pub max_non_print_ratio: f32,
    /// Beam search overrides
    pub beam_size: Option<usize>,
    pub top_expansion: Option<usize>,
    pub len_penalty: Option<f32>,
    pub no_repeat_ngram: Option<usize>,
    pub min_len_tokens: Option<usize>,
    pub max_len_tokens: Option<usize>,
    /// Optional vocab file path (defaults to models/local/char2idx.json)
    pub vocab_path: Option<String>,
    /// Optional explicit model path; registry will be used when None
    pub model_path: Option<PathBuf>,
}

impl Default for DeobfComposeConfig {
    fn default() -> Self {
        Self {
            min_generated_len: 20,
            max_non_print_ratio: 0.2,
            beam_size: None,
            top_expansion: None,
            len_penalty: None,
            no_repeat_ngram: None,
            min_len_tokens: None,
            max_len_tokens: None,
            vocab_path: Some("models/local/char2idx.json".to_string()),
            model_path: None,
        }
    }
}

/// Deobfuscation + composition service (AI + safe fallback)
pub struct DeobfComposeService {
    #[cfg(feature = "ai")]
    session: ort::session::Session,
    #[cfg(feature = "ai")]
    tokenizer: CharTokenizer,
    #[cfg(feature = "ai")]
    beam: BeamSearchParams,
    /// Model name/path for feedback logging
    #[cfg(feature = "ai")]
    model_name: String,
    /// Optional feedback pipeline for telemetry
    #[cfg(feature = "ai")]
    feedback: Option<browerai_ai_core::feedback_pipeline::FeedbackPipeline>,
    #[cfg_attr(not(feature = "ai"), allow(dead_code))]
    cfg: DeobfComposeConfig,
}

impl DeobfComposeService {
    /// Create the service by loading from ModelRegistry (task: DeobfuscateJs).
    #[cfg(feature = "ai")]
    pub fn from_registry(cfg: DeobfComposeConfig) -> Result<Self> {
        // Resolve model path
        let resolved_model: PathBuf = if let Some(mp) = &cfg.model_path {
            mp.clone()
        } else {
            let reg = ModelRegistry::load_from("models/model_config.toml").unwrap_or_default();
            let mut selected: Option<PathBuf> = None;
            if let Some(spec) = reg.preferred(TaskKind::DeobfuscateJs) {
                if let Some(p) = spec.path.to_str() {
                    let p = if p.contains('/') || p.contains('\\') {
                        p.to_string()
                    } else {
                        format!("models/local/{}", p)
                    };
                    let pb = PathBuf::from(&p);
                    if pb.exists() {
                        selected = Some(pb);
                    }
                }
            }
            selected.unwrap_or_else(|| PathBuf::from("models/local/website_deobf_model.onnx"))
        };

        // Load session via ORT directly (consistent with examples)
        let session = ort::session::Session::builder()?.commit_from_file(&resolved_model)?;

        // Load tokenizer
        let vocab = cfg
            .vocab_path
            .clone()
            .unwrap_or_else(|| "models/local/char2idx.json".to_string());
        let tokenizer = CharTokenizer::load_from_file(&vocab)?;

        // Build beam params with overrides
        let mut beam = BeamSearchParams::default();
        if let Some(v) = cfg.beam_size {
            beam.beam_size = v;
        }
        if let Some(v) = cfg.top_expansion {
            beam.top_expansion = v;
        }
        if let Some(v) = cfg.len_penalty {
            beam.len_penalty = v;
        }
        if let Some(v) = cfg.no_repeat_ngram {
            beam.no_repeat_ngram = v;
        }
        if let Some(v) = cfg.min_len_tokens {
            beam.min_len = v;
        }
        if let Some(v) = cfg.max_len_tokens {
            beam.max_len = v;
        }

        let model_name = resolved_model.to_string_lossy().to_string();

        Ok(Self {
            session,
            tokenizer,
            beam,
            model_name,
            feedback: None,
            cfg,
        })
    }

    /// Create the service by loading from ModelRegistry with feedback pipeline wired.
    #[cfg(feature = "ai")]
    pub fn from_registry_with_feedback(
        cfg: DeobfComposeConfig,
        feedback: browerai_ai_core::feedback_pipeline::FeedbackPipeline,
    ) -> Result<Self> {
        let mut s = Self::from_registry(cfg)?;
        s.feedback = Some(feedback);
        Ok(s)
    }

    /// Compose a snippet using AI generation with safe fallback to original text.
    #[cfg(feature = "ai")]
    pub fn compose(&mut self, snippet: &str) -> Result<String> {
        // Run generation with timing
        let t0 = Instant::now();
        let gen_res =
            generate_with_beam_search(&mut self.session, &self.tokenizer, snippet, &self.beam);
        let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let gen = match gen_res {
            Ok(s) => {
                if let Some(fb) = &self.feedback {
                    fb.record_model_inference(self.model_name.clone(), true, dur_ms, None);
                }
                s
            }
            Err(e) => {
                if let Some(fb) = &self.feedback {
                    fb.record_model_inference(
                        self.model_name.clone(),
                        false,
                        dur_ms,
                        Some(e.to_string()),
                    );
                }
                // On failure, fallback to original text composition directly
                let generator = LayoutRegeneration;
                let composed = generator.generate_layout(snippet, None);
                return Ok(composed);
            }
        };
        // Quality gate
        let _non_print = gen.chars().filter(|c| c.is_control()).count();
        let fallback = should_fallback(
            &gen,
            self.cfg.min_generated_len,
            self.cfg.max_non_print_ratio,
        );
        let layout_hint = if fallback { None } else { Some(gen.as_str()) };
        let generator = LayoutRegeneration;
        let composed = generator.generate_layout(snippet, layout_hint);
        Ok(composed)
    }

    #[cfg(not(feature = "ai"))]
    pub fn from_registry(_cfg: DeobfComposeConfig) -> Result<Self> {
        Err(anyhow::anyhow!("AI feature disabled"))
    }

    #[cfg(not(feature = "ai"))]
    pub fn compose(&mut self, _snippet: &str) -> Result<String> {
        Err(anyhow::anyhow!("AI feature disabled"))
    }
}

/// Decide whether to fallback based on length and non-printing ratio
#[cfg_attr(not(feature = "ai"), allow(dead_code))]
fn should_fallback(generated: &str, min_len: usize, max_non_print_ratio: f32) -> bool {
    if generated.len() < min_len {
        return true;
    }
    if generated.is_empty() {
        return true;
    }
    let non_print = generated.chars().filter(|c| c.is_control()).count();
    let ratio = non_print as f32 / (generated.len().max(1) as f32);
    ratio > max_non_print_ratio
}

#[cfg(test)]
mod tests {
    use super::should_fallback;

    #[test]
    fn test_should_fallback_short() {
        assert!(should_fallback("abc", 10, 0.2));
        assert!(!should_fallback(&"a".repeat(15), 10, 0.2));
    }

    #[test]
    fn test_should_fallback_nonprint() {
        let s = "a".repeat(100) + &"\u{0001}".repeat(30); // 23% 控制字符
        assert!(should_fallback(&s, 10, 0.2));
        let s2 = "a".repeat(100) + &"\u{0001}".repeat(10); // 9%
        assert!(!should_fallback(&s2, 10, 0.2));
    }
}
