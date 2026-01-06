#[cfg(feature = "candle")]
use anyhow::{Context, Result};
#[cfg(feature = "candle")]
use candle_core::quantized::gguf_file;
#[cfg(feature = "candle")]
use candle_core::Device;
#[cfg(feature = "candle")]
use candle_transformers::models::quantized_qwen2::ModelWeights;
#[cfg(feature = "candle")]
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    utils::apply_repeat_penalty,
};
#[cfg(feature = "candle")]
use std::fs::File;
#[cfg(feature = "candle")]
use std::path::Path;
#[cfg(feature = "candle")]
use tokenizers::Tokenizer;

/// Minimal GGUF loader helpers for Candle-backed LLMs (Qwen2/Qwen2.5 GGUF).
///
/// This is intentionally lightweight: we only validate files, pick a device,
/// and rely on Candle's quantized Qwen2 loader to bring the weights in. Callers
/// can wrap the returned `ModelWeights` into their own generation pipeline.
#[cfg(feature = "candle")]
pub struct CandleModelLoader;

/// Simple wrapper around Candle quantized LLaMA/Qwen weights + tokenizer for generation.
#[cfg(feature = "candle")]
pub struct CandleCodeLlm {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    eos_token: Option<u32>,
}

#[cfg(feature = "candle")]
impl CandleModelLoader {
    /// Pick a device. Prefer CUDA:0 if available, otherwise fall back to CPU.
    pub fn device(prefer_gpu: bool) -> Device {
        if prefer_gpu {
            log::info!("Attempting to use CUDA GPU (device 0)...");
            match Device::new_cuda(0) {
                Ok(dev) => {
                    log::info!("✓ Successfully using CUDA GPU");
                    dev
                }
                Err(e) => {
                    log::warn!("✗ CUDA initialization failed: {}. Falling back to CPU.", e);
                    Device::Cpu
                }
            }
        } else {
            log::info!("Using CPU (PREFER_GPU=0 or not set)");
            Device::Cpu
        }
    }

    /// Load quantized GGUF weights (Qwen2-style) for code LLMs.
    ///
    /// Note: Qwen2.5-Coder-7B GGUF exposes `qwen2.*` metadata keys; we use the
    /// dedicated quantized_qwen2 loader in Candle.
    pub fn load_gguf_weights(model_path: &Path, device: &Device) -> Result<ModelWeights> {
        if !model_path.exists() {
            anyhow::bail!("Model file not found: {:?}", model_path);
        }

        let mut file = File::open(model_path)
            .with_context(|| format!("Failed to open model file: {:?}", model_path))?;

        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to read GGUF header from {:?}", model_path))?;

        // Build weights from GGUF content + reader (new Candle API)
        let weights = ModelWeights::from_gguf(content, &mut file, device)
            .with_context(|| format!("Failed to load GGUF weights from {:?}", model_path))?;

        Ok(weights)
    }

    /// Load a tokenizer JSON that accompanies the GGUF checkpoint.
    pub fn load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer> {
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer file not found: {:?}", tokenizer_path);
        }

        Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))
    }
}

#[cfg(feature = "candle")]
impl CandleCodeLlm {
    /// Build an inference-ready code LLM from GGUF weights and tokenizer.
    pub fn new(model_path: &Path, tokenizer_path: &Path, prefer_gpu: bool) -> Result<Self> {
        let device = CandleModelLoader::device(prefer_gpu);
        log::info!("Device selected: {:?}", device);
        log::info!("Loading GGUF weights from {:?}...", model_path);
        let model = CandleModelLoader::load_gguf_weights(model_path, &device)?;
        log::info!("Loading tokenizer from {:?}...", tokenizer_path);
        let tokenizer = CandleModelLoader::load_tokenizer(tokenizer_path)?;

        // Try common EOS tokens; fall back to tokenizer's default if present.
        let vocab = tokenizer.get_vocab(true);
        let eos_token = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .find_map(|tok| vocab.get(*tok).copied());

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token,
        })
    }

    /// Greedy / top-k / top-p generation (single batch) with repeat penalty.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        repeat_penalty: f64,
        repeat_last_n: usize,
    ) -> Result<String> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = tokens.len();

        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match (top_k, top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(42, sampling);

        // First step: feed full prompt to prime KV cache and sample next token
        let input = candle_core::Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?.squeeze(0)?;
        let mut next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);

        // Generate remaining tokens
        for idx in 0..max_new_tokens.saturating_sub(1) {
            let input = candle_core::Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let mut logits = self.model.forward(&input, prompt_len + idx)?.squeeze(0)?;

            if repeat_penalty != 1.0 {
                let start_at = tokens.len().saturating_sub(repeat_last_n);
                logits = apply_repeat_penalty(&logits, repeat_penalty as f32, &tokens[start_at..])?;
            }

            next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);

            if let Some(eos) = self.eos_token {
                if next_token == eos {
                    break;
                }
            }
        }

        // Decode full sequence; callers can trim the prompt prefix if需要
        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;
        Ok(text)
    }
}
