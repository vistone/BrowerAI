#[cfg(feature = "ai")]
use anyhow::Context;
use anyhow::Result;
use std::collections::HashMap;
use std::mem::size_of;
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "ai")]
use ort::{session::input::SessionInputValue, session::Session, value::Value};

use super::InferenceEngine;
use browerai_ai_core::performance_monitor::{InferenceMetrics, PerformanceMonitor};

/// Number of features expected by the code understanding model
const CODE_UNDERSTANDING_FEATURE_DIM: usize = 35;
/// Default output dimension for the code understanding model (site/category logits)
const CODE_UNDERSTANDING_OUTPUT_DIM: usize = 10;

/// Maximum sequence length for JS deobfuscator (must match ONNX model)
const JS_DEOBFUSCATOR_MAX_LEN: usize = 60;
/// Special token IDs
const PAD_ID: i64 = 0;
const SOS_ID: i64 = 1;
const EOS_ID: i64 = 2;
const UNK_ID: i64 = 3;

/// Model integration helper for HTML parsing
pub struct HtmlModelIntegration {
    #[cfg(feature = "ai")]
    session: Option<Session>,
    #[cfg_attr(not(feature = "ai"), allow(dead_code))]
    monitor: Option<PerformanceMonitor>,
    enabled: bool,
}

impl HtmlModelIntegration {
    /// Create a new HTML model integration
    pub fn new(
        engine: &InferenceEngine,
        model_path: Option<&Path>,
        monitor: Option<PerformanceMonitor>,
    ) -> Result<Self> {
        #[cfg(feature = "ai")]
        {
            let session = if let Some(path) = model_path {
                if path.exists() {
                    Some(engine.load_model(path)?)
                } else {
                    log::warn!("HTML model not found at {:?}, running without AI", path);
                    None
                }
            } else {
                None
            };

            let enabled = session.is_some();

            Ok(Self {
                session,
                monitor,
                enabled,
            })
        }

        #[cfg(not(feature = "ai"))]
        {
            let _ = (engine, model_path, monitor);
            Ok(Self {
                enabled: false,
                monitor: None,
            })
        }
    }

    /// Validate HTML structure using AI model
    #[cfg(feature = "ai")]
    pub fn validate_structure(&mut self, html: &str) -> Result<(bool, f32)> {
        let start = Instant::now();

        if !self.enabled || self.session.is_none() {
            if let Some(m) = &self.monitor {
                m.record_inference(InferenceMetrics {
                    model_name: "html_model".to_string(),
                    inference_time: start.elapsed(),
                    input_size: html.len(),
                    output_size: 2 * size_of::<f32>(),
                    success: false,
                    timestamp: start,
                });
            }
            return Ok((true, 0.5)); // Fallback: assume valid, medium complexity
        }

        // Tokenize HTML (simple character-level tokenization)
        let tokens = self.tokenize_html(html, 100); // Changed from 512 to 100 to match model

        let session = self.session.as_mut().unwrap();

        // Create input tensor with f32 (matching model training)
        let input_shape = vec![1, 100];
        let input_data: Vec<f32> = tokens.iter().map(|&x| x as f32).collect();

        let input_tensor = Value::from_array((input_shape.clone(), input_data.clone()))
            .context("Failed to create input tensor")?;

        // Run inference - wrap in SessionInputValue and pass as array
        let inputs = [SessionInputValue::from(input_tensor)];
        let outputs = session.run(inputs).map_err(|e| {
            log::error!("ONNX inference detailed error: {:?}", e);
            anyhow::anyhow!("Failed to run inference: {}", e)
        })?;

        // Parse outputs - extract as 1D array first, then reshape
        let output_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        let output_data: Vec<f32> = output_tensor.1.to_vec();

        if output_data.len() < 2 {
            log::warn!(
                "Unexpected output size: {}, expected at least 2",
                output_data.len()
            );
            return Ok((true, 0.5)); // Fallback
        }

        let validity = output_data[0] > 0.5;
        let complexity = output_data[1];

        log::debug!(
            "HTML validation: valid={}, complexity={}",
            validity,
            complexity
        );

        if let Some(m) = &self.monitor {
            // Record a synthetic metrics entry to keep observability consistent
            m.record_inference(InferenceMetrics {
                model_name: "html_model".to_string(),
                inference_time: start.elapsed(),
                input_size: input_data.len() * size_of::<i64>(),
                output_size: 2 * size_of::<f32>(),
                success: true,
                timestamp: start,
            });
        }

        Ok((validity, complexity))
    }

    #[cfg(not(feature = "ai"))]
    pub fn validate_structure(&mut self, _html: &str) -> Result<(bool, f32)> {
        Ok((true, 0.5))
    }

    /// Tokenize HTML to indices
    #[allow(dead_code)]
    fn tokenize_html(&self, html: &str, max_length: usize) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(max_length);

        for ch in html.chars().take(max_length) {
            // Simple character encoding
            tokens.push(ch as u32 % 256);
        }

        // Pad to max_length
        while tokens.len() < max_length {
            tokens.push(0);
        }

        tokens
    }

    /// Check if AI enhancement is enabled
    #[allow(dead_code)]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Model integration helper for CSS parsing
pub struct CssModelIntegration {
    #[cfg(feature = "ai")]
    #[allow(dead_code)]
    session: Option<Session>,
    #[cfg_attr(not(feature = "ai"), allow(dead_code))]
    monitor: Option<PerformanceMonitor>,
    enabled: bool,
}

impl CssModelIntegration {
    #[allow(dead_code)]
    pub fn new(
        engine: &InferenceEngine,
        model_path: Option<&Path>,
        monitor: Option<PerformanceMonitor>,
    ) -> Result<Self> {
        #[cfg(feature = "ai")]
        {
            let session = if let Some(path) = model_path {
                if path.exists() {
                    Some(engine.load_model(path)?)
                } else {
                    log::warn!("CSS model not found at {:?}, running without AI", path);
                    None
                }
            } else {
                None
            };

            let enabled = session.is_some();

            Ok(Self {
                session,
                monitor,
                enabled,
            })
        }

        #[cfg(not(feature = "ai"))]
        {
            let _ = (engine, model_path, monitor);
            Ok(Self {
                enabled: false,
                monitor: None,
            })
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Optimize CSS rules using AI
    #[allow(dead_code)]
    pub fn optimize_rules(&mut self, css: &str) -> Result<Vec<String>> {
        let start = Instant::now();

        // Placeholder: return original CSS split by rules
        let rules: Vec<String> = css
            .split('}')
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                let mut rule = s.trim().to_string();
                rule.push_str("},");
                rule
            })
            .collect();

        if let Some(m) = &self.monitor {
            m.record_inference(InferenceMetrics {
                model_name: "css_model".to_string(),
                inference_time: start.elapsed(),
                input_size: css.len(),
                output_size: rules.iter().map(|r| r.len()).sum(),
                success: self.enabled,
                timestamp: start,
            });
        }

        Ok(rules)
    }
}

/// Model integration helper for JavaScript parsing
pub struct JsModelIntegration {
    #[cfg(feature = "ai")]
    #[allow(dead_code)]
    session: Option<Session>,
    #[cfg_attr(not(feature = "ai"), allow(dead_code))]
    monitor: Option<PerformanceMonitor>,
    enabled: bool,
}

impl JsModelIntegration {
    #[allow(dead_code)]
    pub fn new(
        engine: &InferenceEngine,
        model_path: Option<&Path>,
        monitor: Option<PerformanceMonitor>,
    ) -> Result<Self> {
        #[cfg(feature = "ai")]
        {
            let session = if let Some(path) = model_path {
                if path.exists() {
                    Some(engine.load_model(path)?)
                } else {
                    log::warn!("JS model not found at {:?}, running without AI", path);
                    None
                }
            } else {
                None
            };

            let enabled = session.is_some();

            Ok(Self {
                session,
                monitor,
                enabled,
            })
        }

        #[cfg(not(feature = "ai"))]
        {
            let _ = (engine, model_path, monitor);
            Ok(Self {
                enabled: false,
                monitor: None,
            })
        }
    }

    #[allow(dead_code)]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Analyze JavaScript code patterns
    #[allow(dead_code)]
    pub fn analyze_patterns(&self, _js: &str) -> Result<Vec<String>> {
        let start = Instant::now();
        // Placeholder: return basic patterns
        let patterns = ["function_declaration", "variable_assignment"];

        if let Some(m) = &self.monitor {
            m.record_inference(InferenceMetrics {
                model_name: "js_model".to_string(),
                inference_time: start.elapsed(),
                input_size: _js.len(),
                output_size: patterns.len(),
                success: self.enabled,
                timestamp: start,
            });
        }

        Ok(patterns.iter().map(|s| s.to_string()).collect())
    }
}

/// Model integration helper for code understanding / site classification
pub struct CodeUnderstandingIntegration {
    #[cfg(feature = "ai")]
    session: Option<Session>,
    #[cfg_attr(not(feature = "ai"), allow(dead_code))]
    monitor: Option<PerformanceMonitor>,
    enabled: bool,
}

impl CodeUnderstandingIntegration {
    #[allow(dead_code)]
    pub fn new(
        engine: &InferenceEngine,
        model_path: Option<&Path>,
        monitor: Option<PerformanceMonitor>,
    ) -> Result<Self> {
        #[cfg(feature = "ai")]
        {
            let session = if let Some(path) = model_path {
                if path.exists() {
                    Some(engine.load_model(path)?)
                } else {
                    log::warn!(
                        "Code understanding model not found at {:?}, running without AI",
                        path
                    );
                    None
                }
            } else {
                None
            };

            let enabled = session.is_some();

            Ok(Self {
                session,
                monitor,
                enabled,
            })
        }

        #[cfg(not(feature = "ai"))]
        {
            let _ = (engine, model_path, monitor);
            Ok(Self {
                enabled: false,
                monitor: None,
            })
        }
    }

    #[allow(dead_code)]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Classify a page using the 35-dim feature vector; returns logits for categories
    #[allow(dead_code)]
    pub fn classify(&mut self, features: &[f32]) -> Result<Vec<f32>> {
        let start = Instant::now();

        // Fallback path when AI is disabled
        if !self.enabled {
            if let Some(m) = &self.monitor {
                m.record_inference(InferenceMetrics {
                    model_name: "code_understanding".to_string(),
                    inference_time: start.elapsed(),
                    input_size: std::mem::size_of_val(features),
                    output_size: CODE_UNDERSTANDING_OUTPUT_DIM * size_of::<f32>(),
                    success: false,
                    timestamp: start,
                });
            }
            return Ok(vec![0.0; CODE_UNDERSTANDING_OUTPUT_DIM]);
        }

        // Pad or truncate to fixed length expected by ONNX model
        let mut input_data: Vec<f32> = features.to_vec();
        input_data.truncate(CODE_UNDERSTANDING_FEATURE_DIM);
        while input_data.len() < CODE_UNDERSTANDING_FEATURE_DIM {
            input_data.push(0.0);
        }

        let shape = vec![1, CODE_UNDERSTANDING_FEATURE_DIM as i64];

        #[cfg(feature = "ai")]
        {
            let session = self
                .session
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Code understanding model session missing"))?;

            let input_tensor = Value::from_array((shape.clone(), input_data.clone()))
                .context("Failed to create input tensor")?;

            let outputs = session
                .run([SessionInputValue::from(input_tensor)])
                .map_err(|e| anyhow::anyhow!("Failed to run inference: {}", e))?;

            let output_tensor = outputs[0]
                .try_extract_tensor::<f32>()
                .context("Failed to extract output tensor")?;

            let output_data: Vec<f32> = output_tensor.1.to_vec();

            if let Some(m) = &self.monitor {
                m.record_inference(InferenceMetrics {
                    model_name: "code_understanding".to_string(),
                    inference_time: start.elapsed(),
                    input_size: input_data.len() * size_of::<f32>(),
                    output_size: output_data.len() * size_of::<f32>(),
                    success: true,
                    timestamp: start,
                });
            }

            Ok(output_data)
        }

        #[cfg(not(feature = "ai"))]
        {
            let _ = shape;
            Ok(vec![0.0; CODE_UNDERSTANDING_OUTPUT_DIM])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use browerai_ai_core::InferenceEngine;

    #[test]
    fn test_html_integration_creation() {
        let engine = InferenceEngine::new().unwrap();
        let integration = HtmlModelIntegration::new(&engine, None, None);
        assert!(integration.is_ok());
    }

    #[test]
    fn test_html_validation_fallback() {
        let engine = InferenceEngine::new().unwrap();
        let mut integration = HtmlModelIntegration::new(&engine, None, None).unwrap();
        let (valid, complexity) = integration
            .validate_structure("<html><body>Test</body></html>")
            .unwrap();
        assert!(valid); // Should fallback to valid
        assert!((0.0..=1.0).contains(&complexity));
    }

    #[test]
    fn test_css_integration_creation() {
        let engine = InferenceEngine::new().unwrap();
        let integration = CssModelIntegration::new(&engine, None, None);
        assert!(integration.is_ok());
    }

    #[test]
    fn test_js_integration_creation() {
        let engine = InferenceEngine::new().unwrap();
        let integration = JsModelIntegration::new(&engine, None, None);
        assert!(integration.is_ok());
    }
}

/// JavaScript tokenizer for deobfuscator model
pub struct JsTokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, i64>,
}

impl Default for JsTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl JsTokenizer {
    pub fn new() -> Self {
        let vocab = Self::build_vocab();
        let token_to_id: HashMap<String, i64> = vocab
            .iter()
            .enumerate()
            .map(|(i, token)| (token.clone(), i as i64))
            .collect();

        Self { vocab, token_to_id }
    }

    fn build_vocab() -> Vec<String> {
        vec![
            // Special tokens
            "<PAD>",
            "<SOS>",
            "<EOS>",
            "<UNK>",
            // Keywords
            "as",
            "async",
            "await",
            "break",
            "case",
            "catch",
            "class",
            "const",
            "constructor",
            "continue",
            "default",
            "do",
            "else",
            "export",
            "extends",
            "finally",
            "for",
            "from",
            "function",
            "if",
            "import",
            "let",
            "new",
            "promise",
            "return",
            "super",
            "switch",
            "then",
            "this",
            "try",
            "var",
            "while",
            // Operators
            "!",
            "!=",
            "!==",
            "%",
            "&",
            "&&",
            "*",
            "**",
            "+",
            "++",
            ",",
            "-",
            "--",
            ".",
            "/",
            ":",
            ";",
            "<",
            "<<",
            "<=",
            "=",
            "==",
            "===",
            "=>",
            ">",
            ">=",
            ">>",
            "?",
            "^",
            "|",
            "||",
            "~",
            "(",
            ")",
            "{",
            "}",
            "[",
            "]",
            // Variables
            "var0",
            "var1",
            "var2",
            "var3",
            "var4",
            "var5",
            "var6",
            "var7",
            "var8",
            "var9",
            "tmp0",
            "tmp1",
            "tmp2",
            "tmp3",
            "tmp4",
            "tmp5",
            "tmp6",
            "tmp7",
            "tmp8",
            "tmp9",
            "val0",
            "val1",
            "val2",
            "val3",
            "val4",
            "val5",
            "val6",
            "val7",
            "val8",
            "val9",
            "data0",
            "data1",
            "data2",
            "data3",
            "data4",
            "data5",
            "data6",
            "data7",
            "data8",
            "data9",
            "result0",
            "result1",
            "result2",
            "result3",
            "result4",
            "result5",
            "result6",
            "result7",
            "result8",
            "result9",
            "item0",
            "item1",
            "item2",
            "item3",
            "item4",
            "item5",
            "item6",
            "item7",
            "item8",
            "item9",
            // Single letters
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Tokenize JS code into token strings (simple whitespace + operator splitting)
    pub fn tokenize(&self, code: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in code.chars() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            } else if "(){}[];,".contains(ch) {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(ch.to_string());
            } else {
                current.push(ch);
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    /// Encode tokens to IDs
    pub fn encode(&self, tokens: &[String]) -> Vec<i64> {
        tokens
            .iter()
            .map(|token| *self.token_to_id.get(token).unwrap_or(&UNK_ID))
            .collect()
    }

    /// Decode IDs to tokens
    pub fn decode(&self, ids: &[i64]) -> Vec<String> {
        ids.iter()
            .filter_map(|&id| {
                if id >= 0 && (id as usize) < self.vocab.len() {
                    Some(self.vocab[id as usize].clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Model integration helper for JavaScript deobfuscation
pub struct JsDeobfuscatorIntegration {
    #[cfg(feature = "ai")]
    session: Option<Session>,
    tokenizer: JsTokenizer,
    #[cfg_attr(not(feature = "ai"), allow(dead_code))]
    monitor: Option<PerformanceMonitor>,
    enabled: bool,
}

impl JsDeobfuscatorIntegration {
    pub fn new(
        engine: &InferenceEngine,
        model_path: Option<&Path>,
        monitor: Option<PerformanceMonitor>,
    ) -> Result<Self> {
        #[cfg(feature = "ai")]
        {
            let session = if let Some(path) = model_path {
                if path.exists() {
                    Some(engine.load_model(path)?)
                } else {
                    log::warn!(
                        "JS deobfuscator model not found at {:?}, running without AI",
                        path
                    );
                    None
                }
            } else {
                None
            };

            let enabled = session.is_some();
            let tokenizer = JsTokenizer::new();

            Ok(Self {
                session,
                tokenizer,
                monitor,
                enabled,
            })
        }

        #[cfg(not(feature = "ai"))]
        {
            let _ = (engine, model_path);
            Ok(Self {
                enabled: false,
                tokenizer: JsTokenizer::new(),
                monitor,
            })
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Deobfuscate JavaScript code using the Seq2Seq model
    #[allow(dead_code)]
    pub fn deobfuscate(&mut self, obfuscated_js: &str) -> Result<String> {
        let start = Instant::now();

        // Fallback: return original if AI disabled
        if !self.enabled {
            if let Some(m) = &self.monitor {
                m.record_inference(InferenceMetrics {
                    model_name: "js_deobfuscator".to_string(),
                    inference_time: start.elapsed(),
                    input_size: obfuscated_js.len(),
                    output_size: obfuscated_js.len(),
                    success: false,
                    timestamp: start,
                });
            }
            return Ok(obfuscated_js.to_string());
        }

        // Tokenize input
        let tokens = self.tokenizer.tokenize(obfuscated_js);
        let mut token_ids = vec![SOS_ID];
        token_ids.extend(self.tokenizer.encode(&tokens));
        token_ids.push(EOS_ID);

        // Pad or truncate to max length
        token_ids.truncate(JS_DEOBFUSCATOR_MAX_LEN);
        while token_ids.len() < JS_DEOBFUSCATOR_MAX_LEN {
            token_ids.push(PAD_ID);
        }

        #[cfg(feature = "ai")]
        {
            let session = self
                .session
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("JS deobfuscator session missing"))?;

            // Create input tensor [1, seq_len]
            let input_tensor =
                Value::from_array((vec![1, JS_DEOBFUSCATOR_MAX_LEN as i64], token_ids.clone()))
                    .context("Failed to create input tensor")?;

            let outputs = session
                .run([SessionInputValue::from(input_tensor)])
                .map_err(|e| anyhow::anyhow!("Failed to run deobfuscator inference: {}", e))?;

            let output_tensor = outputs[0]
                .try_extract_tensor::<i64>()
                .context("Failed to extract output tensor")?;

            let output_ids: Vec<i64> = output_tensor.1.to_vec();

            // Decode output, stop at EOS
            let decoded_tokens = self.tokenizer.decode(&output_ids);
            let clean_tokens: Vec<String> = decoded_tokens
                .into_iter()
                .take_while(|t| t != "<EOS>")
                .filter(|t| t != "<PAD>" && t != "<SOS>")
                .collect();

            let deobfuscated = clean_tokens.join(" ");

            if let Some(m) = &self.monitor {
                m.record_inference(InferenceMetrics {
                    model_name: "js_deobfuscator".to_string(),
                    inference_time: start.elapsed(),
                    input_size: obfuscated_js.len(),
                    output_size: deobfuscated.len(),
                    success: true,
                    timestamp: start,
                });
            }

            Ok(deobfuscated)
        }

        #[cfg(not(feature = "ai"))]
        {
            Ok(obfuscated_js.to_string())
        }
    }
}
