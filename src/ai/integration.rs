use anyhow::{Context, Result};
use std::path::Path;

#[cfg(feature = "ai")]
use ort::{Session, Value};

use super::InferenceEngine;

/// Model integration helper for HTML parsing
pub struct HtmlModelIntegration {
    #[cfg(feature = "ai")]
    session: Option<Session>,
    enabled: bool,
}

impl HtmlModelIntegration {
    /// Create a new HTML model integration
    pub fn new(engine: &InferenceEngine, model_path: Option<&Path>) -> Result<Self> {
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

            Ok(Self {
                session,
                enabled: session.is_some(),
            })
        }

        #[cfg(not(feature = "ai"))]
        {
            Ok(Self { enabled: false })
        }
    }

    /// Validate HTML structure using AI model
    #[cfg(feature = "ai")]
    pub fn validate_structure(&self, html: &str) -> Result<(bool, f32)> {
        if !self.enabled || self.session.is_none() {
            return Ok((true, 0.5)); // Fallback: assume valid, medium complexity
        }

        let session = self.session.as_ref().unwrap();

        // Tokenize HTML (simple character-level tokenization)
        let tokens = self.tokenize_html(html, 512);

        // Create input tensor
        let input_shape = vec![1, 512];
        let input_data: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();

        let input_tensor =
            Value::from_array(session.allocator(), &input_data, &input_shape[..])
                .context("Failed to create input tensor")?;

        // Run inference
        let outputs = session
            .run(vec![input_tensor])
            .context("Failed to run inference")?;

        // Parse outputs
        let output_data = outputs[0]
            .try_extract::<f32>()
            .context("Failed to extract output")?;
        let output_slice = output_data.view();

        let validity = output_slice[[0, 0]] > 0.5;
        let complexity = output_slice[[0, 1]];

        log::debug!(
            "HTML validation: valid={}, complexity={}",
            validity,
            complexity
        );

        Ok((validity, complexity))
    }

    #[cfg(not(feature = "ai"))]
    pub fn validate_structure(&self, _html: &str) -> Result<(bool, f32)> {
        Ok((true, 0.5))
    }

    /// Tokenize HTML to indices
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
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Model integration helper for CSS parsing
pub struct CssModelIntegration {
    #[cfg(feature = "ai")]
    session: Option<Session>,
    enabled: bool,
}

impl CssModelIntegration {
    pub fn new(engine: &InferenceEngine, model_path: Option<&Path>) -> Result<Self> {
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

            Ok(Self {
                session,
                enabled: session.is_some(),
            })
        }

        #[cfg(not(feature = "ai"))]
        {
            Ok(Self { enabled: false })
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Optimize CSS rules using AI
    pub fn optimize_rules(&self, css: &str) -> Result<Vec<String>> {
        // Placeholder: return original CSS split by rules
        let rules: Vec<String> = css
            .split('}')
            .filter(|s| !s.trim().is_empty())
            .map(|s| format!("{}}}", s.trim()))
            .collect();

        Ok(rules)
    }
}

/// Model integration helper for JavaScript parsing
pub struct JsModelIntegration {
    #[cfg(feature = "ai")]
    session: Option<Session>,
    enabled: bool,
}

impl JsModelIntegration {
    pub fn new(engine: &InferenceEngine, model_path: Option<&Path>) -> Result<Self> {
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

            Ok(Self {
                session,
                enabled: session.is_some(),
            })
        }

        #[cfg(not(feature = "ai"))]
        {
            Ok(Self { enabled: false })
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Analyze JavaScript code patterns
    pub fn analyze_patterns(&self, js: &str) -> Result<Vec<String>> {
        // Placeholder: return basic patterns
        let patterns = vec!["function_declaration", "variable_assignment"];

        Ok(patterns.iter().map(|s| s.to_string()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::InferenceEngine;

    #[test]
    fn test_html_integration_creation() {
        let engine = InferenceEngine::new().unwrap();
        let integration = HtmlModelIntegration::new(&engine, None);
        assert!(integration.is_ok());
    }

    #[test]
    fn test_html_validation_fallback() {
        let engine = InferenceEngine::new().unwrap();
        let integration = HtmlModelIntegration::new(&engine, None).unwrap();
        let (valid, complexity) = integration
            .validate_structure("<html><body>Test</body></html>")
            .unwrap();
        assert!(valid); // Should fallback to valid
        assert!(complexity >= 0.0 && complexity <= 1.0);
    }

    #[test]
    fn test_css_integration_creation() {
        let engine = InferenceEngine::new().unwrap();
        let integration = CssModelIntegration::new(&engine, None);
        assert!(integration.is_ok());
    }

    #[test]
    fn test_js_integration_creation() {
        let engine = InferenceEngine::new().unwrap();
        let integration = JsModelIntegration::new(&engine, None);
        assert!(integration.is_ok());
    }
}
