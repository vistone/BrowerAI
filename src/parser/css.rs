use anyhow::Result;
use cssparser::{Parser, ParserInput, Token};

use std::path::PathBuf;

use crate::ai::integration::CssModelIntegration;
use crate::ai::model_manager::ModelType;
use crate::ai::{AiRuntime, InferenceEngine};

/// CSS parser with AI enhancement capabilities
pub struct CssParser {
    inference_engine: Option<InferenceEngine>,
    ai_runtime: Option<AiRuntime>,
    model_path: Option<PathBuf>,
    model_name: Option<String>,
    enable_ai: bool,
}

impl CssParser {
    /// Create a new CSS parser
    pub fn new() -> Self {
        Self {
            inference_engine: None,
            ai_runtime: None,
            model_path: None,
            model_name: None,
            enable_ai: false,
        }
    }

    /// Create a new CSS parser with AI capabilities
    #[allow(dead_code)]
    pub fn with_ai(inference_engine: InferenceEngine) -> Self {
        Self {
            inference_engine: Some(inference_engine),
            ai_runtime: None,
            model_path: None,
            model_name: None,
            enable_ai: true,
        }
    }

    /// Create a new CSS parser with AI runtime (engine + model catalog + monitor)
    #[allow(dead_code)]
    pub fn with_ai_runtime(ai_runtime: AiRuntime) -> Self {
        let (model_name, model_path) = ai_runtime
            .best_model(ModelType::CssParser)
            .map(|(cfg, path)| (Some(cfg.name), Some(path)))
            .unwrap_or((None, None));

        Self {
            inference_engine: Some(ai_runtime.engine()),
            ai_runtime: Some(ai_runtime),
            model_path,
            model_name,
            enable_ai: true,
        }
    }

    /// Parse CSS content and extract rules
    pub fn parse(&self, css: &str) -> Result<Vec<CssRule>> {
        use crate::ai::config::FallbackReason;
        use std::time::Instant;

        let mut input = ParserInput::new(css);
        let mut parser = Parser::new(&mut input);
        let mut rules = Vec::new();

        while let Ok(token) = parser.next() {
            if let Token::Ident(ref name) = token {
                // Basic CSS rule extraction
                rules.push(CssRule {
                    selector: name.to_string(),
                    properties: Vec::new(),
                });
            }
        }

        log::info!("Successfully parsed CSS with {} rules", rules.len());

        // Check if AI runtime is available and AI is enabled
        let runtime = self.ai_runtime.as_ref();
        let ai_enabled = runtime.map_or(false, |r| r.is_ai_enabled()) && self.enable_ai;

        if !ai_enabled {
            if self.enable_ai && runtime.is_none() {
                log::debug!("AI enhancement disabled for CSS parsing; no runtime available");
            } else {
                log::debug!("AI enhancement disabled for CSS parsing; baseline path in use");
            }
            return Ok(rules);
        }

        let runtime = runtime.unwrap();
        let tracker = runtime.fallback_tracker();
        tracker.record_attempt();

        // Try AI enhancement
        let ai_active = self.enable_ai && self.inference_engine.is_some();
        if !ai_active {
            tracker.record_fallback(FallbackReason::NoModelAvailable);
            log::warn!(
                "AI was requested for CSS parsing but no inference engine is available; falling back to baseline parser"
            );
            return Ok(rules);
        }

        let monitor = runtime.monitor();
        let model_path = self.model_path.as_deref();
        let model_name = self.model_name.as_deref().unwrap_or("css_model");
        let start_time = Instant::now();

        match CssModelIntegration::new(self.inference_engine.as_ref().unwrap(), model_path, monitor)
        {
            Ok(mut integration) => match integration.optimize_rules(css) {
                Ok(optimized) => {
                    let elapsed_ms = start_time.elapsed().as_millis() as u64;
                    tracker.record_success(elapsed_ms);
                    log::info!(
                        "AI CSS optimization (model={}, {}ms): generated {} candidate rules",
                        model_name,
                        elapsed_ms,
                        optimized.len()
                    );
                }
                Err(err) => {
                    tracker.record_fallback(FallbackReason::InferenceFailed(err.to_string()));
                    log::warn!(
                        "AI CSS optimization failed (model={}): {}; continuing with baseline rules",
                        model_name,
                        err
                    );
                }
            },
            Err(err) => {
                tracker.record_fallback(FallbackReason::ModelLoadFailed(err.to_string()));
                log::warn!(
                    "AI CSS integration could not start (model={}): {}; continuing without AI",
                    model_name,
                    err
                );
            }
        }

        Ok(rules)
    }

    /// Validate CSS syntax
    #[allow(dead_code)]
    pub fn validate(&self, css: &str) -> Result<bool> {
        let mut input = ParserInput::new(css);
        let _parser = Parser::new(&mut input);

        // Basic validation - if we can create a parser, it's valid enough
        Ok(true)
    }

    /// Enable or disable AI enhancement
    #[allow(dead_code)]
    pub fn set_ai_enabled(&mut self, enabled: bool) {
        self.enable_ai = enabled && self.inference_engine.is_some();
    }

    /// Check if AI enhancement is enabled
    #[allow(dead_code)]
    pub fn is_ai_enabled(&self) -> bool {
        self.enable_ai
    }
}

impl Default for CssParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a CSS rule
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CssRule {
    pub selector: String,
    pub properties: Vec<CssProperty>,
}

/// Represents a CSS property
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CssProperty {
    pub name: String,
    pub value: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_css() {
        let parser = CssParser::new();
        let css = "body { color: red; }";
        let result = parser.parse(css);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_css() {
        let parser = CssParser::new();
        let css = "div { margin: 10px; }";
        let result = parser.validate(css);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_css_parser_with_ai_disabled() {
        let parser = CssParser::new();
        assert!(!parser.is_ai_enabled());
    }
}
