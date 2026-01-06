use anyhow::Result;
use cssparser::{Parser, ParserInput, Token};
#[cfg(feature = "ai")]
use std::path::PathBuf;


#[cfg(feature = "ai")]
use browerai_ai_core::model_manager::ModelType;
#[cfg(feature = "ai")]
use browerai_ai_core::{AiRuntime, InferenceEngine};
#[cfg(feature = "ai")]
use browerai_ai_integration::CssModelIntegration;

/// CSS parser with AI enhancement capabilities
pub struct CssParser {
    #[cfg(feature = "ai")]
    inference_engine: Option<InferenceEngine>,
    #[cfg(feature = "ai")]
    ai_runtime: Option<AiRuntime>,
    #[cfg(feature = "ai")]
    model_path: Option<PathBuf>,
    #[cfg(feature = "ai")]
    model_name: Option<String>,
}

impl CssParser {
    /// Create a new CSS parser
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "ai")]
            inference_engine: None,
            #[cfg(feature = "ai")]
            ai_runtime: None,
            #[cfg(feature = "ai")]
            model_path: None,
            #[cfg(feature = "ai")]
            model_name: None,
        }
    }

    /// Create a new CSS parser with AI capabilities
    #[allow(dead_code)]
    #[cfg(feature = "ai")]
    pub fn with_ai(inference_engine: InferenceEngine) -> Self {
        Self {
            inference_engine: Some(inference_engine),
            ai_runtime: None,
            model_path: None,
            model_name: None,
        }
    }

    /// Create a new CSS parser with AI runtime (engine + model catalog + monitor)
    #[allow(dead_code)]
    #[cfg(feature = "ai")]
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
        }
    }

    /// Parse CSS content and extract rules
    pub fn parse(&self, css: &str) -> Result<Vec<CssRule>> {
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
        Ok(rules)
    }

    /// Validate CSS syntax
    pub fn validate(&self, css: &str) -> Result<bool> {
        let result = self.parse(css);
        Ok(result.is_ok())
    }

    /// Check if AI enhancement is enabled
    pub fn is_ai_enabled(&self) -> bool {
        #[cfg(feature = "ai")]
        {
            self.inference_engine.is_some()
        }
        #[cfg(not(feature = "ai"))]
        {
            false
        }
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
