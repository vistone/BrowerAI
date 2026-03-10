use browerai_css_parser::CssParser;

#[cfg(feature = "onnx")]
use browerai_ai_core::Phase2ModelLoader;

/// Application state shared across handlers
pub struct AppState {
    css_parser: CssParser,
    #[cfg(feature = "onnx")]
    model_dir: Option<String>,
}

impl AppState {
    /// Create new application state
    pub fn new() -> Self {
        Self {
            css_parser: CssParser::new(),
            #[cfg(feature = "onnx")]
            model_dir: None,
        }
    }

    /// Create application state with AI models
    #[cfg(feature = "onnx")]
    pub fn with_models(model_dir: impl Into<String>) -> anyhow::Result<Self> {
        let model_dir_str = model_dir.into();
        let css_parser = CssParser::from_model_dir(&model_dir_str)?;

        Ok(Self {
            css_parser,
            model_dir: Some(model_dir_str),
        })
    }

    /// Get CSS parser reference
    pub fn css_parser(&self) -> &CssParser {
        &self.css_parser
    }

    /// Check if AI features are enabled (simplified)
    pub fn is_ai_enabled(&self) -> bool {
        // Simplified: AI not directly available in CssParser
        false
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
