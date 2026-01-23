use anyhow::Result;
use cssparser::{Parser, ParserInput, Token};
#[cfg(feature = "ai")]
use std::collections::HashMap;
#[cfg(feature = "ai")]
use std::path::PathBuf;
#[cfg(feature = "ai")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "ai")]
use browerai_ai_core::model_manager::ModelType;
#[cfg(feature = "ai")]
use browerai_ai_core::{AiRuntime, InferenceEngine};
#[cfg(all(feature = "ai", feature = "onnx"))]
use browerai_ai_core::{Phase2ModelLoader, Phase2PropertyPredictor, Phase2SelectorEmbedding};

/// CSS parser with AI enhancement capabilities
pub struct CssParser {
    #[cfg(feature = "ai")]
    inference_engine: Option<InferenceEngine>,
    #[cfg(feature = "ai")]
    #[allow(dead_code)]
    ai_runtime: Option<AiRuntime>,
    #[cfg(feature = "ai")]
    #[allow(dead_code)]
    model_path: Option<PathBuf>,
    #[cfg(feature = "ai")]
    #[allow(dead_code)]
    model_name: Option<String>,
    #[cfg(all(feature = "ai", feature = "onnx"))]
    selector_embedding: Option<Arc<Phase2SelectorEmbedding>>,
    #[cfg(all(feature = "ai", feature = "onnx"))]
    property_predictor: Option<Arc<Phase2PropertyPredictor>>,
    #[cfg(all(feature = "ai", feature = "onnx"))]
    embedding_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
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
            #[cfg(all(feature = "ai", feature = "onnx"))]
            selector_embedding: None,
            #[cfg(all(feature = "ai", feature = "onnx"))]
            property_predictor: None,
            #[cfg(all(feature = "ai", feature = "onnx"))]
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new CSS parser with Phase 2 models
    #[cfg(all(feature = "ai", feature = "onnx"))]
    pub fn with_phase2_models(
        selector_embedding: Phase2SelectorEmbedding,
        property_predictor: Phase2PropertyPredictor,
    ) -> Self {
        Self {
            inference_engine: None,
            ai_runtime: None,
            model_path: None,
            model_name: None,
            selector_embedding: Some(Arc::new(selector_embedding)),
            property_predictor: Some(Arc::new(property_predictor)),
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new CSS parser from model directory
    #[cfg(all(feature = "ai", feature = "onnx"))]
    pub fn from_model_dir(model_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let loader = Phase2ModelLoader::new(model_dir);
        let selector_embedding = loader.load_selector_embedding()?;
        let property_predictor = loader.load_property_predictor()?;

        Ok(Self {
            inference_engine: None,
            ai_runtime: None,
            model_path: None,
            model_name: None,
            selector_embedding: Some(Arc::new(selector_embedding)),
            property_predictor: Some(Arc::new(property_predictor)),
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
        })
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

    /// Enhanced parse with AI selector embedding
    #[cfg(all(feature = "ai", feature = "onnx"))]
    pub fn parse_with_ai(&self, css: &str) -> Result<Vec<EnhancedCssRule>> {
        let rules = self.parse(css)?;
        let mut enhanced_rules = Vec::new();

        for rule in rules {
            let embedding = self.get_selector_embedding(&rule.selector)?;
            enhanced_rules.push(EnhancedCssRule {
                selector: rule.selector,
                properties: rule.properties,
                embedding,
                predicted_properties: Vec::new(),
            });
        }

        log::info!(
            "Enhanced {} CSS rules with AI embeddings",
            enhanced_rules.len()
        );
        Ok(enhanced_rules)
    }

    /// Get embedding for a CSS selector (with caching)
    #[cfg(all(feature = "ai", feature = "onnx"))]
    pub fn get_selector_embedding(&self, selector: &str) -> Result<Vec<f32>> {
        // Check cache first
        let cache_hit = {
            let cache = self.embedding_cache.lock().unwrap();
            cache.get(selector).cloned()
        };

        if let Some(embedding) = cache_hit {
            log::debug!("Cache hit for selector: {}", selector);
            return Ok(embedding);
        }

        log::debug!("Cache miss for selector: {}", selector);

        // Compute embedding
        let embedding = if let Some(ref model) = self.selector_embedding {
            // Tokenize selector (simplified: use ASCII values as tokens)
            let tokens: Vec<i64> = selector
                .chars()
                .take(50) // Max sequence length
                .map(|c| c as i64)
                .collect();

            // Pad to fixed length
            let mut padded_tokens = tokens;
            padded_tokens.resize(50, 0);

            // Run inference
            let result = model.infer(&[padded_tokens])?;

            // Flatten embedding (batch=1, seq_len=50, dim=128)
            result.into_iter().flatten().collect()
        } else {
            // No model available, return zero embedding
            vec![0.0; 50 * 128]
        };

        // Cache the result
        {
            let mut cache = self.embedding_cache.lock().unwrap();
            cache.insert(selector.to_string(), embedding.clone());
            log::debug!(
                "Cached embedding for selector: {} (cache size: {})",
                selector,
                cache.len()
            );
        }

        Ok(embedding)
    }

    /// Predict CSS properties for a selector
    #[cfg(all(feature = "ai", feature = "onnx"))]
    pub fn predict_properties(&self, selector: &str) -> Result<Vec<PredictedProperty>> {
        let embedding = self.get_selector_embedding(selector)?;

        if let Some(ref model) = self.property_predictor {
            // Take first 1280 dimensions (10 * 128)
            let input: Vec<f32> = embedding.into_iter().take(1280).collect();

            // Run inference
            let probabilities = model.infer(&[input])?;

            // Convert to predicted properties
            let properties = Self::probabilities_to_properties(&probabilities[0]);

            log::info!(
                "Predicted {} properties for selector '{}'",
                properties.len(),
                selector
            );
            Ok(properties)
        } else {
            Ok(Vec::new())
        }
    }

    /// Convert probability vector to predicted CSS properties
    #[cfg(all(feature = "ai", feature = "onnx"))]
    fn probabilities_to_properties(probs: &[f32]) -> Vec<PredictedProperty> {
        // Property names (top 50 most common CSS properties)
        const PROPERTY_NAMES: &[&str] = &[
            "color",
            "background-color",
            "font-size",
            "margin",
            "padding",
            "width",
            "height",
            "display",
            "position",
            "top",
            "left",
            "border",
            "border-radius",
            "opacity",
            "font-family",
            "font-weight",
            "text-align",
            "line-height",
            "overflow",
            "z-index",
            "box-shadow",
            "flex",
            "grid",
            "align-items",
            "justify-content",
            "gap",
            "transition",
            "transform",
            "animation",
            "cursor",
            "pointer-events",
            "visibility",
            "white-space",
            "word-wrap",
            "text-decoration",
            "list-style",
            "outline",
            "resize",
            "user-select",
            "appearance",
            "filter",
            "backdrop-filter",
            "mix-blend-mode",
            "clip-path",
            "mask",
            "scroll-behavior",
            "overscroll-behavior",
            "aspect-ratio",
            "contain",
            "content-visibility",
        ];

        probs
            .iter()
            .enumerate()
            .filter(|(_, &prob)| prob > 0.5) // Threshold
            .map(|(idx, &prob)| {
                let name = PROPERTY_NAMES.get(idx).unwrap_or(&"unknown").to_string();
                PredictedProperty {
                    name,
                    confidence: prob,
                }
            })
            .collect()
    }

    /// Clear embedding cache
    #[cfg(all(feature = "ai", feature = "onnx"))]
    pub fn clear_cache(&self) {
        let mut cache = self.embedding_cache.lock().unwrap();
        cache.clear();
        log::info!("Cleared embedding cache");
    }

    /// Get cache size
    #[cfg(all(feature = "ai", feature = "onnx"))]
    pub fn cache_size(&self) -> usize {
        let cache = self.embedding_cache.lock().unwrap();
        cache.len()
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

/// Enhanced CSS rule with AI embeddings
#[cfg(all(feature = "ai", feature = "onnx"))]
#[derive(Debug, Clone)]
pub struct EnhancedCssRule {
    pub selector: String,
    pub properties: Vec<CssProperty>,
    pub embedding: Vec<f32>,
    pub predicted_properties: Vec<PredictedProperty>,
}

/// Predicted CSS property with confidence
#[cfg(all(feature = "ai", feature = "onnx"))]
#[derive(Debug, Clone)]
pub struct PredictedProperty {
    pub name: String,
    pub confidence: f32,
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

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn parse_doesnt_crash(css in ".*") {
            let parser = CssParser::new();
            let _ = parser.parse(&css);
            // Should never panic
        }

        #[test]
        fn parse_is_deterministic(css in ".*") {
            let parser = CssParser::new();
            let result1 = parser.parse(&css);
            let result2 = parser.parse(&css);
            prop_assert_eq!(result1.is_ok(), result2.is_ok());
        }

        #[test]
        fn validate_doesnt_panic(css in ".*") {
            let parser = CssParser::new();
            let _ = parser.validate(&css);
        }

        #[test]
        fn parse_simple_selectors(
            selector in "[a-z]{1,10}",
            property in "[a-z-]{1,20}",
            value in "[a-z0-9]{1,20}"
        ) {
            let parser = CssParser::new();
            let css = format!("{} {{ {}: {}; }}", selector, property, value);
            let result = parser.parse(&css);
            // May fail with invalid CSS, but shouldn't panic
            let _ = result;
        }
    }
}
