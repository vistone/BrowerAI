//! Enhanced HTML Parser with Real AI Validation Hook
//! 增强型 HTML 解析器 - 支持真实 AI 验证

use anyhow::{Context, Result};
use browerai_core::error::parse::HtmlParseError;
use html5ever::parse_document;
use html5ever::tendril::TendrilSink;
use markup5ever_rcdom::{Handle, NodeData, RcDom};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::default::Default;
use std::io::Cursor;
use std::path::PathBuf;

/// Validation result from AI model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlValidationResult {
    pub is_valid: bool,
    pub confidence: f32,
    pub complexity_score: f32,
    pub structural_score: f32,
    pub suggestions: Vec<String>,
    pub warnings: Vec<String>,
    pub detected_features: Vec<String>,
    pub processing_time_ms: u64,
}

/// AI validation hook trait for real AI integration
pub trait HtmlAiValidationHook: Send {
    /// Validate HTML structure with real AI model
    fn validate_structure(&self, html: &str) -> Result<HtmlValidationResult>;

    /// Get model name
    fn model_name(&self) -> &str;

    /// Check if enabled
    fn is_enabled(&self) -> bool;
}

/// Simple heuristic-based validator for testing
#[derive(Debug, Default)]
pub struct HeuristicHtmlValidator;

impl HtmlAiValidationHook for HeuristicHtmlValidator {
    fn validate_structure(&self, html: &str) -> Result<HtmlValidationResult> {
        let start = std::time::Instant::now();

        let mut suggestions = Vec::new();
        let mut warnings = Vec::new();
        let mut detected_features = Vec::new();

        // Check for DOCTYPE
        if !html.to_lowercase().starts_with("<!doctype") {
            warnings.push("Missing DOCTYPE declaration".to_string());
        } else {
            detected_features.push("DOCTYPE".to_string());
        }

        // Check for common tags
        let lower = html.to_lowercase();
        if lower.contains("<html") {
            detected_features.push("HTML".to_string());
        }
        if lower.contains("<head") {
            detected_features.push("HEAD".to_string());
        }
        if lower.contains("<body") {
            detected_features.push("BODY".to_string());
        }
        if lower.contains("<script") {
            detected_features.push("SCRIPT".to_string());
            if !lower.contains("</script>") {
                warnings.push("Unclosed script tag detected".to_string());
            }
        }
        if lower.contains("<style") {
            detected_features.push("STYLE".to_string());
        }

        // Check for common issues
        if lower.contains("<div></div>") {
            suggestions.push("Consider using more semantic elements".to_string());
        }

        // Check for inline styles (anti-pattern)
        if lower.contains("style=\"") {
            warnings.push("Inline styles detected - consider using external CSS".to_string());
        }

        // Check for accessibility issues
        if lower.contains("<img") && !lower.contains("alt=\"") {
            warnings.push("Images missing alt attributes".to_string());
            suggestions.push("Add alt attributes to images for accessibility".to_string());
        }

        // Calculate complexity (simple heuristic based on tag nesting)
        let depth = count_max_depth(html);
        let complexity_score = (depth as f32 / 20.0).clamp(0.0, 1.0);

        // Calculate structural score
        let structural_score = if warnings.is_empty() {
            1.0
        } else if warnings.len() <= 2 {
            0.8
        } else {
            0.5
        };

        // Overall validity
        let is_valid = !warnings.iter().any(|w| w.contains("Unclosed"));

        // Confidence based on checks performed
        let confidence = 0.85;

        let processing_time = start.elapsed().as_millis();

        Ok(HtmlValidationResult {
            is_valid,
            confidence,
            complexity_score,
            structural_score,
            suggestions,
            warnings,
            detected_features,
            processing_time_ms: processing_time,
        })
    }

    fn model_name(&self) -> &str {
        "HeuristicHtmlValidator"
    }

    fn is_enabled(&self) -> bool {
        true
    }
}

/// Count maximum nesting depth
fn count_max_depth(html: &str) -> usize {
    let mut max_depth = 0;
    let mut current_depth = 0;
    let mut in_tag = false;
    let mut tag_name = String::new();

    for c in html.chars() {
        if c == '<' {
            in_tag = true;
            tag_name.clear();
        } else if c == '>' {
            in_tag = false;
            if !tag_name.starts_with('/')
                && !tag_name.starts_with('!')
                && !tag_name.starts_with('?')
            {
                // Opening tag
                current_depth += 1;
                max_depth = max_depth.max(current_depth);
            } else if tag_name.starts_with('/') {
                // Closing tag
                current_depth = current_depth.saturating_sub(1);
            }
        } else if in_tag {
            if c.is_alphanumeric() || c == '/' {
                tag_name.push(c);
            }
        }
    }

    max_depth
}

/// Enhanced HTML parser with AI validation
pub struct EnhancedHtmlParser {
    ai_enabled: bool,
    validation_hook: RefCell<Option<Box<dyn HtmlAiValidationHook>>>,
    model_path: Option<PathBuf>,
}

impl EnhancedHtmlParser {
    /// Create a new enhanced HTML parser
    pub fn new() -> Self {
        Self {
            ai_enabled: false,
            validation_hook: RefCell::new(None),
            model_path: None,
        }
    }

    /// Create parser with heuristic validator
    pub fn with_heuristic_validator() -> Self {
        let mut parser = Self::new();
        parser.set_validation_hook(Box::new(HeuristicHtmlValidator));
        parser
    }

    /// Set validation hook
    pub fn set_validation_hook(&mut self, hook: Box<dyn HtmlAiValidationHook>) {
        self.ai_enabled = hook.is_enabled();
        self.validation_hook = RefCell::new(Some(hook));
    }

    /// Parse HTML with AI validation
    pub fn parse_with_validation(&self, html: &str) -> Result<(RcDom, HtmlValidationResult)> {
        let validation_result = if self.ai_enabled {
            if let Some(ref hook) = *self.validation_hook.borrow() {
                hook.validate_structure(html)?
            } else {
                HtmlValidationResult {
                    is_valid: true,
                    confidence: 1.0,
                    complexity_score: 0.5,
                    structural_score: 1.0,
                    suggestions: Vec::new(),
                    warnings: Vec::new(),
                    detected_features: Vec::new(),
                    processing_time_ms: 0,
                }
            }
        } else {
            HtmlValidationResult {
                is_valid: true,
                confidence: 1.0,
                complexity_score: 0.5,
                structural_score: 1.0,
                suggestions: Vec::new(),
                warnings: Vec::new(),
                detected_features: Vec::new(),
                processing_time_ms: 0,
            }
        };

        // Parse HTML
        let dom = self.parse(html)?;

        Ok((dom, validation_result))
    }

    /// Parse HTML content into a DOM tree
    pub fn parse(&self, html: &str) -> Result<RcDom> {
        let input = Cursor::new(html.as_bytes());
        let dom = parse_document(RcDom::default(), Default::default())
            .from_utf8()
            .read_from(&mut input.clone())
            .map_err(|e| HtmlParseError::new(e.to_string(), 0, 0))?;

        Ok(dom)
    }

    /// Parse HTML with detailed error information
    pub fn parse_with_error_info(&self, html: &str) -> Result<RcDom, HtmlParseError> {
        let input = Cursor::new(html.as_bytes());

        match parse_document(RcDom::default(), Default::default())
            .from_utf8()
            .read_from(&mut input.clone())
        {
            Ok(dom) => Ok(dom),
            Err(e) => {
                let line = html.lines().count();
                Err(HtmlParseError::new(e.to_string(), line, 0))
            }
        }
    }

    /// Extract text content from the DOM
    pub fn extract_text(&self, dom: &RcDom) -> String {
        let mut text = String::new();
        self.walk_tree(&dom.document, &mut text);
        text
    }

    /// Walk the DOM tree and collect text
    fn walk_tree(&self, handle: &Handle, text: &mut String) {
        let node = handle;

        if let NodeData::Text { ref contents } = node.data {
            text.push_str(&contents.borrow());
        }

        for child in node.children.borrow().iter() {
            self.walk_tree(child, text);
        }
    }

    /// Get parsing statistics
    pub fn get_stats(&self, html: &str) -> HtmlParseStats {
        let char_count = html.len();
        let line_count = html.lines().count();
        let tag_count = html.matches('<').count();
        let max_depth = count_max_depth(html);

        HtmlParseStats {
            char_count,
            line_count,
            tag_count,
            max_depth,
            estimated_complexity: (max_depth as f32 / 20.0).clamp(0.0, 1.0),
        }
    }

    /// Check if AI enhancement is enabled
    pub fn is_ai_enabled(&self) -> bool {
        self.ai_enabled
    }
}

impl Default for EnhancedHtmlParser {
    fn default() -> Self {
        Self::new()
    }
}

/// HTML parsing statistics
#[derive(Debug, Clone)]
pub struct HtmlParseStats {
    pub char_count: usize,
    pub line_count: usize,
    pub tag_count: usize,
    pub max_depth: usize,
    pub estimated_complexity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_parse_simple_html() {
        let parser = EnhancedHtmlParser::new();
        let html = "<html><body><h1>Hello, World!</h1></body></html>";
        let result = parser.parse(html);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_with_heuristic_validation() {
        let parser = EnhancedHtmlParser::with_heuristic_validator();
        let html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>Test</title>
</head>
<body>
    <h1>Hello</h1>
    <img src="test.jpg" alt="Test image">
</body>
</html>
"#;

        let (dom, validation) = parser.parse_with_validation(html).unwrap();
        assert!(validation.is_valid);
        assert!(validation.confidence > 0.0);
        assert!(validation
            .detected_features
            .contains(&"DOCTYPE".to_string()));
    }

    #[test]
    fn test_parse_with_warnings() {
        let parser = EnhancedHtmlParser::with_heuristic_validator();
        let html = r#"
<html>
<head>
    <title>No DOCTYPE</title>
</head>
<body>
    <img src="test.jpg">
</body>
</html>
"#;

        let (_, validation) = parser.parse_with_validation(html).unwrap();
        // Should have warnings about missing DOCTYPE and img alt
        assert!(!validation.warnings.is_empty());
    }

    #[test]
    fn test_parse_complex_html() {
        let parser = EnhancedHtmlParser::new();

        // Read real test data
        let test_data: Vec<HtmlTestSample> = serde_json::from_str(
            &fs::read_to_string("test_data/real_world/html/test_samples.json").unwrap(),
        )
        .unwrap();

        for sample in test_data {
            let result = parser.parse(&sample.input);
            assert!(
                result.is_ok(),
                "Failed to parse: {}",
                sample.input.chars().take(50).collect::<String>()
            );
        }
    }

    #[test]
    fn test_depth_calculation() {
        let html = "<div><div><div><span>Deep</span></div></div></div>";
        let depth = count_max_depth(html);
        assert!(depth >= 3);
    }

    #[test]
    fn test_stats_collection() {
        let parser = EnhancedHtmlParser::new();
        let html = "<html><body><div><p>Text</p></div></body></html>";
        let stats = parser.get_stats(html);

        assert!(stats.char_count > 0);
        assert!(stats.line_count > 0);
        assert!(stats.tag_count > 0);
        assert!(stats.estimated_complexity >= 0.0);
    }
}

/// Test sample structure for HTML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlTestSample {
    pub input: String,
    pub expected_complexity: f32,
    pub tags: Vec<String>,
}

#[cfg(test)]
impl<'a> From<&'a str> for HtmlTestSample {
    fn from(s: &'a str) -> Self {
        Self {
            input: s.to_string(),
            expected_complexity: 0.5,
            tags: vec![],
        }
    }
}
