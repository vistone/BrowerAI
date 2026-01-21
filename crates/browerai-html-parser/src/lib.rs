// Enhanced HTML Parser module

use anyhow::Result;
use browerai_core::error::parse::HtmlParseError;
use html5ever::parse_document;
use html5ever::tendril::TendrilSink;
use markup5ever_rcdom::{Handle, NodeData, RcDom};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::default::Default;
use std::io::Cursor;

/// Trait for pluggable HTML validation hooks
pub trait HtmlValidationHook: Send {
    fn is_enabled(&self) -> bool;
    fn validate_structure(&mut self, html: &str) -> Result<(bool, f32)>;
}

/// HTML validation result
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

/// HTML parsing statistics
#[derive(Debug, Clone)]
pub struct HtmlParseStats {
    pub char_count: usize,
    pub line_count: usize,
    pub tag_count: usize,
    pub max_depth: usize,
    pub estimated_complexity: f32,
}

/// Test sample structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlTestSample {
    pub input: String,
    pub expected_complexity: f32,
    pub tags: Vec<String>,
}

/// Heuristic-based HTML validator
#[derive(Debug, Default)]
pub struct HeuristicHtmlValidator;

impl HtmlValidationHook for HeuristicHtmlValidator {
    fn is_enabled(&self) -> bool {
        true
    }

    fn validate_structure(&mut self, html: &str) -> Result<(bool, f32)> {
        let mut warnings = 0;
        let lower = html.to_lowercase();

        if !lower.starts_with("<!doctype") {
            warnings += 1;
        }
        if lower.contains("<script") && !lower.contains("</script>") {
            warnings += 1;
        }
        if lower.contains("<img") && !lower.contains("alt=\"") {
            warnings += 1;
        }

        let is_valid = warnings < 2;
        let confidence = 0.85 - (warnings as f32 * 0.1);

        Ok((is_valid, confidence))
    }
}

/// HTML parser with AI enhancement
pub struct HtmlParser {
    ai_enabled: bool,
    validation_hook: RefCell<Option<Box<dyn HtmlValidationHook + Send>>>,
}

impl HtmlParser {
    pub fn new() -> Self {
        Self {
            ai_enabled: false,
            validation_hook: RefCell::new(None),
        }
    }

    pub fn with_validation_hook(mut self, hook: Box<dyn HtmlValidationHook + Send>) -> Self {
        self.ai_enabled = hook.is_enabled();
        *self.validation_hook.borrow_mut() = Some(hook);
        self
    }

    pub fn set_validation_hook(&mut self, hook: Box<dyn HtmlValidationHook + Send>) {
        self.ai_enabled = hook.is_enabled();
        *self.validation_hook.borrow_mut() = Some(hook);
    }

    pub fn parse(&self, html: &str) -> Result<RcDom> {
        if self.ai_enabled {
            if let Some(hook) = self.validation_hook.borrow_mut().as_mut() {
                match hook.validate_structure(html) {
                    Ok((valid, complexity)) => {
                        log::info!(
                            "HTML validation: valid={} complexity={:.3}",
                            valid,
                            complexity
                        );
                    }
                    Err(e) => {
                        log::warn!("HTML validation failed: {}", e);
                    }
                }
            }
        }

        let input = Cursor::new(html.as_bytes());
        let dom = parse_document(RcDom::default(), Default::default())
            .from_utf8()
            .read_from(&mut input.clone())
            .map_err(|e| HtmlParseError::new(e.to_string(), 0, 0))?;

        log::info!("Successfully parsed HTML document");

        Ok(dom)
    }

    pub fn parse_with_error_info(&self, html: &str) -> Result<RcDom, HtmlParseError> {
        let input = Cursor::new(html.as_bytes());

        match parse_document(RcDom::default(), Default::default())
            .from_utf8()
            .read_from(&mut input.clone())
        {
            Ok(dom) => {
                log::info!("Successfully parsed HTML document");
                Ok(dom)
            }
            Err(e) => {
                let line = html.lines().count();
                Err(HtmlParseError::new(e.to_string(), line, 0))
            }
        }
    }

    pub fn extract_text(&self, dom: &RcDom) -> String {
        let mut text = String::new();
        self.walk_tree(&dom.document, &mut text);
        text
    }

    fn walk_tree(&self, handle: &Handle, text: &mut String) {
        let node = handle;

        if let NodeData::Text { ref contents } = node.data {
            text.push_str(&contents.borrow());
        }

        for child in node.children.borrow().iter() {
            self.walk_tree(child, text);
        }
    }

    pub fn set_ai_enabled(&mut self, enabled: bool) {
        self.ai_enabled = enabled;
    }

    pub fn is_ai_enabled(&self) -> bool {
        self.ai_enabled
    }

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
}

impl Default for HtmlParser {
    fn default() -> Self {
        Self::new()
    }
}

fn count_max_depth(html: &str) -> usize {
    let mut max_depth: usize = 0;
    let mut current_depth: usize = 0;
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
                current_depth += 1;
                max_depth = max_depth.max(current_depth);
            } else if tag_name.starts_with('/') && current_depth > 0 {
                current_depth = current_depth.saturating_sub(1);
            }
        } else if in_tag && (c.is_alphanumeric() || c == '/') {
            tag_name.push(c);
        }
    }

    max_depth
}

pub struct EnhancedHtmlParser {
    parser: HtmlParser,
}

impl EnhancedHtmlParser {
    pub fn new() -> Self {
        Self {
            parser: HtmlParser::new(),
        }
    }

    pub fn with_heuristic_validator() -> Self {
        let mut parser = HtmlParser::new();
        parser.set_validation_hook(Box::new(HeuristicHtmlValidator));
        Self { parser }
    }

    pub fn set_validation_hook(&mut self, hook: Box<dyn HtmlValidationHook + Send>) {
        self.parser.set_validation_hook(hook);
    }

    pub fn parse_with_validation(&self, html: &str) -> Result<(RcDom, HtmlValidationResult)> {
        let validation_result = if self.parser.ai_enabled {
            if let Some(hook) = self.parser.validation_hook.borrow_mut().as_mut() {
                let start = std::time::Instant::now();
                let (is_valid, confidence) = hook.validate_structure(html)?;
                let processing_time = start.elapsed().as_millis() as u64;

                HtmlValidationResult {
                    is_valid,
                    confidence,
                    complexity_score: self.parser.get_stats(html).estimated_complexity,
                    structural_score: if is_valid { 1.0 } else { 0.7 },
                    suggestions: vec![],
                    warnings: vec![],
                    detected_features: vec![],
                    processing_time_ms: processing_time,
                }
            } else {
                HtmlValidationResult {
                    is_valid: true,
                    confidence: 1.0,
                    complexity_score: 0.5,
                    structural_score: 1.0,
                    suggestions: vec![],
                    warnings: vec![],
                    detected_features: vec![],
                    processing_time_ms: 0,
                }
            }
        } else {
            HtmlValidationResult {
                is_valid: true,
                confidence: 1.0,
                complexity_score: 0.5,
                structural_score: 1.0,
                suggestions: vec![],
                warnings: vec![],
                detected_features: vec![],
                processing_time_ms: 0,
            }
        };

        let dom = self.parser.parse(html)?;
        Ok((dom, validation_result))
    }

    pub fn parse(&self, html: &str) -> Result<RcDom> {
        self.parser.parse(html)
    }

    pub fn parse_with_error_info(&self, html: &str) -> Result<RcDom, HtmlParseError> {
        self.parser.parse_with_error_info(html)
    }

    pub fn extract_text(&self, dom: &RcDom) -> String {
        self.parser.extract_text(dom)
    }

    pub fn is_ai_enabled(&self) -> bool {
        self.parser.is_ai_enabled()
    }

    pub fn get_stats(&self, html: &str) -> HtmlParseStats {
        self.parser.get_stats(html)
    }
}

impl Default for EnhancedHtmlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_html() {
        let parser = HtmlParser::new();
        let html = "<html><body><h1>Hello, World!</h1></body></html>";
        let result = parser.parse(html);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_text() {
        let parser = HtmlParser::new();
        let html = "<html><body><p>Hello</p><p>World</p></body></html>";
        let dom = parser.parse(html).unwrap();
        let text = parser.extract_text(&dom);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_parse_malformed_html() {
        let parser = HtmlParser::new();
        let html = "<div><p>Unclosed paragraph<div>Nested</div>";
        let result = parser.parse(html);
        assert!(result.is_ok());
    }

    #[test]
    fn test_enhanced_parser() {
        let parser = EnhancedHtmlParser::with_heuristic_validator();
        let html = r#"<!DOCTYPE html><html><body><h1>Test</h1></body></html>"#;

        let (_dom, validation) = parser.parse_with_validation(html).unwrap();
        assert!(validation.is_valid);
        assert!(validation.confidence > 0.0);
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

    #[test]
    fn test_depth_calculation() {
        let html = "<div><div><div><span>Deep</span></div></div></div>";
        let depth = count_max_depth(html);
        assert!(depth >= 3);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn parse_doesnt_crash(html in ".*") {
            let parser = HtmlParser::new();
            let _ = parser.parse(&html);
        }

        #[test]
        fn parse_is_deterministic(html in ".*") {
            let parser = HtmlParser::new();
            let result1 = parser.parse(&html);
            let result2 = parser.parse(&html);
            prop_assert_eq!(result1.is_ok(), result2.is_ok());
        }

        #[test]
        fn parse_empty_or_whitespace_succeeds(s in r"[ \t\n\r]*") {
            let parser = HtmlParser::new();
            let result = parser.parse(&s);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn parse_simple_tags(tag in "[a-z]{1,10}", content in "[ -~]{0,100}") {
            let parser = HtmlParser::new();
            let html = format!("<{0}>{1}</{0}>", tag, content);
            let result = parser.parse(&html);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn extract_text_never_panics(html in ".*") {
            let parser = HtmlParser::new();
            if let Ok(dom) = parser.parse(&html) {
                let _ = parser.extract_text(&dom);
            }
        }
    }
}
