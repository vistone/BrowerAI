use anyhow::Result;
use html5ever::parse_document;
use html5ever::tendril::TendrilSink;
use markup5ever_rcdom::{Handle, NodeData, RcDom};
use std::default::Default;
use std::io::Cursor;

use std::path::PathBuf;

use crate::ai::integration::HtmlModelIntegration;
use crate::ai::model_manager::ModelType;
use crate::ai::{AiRuntime, InferenceEngine};

/// HTML parser with AI enhancement capabilities
pub struct HtmlParser {
    inference_engine: Option<InferenceEngine>,
    ai_runtime: Option<AiRuntime>,
    model_path: Option<PathBuf>,
    model_name: Option<String>,
    enable_ai: bool,
}

impl HtmlParser {
    /// Create a new HTML parser
    pub fn new() -> Self {
        Self {
            inference_engine: None,
            ai_runtime: None,
            model_path: None,
            model_name: None,
            enable_ai: false,
        }
    }

    /// Create a new HTML parser with AI capabilities
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

    /// Create a new HTML parser with AI runtime (engine + model catalog + monitor)
    #[allow(dead_code)]
    pub fn with_ai_runtime(ai_runtime: AiRuntime) -> Self {
        let (model_name, model_path) = ai_runtime
            .best_model(ModelType::HtmlParser)
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

    /// Parse HTML content into a DOM tree
    pub fn parse(&self, html: &str) -> Result<RcDom> {
        use std::time::Instant;
        use crate::ai::config::FallbackReason;

        let input = Cursor::new(html.as_bytes());
        let dom = parse_document(RcDom::default(), Default::default())
            .from_utf8()
            .read_from(&mut input.clone())?;

        log::info!("Successfully parsed HTML document");

        // Check if AI runtime is available and AI is enabled
        let runtime = self.ai_runtime.as_ref();
        let ai_enabled = runtime.map_or(false, |r| r.is_ai_enabled()) && self.enable_ai;

        if !ai_enabled {
            if self.enable_ai && runtime.is_none() {
                log::debug!("AI enhancement disabled for HTML parsing; no runtime available");
            } else {
                log::debug!("AI enhancement disabled for HTML parsing; baseline path in use");
            }
            return Ok(dom);
        }

        let runtime = runtime.unwrap();
        let tracker = runtime.fallback_tracker();
        tracker.record_attempt();

        // Try AI enhancement
        let ai_active = self.enable_ai && self.inference_engine.is_some();
        if !ai_active {
            tracker.record_fallback(FallbackReason::NoModelAvailable);
            log::warn!(
                "AI was requested for HTML parsing but no inference engine is available; falling back to baseline parser"
            );
            return Ok(dom);
        }

        let monitor = runtime.monitor();
        let model_path = self.model_path.as_deref();
        let model_name = self.model_name.as_deref().unwrap_or("html_model");
        let start_time = Instant::now();

        match HtmlModelIntegration::new(
            self.inference_engine.as_ref().unwrap(),
            model_path,
            monitor,
        ) {
            Ok(mut integration) => {
                match integration.validate_structure(html) {
                    Ok((valid, complexity)) => {
                        let elapsed_ms = start_time.elapsed().as_millis() as u64;
                        tracker.record_success(elapsed_ms);
                        log::info!(
                            "AI HTML validation (model={}, {}ms): valid={} complexity={:.3}",
                            model_name, elapsed_ms, valid, complexity
                        );
                    }
                    Err(err) => {
                        tracker.record_fallback(FallbackReason::InferenceFailed(err.to_string()));
                        log::warn!(
                            "AI HTML validation failed (model={}): {}; using baseline output",
                            model_name,
                            err
                        );
                    }
                }
            }
            Err(err) => {
                tracker.record_fallback(FallbackReason::ModelLoadFailed(err.to_string()));
                log::warn!(
                    "AI HTML integration could not start (model={}): {}; continuing without AI",
                    model_name,
                    err
                );
            }
        }

        Ok(dom)
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

impl Default for HtmlParser {
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
        assert!(result.is_ok()); // html5ever is forgiving
    }
}
