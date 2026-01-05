use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use crate::learning::feedback::{Feedback, FeedbackType};

/// Online feedback pipeline for collecting various browser events for subsequent learning
#[derive(Clone)]
pub struct FeedbackPipeline {
    events: Arc<RwLock<VecDeque<FeedbackEvent>>>,
    max_capacity: usize,
}

/// Feedback event types
#[derive(Debug, Clone)]
pub enum FeedbackEvent {
    /// HTML parsing event
    HtmlParsing {
        success: bool,
        complexity: f32,
        ai_used: bool,
        error: Option<String>,
        /// Actual HTML content (for obfuscation analysis)
        content: Option<String>,
        /// HTML size
        size: usize,
    },
    /// CSS parsing event
    CssParsing {
        success: bool,
        rule_count: usize,
        ai_used: bool,
        error: Option<String>,
        /// Actual CSS content
        content: Option<String>,
    },
    /// JavaScript parsing event
    JsParsing {
        success: bool,
        statement_count: usize,
        compatibility_warnings: Vec<String>,
        ai_used: bool,
        error: Option<String>,
        /// Actual JS content (for obfuscation detection)
        content: Option<String>,
    },
    /// JavaScript execution compatibility violation
    JsCompatibilityViolation {
        feature: String,
        detail: String,
        enforced: bool,
    },
    /// Rendering performance event
    RenderingPerformance {
        node_count: usize,
        duration_ms: f64,
        ai_optimized: bool,
    },
    /// Layout performance event
    LayoutPerformance {
        element_count: usize,
        duration_ms: f64,
        ai_optimized: bool,
    },
    /// AI model inference event
    ModelInference {
        model_name: String,
        success: bool,
        duration_ms: f64,
        error: Option<String>,
    },
}

impl FeedbackPipeline {
    /// Create a new feedback pipeline
    pub fn new(max_capacity: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(VecDeque::with_capacity(max_capacity))),
            max_capacity,
        }
    }

    /// Add event to pipeline
    pub fn push_event(&self, event: FeedbackEvent) {
        let mut events = self.events.write().unwrap();
        
        if events.len() >= self.max_capacity {
            events.pop_front(); // Remove oldest event
        }
        
        events.push_back(event);
    }

    /// Record HTML parsing event
    pub fn record_html_parsing(&self, success: bool, complexity: f32, ai_used: bool, error: Option<String>, content: Option<String>, size: usize) {
        self.push_event(FeedbackEvent::HtmlParsing {
            success,
            complexity,
            ai_used,
            error,
            content,
            size,
        });
    }

    /// Record CSS parsing event
    pub fn record_css_parsing(&self, success: bool, rule_count: usize, ai_used: bool, error: Option<String>, content: Option<String>) {
        self.push_event(FeedbackEvent::CssParsing {
            success,
            rule_count,
            ai_used,
            error,
            content,
        });
    }

    /// Record JavaScript parsing event
    pub fn record_js_parsing(
        &self,
        success: bool,
        statement_count: usize,
        compatibility_warnings: Vec<String>,
        ai_used: bool,
        error: Option<String>,
        content: Option<String>,
    ) {
        self.push_event(FeedbackEvent::JsParsing {
            success,
            statement_count,
            compatibility_warnings,
            ai_used,
            error,
            content,
        });
    }

    /// Record JS compatibility violation
    pub fn record_js_compatibility_violation(&self, feature: String, detail: String, enforced: bool) {
        self.push_event(FeedbackEvent::JsCompatibilityViolation {
            feature,
            detail,
            enforced,
        });
    }

    /// Record rendering performance event
    pub fn record_rendering_performance(&self, node_count: usize, duration_ms: f64, ai_optimized: bool) {
        self.push_event(FeedbackEvent::RenderingPerformance {
            node_count,
            duration_ms,
            ai_optimized,
        });
    }

    /// Record layout performance event
    pub fn record_layout_performance(&self, element_count: usize, duration_ms: f64, ai_optimized: bool) {
        self.push_event(FeedbackEvent::LayoutPerformance {
            element_count,
            duration_ms,
            ai_optimized,
        });
    }

    /// Record model inference event
    pub fn record_model_inference(&self, model_name: String, success: bool, duration_ms: f64, error: Option<String>) {
        self.push_event(FeedbackEvent::ModelInference {
            model_name,
            success,
            duration_ms,
            error,
        });
    }

    /// Get all events
    pub fn get_events(&self) -> Vec<FeedbackEvent> {
        self.events.read().unwrap().iter().cloned().collect()
    }

    /// Clear event queue
    pub fn clear(&self) {
        self.events.write().unwrap().clear();
    }

    /// Get event count
    pub fn len(&self) -> usize {
        self.events.read().unwrap().len()
    }

    /// Convert to training sample format (exportable to JSON)
    pub fn export_training_samples(&self) -> Result<String> {
        let events = self.get_events();
        let samples: Vec<serde_json::Value> = events
            .iter()
            .map(|event| self.event_to_json(event))
            .collect();

        Ok(serde_json::to_string_pretty(&samples)?)
    }

    /// Convert event to JSON
    fn event_to_json(&self, event: &FeedbackEvent) -> serde_json::Value {
        match event {
            FeedbackEvent::HtmlParsing { success, complexity, ai_used, error, content, size } => {
                serde_json::json!({
                    "type": "html_parsing",
                    "success": success,
                    "complexity": complexity,
                    "ai_used": ai_used,
                    "error": error,
                    "content": content,
                    "size": size,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::CssParsing { success, rule_count, ai_used, error, content } => {
                serde_json::json!({
                    "type": "css_parsing",
                    "success": success,
                    "rule_count": rule_count,
                    "ai_used": ai_used,
                    "error": error,
                    "content": content,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::JsParsing { success, statement_count, compatibility_warnings, ai_used, error, content } => {
                serde_json::json!({
                    "type": "js_parsing",
                    "success": success,
                    "statement_count": statement_count,
                    "compatibility_warnings": compatibility_warnings,
                    "ai_used": ai_used,
                    "error": error,
                    "content": content,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::JsCompatibilityViolation { feature, detail, enforced } => {
                serde_json::json!({
                    "type": "js_compatibility_violation",
                    "feature": feature,
                    "detail": detail,
                    "enforced": enforced,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::RenderingPerformance { node_count, duration_ms, ai_optimized } => {
                serde_json::json!({
                    "type": "rendering_performance",
                    "node_count": node_count,
                    "duration_ms": duration_ms,
                    "ai_optimized": ai_optimized,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::LayoutPerformance { element_count, duration_ms, ai_optimized } => {
                serde_json::json!({
                    "type": "layout_performance",
                    "element_count": element_count,
                    "duration_ms": duration_ms,
                    "ai_optimized": ai_optimized,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            FeedbackEvent::ModelInference { model_name, success, duration_ms, error } => {
                serde_json::json!({
                    "type": "model_inference",
                    "model_name": model_name,
                    "success": success,
                    "duration_ms": duration_ms,
                    "error": error,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
        }
    }

    /// Generate statistics summary
    pub fn generate_summary(&self) -> String {
        let events = self.get_events();
        let mut summary = String::new();

        summary.push_str("【Feedback Pipeline Statistics】\n");
        summary.push_str(&format!("  Total events: {}\n", events.len()));

        // Count each event type
        let html_count = events.iter().filter(|e| matches!(e, FeedbackEvent::HtmlParsing { .. })).count();
        let css_count = events.iter().filter(|e| matches!(e, FeedbackEvent::CssParsing { .. })).count();
        let js_count = events.iter().filter(|e| matches!(e, FeedbackEvent::JsParsing { .. })).count();
        let violation_count = events.iter().filter(|e| matches!(e, FeedbackEvent::JsCompatibilityViolation { .. })).count();

        summary.push_str(&format!("  HTML parsing events: {}\n", html_count));
        summary.push_str(&format!("  CSS parsing events: {}\n", css_count));
        summary.push_str(&format!("  JS parsing events: {}\n", js_count));
        summary.push_str(&format!("  Compatibility violations: {}\n", violation_count));

        summary
    }
}

impl Default for FeedbackPipeline {
    fn default() -> Self {
        Self::new(10000) // Default save 10000 events
    }
}
