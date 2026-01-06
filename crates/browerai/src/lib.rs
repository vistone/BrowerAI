//! BrowerAI - AI-Powered Browser Engine
//!
//! This is the main entry point that re-exports all public APIs from workspace crates.
//!
//! # Architecture
//!
//! The BrowerAI workspace consists of multiple specialized crates:
//! - **Core**: `browerai-core`, `browerai-dom`
//! - **Parsers**: `browerai-html-parser`, `browerai-css-parser`, `browerai-js-parser`, `browerai-js-analyzer`
//! - **AI**: `browerai-ai-core`, `browerai-ai-integration` (optional, enabled with `ai` feature)
//! - **Rendering**: `browerai-renderer-core`, `browerai-renderer-predictive`, `browerai-intelligent-rendering`
//! - **Learning**: `browerai-learning`
//! - **Utilities**: `browerai-network`, `browerai-devtools`, `browerai-testing`, `browerai-plugins`
//!
//! # Usage
//!
//! ```ignore
//! use browerai::prelude::*;
//!
//! let parser = HtmlParser::new();
//! let dom = parser.parse("<html><body>Hello!</body></html>")?;
//! ```

// Re-export core types
pub use browerai_core::{self as core, *};

// Re-export DOM
pub use browerai_dom as dom;

// Re-export parsers
pub use browerai_css_parser as css_parser;
pub use browerai_html_parser as html_parser;
pub use browerai_js_analyzer as js_analyzer;
pub use browerai_js_parser as js_parser;

// Re-export AI (conditional)
#[cfg(feature = "ai")]
pub use browerai_ai_core as ai;
#[cfg(feature = "ai")]
pub use browerai_ai_integration as ai_integration;

// Re-export renderers
pub use browerai_intelligent_rendering as intelligent_rendering;
pub use browerai_renderer_core as renderer;
pub use browerai_renderer_predictive as renderer_predictive;

// Re-export learning
pub use browerai_learning as learning;

// Re-export utilities
pub use browerai_devtools as devtools;
pub use browerai_network as network;
pub use browerai_plugins as plugins;
pub use browerai_testing as testing;

// Re-export ML toolkit (conditional)
#[cfg(feature = "ml")]
pub use browerai_ml as ml;

/// Prelude module for convenient imports
pub mod prelude {
    // Core
    pub use browerai_core::{BrowserError, Result};

    // DOM
    pub use browerai_dom::{Document, DomElement, DomNode, JsSandbox};

    // Parsers
    pub use browerai_css_parser::CssParser;
    pub use browerai_html_parser::HtmlParser;
    pub use browerai_js_analyzer::JsDeepAnalyzer;
    pub use browerai_js_parser::JsParser;

    // AI (conditional)
    #[cfg(feature = "ai")]
    pub use browerai_ai_core::{AiRuntime, InferenceEngine, ModelManager};

    // Renderers
    pub use browerai_renderer_core::RenderEngine;
    pub use browerai_renderer_predictive::PredictiveRenderer;

    // Learning
    pub use browerai_learning::{CodeGenerator, FeedbackCollector, JsDeobfuscator};

    // Network
    pub use browerai_network::{HttpClient, ResourceCache};

    // Testing
    pub use browerai_testing::{BenchmarkRunner, WebsiteTester};
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
