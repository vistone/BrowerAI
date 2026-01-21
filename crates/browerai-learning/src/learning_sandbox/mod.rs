//! Learning Sandbox Module
//!
//! Analyzes and learns from captured page content and behavior records.

pub mod intent_analyzer;
// pub mod tech_stack_detector;
// pub mod feature_recognizer;
// pub mod structure_analyzer;
// pub mod knowledge_graph;

pub use intent_analyzer::{IntentAnalyzer, WebsiteIntent};
