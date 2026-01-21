//! Safe Sandbox Module
//!
//! Sandboxed environment for safe web page fetching and analysis.
//! This module provides the foundation for capturing page content without
//! modifying the original page logic.

pub mod behavior_recorder;
pub mod page_fetcher;

pub use behavior_recorder::BehaviorRecorder;
pub use page_fetcher::PageFetcher;
