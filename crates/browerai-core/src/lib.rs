pub mod config;
/// Core types and traits for BrowerAI browser engine
pub mod error;
pub mod traits;

pub use config::BrowserConfig;
pub use error::{BrowserError, Result};
pub use traits::{AiModel, Parser, Renderer};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
