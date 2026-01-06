/// Core types and traits for BrowerAI browser engine
pub mod error;
pub mod traits;
pub mod config;

pub use error::{BrowserError, Result};
pub use traits::{Parser, Renderer, AiModel};
pub use config::BrowserConfig;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
