pub mod code_type;
pub mod config;
/// Core types and traits for BrowerAI browser engine
pub mod error;
pub mod health;
pub mod metrics;
pub mod source_loc;
pub mod traits;

pub use code_type::CodeType;
pub use config::BrowserConfig;
pub use error::ai::{AiError, ModelLoadError};
pub use error::config::{ConfigError, ValidationError};
pub use error::network::{NetworkError, TimeoutError};
pub use error::parse::{CssParseError, HtmlParseError, JsParseError};
pub use error::render::{LayoutError, PaintError};
pub use error::{BrowserError, ErrorKind, Result};
pub use health::{FallbackReason, HealthStatus, HealthSummary, Issue, Severity};
pub use metrics::{Histogram, Metric, MetricStats, MetricType, MetricsDashboard};
pub use source_loc::{SourceInfo, SourceKind, SourceLocation, SourceSpan};
pub use traits::{AiModel, Parser, Renderer};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
