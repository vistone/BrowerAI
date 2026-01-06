pub mod advanced_monitor;
#[cfg(feature = "candle")]
pub mod candle_loader;
pub mod config;
pub mod feedback_pipeline;
pub mod gpu_support;
pub mod hot_reload;
pub mod inference;
pub mod model_loader;
pub mod model_manager;
pub mod performance_monitor;
pub mod reporter;
pub mod runtime;
pub mod smart_features;

pub use advanced_monitor::AdvancedPerformanceMonitor;
#[cfg(feature = "candle")]
pub use candle_loader::{CandleCodeLlm, CandleModelLoader};
pub use config::{AiConfig, AiStats, FallbackReason, FallbackTracker};
pub use feedback_pipeline::FeedbackPipeline;
pub use gpu_support::{GpuConfig, GpuProvider, GpuStats};
pub use hot_reload::HotReloadManager;
pub use inference::InferenceEngine;
pub use model_loader::ModelLoader;
pub use model_manager::{ModelHealth, ModelHealthSummary, ModelManager};
pub use reporter::AiReporter;
pub use runtime::AiRuntime;
