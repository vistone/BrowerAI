pub mod integration;

pub use integration::{CssModelIntegration, HtmlModelIntegration, JsDeobfuscatorIntegration};

pub use browerai_ai_core::advanced_monitor::AdvancedPerformanceMonitor;
pub use browerai_ai_core::config::{AiConfig, AiStats, FallbackReason, FallbackTracker};
pub use browerai_ai_core::feedback_pipeline::FeedbackPipeline;
pub use browerai_ai_core::gpu_support::{GpuConfig, GpuProvider, GpuStats};
pub use browerai_ai_core::hot_reload::HotReloadManager;
pub use browerai_ai_core::inference::InferenceEngine;
pub use browerai_ai_core::model_loader::ModelLoader;
pub use browerai_ai_core::model_manager::{ModelHealth, ModelHealthSummary, ModelManager};
pub use browerai_ai_core::performance_monitor::PerformanceMonitor;
pub use browerai_ai_core::reporter::AiReporter;
pub use browerai_ai_core::runtime::AiRuntime;
