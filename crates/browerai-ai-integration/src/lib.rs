pub mod decoder;
pub mod integration;
#[cfg(feature = "js_orchestrator")]
pub mod js_orchestrator;
pub mod services;
pub mod tokenizer;

pub use decoder::beam_search::BeamSearchParams;
pub use integration::{CssModelIntegration, HtmlModelIntegration, JsDeobfuscatorIntegration};
#[cfg(feature = "js_orchestrator")]
pub use js_orchestrator::{HybridJsOrchestrator, OrchestrationPolicy};
pub use services::deobf_compose_service::{DeobfComposeConfig, DeobfComposeService};
pub use tokenizer::CharTokenizer;

#[cfg(feature = "ai")]
pub use browerai_ai_core::advanced_monitor::AdvancedPerformanceMonitor;
pub use browerai_ai_core::config::{AiConfig, AiStats, FallbackReason, FallbackTracker};
pub use browerai_ai_core::feedback_pipeline::FeedbackPipeline;
pub use browerai_ai_core::gpu_support::{GpuConfig, GpuProvider, GpuStats};
pub use browerai_ai_core::hot_reload::HotReloadManager;

#[cfg(feature = "ai")]
pub use browerai_ai_core::inference::InferenceEngine;
#[cfg(feature = "ai")]
pub use browerai_ai_core::model_loader::ModelLoader;
pub use browerai_ai_core::model_manager::{ModelHealth, ModelHealthSummary, ModelManager};
pub use browerai_ai_core::performance_monitor::PerformanceMonitor;

#[cfg(feature = "ai")]
pub use browerai_ai_core::reporter::AiReporter;
#[cfg(feature = "ai")]
pub use browerai_ai_core::runtime::AiRuntime;
