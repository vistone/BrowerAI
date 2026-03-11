pub mod decoder;
pub mod framework_detector;
pub mod http_client;
pub mod hybrid_framework_integration;
pub mod integration;
#[cfg(feature = "js_orchestrator")]
pub mod js_orchestrator;
pub mod services;
pub mod tokenizer;

pub use decoder::beam_search::BeamSearchParams;
pub use framework_detector::FrameworkDetectorIntegration;
pub use http_client::{BatchDetectResponse, DetectResponse, FrameworkDetectorClient};
pub use hybrid_framework_integration::{
    DetectionMethod, FrameworkDetectionResult, HybridFrameworkIntegration,
};
pub use integration::{CssModelIntegration, HtmlModelIntegration, JsDeobfuscatorIntegration};
#[cfg(feature = "js_orchestrator")]
pub use js_orchestrator::{HybridJsOrchestrator, OrchestrationPolicy};
pub use services::deobf_compose_service::{DeobfComposeConfig, DeobfComposeService};
pub use tokenizer::CharTokenizer;

// Re-exports from browerai_ai_core
pub use browerai_ai_core::{
    AiCore, FeatureExtractor, FeatureType, FeatureVector, InferenceEngine, InferenceRequest,
    InferenceResult, LearningConfig, LearningEngine, ModelConfig, ModelInfo, ModelManager,
    ModelType, TrainingSample,
};
