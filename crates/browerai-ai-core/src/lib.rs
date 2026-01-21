pub mod advanced_monitor;
pub mod ai_trait;
#[cfg(feature = "candle")]
pub mod candle_loader;
pub mod code_predictor;
pub mod config;
pub mod feedback_pipeline;
pub mod feedback_training_loop;
pub mod gpu_support;
pub mod hot_reload;
#[cfg(feature = "onnx")]
pub mod inference;
#[cfg(feature = "onnx")]
pub mod model_loader;
pub mod model_manager;
#[cfg(feature = "onnx")]
pub mod onnx_model;
pub mod performance_monitor;
#[cfg(feature = "onnx")]
pub mod reporter;
#[cfg(feature = "onnx")]
pub mod runtime;
pub mod smart_features;
#[cfg(feature = "onnx")]
pub mod tech_model_library;
pub mod trainable_model;

pub mod llm_gateway;

pub use advanced_monitor::AdvancedPerformanceMonitor;
pub use ai_trait::{
    AiInferenceError, AiModel, DataType, ModelMetadata, ModelType, Tensor, TensorBuilder,
    TensorSpec,
};
#[cfg(feature = "candle")]
pub use candle_loader::{CandleCodeLlm, CandleModelLoader};
pub use code_predictor::CodePredictorModel;
pub use config::{AiConfig, AiStats, FallbackReason, FallbackTracker};
pub use feedback_pipeline::FeedbackPipeline;
pub use feedback_training_loop::{
    FeedbackLoopConfig, FeedbackLoopStats, FeedbackTrainingLoop, LayoutFeedback, TrainingBatch,
    TrainingSignal,
};
pub use gpu_support::{GpuConfig, GpuProvider, GpuStats};
pub use hot_reload::HotReloadManager;
#[cfg(feature = "onnx")]
pub use inference::InferenceEngine;
pub use llm_gateway::{LlmConfig, LlmGatewayConfig, LlmProvider, LlmResponse, ProviderConfig};
#[cfg(feature = "onnx")]
pub use model_loader::ModelLoader;
pub use model_manager::{ModelHealth, ModelHealthSummary, ModelManager};
#[cfg(feature = "onnx")]
pub use onnx_model::{OnnxModel, OnnxModelStub};
#[cfg(feature = "onnx")]
pub use reporter::AiReporter;
#[cfg(feature = "onnx")]
pub use runtime::AiRuntime;
#[cfg(feature = "onnx")]
pub use tech_model_library::{
    GenerateTask, InferTask, JsDeobfuscationInfer, LayoutRegeneration, LearnTask, ModelRegistry,
    ModelSpec, TaskKind,
};
pub use trainable_model::{TrainableModelConfig, TrainableOnnxModel, TrainingStats};
