use std::path::PathBuf;
use std::sync::Arc;

use crate::ai::config::{AiConfig, FallbackTracker};
use crate::ai::feedback_pipeline::FeedbackPipeline;
use crate::ai::model_manager::{ModelConfig, ModelType};
use crate::ai::performance_monitor::PerformanceMonitor;
use crate::ai::InferenceEngine;
use crate::ai::ModelManager;

/// Shared AI runtime context that bundles the inference engine, optional model
/// catalog, and monitor handles for downstream components.
#[derive(Clone)]
pub struct AiRuntime {
    engine: InferenceEngine,
    models: Option<ModelManager>,
    feedback: FeedbackPipeline,
    config: Arc<AiConfig>,
    fallback_tracker: Arc<FallbackTracker>,
}

impl AiRuntime {
    /// Create a runtime with an inference engine but no model catalog.
    pub fn new(engine: InferenceEngine) -> Self {
        Self {
            engine,
            models: None,
            feedback: FeedbackPipeline::default(),
            config: Arc::new(AiConfig::default()),
            fallback_tracker: Arc::new(FallbackTracker::default()),
        }
    }

    /// Create a runtime with an inference engine and a model catalog.
    pub fn with_models(engine: InferenceEngine, model_manager: ModelManager) -> Self {
        Self {
            engine,
            models: Some(model_manager),
            feedback: FeedbackPipeline::default(),
            config: Arc::new(AiConfig::default()),
            fallback_tracker: Arc::new(FallbackTracker::default()),
        }
    }

    /// Create a runtime with custom configuration.
    pub fn with_config(engine: InferenceEngine, config: AiConfig) -> Self {
        Self {
            engine,
            models: None,
            feedback: FeedbackPipeline::default(),
            config: Arc::new(config),
            fallback_tracker: Arc::new(FallbackTracker::default()),
        }
    }

    /// Get a cloned inference engine for component use.
    pub fn engine(&self) -> InferenceEngine {
        self.engine.clone()
    }

    /// Get the performance monitor if available.
    pub fn monitor(&self) -> Option<PerformanceMonitor> {
        self.engine.monitor_handle()
    }

    /// Get the best model config and resolved file path for a given type.
    pub fn best_model(&self, model_type: ModelType) -> Option<(ModelConfig, PathBuf)> {
        self.models.as_ref().and_then(|manager| {
            manager.get_best_model(&model_type).map(|cfg| {
                let resolved_path = manager.model_dir().join(&cfg.path);
                // Canonicalize to absolute path for ONNX Runtime
                let absolute_path = resolved_path.canonicalize().unwrap_or(resolved_path);
                (cfg.clone(), absolute_path)
            })
        })
    }

    /// Check whether a model catalog is present.
    pub fn has_models(&self) -> bool {
        self.models.is_some()
    }

    /// Get the feedback pipeline for recording events.
    pub fn feedback(&self) -> &FeedbackPipeline {
        &self.feedback
    }

    /// Get the AI configuration.
    pub fn config(&self) -> &AiConfig {
        &self.config
    }

    /// Get the fallback tracker.
    pub fn fallback_tracker(&self) -> &FallbackTracker {
        &self.fallback_tracker
    }

    /// Check if AI is enabled.
    pub fn is_ai_enabled(&self) -> bool {
        self.config.enable_ai
    }
}
