//! BrowerAI AI Core
//!
//! AI驱动的浏览器引擎核心，提供：
//! - 模型管理 (ModelManager)
//! - ONNX推理引擎
//! - 特征提取 (FeatureExtractor)
//! - AI辅助渲染
//! - 学习系统
//!
//! # 特性
//! - `onnx`: 启用ONNX运行时支持
//! - `candle`: 启用Candle GGUF模型支持
//!
//! # 示例
//! ```
//! use browerai_ai_core::{AiCore, ModelConfig};
//!
//! let ai = AiCore::new().unwrap();
//! // 使用AI功能...
//! ```

#![warn(missing_docs)]

use browerai_core::Result;

pub mod models;
pub mod features;
pub mod learning;
pub mod inference;

pub use models::{ModelManager, ModelConfig, ModelInfo, ModelType};
pub use features::{FeatureExtractor, FeatureVector, FeatureType};
pub use learning::{LearningEngine, TrainingSample, LearningConfig};
pub use inference::{InferenceEngine, InferenceRequest, InferenceResult};

/// AI核心
///
/// BrowerAI的AI功能入口点，整合模型管理、特征提取、推理和学习
#[derive(Debug)]
pub struct AiCore {
    /// 模型管理器
    model_manager: ModelManager,
    /// 特征提取器
    feature_extractor: FeatureExtractor,
    /// 推理引擎
    inference_engine: InferenceEngine,
    /// 学习引擎
    learning_engine: LearningEngine,
}

impl AiCore {
    /// 创建新的AI核心
    ///
    /// # 示例
    /// ```
    /// use browerai_ai_core::AiCore;
    ///
    /// let ai = AiCore::new().unwrap();
    /// ```
    pub fn new() -> Result<Self> {
        Ok(Self {
            model_manager: ModelManager::new(),
            feature_extractor: FeatureExtractor::new(),
            inference_engine: InferenceEngine::new()?,
            learning_engine: LearningEngine::new(),
        })
    }

    /// 使用配置创建AI核心
    pub fn with_config(config: AiCoreConfig) -> Result<Self> {
        Ok(Self {
            model_manager: ModelManager::with_config(config.model_config)?,
            feature_extractor: FeatureExtractor::with_config(config.feature_config),
            inference_engine: InferenceEngine::with_config(config.inference_config)?,
            learning_engine: LearningEngine::with_config(config.learning_config),
        })
    }

    /// 获取模型管理器
    pub fn model_manager(&self) -> &ModelManager {
        &self.model_manager
    }

    /// 获取特征提取器
    pub fn feature_extractor(&self) -> &FeatureExtractor {
        &self.feature_extractor
    }

    /// 获取推理引擎
    pub fn inference_engine(&self) -> &InferenceEngine {
        &self.inference_engine
    }

    /// 获取学习引擎
    pub fn learning_engine(&self) -> &LearningEngine {
        &self.learning_engine
    }

    /// 执行推理（便捷方法）
    pub fn infer(&self, request: InferenceRequest) -> Result<InferenceResult> {
        self.inference_engine.infer(request)
    }

    /// 提取特征（便捷方法）
    pub fn extract_features(&self, input: &str, feature_type: FeatureType) -> Result<FeatureVector> {
        self.feature_extractor.extract(input, feature_type)
    }

    /// 检查AI核心是否可用
    pub fn is_available(&self) -> bool {
        self.model_manager.has_available_models()
    }

    /// 获取状态信息
    pub fn status(&self) -> AiCoreStatus {
        AiCoreStatus {
            models_loaded: self.model_manager.loaded_model_count(),
            models_available: self.model_manager.available_model_count(),
            features_supported: self.feature_extractor.supported_features(),
            learning_enabled: self.learning_engine.is_enabled(),
        }
    }
}

impl Default for AiCore {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            model_manager: ModelManager::new(),
            feature_extractor: FeatureExtractor::new(),
            inference_engine: InferenceEngine::default(),
            learning_engine: LearningEngine::new(),
        })
    }
}

/// AI核心配置
#[derive(Debug, Clone, Default)]
pub struct AiCoreConfig {
    /// 模型配置
    pub model_config: models::ModelManagerConfig,
    /// 特征配置
    pub feature_config: features::FeatureExtractorConfig,
    /// 推理配置
    pub inference_config: inference::InferenceConfig,
    /// 学习配置
    pub learning_config: learning::LearningConfig,
}

/// AI核心状态
#[derive(Debug, Clone)]
pub struct AiCoreStatus {
    /// 已加载模型数量
    pub models_loaded: usize,
    /// 可用模型数量
    pub models_available: usize,
    /// 支持的特征类型
    pub features_supported: Vec<FeatureType>,
    /// 学习是否启用
    pub learning_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_core_creation() {
        let ai = AiCore::new();
        assert!(ai.is_ok());
    }

    #[test]
    fn test_ai_core_default() {
        let ai: AiCore = Default::default();
        let status = ai.status();
        
        // 默认状态下可能没有加载模型
        assert_eq!(status.models_loaded, 0);
    }

    #[test]
    fn test_ai_core_config() {
        let config = AiCoreConfig::default();
        let ai = AiCore::with_config(config);
        
        assert!(ai.is_ok());
    }
}
