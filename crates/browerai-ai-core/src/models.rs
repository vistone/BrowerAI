//! Model Management - 模型管理
//!
//! 管理AI模型的加载、卸载和生命周期，包括：
//! - 模型注册和发现
//! - 模型加载和缓存
//! - 版本管理
//! - 健康检查

use browerai_core::{BrowserError, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// 模型管理器
#[derive(Debug)]
pub struct ModelManager {
    /// 已加载的模型
    models: HashMap<String, ModelHandle>,
    /// 模型配置
    config: ModelManagerConfig,
    /// 模型存储路径
    model_path: PathBuf,
}

impl ModelManager {
    /// 创建新的模型管理器
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            config: ModelManagerConfig::default(),
            model_path: PathBuf::from("./models"),
        }
    }

    /// 使用配置创建模型管理器
    pub fn with_config(config: ModelManagerConfig) -> Result<Self> {
        Ok(Self {
            models: HashMap::new(),
            model_path: config.model_path.clone(),
            config,
        })
    }

    /// 注册模型
    pub fn register(&mut self, config: ModelConfig) -> Result<()> {
        let model_id = config.id.clone();
        
        let handle = ModelHandle {
            config,
            status: ModelStatus::Registered,
        };
        
        self.models.insert(model_id, handle);
        Ok(())
    }

    /// 加载模型
    pub fn load(&mut self, model_id: &str) -> Result<()> {
        let handle = self.models.get_mut(model_id)
            .ok_or_else(|| BrowserError::ai(format!("Model not found: {}", model_id)))?;
        
        // 实际实现需要加载模型文件到内存
        handle.status = ModelStatus::Loaded;
        
        Ok(())
    }

    /// 卸载模型
    pub fn unload(&mut self, model_id: &str) -> Result<()> {
        let handle = self.models.get_mut(model_id)
            .ok_or_else(|| BrowserError::ai(format!("Model not found: {}", model_id)))?;
        
        handle.status = ModelStatus::Registered;
        Ok(())
    }

    /// 获取模型信息
    pub fn get_model_info(&self, model_id: &str) -> Option<ModelInfo> {
        self.models.get(model_id).map(|h| ModelInfo {
            id: h.config.id.clone(),
            name: h.config.name.clone(),
            model_type: h.config.model_type,
            status: h.status,
            version: h.config.version.clone(),
        })
    }

    /// 列出所有模型
    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.models.values()
            .map(|h| ModelInfo {
                id: h.config.id.clone(),
                name: h.config.name.clone(),
                model_type: h.config.model_type,
                status: h.status,
                version: h.config.version.clone(),
            })
            .collect()
    }

    /// 列出已加载的模型
    pub fn list_loaded_models(&self) -> Vec<ModelInfo> {
        self.models.values()
            .filter(|h| matches!(h.status, ModelStatus::Loaded))
            .map(|h| ModelInfo {
                id: h.config.id.clone(),
                name: h.config.name.clone(),
                model_type: h.config.model_type,
                status: h.status,
                version: h.config.version.clone(),
            })
            .collect()
    }

    /// 检查模型是否存在
    pub fn has_model(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    /// 检查模型是否已加载
    pub fn is_loaded(&self, model_id: &str) -> bool {
        self.models.get(model_id)
            .map(|h| matches!(h.status, ModelStatus::Loaded))
            .unwrap_or(false)
    }

    /// 获取已加载模型数量
    pub fn loaded_model_count(&self) -> usize {
        self.models.values()
            .filter(|h| matches!(h.status, ModelStatus::Loaded))
            .count()
    }

    /// 获取可用模型数量
    pub fn available_model_count(&self) -> usize {
        self.models.len()
    }

    /// 检查是否有可用模型
    pub fn has_available_models(&self) -> bool {
        !self.models.is_empty()
    }

    /// 扫描模型目录
    pub fn scan_models(&mut self, path: &Path) -> Result<Vec<String>> {
        let discovered = Vec::new();
        
        // 简化实现：扫描目录中的.onnx文件
        if path.exists() {
            // 实际实现需要遍历目录
            log::info!("Scanning models in: {:?}", path);
        }
        
        Ok(discovered)
    }

    /// 获取模型路径
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// 设置模型路径
    pub fn set_model_path(&mut self, path: PathBuf) {
        self.model_path = path;
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

/// 模型句柄
#[derive(Debug, Clone)]
struct ModelHandle {
    /// 模型配置
    config: ModelConfig,
    /// 模型状态
    status: ModelStatus,
}

/// 模型配置
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// 模型ID
    pub id: String,
    /// 模型名称
    pub name: String,
    /// 模型类型
    pub model_type: ModelType,
    /// 模型版本
    pub version: String,
    /// 模型文件路径
    pub path: PathBuf,
    /// 输入维度
    pub input_shape: Vec<usize>,
    /// 输出维度
    pub output_shape: Vec<usize>,
    /// 描述
    pub description: Option<String>,
}

impl ModelConfig {
    /// 创建新的模型配置
    pub fn new(id: impl Into<String>, name: impl Into<String>, model_type: ModelType) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            model_type,
            version: "1.0.0".to_string(),
            path: PathBuf::new(),
            input_shape: vec![],
            output_shape: vec![],
            description: None,
        }
    }

    /// 设置版本
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// 设置路径
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = path.into();
        self
    }

    /// 设置输入形状
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = shape;
        self
    }

    /// 设置输出形状
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = shape;
        self
    }
}

/// 模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// ONNX模型
    Onnx,
    /// TensorFlow模型
    TensorFlow,
    /// PyTorch模型
    PyTorch,
    /// GGUF模型(Candle)
    Gguf,
    /// 自定义模型
    Custom,
}

/// 模型状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    /// 已注册但未加载
    Registered,
    /// 正在加载
    Loading,
    /// 已加载
    Loaded,
    /// 加载失败
    Failed,
    /// 已卸载
    Unloaded,
}

/// 模型信息
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// 模型ID
    pub id: String,
    /// 模型名称
    pub name: String,
    /// 模型类型
    pub model_type: ModelType,
    /// 模型状态
    pub status: ModelStatus,
    /// 版本
    pub version: String,
}

/// 模型管理器配置
#[derive(Debug, Clone)]
pub struct ModelManagerConfig {
    /// 模型存储路径
    pub model_path: PathBuf,
    /// 最大并发加载数
    pub max_concurrent_loads: usize,
    /// 自动加载已注册模型
    pub auto_load: bool,
    /// 缓存策略
    pub cache_policy: CachePolicy,
}

impl Default for ModelManagerConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("./models"),
            max_concurrent_loads: 2,
            auto_load: false,
            cache_policy: CachePolicy::LRU,
        }
    }
}

/// 缓存策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// 最近最少使用
    LRU,
    /// 先进先出
    FIFO,
    /// 不缓存
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_manager_creation() {
        let manager = ModelManager::new();
        assert_eq!(manager.loaded_model_count(), 0);
    }

    #[test]
    fn test_model_registration() {
        let mut manager = ModelManager::new();
        let config = ModelConfig::new("test-model", "Test Model", ModelType::Onnx);
        
        assert!(manager.register(config).is_ok());
        assert!(manager.has_model("test-model"));
    }

    #[test]
    fn test_model_load_unload() {
        let mut manager = ModelManager::new();
        let config = ModelConfig::new("test-model", "Test Model", ModelType::Onnx);
        
        manager.register(config).unwrap();
        
        assert!(!manager.is_loaded("test-model"));
        
        // 加载模型
        assert!(manager.load("test-model").is_ok());
        assert!(manager.is_loaded("test-model"));
        
        // 卸载模型
        assert!(manager.unload("test-model").is_ok());
        assert!(!manager.is_loaded("test-model"));
    }

    #[test]
    fn test_model_info() {
        let mut manager = ModelManager::new();
        let config = ModelConfig::new("test-model", "Test Model", ModelType::Onnx)
            .with_version("2.0.0");
        
        manager.register(config).unwrap();
        
        let info = manager.get_model_info("test-model").unwrap();
        assert_eq!(info.id, "test-model");
        assert_eq!(info.name, "Test Model");
        assert_eq!(info.version, "2.0.0");
    }

    #[test]
    fn test_list_models() {
        let mut manager = ModelManager::new();
        
        manager.register(ModelConfig::new("model-1", "Model 1", ModelType::Onnx)).unwrap();
        manager.register(ModelConfig::new("model-2", "Model 2", ModelType::Gguf)).unwrap();
        
        let models = manager.list_models();
        assert_eq!(models.len(), 2);
    }
}
