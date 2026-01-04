use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Configuration for a model in the local model library
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub model_type: ModelType,
    pub path: PathBuf,
    pub description: String,
    pub version: String,
}

/// Types of models supported by BrowerAI
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelType {
    HtmlParser,
    CssParser,
    JsParser,
    LayoutOptimizer,
    RenderingOptimizer,
}

/// Manages the local model library for AI-powered browser operations
pub struct ModelManager {
    models: HashMap<ModelType, Vec<ModelConfig>>,
    model_dir: PathBuf,
}

impl ModelManager {
    /// Create a new ModelManager with the specified model directory
    pub fn new(model_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&model_dir)
            .context("Failed to create model directory")?;
        
        Ok(Self {
            models: HashMap::new(),
            model_dir,
        })
    }

    /// Register a model in the model library
    pub fn register_model(&mut self, config: ModelConfig) -> Result<()> {
        let model_path = self.model_dir.join(&config.path);
        
        if !model_path.exists() {
            log::warn!("Model file does not exist yet: {:?}", model_path);
        }

        self.models
            .entry(config.model_type.clone())
            .or_insert_with(Vec::new)
            .push(config);

        Ok(())
    }

    /// Get all models of a specific type
    pub fn get_models(&self, model_type: &ModelType) -> Vec<&ModelConfig> {
        self.models
            .get(model_type)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get the default model for a specific type (returns the first one)
    pub fn get_default_model(&self, model_type: &ModelType) -> Option<&ModelConfig> {
        self.models
            .get(model_type)
            .and_then(|v| v.first())
    }

    /// Load model configuration from a TOML file
    pub fn load_config(&mut self, config_path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(config_path)
            .context("Failed to read model config file")?;
        
        let configs: Vec<ModelConfig> = toml::from_str(&content)
            .context("Failed to parse model config")?;

        for config in configs {
            self.register_model(config)?;
        }

        Ok(())
    }

    /// Save current model configurations to a TOML file
    pub fn save_config(&self, config_path: &Path) -> Result<()> {
        let all_configs: Vec<ModelConfig> = self.models
            .values()
            .flatten()
            .cloned()
            .collect();

        let content = toml::to_string_pretty(&all_configs)
            .context("Failed to serialize model config")?;

        std::fs::write(config_path, content)
            .context("Failed to write model config file")?;

        Ok(())
    }

    /// Get the model directory path
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_model_manager_creation() {
        let temp_dir = tempdir().unwrap();
        let manager = ModelManager::new(temp_dir.path().to_path_buf());
        assert!(manager.is_ok());
    }

    #[test]
    fn test_register_and_get_model() {
        let temp_dir = tempdir().unwrap();
        let mut manager = ModelManager::new(temp_dir.path().to_path_buf()).unwrap();

        let config = ModelConfig {
            name: "test_html_parser".to_string(),
            model_type: ModelType::HtmlParser,
            path: PathBuf::from("html_parser_v1.onnx"),
            description: "Test HTML parser model".to_string(),
            version: "1.0.0".to_string(),
        };

        manager.register_model(config.clone()).unwrap();
        let models = manager.get_models(&ModelType::HtmlParser);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test_html_parser");
    }
}
