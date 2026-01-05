use anyhow::{Context, Result};
use log;
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
    #[serde(default)]
    pub priority: u8,
    #[serde(default)]
    pub health: ModelHealth,
}

/// Types of models supported by BrowerAI
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelType {
    HtmlParser,
    CssParser,
    JsParser,
    LayoutOptimizer,
    RenderingOptimizer,
    CodeUnderstanding,
    JsDeobfuscator,
}

/// Manages the local model library for AI-powered browser operations
#[derive(Clone)]
pub struct ModelManager {
    models: HashMap<ModelType, Vec<ModelConfig>>,
    model_dir: PathBuf,
}

/// Health status for a model record
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelHealth {
    Ready,
    MissingFile,
    Unknown,
}

impl Default for ModelHealth {
    fn default() -> Self {
        Self::Unknown
    }
}

impl ModelManager {
    /// Create a new ModelManager with the specified model directory
    pub fn new(model_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&model_dir).context("Failed to create model directory")?;

        Ok(Self {
            models: HashMap::new(),
            model_dir,
        })
    }

    /// Register a model in the model library
    pub fn register_model(&mut self, config: ModelConfig) -> Result<()> {
        let model_path = self.model_dir.join(&config.path);

        let mut config = config;

        if !model_path.exists() {
            log::warn!("Model file does not exist yet: {:?}", model_path);
            config.health = ModelHealth::MissingFile;
        } else {
            config.health = ModelHealth::Ready;
        }

        self.models
            .entry(config.model_type.clone())
            .or_default()
            .push(config);

        Ok(())
    }

    /// Get all models of a specific type
    #[allow(dead_code)]
    pub fn get_models(&self, model_type: &ModelType) -> Vec<&ModelConfig> {
        self.models
            .get(model_type)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get the default model for a specific type (returns the first one)
    #[allow(dead_code)]
    pub fn get_default_model(&self, model_type: &ModelType) -> Option<&ModelConfig> {
        self.get_best_model(model_type)
    }

    /// Select the best model based on priority (higher wins) and Ready health
    #[allow(dead_code)]
    pub fn get_best_model(&self, model_type: &ModelType) -> Option<&ModelConfig> {
        self.models.get(model_type).and_then(|candidates| {
            let total = candidates.len();
            let ready_count = candidates
                .iter()
                .filter(|cfg| cfg.health == ModelHealth::Ready)
                .count();

            let best_ready = candidates
                .iter()
                .filter(|cfg| cfg.health == ModelHealth::Ready)
                .max_by(|a, b| a.priority.cmp(&b.priority));

            let best_any = candidates
                .iter()
                .max_by(|a, b| a.priority.cmp(&b.priority));

            let chosen = best_ready.or(best_any);

            match chosen {
                Some(cfg) => {
                    log::debug!(
                        "Model selected: type={:?} name={} priority={} health={:?} (ready {}/{} total {})",
                        model_type,
                        cfg.name,
                        cfg.priority,
                        cfg.health,
                        ready_count,
                        total,
                        total
                    );
                    Some(cfg)
                }
                None => {
                    log::warn!(
                        "No models available for type={:?}; candidates={} ready={} (falling back to none)",
                        model_type,
                        total,
                        ready_count
                    );
                    None
                }
            }
        })
    }

    /// Load model configuration from a TOML file
    #[allow(dead_code)]
    pub fn load_config(&mut self, config_path: &Path) -> Result<()> {
        let content =
            std::fs::read_to_string(config_path).context("Failed to read model config file")?;

        // Try to parse as format with models key or direct array
        #[derive(serde::Deserialize)]
        struct ConfigWrapper {
            models: Vec<ModelConfig>,
        }

        let configs: Vec<ModelConfig> = if let Ok(wrapper) = toml::from_str::<ConfigWrapper>(&content) {
            wrapper.models
        } else if let Ok(direct) = toml::from_str::<Vec<ModelConfig>>(&content) {
            direct
        } else {
            // If both fail, file may be empty or contain only comments
            log::info!("No models found in config file (may be empty or all comments)");
            Vec::new()
        };

        for config in configs {
            self.register_model(config)?;
        }

        Ok(())
    }

    /// Save current model configurations to a TOML file
    #[allow(dead_code)]
    pub fn save_config(&self, config_path: &Path) -> Result<()> {
        let all_configs: Vec<ModelConfig> = self.models.values().flatten().cloned().collect();

        let content =
            toml::to_string_pretty(&all_configs).context("Failed to serialize model config")?;

        std::fs::write(config_path, content).context("Failed to write model config file")?;

        Ok(())
    }

    /// Get the model directory path
    #[allow(dead_code)]
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
            priority: 1,
            health: ModelHealth::Unknown,
        };

        manager.register_model(config.clone()).unwrap();
        let models = manager.get_models(&ModelType::HtmlParser);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test_html_parser");
    }

    #[test]
    fn test_best_model_prefers_ready_and_priority() {
        let temp_dir = tempdir().unwrap();
        let mut manager = ModelManager::new(temp_dir.path().to_path_buf()).unwrap();

        // Ready model with lower priority
        let ready_path = temp_dir.path().join("ready.onnx");
        std::fs::write(&ready_path, b"stub").unwrap();
        let ready_cfg = ModelConfig {
            name: "ready_model".to_string(),
            model_type: ModelType::CssParser,
            path: PathBuf::from("ready.onnx"),
            description: "Ready model".to_string(),
            version: "1.0.0".to_string(),
            priority: 5,
            health: ModelHealth::Unknown,
        };

        // Missing model with higher priority should not win when not ready
        let missing_cfg = ModelConfig {
            name: "missing_model".to_string(),
            model_type: ModelType::CssParser,
            path: PathBuf::from("missing.onnx"),
            description: "Missing model".to_string(),
            version: "1.0.0".to_string(),
            priority: 10,
            health: ModelHealth::Unknown,
        };

        manager.register_model(ready_cfg).unwrap();
        manager.register_model(missing_cfg).unwrap();

        let best = manager.get_best_model(&ModelType::CssParser).unwrap();
        assert_eq!(best.name, "ready_model");
        assert_eq!(best.health, ModelHealth::Ready);
    }
}
