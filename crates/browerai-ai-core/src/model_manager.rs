use anyhow::{Context, Result};
use browerai_core::HealthStatus;
use log;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Type alias for backward compatibility
pub type ModelHealth = HealthStatus;

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
    /// Large code LLMs (e.g., Qwen2.5-Coder) served via Candle/GGUF
    CodeLlm,
}

/// Manages the local model library for AI-powered browser operations
#[derive(Clone)]
pub struct ModelManager {
    models: HashMap<ModelType, Vec<ModelConfig>>,
    model_dir: PathBuf,
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

        let configs: Vec<ModelConfig> =
            if let Ok(wrapper) = toml::from_str::<ConfigWrapper>(&content) {
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

    /// Update the health status of a specific model
    pub fn update_model_health(&mut self, model_name: &str, health: ModelHealth) -> Result<()> {
        let mut found = false;
        for models in self.models.values_mut() {
            for model in models.iter_mut() {
                if model.name == model_name {
                    let old_health = model.health.clone();
                    model.health = health.clone();
                    found = true;

                    // Log health changes
                    if old_health != health {
                        match &health {
                            ModelHealth::Ready => {
                                log::info!(
                                    "Model '{}' health improved: {:?} -> Ready",
                                    model_name,
                                    old_health
                                );
                            }
                            ModelHealth::LoadFailed(reason) => {
                                log::warn!("Model '{}' failed to load: {}", model_name, reason);
                            }
                            ModelHealth::ValidationFailed(reason) => {
                                log::warn!("Model '{}' failed validation: {}", model_name, reason);
                            }
                            ModelHealth::InferenceFailing => {
                                log::error!("Model '{}' consistently failing inference - consider replacing", model_name);
                            }
                            _ => {
                                log::debug!(
                                    "Model '{}' health changed: {:?} -> {:?}",
                                    model_name,
                                    old_health,
                                    health
                                );
                            }
                        }
                    }
                }
            }
        }

        if !found {
            anyhow::bail!("Model '{}' not found in registry", model_name);
        }

        Ok(())
    }

    /// Check for bad models and return their names with reasons
    pub fn detect_bad_models(&self) -> Vec<(String, String)> {
        let mut bad_models = Vec::new();

        for models in self.models.values() {
            for model in models {
                let reason = match &model.health {
                    ModelHealth::MissingFile => {
                        Some(format!("Model file '{}' is missing", model.path.display()))
                    }
                    ModelHealth::LoadFailed(err) => Some(format!("Failed to load: {}", err)),
                    ModelHealth::ValidationFailed(err) => {
                        Some(format!("Failed validation: {}", err))
                    }
                    ModelHealth::InferenceFailing => {
                        Some("Inference consistently failing".to_string())
                    }
                    _ => None,
                };

                if let Some(reason) = reason {
                    bad_models.push((model.name.clone(), reason));
                }
            }
        }

        bad_models
    }

    /// Get health summary for all models
    pub fn health_summary(&self) -> ModelHealthSummary {
        let mut summary = ModelHealthSummary::default();

        for models in self.models.values() {
            for model in models {
                summary.total += 1;
                match model.health {
                    ModelHealth::Ready => summary.ready += 1,
                    ModelHealth::MissingFile => summary.missing_file += 1,
                    ModelHealth::LoadFailed(_) => summary.load_failed += 1,
                    ModelHealth::ValidationFailed(_) => summary.validation_failed += 1,
                    ModelHealth::InferenceFailing => summary.inference_failing += 1,
                    ModelHealth::Unknown => summary.unknown += 1,
                    // New HealthStatus variants map to unknown for backward compatibility
                    ModelHealth::Warning(_) => summary.unknown += 1,
                    ModelHealth::Disabled => summary.unknown += 1,
                    ModelHealth::OperationFailed(_) => summary.unknown += 1,
                    ModelHealth::Unhealthy(_) => summary.unknown += 1,
                }
            }
        }

        summary
    }
}

/// Summary of model health across all registered models
#[derive(Debug, Clone, Default)]
pub struct ModelHealthSummary {
    pub total: usize,
    pub ready: usize,
    pub missing_file: usize,
    pub load_failed: usize,
    pub validation_failed: usize,
    pub inference_failing: usize,
    pub unknown: usize,
}

impl ModelHealthSummary {
    /// Get the percentage of healthy models
    pub fn health_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.ready as f64 / self.total as f64
        }
    }

    /// Check if there are any bad models
    pub fn has_issues(&self) -> bool {
        self.missing_file > 0
            || self.load_failed > 0
            || self.validation_failed > 0
            || self.inference_failing > 0
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

    #[test]
    fn test_update_model_health() {
        let temp_dir = tempdir().unwrap();
        let mut manager = ModelManager::new(temp_dir.path().to_path_buf()).unwrap();

        let config = ModelConfig {
            name: "test_model".to_string(),
            model_type: ModelType::HtmlParser,
            path: PathBuf::from("test.onnx"),
            description: "Test model".to_string(),
            version: "1.0.0".to_string(),
            priority: 1,
            health: ModelHealth::Unknown,
        };

        manager.register_model(config).unwrap();

        // Update health to Ready
        manager
            .update_model_health("test_model", ModelHealth::Ready)
            .unwrap();
        let model = manager.get_models(&ModelType::HtmlParser)[0];
        assert_eq!(model.health, ModelHealth::Ready);

        // Update health to LoadFailed
        manager
            .update_model_health(
                "test_model",
                ModelHealth::LoadFailed("Test error".to_string()),
            )
            .unwrap();
        let model = manager.get_models(&ModelType::HtmlParser)[0];
        // 修复: 使用更安全的断言，避免panic
        assert!(
            matches!(model.health, ModelHealth::LoadFailed(_)),
            "Expected LoadFailed variant, got {:?}",
            model.health
        );
    }

    #[test]
    fn test_detect_bad_models() {
        let temp_dir = tempdir().unwrap();
        let mut manager = ModelManager::new(temp_dir.path().to_path_buf()).unwrap();

        // Add a good model (create the file)
        let good_path = temp_dir.path().join("good.onnx");
        std::fs::write(&good_path, b"fake model data").unwrap();

        let good_config = ModelConfig {
            name: "good_model".to_string(),
            model_type: ModelType::HtmlParser,
            path: PathBuf::from("good.onnx"),
            description: "Good model".to_string(),
            version: "1.0.0".to_string(),
            priority: 1,
            health: ModelHealth::Unknown, // Will be set to Ready by register_model
        };
        manager.register_model(good_config).unwrap();

        // Add bad models (don't create files - they will be marked as MissingFile)
        let missing_config = ModelConfig {
            name: "missing_model".to_string(),
            model_type: ModelType::CssParser,
            path: PathBuf::from("missing.onnx"),
            description: "Missing model".to_string(),
            version: "1.0.0".to_string(),
            priority: 1,
            health: ModelHealth::Unknown, // Will be set to MissingFile by register_model
        };
        manager.register_model(missing_config).unwrap();

        // Update one model to LoadFailed status
        manager
            .update_model_health(
                "missing_model",
                ModelHealth::LoadFailed("Test error".to_string()),
            )
            .unwrap();

        let bad_models = manager.detect_bad_models();
        assert_eq!(bad_models.len(), 1); // Only the LoadFailed one

        let names: Vec<String> = bad_models.iter().map(|(n, _)| n.clone()).collect();
        assert!(names.contains(&"missing_model".to_string()));
    }

    #[test]
    fn test_health_summary() {
        let temp_dir = tempdir().unwrap();
        let mut manager = ModelManager::new(temp_dir.path().to_path_buf()).unwrap();

        // Create files for "ready" models
        std::fs::write(temp_dir.path().join("ready1.onnx"), b"fake").unwrap();
        std::fs::write(temp_dir.path().join("ready2.onnx"), b"fake").unwrap();

        // Add models with various health statuses
        let configs = vec![
            ModelConfig {
                name: "ready1".to_string(),
                model_type: ModelType::HtmlParser,
                path: PathBuf::from("ready1.onnx"),
                description: "".to_string(),
                version: "1.0.0".to_string(),
                priority: 1,
                health: ModelHealth::Unknown, // Will be set to Ready
            },
            ModelConfig {
                name: "ready2".to_string(),
                model_type: ModelType::CssParser,
                path: PathBuf::from("ready2.onnx"),
                description: "".to_string(),
                version: "1.0.0".to_string(),
                priority: 1,
                health: ModelHealth::Unknown, // Will be set to Ready
            },
            ModelConfig {
                name: "missing".to_string(),
                model_type: ModelType::JsParser,
                path: PathBuf::from("missing.onnx"),
                description: "".to_string(),
                version: "1.0.0".to_string(),
                priority: 1,
                health: ModelHealth::Unknown, // Will be set to MissingFile
            },
            ModelConfig {
                name: "failing".to_string(),
                model_type: ModelType::LayoutOptimizer,
                path: PathBuf::from("failing.onnx"),
                description: "".to_string(),
                version: "1.0.0".to_string(),
                priority: 1,
                health: ModelHealth::Unknown, // Will be set to MissingFile, then we'll update
            },
        ];

        for config in configs {
            manager.register_model(config).unwrap();
        }

        // Update the failing model manually
        manager
            .update_model_health("failing", ModelHealth::InferenceFailing)
            .unwrap();

        let summary = manager.health_summary();
        assert_eq!(summary.total, 4);
        assert_eq!(summary.ready, 2);
        assert_eq!(summary.missing_file, 1);
        assert_eq!(summary.inference_failing, 1);
        assert_eq!(summary.health_rate(), 0.5);
        assert!(summary.has_issues());
    }

    #[test]
    fn test_health_summary_all_healthy() {
        let temp_dir = tempdir().unwrap();
        let mut manager = ModelManager::new(temp_dir.path().to_path_buf()).unwrap();

        // Create the model file so it's marked as Ready
        std::fs::write(temp_dir.path().join("healthy.onnx"), b"fake").unwrap();

        let config = ModelConfig {
            name: "healthy".to_string(),
            model_type: ModelType::HtmlParser,
            path: PathBuf::from("healthy.onnx"),
            description: "".to_string(),
            version: "1.0.0".to_string(),
            priority: 1,
            health: ModelHealth::Unknown, // Will be set to Ready
        };
        manager.register_model(config).unwrap();

        let summary = manager.health_summary();
        assert_eq!(summary.total, 1);
        assert_eq!(summary.ready, 1);
        assert_eq!(summary.health_rate(), 1.0);
        assert!(!summary.has_issues());
    }
}
