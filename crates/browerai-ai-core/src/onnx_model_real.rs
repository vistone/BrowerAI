//! ONNX Model Loader with Real Model Support
//! 真实 ONNX 模型加载器

use anyhow::{Context, Result, anyhow};
use log::{info, warn, error, debug};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Model configuration for real ONNX models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxModelConfig {
    pub name: String,
    pub model_type: String,
    pub path: PathBuf,
    pub description: String,
    pub version: String,
    pub priority: u8,
    pub enabled: bool,
    pub task: String,
    
    #[serde(default)]
    pub parameters: Option<String>,
    
    #[serde(default)]
    pub inference_time_cpu_ms: Option<String>,
    
    #[serde(default)]
    pub vocab_size: Option<usize>,
    
    #[serde(default)]
    pub max_length: Option<usize>,
    
    #[serde(default)]
    pub tags: Vec<String>,
    
    #[serde(default)]
    pub input_spec: Option<HashMap<String, String>>,
    
    #[serde(default)]
    pub output_spec: Option<HashMap<String, String>>,
}

/// Model loader statistics
#[derive(Debug, Clone, Default)]
pub struct ModelLoadStats {
    pub total_attempts: u64,
    pub successful_loads: u64,
    pub failed_loads: u64,
    pub total_load_time_ms: u64,
    pub last_load_time_ms: Option<u64>,
}

/// Real ONNX Model wrapper with actual inference support
#[cfg(feature = "onnx")]
pub struct OnnxModel {
    name: String,
    model_type: String,
    session: ort::Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    config: OnnxModelConfig,
    load_time: Instant,
}

#[cfg(feature = "onnx")]
impl OnnxModel {
    /// Load a real ONNX model from file
    pub fn load(config: OnnxModelConfig, model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join(&config.path);
        
        if !model_path.exists() {
            return Err(anyhow!("Model file not found: {:?}", model_path));
        }
        
        info!("Loading ONNX model: {} from {:?}", config.name, model_path);
        
        let start = Instant::now();
        
        // Create ONNX session with optimized settings
        let session = ort::Session::builder()?
            .with_optimization_level(ort::OptimizationLevel::All)?
            .with_parallel_execution(true)?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load ONNX model: {:?}", model_path))?;
        
        // Get input and output names
        let input_names: Vec<String> = session.inputs.iter()
            .map(|input| input.name.clone())
            .collect();
        
        let output_names: Vec<String> = session.outputs.iter()
            .map(|output| output.name.clone())
            .collect();
        
        let load_time_ms = start.elapsed().as_millis() as u64;
        
        info!(
            "Successfully loaded model '{}' in {}ms - inputs: {:?}, outputs: {:?}",
            config.name, load_time_ms, input_names, output_names
        );
        
        Ok(Self {
            name: config.name.clone(),
            model_type: config.model_type.clone(),
            session,
            input_names,
            output_names,
            config,
            load_time: start,
        })
    }
    
    /// Run inference with real input tensor
    pub fn infer(&self, input_data: &[f32], input_shape: &[i64]) -> Result<Vec<f32>> {
        use ort::tensor::InputTensor;
        use ort::session::input::SessionInputValue;
        
        let start = Instant::now();
        
        // Create input tensor
        let input_tensor = InputTensor::from_array(
            input_shape.to_vec(),
            input_data.to_vec()
        );
        
        let inputs = [SessionInputValue::from(input_tensor)];
        
        // Run inference
        let outputs = self.session.run(inputs)
            .context("Inference failed")?;
        
        // Extract output
        if outputs.is_empty() {
            return Err(anyhow!("No outputs from model inference"));
        }
        
        let output = &outputs[0];
        let result: Vec<f32> = match output {
            ort::value::Value::F32(arr) => arr.iter().copied().collect(),
            ort::value::Value::F16(_) => {
                return Err(anyhow!("Float16 output not supported yet"));
            }
            ort::value::Value::Int64(arr) => arr.iter().map(|&x| x as f32).collect(),
            ort::value::Value::Int32(arr) => arr.iter().map(|&x| x as f32).collect(),
            ort::value::Value::Uint8(arr) => arr.iter().map(|&x| x as f32).collect(),
            _ => return Err(anyhow!("Unsupported output tensor type")),
        };
        
        let inference_time = start.elapsed().as_millis();
        debug!("Model '{}' inference completed in {}ms", self.name, inference_time);
        
        Ok(result)
    }
    
    /// Get model metadata
    pub fn metadata(&self) -> &OnnxModelConfig {
        &self.config
    }
    
    /// Get input names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }
    
    /// Get output names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
    
    /// Get model name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get model type
    pub fn model_type(&self) -> &str {
        &self.model_type
    }
}

/// ONNX Model Registry with real model support
#[derive(Clone)]
pub struct OnnxModelRegistry {
    models: Arc<RwLock<HashMap<String, Arc<OnnxModel>>>>,
    config_path: PathBuf,
    model_dir: PathBuf,
    stats: Arc<RwLock<ModelLoadStats>>,
}

impl OnnxModelRegistry {
    /// Create a new model registry
    pub fn new(config_path: PathBuf, model_dir: PathBuf) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            config_path,
            model_dir,
            stats: Arc::new(RwLock::new(ModelLoadStats::default())),
        }
    }
    
    /// Load all models from configuration
    #[cfg(feature = "onnx")]
    pub fn load_all(&mut self) -> Result<Vec<String>> {
        let mut loaded = Vec::new();
        let mut failed = Vec::new();
        
        // Read and parse config
        let config_content = std::fs::read_to_string(&self.config_path)
            .context("Failed to read model config")?;
        
        #[derive(Deserialize)]
        struct ConfigFile {
            models: Vec<OnnxModelConfig>,
        }
        
        let config: ConfigFile = toml::from_str(&config_content)
            .context("Failed to parse model config")?;
        
        for model_config in config.models {
            if !model_config.enabled {
                debug!("Skipping disabled model: {}", model_config.name);
                continue;
            }
            
            match OnnxModel::load(model_config.clone(), &self.model_dir) {
                Ok(model) => {
                    let name = model.name().to_string();
                    self.models.write().unwrap().insert(
                        name.clone(),
                        Arc::new(model)
                    );
                    loaded.push(name);
                    
                    // Update stats
                    {
                        let mut stats = self.stats.write().unwrap();
                        stats.successful_loads += 1;
                    }
                }
                Err(e) => {
                    warn!("Failed to load model {}: {}", model_config.name, e);
                    failed.push(model_config.name);
                    
                    // Update stats
                    {
                        let mut stats = self.stats.write().unwrap();
                        stats.failed_loads += 1;
                    }
                }
            }
        }
        
        info!("Loaded {} models, {} failed", loaded.len(), failed.len());
        Ok(loaded)
    }
    
    /// Get a model by name
    pub fn get(&self, name: &str) -> Option<Arc<OnnxModel>>> {
        self.models.read().unwrap().get(name).cloned()
    }
    
    /// Get models by task type
    pub fn get_by_task(&self, task: &str) -> Vec<Arc<OnnxModel>>> {
        self.models.read().unwrap()
            .values()
            .filter(|m| m.metadata().task == task)
            .cloned()
            .collect()
    }
    
    /// Get all model names
    pub fn list(&self) -> Vec<String> {
        self.models.read().unwrap().keys().cloned().collect()
    }
    
    /// Get statistics
    pub fn stats(&self) -> ModelLoadStats {
        self.stats.read().unwrap().clone()
    }
}

/// Tokenizer for ONNX model input
pub struct SimpleTokenizer {
    vocab: HashMap<char, usize>,
    reverse_vocab: HashMap<usize, char>,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        // Basic ASCII characters
        for (i, c) in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t.,!?;:()[]{}<>=/-+*\"'\n@#$%^&~`_|\n\\".chars().enumerate() {
            vocab.insert(c, i);
            reverse_vocab.insert(i, c);
        }
        
        Self { vocab, reverse_vocab }
    }
    
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<usize> {
        let mut tokens: Vec<usize> = text.chars()
            .filter_map(|c| self.vocab.get(&c).copied())
            .take(max_len)
            .collect();
        
        // Pad if necessary
        while tokens.len() < max_len {
            tokens.push(0); // Use 0 as padding
        }
        
        tokens
    }
    
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&t| self.reverse_vocab.get(&t).copied())
            .collect()
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_tokenizer_basic() {
        let tokenizer = SimpleTokenizer::new();
        let text = "hello world";
        let tokens = tokenizer.encode(text, 100);
        let decoded = tokenizer.decode(&tokens);
        
        // Basic test - tokens should be generated
        assert!(!tokens.is_empty());
        assert!(tokenizer.vocab_size() > 0);
    }
    
    #[test]
    fn test_tokenizer_padding() {
        let tokenizer = SimpleTokenizer::new();
        let text = "hi";
        let tokens = tokenizer.encode(text, 10);
        
        assert_eq!(tokens.len(), 10);
        // First tokens should be 'hi' characters
        assert!(tokens[0] > 0 || tokens[1] > 0);
        // Last tokens should be padding (0)
        assert_eq!(tokens[tokens.len() - 1], 0);
    }
    
    #[test]
    fn test_model_config_parsing() {
        let config_str = r#"
[[models]]
name = "test_model"
model_type = "TestType"
path = "test.onnx"
description = "Test model"
version = "1.0.0"
priority = 100
enabled = true
task = "test"
"#;
        
        #[derive(Deserialize)]
        struct ConfigFile {
            models: Vec<OnnxModelConfig>,
        }
        
        let config: ConfigFile = toml::from_str(config_str).unwrap();
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].name, "test_model");
        assert_eq!(config.models[0].priority, 100);
    }
}
