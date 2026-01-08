/// Model Provider Abstraction Layer
///
/// This module defines the trait system for pluggable model providers,
/// allowing BrowerAI to support multiple inference backends (ONNX, Candle, Custom, etc.)
/// while maintaining a consistent interface.
///
/// # Architecture
///
/// The provider system is built on several key traits:
/// - `ModelProvider`: Factory for creating models from configs
/// - `Model`: Interface for inference operations
/// - `ModelMetadata`: Introspection and validation
///
/// # Examples
///
/// ```ignore
/// use browerai_ai_core::{ModelProvider, Model};
/// use std::sync::Arc;
///
/// // Create a provider
/// let provider = OnnxModelProvider::new();
///
/// // Load a model
/// let model = provider.load_model(&path, &config)?;
///
/// // Run inference
/// let output = model.infer(&input)?;
/// ```
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Metadata about a model's input/output structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Name of the tensor
    pub name: String,
    /// Data type (e.g., "float32", "int64")
    pub dtype: String,
    /// Shape specification (can include -1 for variable dimensions)
    pub shape: Vec<i64>,
}

/// Complete metadata about a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Name of the model
    pub name: String,
    /// Version string following semantic versioning
    pub version: String,
    /// Input tensor specifications
    pub inputs: Vec<TensorInfo>,
    /// Output tensor specifications
    pub outputs: Vec<TensorInfo>,
    /// Custom properties (e.g., batch_size, quantization)
    pub properties: HashMap<String, String>,
    /// Framework/backend that created this model
    pub framework: String,
}

/// Result type for model operations
pub type ModelResult<T> = Result<T>;

/// Information about a model provider
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Name of the provider
    pub name: String,
    /// Version of the provider
    pub version: String,
    /// Supported model types
    pub supported_formats: Vec<String>,
    /// Hardware acceleration support
    pub capabilities: ProviderCapabilities,
}

/// Capabilities of a model provider
#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    /// Supports GPU acceleration
    pub gpu: bool,
    /// Supports quantization
    pub quantization: bool,
    /// Supports dynamic shapes
    pub dynamic_shapes: bool,
    /// Supports batch inference
    pub batch_inference: bool,
    /// Maximum model size in MB
    pub max_model_size_mb: Option<u64>,
}

/// Configuration for loading a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadConfig {
    /// Path to model file
    pub path: PathBuf,
    /// Provider-specific options
    pub options: HashMap<String, String>,
    /// Enable GPU if available
    pub use_gpu: bool,
    /// Enable model validation before loading
    pub validate: bool,
    /// Warm-up model after loading
    pub warmup: bool,
}

impl ModelLoadConfig {
    /// Create a new config with defaults
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            options: HashMap::new(),
            use_gpu: false,
            validate: true,
            warmup: false,
        }
    }

    /// Enable GPU usage
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set a provider-specific option
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// Enable warmup after loading
    pub fn with_warmup(mut self, warmup: bool) -> Self {
        self.warmup = warmup;
        self
    }
}

/// Main trait for model providers - responsible for loading models
///
/// Implementations should handle provider-specific initialization,
/// validation, and loading of model files.
///
/// # Thread Safety
///
/// All providers must be `Send + Sync` for safe use in concurrent contexts.
pub trait ModelProvider: Send + Sync {
    /// Load a model from the specified configuration
    ///
    /// # Arguments
    /// * `config` - Model loading configuration
    ///
    /// # Returns
    /// A boxed model ready for inference
    ///
    /// # Errors
    /// Returns an error if:
    /// - Model file not found
    /// - Format not supported by this provider
    /// - Validation fails (if enabled)
    /// - Hardware requirements not met
    fn load_model(&self, config: &ModelLoadConfig) -> ModelResult<Arc<dyn Model>>;

    /// Validate a model file without loading it
    ///
    /// This is useful for checking compatibility before actual loading.
    fn validate_model(&self, path: &Path) -> ModelResult<ModelMetadata>;

    /// Get information about this provider
    fn info(&self) -> ProviderInfo;

    /// Check if this provider can handle the given file
    ///
    /// Default implementation checks file extension, but providers
    /// may override for more sophisticated checks.
    fn can_load(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            if let Some(ext_str) = ext.to_str() {
                self.info()
                    .supported_formats
                    .iter()
                    .any(|fmt| fmt.to_lowercase() == ext_str.to_lowercase())
            } else {
                false
            }
        } else {
            false
        }
    }
}

/// Trait for models - responsible for running inference
///
/// Models are created by providers and encapsulate the actual
/// inference computation.
///
/// # Thread Safety
///
/// Models must be `Send + Sync` to allow sharing across threads.
pub trait Model: Send + Sync {
    /// Run inference on a single input
    ///
    /// # Arguments
    /// * `input` - Input data as Vec<f32>
    /// * `shape` - Shape specification for the input
    ///
    /// # Returns
    /// Output data as Vec<f32>
    fn infer(&self, input: &[f32], shape: &[i64]) -> ModelResult<Vec<f32>>;

    /// Run batch inference on multiple inputs
    ///
    /// Default implementation calls `infer` for each input.
    /// Providers may override for optimized batch processing.
    fn infer_batch(&self, inputs: &[(Vec<f32>, Vec<i64>)]) -> ModelResult<Vec<Vec<f32>>> {
        inputs
            .iter()
            .map(|(data, shape)| self.infer(data, shape))
            .collect()
    }

    /// Get metadata about this model
    fn metadata(&self) -> &ModelMetadata;

    /// Warm up the model for optimal performance
    ///
    /// This allocates buffers, compiles kernels, etc.
    /// Optional but recommended before production use.
    fn warmup(&self) -> ModelResult<()> {
        Ok(())
    }

    /// Check if the model is healthy and ready for inference
    fn health_check(&self) -> ModelResult<()>;

    /// Get memory usage statistics (optional)
    fn memory_stats(&self) -> Option<MemoryStats> {
        None
    }
}

/// Memory statistics for a model
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current memory usage in MB
    pub usage_mb: u64,
    /// Peak memory usage in MB
    pub peak_mb: u64,
    /// Estimated model size in MB
    pub model_size_mb: u64,
}

/// Registry for managing multiple model providers
#[derive(Clone)]
pub struct ModelProviderRegistry {
    providers: Arc<std::sync::RwLock<Vec<Arc<dyn ModelProvider>>>>,
}

impl ModelProviderRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            providers: Arc::new(std::sync::RwLock::new(Vec::new())),
        }
    }

    /// Register a provider
    pub fn register(&self, provider: Arc<dyn ModelProvider>) -> Result<()> {
        if let Ok(mut providers) = self.providers.write() {
            providers.push(provider);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Failed to acquire lock on providers"))
        }
    }

    /// Find a provider that can load the given model
    pub fn find_provider_for(&self, path: &Path) -> Option<Arc<dyn ModelProvider>> {
        if let Ok(providers) = self.providers.read() {
            providers.iter().find(|p| p.can_load(path)).map(Arc::clone)
        } else {
            None
        }
    }

    /// Load a model using the first suitable provider
    pub fn load_model(&self, config: &ModelLoadConfig) -> ModelResult<Arc<dyn Model>> {
        if let Ok(providers) = self.providers.read() {
            for provider in providers.iter() {
                if provider.can_load(&config.path) {
                    return provider.load_model(config);
                }
            }
        }

        Err(anyhow::anyhow!(
            "No suitable provider found for model at {:?}",
            config.path
        ))
    }

    /// Validate a model using the first suitable provider
    pub fn validate_model(&self, path: &Path) -> ModelResult<ModelMetadata> {
        if let Ok(providers) = self.providers.read() {
            for provider in providers.iter() {
                if provider.can_load(path) {
                    return provider.validate_model(path);
                }
            }
        }

        Err(anyhow::anyhow!(
            "No suitable provider found for model at {:?}",
            path
        ))
    }

    /// Get info about all registered providers
    pub fn list_providers(&self) -> Vec<ProviderInfo> {
        if let Ok(providers) = self.providers.read() {
            providers.iter().map(|p| p.info()).collect()
        } else {
            Vec::new()
        }
    }
}

impl Default for ModelProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_load_config() {
        let config = ModelLoadConfig::new(PathBuf::from("model.onnx"))
            .with_gpu(true)
            .with_option("precision", "fp16")
            .with_warmup(true);

        assert_eq!(config.path, PathBuf::from("model.onnx"));
        assert!(config.use_gpu);
        assert!(config.warmup);
        assert_eq!(config.options.get("precision"), Some(&"fp16".to_string()));
    }

    #[test]
    fn test_provider_capabilities() {
        let caps = ProviderCapabilities {
            gpu: true,
            quantization: true,
            dynamic_shapes: true,
            batch_inference: true,
            max_model_size_mb: Some(2048),
        };

        assert!(caps.gpu);
        assert_eq!(caps.max_model_size_mb, Some(2048));
    }

    #[test]
    fn test_tensor_info() {
        let tensor = TensorInfo {
            name: "input".to_string(),
            dtype: "float32".to_string(),
            shape: vec![1, 224, 224, 3],
        };

        assert_eq!(tensor.name, "input");
        assert_eq!(tensor.shape.len(), 4);
    }

    #[test]
    fn test_model_metadata() {
        let inputs = vec![TensorInfo {
            name: "image".to_string(),
            dtype: "float32".to_string(),
            shape: vec![1, 224, 224, 3],
        }];

        let outputs = vec![TensorInfo {
            name: "output".to_string(),
            dtype: "float32".to_string(),
            shape: vec![1, 1000],
        }];

        let metadata = ModelMetadata {
            name: "ResNet50".to_string(),
            version: "1.0.0".to_string(),
            inputs,
            outputs,
            properties: HashMap::new(),
            framework: "ONNX".to_string(),
        };

        assert_eq!(metadata.name, "ResNet50");
        assert_eq!(metadata.version, "1.0.0");
    }
}
