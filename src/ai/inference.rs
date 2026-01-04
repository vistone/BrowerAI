use anyhow::Result;
use std::path::Path;

#[cfg(feature = "ai")]
use ort::{Environment, Session, SessionBuilder, Value};
#[cfg(feature = "ai")]
use std::sync::Arc;

/// Inference engine for running ONNX models
pub struct InferenceEngine {
    #[cfg(feature = "ai")]
    environment: Arc<Environment>,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new() -> Result<Self> {
        #[cfg(feature = "ai")]
        {
            let environment = Environment::builder()
                .with_name("BrowerAI")
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to create ONNX environment: {}", e))?;

            Ok(Self {
                environment: Arc::new(environment),
            })
        }
        
        #[cfg(not(feature = "ai"))]
        {
            log::warn!("AI feature not enabled. Inference engine will run in stub mode.");
            log::warn!("To enable AI features, compile with: cargo build --features ai");
            Ok(Self {})
        }
    }

    /// Load an ONNX model from the specified path
    #[cfg(feature = "ai")]
    pub fn load_model(&self, model_path: &Path) -> Result<Session> {
        if !model_path.exists() {
            return Err(anyhow::anyhow!("Model file does not exist: {:?}", model_path));
        }

        let session = SessionBuilder::new(&self.environment)
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_model_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model from file: {}", e))?;

        log::info!("Successfully loaded model from {:?}", model_path);
        Ok(session)
    }

    /// Load an ONNX model from the specified path (stub version)
    #[cfg(not(feature = "ai"))]
    pub fn load_model(&self, _model_path: &Path) -> Result<()> {
        Err(anyhow::anyhow!("AI feature not enabled. Cannot load models."))
    }

    /// Run inference on the model with input data
    #[cfg(feature = "ai")]
    pub fn infer(&self, session: &Session, input_name: &str, input_data: Vec<f32>, shape: Vec<i64>) -> Result<Vec<f32>> {
        // Create input tensor
        let input_tensor = Value::from_array(session.allocator(), &shape, &input_data)
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?;

        // Run inference
        let outputs = session.run(vec![input_tensor])
            .map_err(|e| anyhow::anyhow!("Failed to run inference: {}", e))?;

        // Extract output data
        if outputs.is_empty() {
            return Err(anyhow::anyhow!("No outputs from inference"));
        }

        let output = &outputs[0];
        let output_data = output.try_extract::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output data: {}", e))?;

        let result = output_data.view().iter().copied().collect();
        
        log::debug!("Inference completed successfully");
        Ok(result)
    }

    /// Run inference on the model with input data (stub version)
    #[cfg(not(feature = "ai"))]
    pub fn infer(&self, _session: &(), _input_name: &str, _input_data: Vec<f32>, _shape: Vec<i64>) -> Result<Vec<f32>> {
        Err(anyhow::anyhow!("AI feature not enabled. Cannot run inference."))
    }
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default inference engine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_engine_creation() {
        let engine = InferenceEngine::new();
        assert!(engine.is_ok());
    }
}
