use anyhow::Result;
use std::path::Path;

#[cfg(feature = "ai")]
use std::time::Instant;

#[cfg(feature = "ai")]
use ort::{session::input::SessionInputValue, session::Session, value::Value};

#[cfg(feature = "ai")]
use crate::performance_monitor::InferenceMetrics;
use crate::performance_monitor::PerformanceMonitor;

/// Inference engine for running ONNX models
#[derive(Clone)]
pub struct InferenceEngine {
    monitor: Option<PerformanceMonitor>,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new() -> Result<Self> {
        #[cfg(feature = "ai")]
        {
            // Initialize global environment once
            // commit() returns a boolean indicating whether initialization occurred; ignore value
            let _ = ort::init().with_name("BrowerAI").commit();

            Ok(Self { monitor: None })
        }

        #[cfg(not(feature = "ai"))]
        {
            log::warn!("AI feature not enabled. Inference engine will run in stub mode.");
            log::warn!("To enable AI features, compile with: cargo build --features ai");
            Ok(Self { monitor: None })
        }
    }

    /// Create a new inference engine with a performance monitor
    pub fn with_monitor(monitor: PerformanceMonitor) -> Result<Self> {
        let mut engine = Self::new()?;
        engine.monitor = Some(monitor);
        Ok(engine)
    }

    /// Access the performance monitor if present
    pub fn monitor_handle(&self) -> Option<PerformanceMonitor> {
        self.monitor.clone()
    }

    /// Load an ONNX model from the specified path
    #[cfg(feature = "ai")]
    pub fn load_model(&self, model_path: &Path) -> Result<Session> {
        if !model_path.exists() {
            return Err(anyhow::anyhow!(
                "Model file does not exist: {:?}",
                model_path
            ));
        }

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model from file: {}", e))?;

        log::info!("Successfully loaded model from {:?}", model_path);
        Ok(session)
    }

    /// Load an ONNX model from the specified path (stub version)
    #[cfg(not(feature = "ai"))]
    #[allow(dead_code)]
    pub fn load_model(&self, _model_path: &Path) -> Result<()> {
        Err(anyhow::anyhow!(
            "AI feature not enabled. Cannot load models."
        ))
    }

    /// Run inference on the model with input data
    #[cfg(feature = "ai")]
    pub fn infer(
        &self,
        session: &mut Session,
        model_name: &str,
        _input_name: &str,
        input_data: Vec<f32>,
        shape: Vec<i64>,
    ) -> Result<Vec<f32>> {
        let input_bytes = input_data.len() * std::mem::size_of::<f32>();
        let start_time = Instant::now();

        // Create input tensor (shape, data)
        let input_tensor = Value::from_array((shape.clone(), input_data.clone()))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?;

        // Run inference
        let inputs = [SessionInputValue::from(input_tensor)];
        let outputs = match session.run(inputs) {
            Ok(outputs) => outputs,
            Err(e) => {
                if let Some(monitor) = &self.monitor {
                    monitor.record_inference(InferenceMetrics {
                        model_name: model_name.to_string(),
                        inference_time: start_time.elapsed(),
                        input_size: input_bytes,
                        output_size: 0,
                        success: false,
                        timestamp: start_time,
                    });
                }
                return Err(anyhow::anyhow!("Failed to run inference: {}", e));
            }
        };

        // Extract output data
        if outputs.len() == 0 {
            if let Some(monitor) = &self.monitor {
                monitor.record_inference(InferenceMetrics {
                    model_name: model_name.to_string(),
                    inference_time: start_time.elapsed(),
                    input_size: input_bytes,
                    output_size: 0,
                    success: false,
                    timestamp: start_time,
                });
            }
            return Err(anyhow::anyhow!("No outputs from inference"));
        }

        let output = &outputs[0];
        let output_data = output.try_extract_array::<f32>().map_err(|e| {
            if let Some(monitor) = &self.monitor {
                monitor.record_inference(InferenceMetrics {
                    model_name: model_name.to_string(),
                    inference_time: start_time.elapsed(),
                    input_size: input_bytes,
                    output_size: 0,
                    success: false,
                    timestamp: start_time,
                });
            }
            anyhow::anyhow!("Failed to extract output data: {}", e)
        })?;

        let result: Vec<f32> = output_data.iter().copied().collect();
        let output_bytes = result.len() * std::mem::size_of::<f32>();

        if let Some(monitor) = &self.monitor {
            monitor.record_inference(InferenceMetrics {
                model_name: model_name.to_string(),
                inference_time: start_time.elapsed(),
                input_size: input_bytes,
                output_size: output_bytes,
                success: true,
                timestamp: start_time,
            });
        }

        log::debug!("Inference completed successfully for model {}", model_name);
        Ok(result)
    }

    /// Run inference on the model with input data (stub version)
    #[cfg(not(feature = "ai"))]
    #[allow(dead_code)]
    pub fn infer(
        &self,
        _session: &(),
        _model_name: &str,
        _input_name: &str,
        _input_data: Vec<f32>,
        _shape: Vec<i64>,
    ) -> Result<Vec<f32>> {
        Err(anyhow::anyhow!(
            "AI feature not enabled. Cannot run inference."
        ))
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
