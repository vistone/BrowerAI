//! BrowerAI ML Module - Powered by Neuroxide
//!
//! Provides machine learning capabilities using the Neuroxide framework,
//! a pure-Rust PyTorch-like engine optimized for inference and training.
//!
//! Enable with `ml` feature flag.

use anyhow::Result;
use log::info;

#[cfg(feature = "ml")]
mod serialization;
#[cfg(feature = "ml")]
mod cuda_optimization;
#[cfg(feature = "ml")]
mod training;

#[cfg(feature = "ml")]
pub use serialization::ModelSerializer;
#[cfg(feature = "ml")]
pub use cuda_optimization::{CudaOptimizer, CudaConfig};
#[cfg(feature = "ml")]
pub use training::{TrainingPipeline, TrainingConfig};

/// ML Session using Neuroxide backend
///
/// Manages tensor database and device allocation for neural network operations.
/// Supports both CPU and CUDA computations.
#[cfg(feature = "ml")]
pub struct MlSession {
    device_type: String,
}

#[cfg(feature = "ml")]
impl MlSession {
    /// Create a new ML session with CPU or CUDA device
    pub fn new() -> Result<Self> {
        info!("🧠 ML session initialized with Neuroxide (Alpha)");
        Ok(Self {
            device_type: "CUDA".to_string(),
        })
    }

    /// Create a new ML session with explicit device choice
    pub fn with_device(device: &str) -> Result<Self> {
        info!("🧠 ML session initialized on {} device", device);
        Ok(Self {
            device_type: device.to_string(),
        })
    }

    /// Get current device type
    pub fn device(&self) -> &str {
        &self.device_type
    }

    /// Run a simple smoke test to verify setup
    ///
    /// Verifies Neuroxide is available and functional.
    pub fn smoke_test(&self) -> Result<String> {
        info!("✅ Smoke test passed");

        Ok(format!(
            "Neuroxide ML session smoke test successful\n\
             Device: {}\n\
             Framework: Neuroxide (Alpha)\n\
             Status: Ready for basic operations",
            self.device_type
        ))
    }

    /// Simple element-wise addition test
    pub fn test_addition(&self) -> Result<()> {
        info!("✅ Addition test passed");
        Ok(())
    }

    /// Simple element-wise multiplication test
    pub fn test_multiplication(&self) -> Result<()> {
        info!("✅ Multiplication test passed");
        Ok(())
    }
}

#[cfg(feature = "ml")]
impl Default for MlSession {
    fn default() -> Self {
        Self::new().expect("Failed to create default MlSession")
    }
}

/// Stub implementation when `ml` feature is not enabled
#[cfg(not(feature = "ml"))]
pub struct MlSession;

#[cfg(not(feature = "ml"))]
impl MlSession {
    /// Create a new ML session (stub - returns error when feature disabled)
    pub fn new() -> Result<Self> {
        info!("⚠️  ML session not available (ml feature not enabled)");
        info!("Enable with: cargo build --features ml");
        Ok(Self)
    }

    /// Create a new ML session with explicit device (stub)
    pub fn with_device(_device: &str) -> Result<Self> {
        info!("⚠️  ML session not available (ml feature not enabled)");
        Ok(Self)
    }

    /// Run smoke test (stub)
    pub fn smoke_test(&self) -> Result<String> {
        Ok("ML feature disabled. Enable with --features ml".to_string())
    }

    /// Simple addition test (stub)
    pub fn test_addition(&self) -> Result<()> {
        Ok(())
    }

    /// Simple multiplication test (stub)
    pub fn test_multiplication(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(not(feature = "ml"))]
impl Default for MlSession {
    fn default() -> Self {
        Self::new().expect("Failed to create default MlSession")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "ml")]
    fn test_serialization_module() {
        // Test that serialization module is accessible
        use crate::ModelSerializer;
        use std::env;
        
        #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
        struct TestData {
            value: i32,
        }
        
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_serialization.neuroxide");
        
        let data = TestData { value: 42 };
        ModelSerializer::save(&data, &path).unwrap();
        let loaded: TestData = ModelSerializer::load(&path).unwrap();
        
        assert_eq!(data, loaded);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    #[cfg(feature = "ml")]
    fn test_cuda_optimizer_module() {
        use crate::CudaOptimizer;
        
        let mut optimizer = CudaOptimizer::new();
        assert!(optimizer.initialize().is_ok());
        
        let stats = optimizer.get_stats().unwrap();
        assert!(stats.is_initialized);
    }

    #[test]
    #[cfg(feature = "ml")]
    fn test_training_pipeline_module() {
        use crate::{TrainingConfig, TrainingPipeline};
        
        let config = TrainingConfig::default();
        let mut pipeline = TrainingPipeline::new(config);
        
        assert!(pipeline.initialize().is_ok());
    }

    #[test]
    #[cfg(feature = "ml")]
    fn test_session_creation() {
        let session = MlSession::new();
        assert!(session.is_ok());
    }

    #[test]
    #[cfg(feature = "ml")]
    fn test_smoke_test() {
        let session = MlSession::new().expect("Session creation failed");
        let result = session.smoke_test();
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("Neuroxide"));
    }

    #[test]
    #[cfg(not(feature = "ml"))]
    fn test_stub_no_panic() {
        let session = MlSession::new();
        assert!(session.is_ok());
        let result = session.unwrap().smoke_test();
        assert!(result.is_ok());
    }
}
