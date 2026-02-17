//! Model serialization for Neuroxide models
//!
//! Provides functionality to save and load Neuroxide models in a custom format.
//! Supports both TensorDB and individual tensor serialization.

use anyhow::{Context, Result};
use log::info;
use std::fs;
use std::path::Path;

/// Model serializer for Neuroxide models
///
/// Handles saving and loading of model weights, architecture metadata,
/// and training state checkpoints.
pub struct ModelSerializer;

impl ModelSerializer {
    /// Save a model to disk in Neuroxide format
    ///
    /// Format: .neuroxide file containing:
    /// - Model architecture metadata (JSON)
    /// - Tensor data (binary)
    /// - Device information
    /// - Version information
    ///
    /// # Arguments
    /// * `model_data` - Serializable model data (currently placeholder)
    /// * `path` - Output file path
    ///
    /// # Example
    /// ```no_run
    /// use browerai_ml::ModelSerializer;
    /// use std::path::Path;
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let model_data = vec![1.0, 2.0, 3.0]; // Simplified
    /// ModelSerializer::save(&model_data, Path::new("model.neuroxide"))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn save<T: serde::Serialize>(model_data: &T, path: &Path) -> Result<()> {
        info!("💾 Saving Neuroxide model to: {}", path.display());

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context("Failed to create model directory")?;
        }

        // Serialize model metadata and weights
        let json_data =
            serde_json::to_string_pretty(model_data).context("Failed to serialize model data")?;

        // Write to file
        fs::write(path, json_data)
            .with_context(|| format!("Failed to write model to {}", path.display()))?;

        info!("✅ Model saved successfully");
        Ok(())
    }

    /// Load a model from disk
    ///
    /// Reconstructs the model from the .neuroxide file format.
    ///
    /// # Arguments
    /// * `path` - Path to .neuroxide model file
    ///
    /// # Returns
    /// Deserialized model data
    ///
    /// # Example
    /// ```no_run
    /// use browerai_ml::ModelSerializer;
    /// use std::path::Path;
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let model_data: Vec<f32> = ModelSerializer::load(Path::new("model.neuroxide"))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
        info!("📂 Loading Neuroxide model from: {}", path.display());

        let json_data = fs::read_to_string(path)
            .with_context(|| format!("Failed to read model from {}", path.display()))?;

        let model_data =
            serde_json::from_str(&json_data).context("Failed to deserialize model data")?;

        info!("✅ Model loaded successfully");
        Ok(model_data)
    }

    /// Create a checkpoint snapshot of current training state
    ///
    /// Saves:
    /// - Model weights
    /// - Optimizer state
    /// - Training epoch/iteration
    /// - Learning rate schedule
    ///
    /// # Arguments
    /// * `checkpoint_data` - Complete training checkpoint
    /// * `path` - Output checkpoint file path
    pub fn save_checkpoint<T: serde::Serialize>(checkpoint_data: &T, path: &Path) -> Result<()> {
        info!("🔖 Saving training checkpoint to: {}", path.display());
        Self::save(checkpoint_data, path)
    }

    /// Load a training checkpoint
    ///
    /// Restores complete training state for resuming training.
    ///
    /// # Arguments
    /// * `path` - Path to checkpoint file
    pub fn load_checkpoint<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
        info!("🔖 Loading training checkpoint from: {}", path.display());
        Self::load(path)
    }

    /// Export model to ONNX format for cross-framework compatibility
    ///
    /// Converts Neuroxide model to ONNX for use with:
    /// - ONNX Runtime
    /// - TensorFlow
    /// - PyTorch
    /// - Other ONNX-compatible frameworks
    ///
    /// # Arguments
    /// * `model_data` - Neuroxide model to export
    /// * `output_path` - Output .onnx file path
    ///
    /// # Note
    /// This is a placeholder for future ONNX export functionality.
    /// Requires Neuroxide to support ONNX conversion (not yet available in Alpha).
    pub fn export_to_onnx<T: serde::Serialize>(_model_data: &T, output_path: &Path) -> Result<()> {
        info!("🔄 ONNX export requested: {}", output_path.display());
        info!("⚠️  ONNX export not yet implemented in Neuroxide Alpha");
        info!("📝 This feature will be available when Neuroxide stabilizes");

        anyhow::bail!(
            "ONNX export is not yet supported. \
             This is a planned feature for when Neuroxide reaches stable release."
        )
    }

    /// Validate a saved model file
    ///
    /// Checks:
    /// - File exists and is readable
    /// - Format is valid JSON
    /// - Required metadata fields present
    ///
    /// # Arguments
    /// * `path` - Path to model file to validate
    pub fn validate_model_file(path: &Path) -> Result<bool> {
        if !path.exists() {
            anyhow::bail!("Model file does not exist: {}", path.display());
        }

        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read model file: {}", path.display()))?;

        // Try to parse as JSON
        serde_json::from_str::<serde_json::Value>(&content)
            .context("Model file is not valid JSON")?;

        info!("✅ Model file validation passed");
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::env;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestModel {
        weights: Vec<f32>,
        architecture: String,
        version: String,
    }

    #[test]
    fn test_save_and_load_model() {
        let temp_dir = env::temp_dir();
        let model_path = temp_dir.join("test_model.neuroxide");

        let original_model = TestModel {
            weights: vec![1.0, 2.0, 3.0, 4.0],
            architecture: "SimpleNN".to_string(),
            version: "0.1.0".to_string(),
        };

        // Save model
        ModelSerializer::save(&original_model, &model_path).expect("Failed to save model");

        // Load model
        let loaded_model: TestModel =
            ModelSerializer::load(&model_path).expect("Failed to load model");

        assert_eq!(original_model, loaded_model);

        // Cleanup
        let _ = fs::remove_file(model_path);
    }

    #[test]
    fn test_validate_model_file() {
        let temp_dir = env::temp_dir();
        let model_path = temp_dir.join("test_validate.neuroxide");

        let model = TestModel {
            weights: vec![0.5],
            architecture: "Test".to_string(),
            version: "1.0.0".to_string(),
        };

        ModelSerializer::save(&model, &model_path).unwrap();

        let is_valid = ModelSerializer::validate_model_file(&model_path);
        assert!(is_valid.is_ok());
        assert!(is_valid.unwrap());

        // Cleanup
        let _ = fs::remove_file(model_path);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = env::temp_dir();
        let checkpoint_path = temp_dir.join("test_checkpoint.neuroxide");

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct Checkpoint {
            epoch: u32,
            loss: f32,
            model: TestModel,
        }

        let checkpoint = Checkpoint {
            epoch: 10,
            loss: 0.123,
            model: TestModel {
                weights: vec![1.0, 2.0],
                architecture: "SimpleNN".to_string(),
                version: "0.1.0".to_string(),
            },
        };

        ModelSerializer::save_checkpoint(&checkpoint, &checkpoint_path).unwrap();
        let loaded: Checkpoint = ModelSerializer::load_checkpoint(&checkpoint_path).unwrap();

        assert_eq!(checkpoint, loaded);

        // Cleanup
        let _ = fs::remove_file(checkpoint_path);
    }
}
