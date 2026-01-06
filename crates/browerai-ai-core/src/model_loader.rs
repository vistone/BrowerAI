// Model hot-reloading system for AI models
// Allows reloading models without restarting the application

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

/// Model metadata for tracking loaded models
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Path to the model file
    pub path: PathBuf,
    /// Last modification time
    pub last_modified: SystemTime,
    /// Model version identifier
    pub version: String,
    /// Model type (e.g., "html_parser", "css_optimizer")
    pub model_type: String,
}

/// Model loader with hot-reloading capability
pub struct ModelLoader {
    /// Registered models with their metadata
    models: Arc<RwLock<HashMap<String, ModelMetadata>>>,
    /// Model directory path
    model_dir: PathBuf,
    /// Enable automatic reloading on file change
    auto_reload: bool,
}

impl ModelLoader {
    /// Creates a new ModelLoader
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing model files
    /// * `auto_reload` - Enable automatic reloading on file changes
    pub fn new<P: AsRef<Path>>(model_dir: P, auto_reload: bool) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            model_dir: model_dir.as_ref().to_path_buf(),
            auto_reload,
        }
    }

    /// Registers a model for tracking
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier for the model
    /// * `filename` - Model filename (e.g., "html_parser_v1.onnx")
    /// * `model_type` - Type of model (e.g., "html_parser")
    ///
    /// # Returns
    /// Result with unit on success, or error message
    pub fn register_model(
        &self,
        model_id: &str,
        filename: &str,
        model_type: &str,
    ) -> Result<(), String> {
        let model_path = self.model_dir.join(filename);

        // Check if model file exists
        if !model_path.exists() {
            return Err(format!("Model file not found: {}", model_path.display()));
        }

        // Get file metadata
        let metadata = fs::metadata(&model_path)
            .map_err(|e| format!("Failed to read model metadata: {}", e))?;

        let last_modified = metadata
            .modified()
            .map_err(|e| format!("Failed to get modification time: {}", e))?;

        let model_meta = ModelMetadata {
            path: model_path,
            last_modified,
            version: "v1".to_string(),
            model_type: model_type.to_string(),
        };

        let mut models = self
            .models
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        models.insert(model_id.to_string(), model_meta);

        Ok(())
    }

    /// Checks if a model has been modified since last load
    ///
    /// # Arguments
    /// * `model_id` - Model identifier to check
    ///
    /// # Returns
    /// Result with true if modified, false otherwise
    pub fn is_model_modified(&self, model_id: &str) -> Result<bool, String> {
        let models = self
            .models
            .read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        let model_meta = models
            .get(model_id)
            .ok_or_else(|| format!("Model not registered: {}", model_id))?;

        let current_metadata = fs::metadata(&model_meta.path)
            .map_err(|e| format!("Failed to read model metadata: {}", e))?;

        let current_modified = current_metadata
            .modified()
            .map_err(|e| format!("Failed to get modification time: {}", e))?;

        Ok(current_modified > model_meta.last_modified)
    }

    /// Reloads a model if it has been modified
    ///
    /// # Arguments
    /// * `model_id` - Model identifier to reload
    ///
    /// # Returns
    /// Result with true if reloaded, false if no reload needed
    pub fn reload_if_modified(&self, model_id: &str) -> Result<bool, String> {
        if !self.is_model_modified(model_id)? {
            return Ok(false);
        }

        self.reload_model(model_id)?;
        Ok(true)
    }

    /// Forces a model reload regardless of modification status
    ///
    /// # Arguments
    /// * `model_id` - Model identifier to reload
    ///
    /// # Returns
    /// Result with unit on success
    pub fn reload_model(&self, model_id: &str) -> Result<(), String> {
        let mut models = self
            .models
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        let model_meta = models
            .get_mut(model_id)
            .ok_or_else(|| format!("Model not registered: {}", model_id))?;

        // Update last modified time
        let current_metadata = fs::metadata(&model_meta.path)
            .map_err(|e| format!("Failed to read model metadata: {}", e))?;

        model_meta.last_modified = current_metadata
            .modified()
            .map_err(|e| format!("Failed to get modification time: {}", e))?;

        Ok(())
    }

    /// Gets metadata for a registered model
    ///
    /// # Arguments
    /// * `model_id` - Model identifier
    ///
    /// # Returns
    /// Result with model metadata
    pub fn get_model_metadata(&self, model_id: &str) -> Result<ModelMetadata, String> {
        let models = self
            .models
            .read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        models
            .get(model_id)
            .cloned()
            .ok_or_else(|| format!("Model not registered: {}", model_id))
    }

    /// Lists all registered models
    ///
    /// # Returns
    /// Vector of model identifiers
    pub fn list_models(&self) -> Vec<String> {
        let models = self.models.read().unwrap_or_else(|e| e.into_inner());
        models.keys().cloned().collect()
    }

    /// Checks all models and reloads any that have been modified
    ///
    /// # Returns
    /// Result with count of models reloaded
    pub fn check_and_reload_all(&self) -> Result<usize, String> {
        if !self.auto_reload {
            return Ok(0);
        }

        let model_ids = self.list_models();
        let mut reloaded_count = 0;

        for model_id in model_ids {
            if self.reload_if_modified(&model_id)? {
                reloaded_count += 1;
            }
        }

        Ok(reloaded_count)
    }

    /// Unregisters a model from tracking
    ///
    /// # Arguments
    /// * `model_id` - Model identifier to unregister
    ///
    /// # Returns
    /// Result with unit on success
    pub fn unregister_model(&self, model_id: &str) -> Result<(), String> {
        let mut models = self
            .models
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        models
            .remove(model_id)
            .ok_or_else(|| format!("Model not registered: {}", model_id))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_model_loader_creation() {
        let temp_dir = TempDir::new().unwrap();
        let loader = ModelLoader::new(temp_dir.path(), false);
        assert_eq!(loader.list_models().len(), 0);
    }

    #[test]
    fn test_register_model() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.onnx");

        // Create a dummy model file
        let mut file = fs::File::create(&model_path).unwrap();
        file.write_all(b"fake model data").unwrap();
        drop(file);

        let loader = ModelLoader::new(temp_dir.path(), false);
        let result = loader.register_model("test_model", "test_model.onnx", "test");

        assert!(result.is_ok());
        assert_eq!(loader.list_models().len(), 1);
    }

    #[test]
    fn test_register_nonexistent_model() {
        let temp_dir = TempDir::new().unwrap();
        let loader = ModelLoader::new(temp_dir.path(), false);

        let result = loader.register_model("missing", "missing.onnx", "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_modification_check() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.onnx");

        // Create initial model
        let mut file = fs::File::create(&model_path).unwrap();
        file.write_all(b"initial data").unwrap();
        drop(file);

        let loader = ModelLoader::new(temp_dir.path(), false);
        loader
            .register_model("test_model", "test_model.onnx", "test")
            .unwrap();

        // Check initially - should not be modified
        assert!(!loader.is_model_modified("test_model").unwrap());
    }

    #[test]
    fn test_list_models() {
        let temp_dir = TempDir::new().unwrap();

        // Create multiple model files
        for i in 1..=3 {
            let model_path = temp_dir.path().join(format!("model{}.onnx", i));
            let mut file = fs::File::create(&model_path).unwrap();
            file.write_all(b"model data").unwrap();
        }

        let loader = ModelLoader::new(temp_dir.path(), false);
        for i in 1..=3 {
            loader
                .register_model(&format!("model{}", i), &format!("model{}.onnx", i), "test")
                .unwrap();
        }

        assert_eq!(loader.list_models().len(), 3);
    }

    #[test]
    fn test_get_model_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.onnx");

        let mut file = fs::File::create(&model_path).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        let loader = ModelLoader::new(temp_dir.path(), false);
        loader
            .register_model("test_model", "test_model.onnx", "html_parser")
            .unwrap();

        let metadata = loader.get_model_metadata("test_model").unwrap();
        assert_eq!(metadata.model_type, "html_parser");
        assert_eq!(metadata.version, "v1");
    }

    #[test]
    fn test_unregister_model() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.onnx");

        let mut file = fs::File::create(&model_path).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        let loader = ModelLoader::new(temp_dir.path(), false);
        loader
            .register_model("test_model", "test_model.onnx", "test")
            .unwrap();
        assert_eq!(loader.list_models().len(), 1);

        loader.unregister_model("test_model").unwrap();
        assert_eq!(loader.list_models().len(), 0);
    }
}
