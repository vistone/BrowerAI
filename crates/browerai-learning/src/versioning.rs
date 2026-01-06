/// Model versioning system for tracking and managing model versions
///
/// Enables rolling back to previous versions and comparing model performance
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Semantic version for models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse version from string (e.g., "1.2.3")
    pub fn from_string(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(format!("Invalid version format: {}", s));
        }

        let major = parts[0]
            .parse()
            .map_err(|_| format!("Invalid major version: {}", parts[0]))?;
        let minor = parts[1]
            .parse()
            .map_err(|_| format!("Invalid minor version: {}", parts[1]))?;
        let patch = parts[2]
            .parse()
            .map_err(|_| format!("Invalid patch version: {}", parts[2]))?;

        Ok(Self::new(major, minor, patch))
    }

    /// Increment major version (breaking changes)
    pub fn increment_major(&mut self) {
        self.major += 1;
        self.minor = 0;
        self.patch = 0;
    }

    /// Increment minor version (new features)
    pub fn increment_minor(&mut self) {
        self.minor += 1;
        self.patch = 0;
    }

    /// Increment patch version (bug fixes)
    pub fn increment_patch(&mut self) {
        self.patch += 1;
    }
}

impl std::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Metadata for a versioned model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedModel {
    /// Model name
    pub name: String,
    /// Model version
    pub version: ModelVersion,
    /// Path to model file
    pub path: PathBuf,
    /// Creation timestamp
    pub created_at: u64,
    /// Performance metrics
    pub metrics: HashMap<String, f32>,
    /// Optional description
    pub description: Option<String>,
    /// Whether this is the active version
    pub is_active: bool,
}

impl VersionedModel {
    /// Create a new versioned model
    pub fn new(name: impl Into<String>, version: ModelVersion, path: PathBuf) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();

        Self {
            name: name.into(),
            version,
            path,
            created_at: timestamp,
            metrics: HashMap::new(),
            description: None,
            is_active: false,
        }
    }

    /// Add a performance metric
    pub fn add_metric(&mut self, name: impl Into<String>, value: f32) {
        self.metrics.insert(name.into(), value);
    }

    /// Get a performance metric
    pub fn get_metric(&self, name: &str) -> Option<f32> {
        self.metrics.get(name).copied()
    }

    /// Set description
    pub fn set_description(&mut self, description: impl Into<String>) {
        self.description = Some(description.into());
    }

    /// Set as active version
    pub fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }
}

/// Manager for model versions
pub struct VersionManager {
    /// All registered versions by model name
    versions: HashMap<String, Vec<VersionedModel>>,
}

impl VersionManager {
    /// Create a new version manager
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
        }
    }

    /// Register a new model version
    pub fn register_version(&mut self, model: VersionedModel) {
        let name = model.name.clone();
        self.versions
            .entry(name)
            .or_default()
            .push(model);
    }

    /// Get all versions for a model
    pub fn get_versions(&self, model_name: &str) -> Option<&[VersionedModel]> {
        self.versions.get(model_name).map(|v| v.as_slice())
    }

    /// Get the latest version for a model
    pub fn get_latest_version(&self, model_name: &str) -> Option<&VersionedModel> {
        self.versions
            .get(model_name)?
            .iter()
            .max_by(|a, b| a.version.cmp(&b.version))
    }

    /// Get the active version for a model
    pub fn get_active_version(&self, model_name: &str) -> Option<&VersionedModel> {
        self.versions.get(model_name)?.iter().find(|v| v.is_active)
    }

    /// Get a specific version
    pub fn get_version(&self, model_name: &str, version: &ModelVersion) -> Option<&VersionedModel> {
        self.versions
            .get(model_name)?
            .iter()
            .find(|v| &v.version == version)
    }

    /// Set active version for a model
    pub fn set_active_version(
        &mut self,
        model_name: &str,
        version: &ModelVersion,
    ) -> Result<(), String> {
        let versions = self
            .versions
            .get_mut(model_name)
            .ok_or_else(|| format!("Model not found: {}", model_name))?;

        let mut found = false;
        for v in versions.iter_mut() {
            if &v.version == version {
                v.is_active = true;
                found = true;
            } else {
                v.is_active = false;
            }
        }

        if found {
            Ok(())
        } else {
            Err(format!("Version not found: {}", version))
        }
    }

    /// Compare metrics between versions
    pub fn compare_versions(
        &self,
        model_name: &str,
        version1: &ModelVersion,
        version2: &ModelVersion,
    ) -> Option<HashMap<String, (f32, f32)>> {
        let v1 = self.get_version(model_name, version1)?;
        let v2 = self.get_version(model_name, version2)?;

        let mut comparison = HashMap::new();

        // Get all metric names from both versions
        let mut metric_names: Vec<String> = v1.metrics.keys().cloned().collect();
        for name in v2.metrics.keys() {
            if !metric_names.contains(name) {
                metric_names.push(name.clone());
            }
        }

        for metric_name in metric_names {
            let value1 = v1.metrics.get(&metric_name).copied().unwrap_or(0.0);
            let value2 = v2.metrics.get(&metric_name).copied().unwrap_or(0.0);
            comparison.insert(metric_name, (value1, value2));
        }

        Some(comparison)
    }

    /// Get count of all models
    pub fn model_count(&self) -> usize {
        self.versions.len()
    }

    /// Get count of all versions
    pub fn version_count(&self) -> usize {
        self.versions.values().map(|v| v.len()).sum()
    }

    /// List all model names
    pub fn list_models(&self) -> Vec<String> {
        self.versions.keys().cloned().collect()
    }
}

impl Default for VersionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_creation() {
        let version = ModelVersion::new(1, 2, 3);
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
    }

    #[test]
    fn test_version_from_string() {
        let version = ModelVersion::from_string("1.2.3").unwrap();
        assert_eq!(version, ModelVersion::new(1, 2, 3));

        assert!(ModelVersion::from_string("1.2").is_err());
        assert!(ModelVersion::from_string("1.2.x").is_err());
    }

    #[test]
    fn test_version_to_string() {
        let version = ModelVersion::new(1, 2, 3);
        assert_eq!(version.to_string(), "1.2.3");
    }

    #[test]
    fn test_version_increment() {
        let mut version = ModelVersion::new(1, 2, 3);

        version.increment_patch();
        assert_eq!(version, ModelVersion::new(1, 2, 4));

        version.increment_minor();
        assert_eq!(version, ModelVersion::new(1, 3, 0));

        version.increment_major();
        assert_eq!(version, ModelVersion::new(2, 0, 0));
    }

    #[test]
    fn test_version_comparison() {
        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(2, 0, 0);
        let v3 = ModelVersion::new(1, 1, 0);

        assert!(v1 < v2);
        assert!(v1 < v3);
        assert!(v3 < v2);
    }

    #[test]
    fn test_versioned_model_creation() {
        let version = ModelVersion::new(1, 0, 0);
        let path = PathBuf::from("/models/test.onnx");
        let model = VersionedModel::new("test_model", version.clone(), path.clone());

        assert_eq!(model.name, "test_model");
        assert_eq!(model.version, version);
        assert_eq!(model.path, path);
        assert!(!model.is_active);
    }

    #[test]
    fn test_versioned_model_metrics() {
        let version = ModelVersion::new(1, 0, 0);
        let path = PathBuf::from("/models/test.onnx");
        let mut model = VersionedModel::new("test_model", version, path);

        model.add_metric("accuracy", 0.95);
        model.add_metric("speed", 100.0);

        assert_eq!(model.get_metric("accuracy"), Some(0.95));
        assert_eq!(model.get_metric("speed"), Some(100.0));
        assert_eq!(model.get_metric("nonexistent"), None);
    }

    #[test]
    fn test_version_manager_register() {
        let mut manager = VersionManager::new();

        let v1 = VersionedModel::new(
            "html_parser",
            ModelVersion::new(1, 0, 0),
            PathBuf::from("/models/v1.onnx"),
        );

        manager.register_version(v1);

        assert_eq!(manager.model_count(), 1);
        assert_eq!(manager.version_count(), 1);
    }

    #[test]
    fn test_version_manager_get_latest() {
        let mut manager = VersionManager::new();

        manager.register_version(VersionedModel::new(
            "html_parser",
            ModelVersion::new(1, 0, 0),
            PathBuf::from("/models/v1.onnx"),
        ));

        manager.register_version(VersionedModel::new(
            "html_parser",
            ModelVersion::new(2, 0, 0),
            PathBuf::from("/models/v2.onnx"),
        ));

        manager.register_version(VersionedModel::new(
            "html_parser",
            ModelVersion::new(1, 5, 0),
            PathBuf::from("/models/v1.5.onnx"),
        ));

        let latest = manager.get_latest_version("html_parser").unwrap();
        assert_eq!(latest.version, ModelVersion::new(2, 0, 0));
    }

    #[test]
    fn test_version_manager_set_active() {
        let mut manager = VersionManager::new();

        manager.register_version(VersionedModel::new(
            "html_parser",
            ModelVersion::new(1, 0, 0),
            PathBuf::from("/models/v1.onnx"),
        ));

        manager.register_version(VersionedModel::new(
            "html_parser",
            ModelVersion::new(2, 0, 0),
            PathBuf::from("/models/v2.onnx"),
        ));

        let version = ModelVersion::new(1, 0, 0);
        manager.set_active_version("html_parser", &version).unwrap();

        let active = manager.get_active_version("html_parser").unwrap();
        assert_eq!(active.version, version);
    }

    #[test]
    fn test_version_manager_compare() {
        let mut manager = VersionManager::new();

        let mut v1 = VersionedModel::new(
            "html_parser",
            ModelVersion::new(1, 0, 0),
            PathBuf::from("/models/v1.onnx"),
        );
        v1.add_metric("accuracy", 0.90);
        v1.add_metric("speed", 100.0);

        let mut v2 = VersionedModel::new(
            "html_parser",
            ModelVersion::new(2, 0, 0),
            PathBuf::from("/models/v2.onnx"),
        );
        v2.add_metric("accuracy", 0.95);
        v2.add_metric("speed", 120.0);

        manager.register_version(v1);
        manager.register_version(v2);

        let comparison = manager
            .compare_versions(
                "html_parser",
                &ModelVersion::new(1, 0, 0),
                &ModelVersion::new(2, 0, 0),
            )
            .unwrap();

        assert_eq!(comparison.get("accuracy"), Some(&(0.90, 0.95)));
        assert_eq!(comparison.get("speed"), Some(&(100.0, 120.0)));
    }

    #[test]
    fn test_version_manager_list_models() {
        let mut manager = VersionManager::new();

        manager.register_version(VersionedModel::new(
            "html_parser",
            ModelVersion::new(1, 0, 0),
            PathBuf::from("/models/v1.onnx"),
        ));

        manager.register_version(VersionedModel::new(
            "css_parser",
            ModelVersion::new(1, 0, 0),
            PathBuf::from("/models/css_v1.onnx"),
        ));

        let models = manager.list_models();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"html_parser".to_string()));
        assert!(models.contains(&"css_parser".to_string()));
    }
}
