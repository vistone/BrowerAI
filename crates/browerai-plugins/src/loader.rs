/// Plugin loader for loading and managing plugins
///
/// Handles plugin discovery, loading, and lifecycle management
use super::{Plugin, PluginError, PluginMetadata};
use std::collections::HashMap;
use std::path::PathBuf;

/// Plugin loader
pub struct PluginLoader {
    /// Search paths for plugins
    search_paths: Vec<PathBuf>,
    /// Loaded plugin metadata cache
    metadata_cache: HashMap<String, PluginMetadata>,
}

impl PluginLoader {
    /// Create a new plugin loader
    pub fn new() -> Self {
        Self {
            search_paths: vec![PathBuf::from("./plugins")],
            metadata_cache: HashMap::new(),
        }
    }

    /// Add a search path
    pub fn add_search_path(&mut self, path: PathBuf) {
        if !self.search_paths.contains(&path) {
            self.search_paths.push(path);
        }
    }

    /// Discover plugins in search paths
    pub fn discover_plugins(&mut self) -> Result<Vec<String>, PluginError> {
        let mut discovered = Vec::new();

        for search_path in &self.search_paths {
            if !search_path.exists() {
                continue;
            }

            // In a real implementation, this would:
            // 1. Scan the directory for plugin files
            // 2. Read plugin metadata
            // 3. Validate plugin compatibility
            // 4. Cache metadata

            // Stub: just return some fake plugin names for testing
            if search_path.to_str() == Some("./plugins") {
                discovered.push("example_plugin".to_string());
            }
        }

        Ok(discovered)
    }

    /// Load a plugin by name (stub implementation)
    pub fn load_plugin(&mut self, name: &str) -> Result<Box<dyn Plugin>, PluginError> {
        // In a real implementation, this would:
        // 1. Find the plugin in search paths
        // 2. Load the plugin binary/library
        // 3. Instantiate the plugin
        // 4. Return the plugin instance

        Err(PluginError::NotFound(name.to_string()))
    }

    /// Get cached metadata for a plugin
    pub fn get_metadata(&self, name: &str) -> Option<&PluginMetadata> {
        self.metadata_cache.get(name)
    }

    /// Cache plugin metadata
    pub fn cache_metadata(&mut self, metadata: PluginMetadata) {
        self.metadata_cache.insert(metadata.name.clone(), metadata);
    }

    /// Get all search paths
    pub fn search_paths(&self) -> &[PathBuf] {
        &self.search_paths
    }
}

impl Default for PluginLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_loader_creation() {
        let loader = PluginLoader::new();
        assert_eq!(loader.search_paths.len(), 1);
    }

    #[test]
    fn test_add_search_path() {
        let mut loader = PluginLoader::new();
        loader.add_search_path(PathBuf::from("/custom/plugins"));

        assert_eq!(loader.search_paths.len(), 2);
        assert!(loader
            .search_paths
            .contains(&PathBuf::from("/custom/plugins")));
    }

    #[test]
    fn test_add_duplicate_search_path() {
        let mut loader = PluginLoader::new();
        let path = PathBuf::from("/custom/plugins");

        loader.add_search_path(path.clone());
        loader.add_search_path(path.clone());

        assert_eq!(loader.search_paths.len(), 2); // Should not add duplicates
    }

    #[test]
    fn test_discover_plugins() {
        let mut loader = PluginLoader::new();
        let result = loader.discover_plugins();

        assert!(result.is_ok());
    }

    #[test]
    fn test_load_plugin_not_found() {
        let mut loader = PluginLoader::new();
        let result = loader.load_plugin("nonexistent_plugin");

        assert!(result.is_err());
        match result {
            Err(PluginError::NotFound(name)) => assert_eq!(name, "nonexistent_plugin"),
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn test_cache_metadata() {
        let mut loader = PluginLoader::new();
        let metadata = PluginMetadata {
            name: "test_plugin".to_string(),
            version: "1.0.0".to_string(),
            author: "Test Author".to_string(),
            description: "Test plugin".to_string(),
            required_version: None,
            dependencies: vec![],
        };

        loader.cache_metadata(metadata.clone());

        let cached = loader.get_metadata("test_plugin");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().name, "test_plugin");
    }

    #[test]
    fn test_get_metadata_not_found() {
        let loader = PluginLoader::new();
        let metadata = loader.get_metadata("nonexistent");
        assert!(metadata.is_none());
    }

    #[test]
    fn test_search_paths() {
        let loader = PluginLoader::new();
        let paths = loader.search_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], PathBuf::from("./plugins"));
    }
}
