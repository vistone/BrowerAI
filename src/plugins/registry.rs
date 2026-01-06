/// Plugin registry for managing active plugins
///
/// Maintains registry of loaded plugins and handles plugin interactions
use super::{HookResult, Plugin, PluginCapability, PluginError, PluginHook};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Plugin registry
pub struct PluginRegistry {
    /// Active plugins by name
    plugins: HashMap<String, Arc<RwLock<Box<dyn Plugin>>>>,
    /// Plugins grouped by capability
    by_capability: HashMap<PluginCapability, Vec<String>>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            by_capability: HashMap::new(),
        }
    }

    /// Register a plugin
    pub fn register(&mut self, plugin: Box<dyn Plugin>) -> Result<(), PluginError> {
        let name = plugin.metadata().name.clone();
        let capabilities = plugin.capabilities();

        // Check if plugin is already registered
        if self.plugins.contains_key(&name) {
            return Err(PluginError::InitializationFailed(format!(
                "Plugin {} is already registered",
                name
            )));
        }

        // Register by capabilities
        for cap in &capabilities {
            self.by_capability
                .entry(*cap)
                .or_insert_with(Vec::new)
                .push(name.clone());
        }

        // Store plugin
        self.plugins.insert(name, Arc::new(RwLock::new(plugin)));

        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister(&mut self, name: &str) -> Result<(), PluginError> {
        // Remove from registry
        let plugin_arc = self
            .plugins
            .remove(name)
            .ok_or_else(|| PluginError::NotFound(name.to_string()))?;

        // Get capabilities before shutdown
        let capabilities = {
            let plugin = plugin_arc.read().unwrap();
            plugin.capabilities()
        };

        // Shutdown plugin
        {
            let mut plugin = plugin_arc.write().unwrap();
            plugin.shutdown()?;
        }

        // Remove from capability index
        for cap in capabilities {
            if let Some(plugins) = self.by_capability.get_mut(&cap) {
                plugins.retain(|n| n != name);
            }
        }

        Ok(())
    }

    /// Get a plugin by name
    pub fn get(&self, name: &str) -> Option<Arc<RwLock<Box<dyn Plugin>>>> {
        self.plugins.get(name).cloned()
    }

    /// Get all plugins with a specific capability
    pub fn get_by_capability(
        &self,
        capability: &PluginCapability,
    ) -> Vec<Arc<RwLock<Box<dyn Plugin>>>> {
        let plugin_names = match self.by_capability.get(capability) {
            Some(names) => names,
            None => return Vec::new(),
        };

        plugin_names
            .iter()
            .filter_map(|name| self.plugins.get(name).cloned())
            .collect()
    }

    /// Execute a hook across all relevant plugins
    pub fn execute_hook(&self, hook: &PluginHook) -> Vec<Result<HookResult, PluginError>> {
        let mut results = Vec::new();

        // Determine which plugins should receive this hook
        let relevant_plugins: Vec<_> = self.plugins.values().cloned().collect();

        for plugin_arc in relevant_plugins {
            let mut plugin = match plugin_arc.write() {
                Ok(guard) => guard,
                Err(_) => continue,
            };

            let result = plugin.on_hook(hook);
            results.push(result);
        }

        results
    }

    /// Get count of registered plugins
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// List all registered plugin names
    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins.keys().cloned().collect()
    }

    /// Check if a plugin is registered
    pub fn is_registered(&self, name: &str) -> bool {
        self.plugins.contains_key(name)
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::PluginMetadata;
    use super::*;

    struct TestPlugin {
        metadata: PluginMetadata,
        initialized: bool,
    }

    impl TestPlugin {
        fn new(name: &str) -> Self {
            Self {
                metadata: PluginMetadata {
                    name: name.to_string(),
                    version: "1.0.0".to_string(),
                    author: "Test".to_string(),
                    description: "Test".to_string(),
                    required_version: None,
                    dependencies: vec![],
                },
                initialized: false,
            }
        }
    }

    impl Plugin for TestPlugin {
        fn metadata(&self) -> &PluginMetadata {
            &self.metadata
        }

        fn initialize(&mut self) -> Result<(), PluginError> {
            self.initialized = true;
            Ok(())
        }

        fn shutdown(&mut self) -> Result<(), PluginError> {
            self.initialized = false;
            Ok(())
        }

        fn capabilities(&self) -> Vec<PluginCapability> {
            vec![PluginCapability::HtmlModification]
        }

        fn on_hook(&mut self, _hook: &PluginHook) -> Result<HookResult, PluginError> {
            Ok(HookResult::Continue)
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = PluginRegistry::new();
        assert_eq!(registry.plugin_count(), 0);
    }

    #[test]
    fn test_register_plugin() {
        let mut registry = PluginRegistry::new();
        let plugin = Box::new(TestPlugin::new("test_plugin"));

        let result = registry.register(plugin);
        assert!(result.is_ok());
        assert_eq!(registry.plugin_count(), 1);
    }

    #[test]
    fn test_register_duplicate_plugin() {
        let mut registry = PluginRegistry::new();
        let plugin1 = Box::new(TestPlugin::new("test_plugin"));
        let plugin2 = Box::new(TestPlugin::new("test_plugin"));

        registry.register(plugin1).unwrap();
        let result = registry.register(plugin2);

        assert!(result.is_err());
    }

    #[test]
    fn test_unregister_plugin() {
        let mut registry = PluginRegistry::new();
        let plugin = Box::new(TestPlugin::new("test_plugin"));

        registry.register(plugin).unwrap();
        assert_eq!(registry.plugin_count(), 1);

        registry.unregister("test_plugin").unwrap();
        assert_eq!(registry.plugin_count(), 0);
    }

    #[test]
    fn test_get_plugin() {
        let mut registry = PluginRegistry::new();
        let plugin = Box::new(TestPlugin::new("test_plugin"));

        registry.register(plugin).unwrap();

        let retrieved = registry.get("test_plugin");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_get_by_capability() {
        let mut registry = PluginRegistry::new();
        let plugin = Box::new(TestPlugin::new("test_plugin"));

        registry.register(plugin).unwrap();

        let plugins = registry.get_by_capability(&PluginCapability::HtmlModification);
        assert_eq!(plugins.len(), 1);
    }

    #[test]
    fn test_execute_hook() {
        let mut registry = PluginRegistry::new();
        let plugin = Box::new(TestPlugin::new("test_plugin"));

        registry.register(plugin).unwrap();

        let hook = PluginHook::BeforeHtmlParse {
            content: "<html></html>".to_string(),
        };

        let results = registry.execute_hook(&hook);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
    }

    #[test]
    fn test_list_plugins() {
        let mut registry = PluginRegistry::new();
        registry
            .register(Box::new(TestPlugin::new("plugin1")))
            .unwrap();
        registry
            .register(Box::new(TestPlugin::new("plugin2")))
            .unwrap();

        let plugins = registry.list_plugins();
        assert_eq!(plugins.len(), 2);
        assert!(plugins.contains(&"plugin1".to_string()));
        assert!(plugins.contains(&"plugin2".to_string()));
    }

    #[test]
    fn test_is_registered() {
        let mut registry = PluginRegistry::new();
        registry
            .register(Box::new(TestPlugin::new("test_plugin")))
            .unwrap();

        assert!(registry.is_registered("test_plugin"));
        assert!(!registry.is_registered("other_plugin"));
    }
}
