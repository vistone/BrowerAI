/// Plugin system for extensibility
/// 
/// Allows third-party extensions to enhance browser functionality

pub mod loader;
pub mod registry;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use loader::PluginLoader;
pub use registry::PluginRegistry;

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin author
    pub author: String,
    /// Plugin description
    pub description: String,
    /// Required browser version
    pub required_version: Option<String>,
    /// Plugin dependencies
    pub dependencies: Vec<String>,
}

/// Plugin capability flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginCapability {
    /// Can modify HTML content
    HtmlModification,
    /// Can modify CSS styles
    CssModification,
    /// Can execute JavaScript
    JavaScriptExecution,
    /// Can intercept network requests
    NetworkInterception,
    /// Can access storage
    StorageAccess,
    /// Can modify rendering
    RenderingModification,
}

/// Plugin lifecycle hooks
pub trait Plugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Initialize the plugin
    fn initialize(&mut self) -> Result<(), PluginError>;

    /// Shutdown the plugin
    fn shutdown(&mut self) -> Result<(), PluginError>;

    /// Get plugin capabilities
    fn capabilities(&self) -> Vec<PluginCapability>;

    /// Handle a hook event
    fn on_hook(&mut self, hook: &PluginHook) -> Result<HookResult, PluginError>;
}

/// Plugin hook types
#[derive(Debug, Clone)]
pub enum PluginHook {
    /// Before HTML parsing
    BeforeHtmlParse { content: String },
    /// After HTML parsing
    AfterHtmlParse,
    /// Before CSS parsing
    BeforeCssParse { content: String },
    /// After CSS parsing
    AfterCssParse,
    /// Before rendering
    BeforeRender,
    /// After rendering
    AfterRender,
    /// Network request
    NetworkRequest { url: String },
    /// Custom hook
    Custom { name: String, data: HashMap<String, String> },
}

/// Hook result
#[derive(Debug, Clone)]
pub enum HookResult {
    /// Continue with original behavior
    Continue,
    /// Replace content/behavior
    Replace(String),
    /// Cancel the operation
    Cancel,
    /// Custom result
    Custom(HashMap<String, String>),
}

/// Plugin errors
#[derive(Debug, Clone)]
pub enum PluginError {
    /// Plugin initialization failed
    InitializationFailed(String),
    /// Plugin not found
    NotFound(String),
    /// Invalid plugin format
    InvalidFormat(String),
    /// Missing dependency
    MissingDependency(String),
    /// Permission denied
    PermissionDenied(String),
    /// Runtime error
    RuntimeError(String),
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            PluginError::NotFound(name) => write!(f, "Plugin not found: {}", name),
            PluginError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            PluginError::MissingDependency(dep) => write!(f, "Missing dependency: {}", dep),
            PluginError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            PluginError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl std::error::Error for PluginError {}

#[cfg(test)]
mod tests {
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
                    author: "Test Author".to_string(),
                    description: "Test plugin".to_string(),
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
    fn test_plugin_metadata() {
        let plugin = TestPlugin::new("test_plugin");
        assert_eq!(plugin.metadata().name, "test_plugin");
        assert_eq!(plugin.metadata().version, "1.0.0");
    }

    #[test]
    fn test_plugin_lifecycle() {
        let mut plugin = TestPlugin::new("test_plugin");
        assert!(!plugin.initialized);

        plugin.initialize().unwrap();
        assert!(plugin.initialized);

        plugin.shutdown().unwrap();
        assert!(!plugin.initialized);
    }

    #[test]
    fn test_plugin_capabilities() {
        let plugin = TestPlugin::new("test_plugin");
        let caps = plugin.capabilities();
        assert_eq!(caps.len(), 1);
        assert!(caps.contains(&PluginCapability::HtmlModification));
    }

    #[test]
    fn test_plugin_hook() {
        let mut plugin = TestPlugin::new("test_plugin");
        let hook = PluginHook::BeforeHtmlParse {
            content: "<html></html>".to_string(),
        };

        let result = plugin.on_hook(&hook);
        assert!(result.is_ok());
    }

    #[test]
    fn test_plugin_error_display() {
        let err = PluginError::NotFound("test_plugin".to_string());
        assert_eq!(err.to_string(), "Plugin not found: test_plugin");

        let err2 = PluginError::RuntimeError("test error".to_string());
        assert_eq!(err2.to_string(), "Runtime error: test error");
    }

    #[test]
    fn test_hook_result_variants() {
        let continue_result = HookResult::Continue;
        let replace_result = HookResult::Replace("new content".to_string());
        let cancel_result = HookResult::Cancel;

        match continue_result {
            HookResult::Continue => {},
            _ => panic!("Expected Continue variant"),
        }

        match replace_result {
            HookResult::Replace(content) => assert_eq!(content, "new content"),
            _ => panic!("Expected Replace variant"),
        }

        match cancel_result {
            HookResult::Cancel => {},
            _ => panic!("Expected Cancel variant"),
        }
    }
}
