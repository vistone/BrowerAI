use serde::{Deserialize, Serialize};

/// Global browser configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserConfig {
    /// Enable AI-powered features
    pub enable_ai: bool,
    
    /// Enable logging
    pub enable_logging: bool,
    
    /// Log level
    pub log_level: String,
    
    /// Model directory path
    pub model_dir: String,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            enable_ai: false,
            enable_logging: true,
            log_level: "info".to_string(),
            model_dir: "models/local".to_string(),
        }
    }
}
