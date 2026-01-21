use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// LLM Gateway Configuration
///
/// This module provides configuration management for LLM providers,
/// including API keys, model settings, and provider-specific options.
use super::types::LlmProvider;

/// Alias for LlmGatewayConfig
pub type LlmConfig = LlmGatewayConfig;

/// Gateway configuration that can be saved/loaded
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmGatewayConfig {
    /// Default provider to use
    pub default_provider: Option<LlmProvider>,
    /// Provider configurations
    pub providers: HashMap<String, ProviderConfigEntry>,
    /// Timeout for requests (seconds)
    pub timeout_seconds: u64,
    /// Enable/disable specific providers
    pub enabled_providers: Vec<String>,
}

/// Provider configuration for runtime use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub api_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

impl ProviderConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            model: None,
            temperature: None,
            max_tokens: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }
}

/// Individual provider configuration entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfigEntry {
    /// API key (can be masked)
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
    /// Base URL override
    pub base_url: Option<String>,
    /// Default model
    pub model: Option<String>,
    /// Temperature setting
    pub temperature: Option<f32>,
    /// Max tokens
    pub max_tokens: Option<u32>,
    /// Whether this provider is enabled
    pub enabled: bool,
}

impl Default for ProviderConfigEntry {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: None,
            model: None,
            temperature: None,
            max_tokens: None,
            enabled: true,
        }
    }
}

/// Environment variable mappings
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvVarMappings {
    pub openai: Option<String>,
    pub anthropic: Option<String>,
    pub google: Option<String>,
    pub azure: Option<String>,
    pub github: Option<String>,
    pub custom: Option<String>,
}

impl EnvVarMappings {
    #[allow(dead_code)]
    pub fn for_provider(&self, provider: &LlmProvider) -> Option<&str> {
        match provider {
            LlmProvider::OpenAI => self.openai.as_deref(),
            LlmProvider::Anthropic => self.anthropic.as_deref(),
            LlmProvider::Google => self.google.as_deref(),
            LlmProvider::AzureOpenAI => self.azure.as_deref(),
            LlmProvider::DeepSeek => None,
            LlmProvider::Kimi => None,
            LlmProvider::Minimax => None,
            LlmProvider::GitHub => self.github.as_deref(),
            LlmProvider::Custom => self.custom.as_deref(),
        }
    }
}

/// Configuration file locations
pub const CONFIG_FILE_NAME: &str = "llm_gateway_config.toml";
#[allow(dead_code)]
pub const LEGACY_CONFIG_FILE_NAME: &str = ".llm_gateway";

impl LlmGatewayConfig {
    /// Create default configuration
    pub fn new() -> Self {
        Self {
            default_provider: Some(LlmProvider::OpenAI),
            providers: HashMap::new(),
            timeout_seconds: 120,
            enabled_providers: vec![
                "openai".to_string(),
                "anthropic".to_string(),
                "google".to_string(),
            ],
        }
    }

    /// Get the default config file path
    pub fn default_config_path() -> PathBuf {
        let mut path = dirs::config_dir().unwrap_or_else(|| PathBuf::from("~/.config"));
        path.push("browerai");
        path.push(CONFIG_FILE_NAME);
        path
    }

    /// Load configuration from file
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, ConfigError> {
        let path = path.into();
        if !path.exists() {
            return Ok(Self::new());
        }

        let content = std::fs::read_to_string(&path)
            .map_err(|e| ConfigError::Io(e.to_string(), path.clone()))?;
        toml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string(), path))
    }

    /// Save configuration to file
    pub fn save(&self, path: impl Into<PathBuf>) -> Result<(), ConfigError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ConfigError::Io(e.to_string(), parent.to_path_buf()))?;
        }

        let content = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::Serialize(e.to_string(), path.clone()))?;
        std::fs::write(&path, content).map_err(|e| ConfigError::Io(e.to_string(), path))?;
        Ok(())
    }

    /// Get provider entry
    pub fn get_provider(&self, provider: &LlmProvider) -> Option<&ProviderConfigEntry> {
        self.providers.get(&provider.to_string())
    }

    /// Set provider configuration
    pub fn set_provider(
        &mut self,
        provider: LlmProvider,
        entry: ProviderConfigEntry,
    ) -> Option<ProviderConfigEntry> {
        self.providers.insert(provider.to_string(), entry)
    }

    /// Load API key from environment variable
    pub fn load_api_key_from_env(&mut self, provider: &LlmProvider) -> Result<(), ConfigError> {
        let env_var = match provider {
            LlmProvider::OpenAI => "OPENAI_API_KEY",
            LlmProvider::Anthropic => "ANTHROPIC_API_KEY",
            LlmProvider::Google => "GOOGLE_API_KEY",
            LlmProvider::AzureOpenAI => "AZURE_OPENAI_API_KEY",
            LlmProvider::DeepSeek => "DEEPSEEK_API_KEY",
            LlmProvider::Kimi => "KIMI_API_KEY",
            LlmProvider::Minimax => "MINIMAX_API_KEY",
            LlmProvider::GitHub => "GITHUB_TOKEN",
            LlmProvider::Custom => "CUSTOM_LLM_API_KEY",
        };

        let key =
            std::env::var(env_var).map_err(|_| ConfigError::EnvVarNotFound(env_var.to_string()))?;

        let entry = self.providers.entry(provider.to_string()).or_default();
        entry.api_key = Some(key);
        Ok(())
    }

    /// Convert to ProviderConfig for gateway
    pub fn to_provider_config(&self, provider: &LlmProvider) -> Option<ProviderConfig> {
        let entry = self.providers.get(&provider.to_string())?;
        Some(ProviderConfig {
            api_key: entry.api_key.clone()?,
            base_url: entry.base_url.clone(),
            model: entry.model.clone(),
            temperature: entry.temperature,
            max_tokens: entry.max_tokens,
        })
    }

    /// Check if provider is configured (has API key)
    pub fn is_provider_configured(&self, provider: &LlmProvider) -> bool {
        self.providers
            .get(&provider.to_string())
            .map(|e| e.api_key.is_some())
            .unwrap_or(false)
    }

    /// Mask API keys for display
    pub fn mask_api_keys(&self) -> String {
        let masked: LlmGatewayConfig = LlmGatewayConfig {
            providers: self
                .providers
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        ProviderConfigEntry {
                            api_key: v.api_key.as_ref().map(|_| "********".to_string()),
                            ..v.clone()
                        },
                    )
                })
                .collect(),
            ..self.clone()
        };
        toml::to_string_pretty(&masked).unwrap_or_default()
    }
}

/// Configuration errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(String, PathBuf),

    #[error("Parse error: {0}")]
    Parse(String, PathBuf),

    #[error("Serialize error: {0}")]
    Serialize(String, PathBuf),

    #[error("Environment variable not found: {0}")]
    EnvVarNotFound(String),

    #[error("Provider not found: {0}")]
    ProviderNotFound(String),
}

/// API Key manager with secure storage
#[allow(dead_code)]
pub struct ApiKeyManager {
    /// Encrypted API keys
    keys: HashMap<String, String>,
    /// Encryption key (should be stored securely)
    encryption_key: Option<[u8; 32]>,
}

impl ApiKeyManager {
    /// Create a new API key manager
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            encryption_key: None,
        }
    }

    /// Store an API key
    #[allow(dead_code)]
    pub fn set_key(&mut self, provider: &str, key: &str) {
        self.keys.insert(provider.to_string(), key.to_string());
    }

    /// Retrieve an API key
    #[allow(dead_code)]
    pub fn get_key(&self, provider: &str) -> Option<&str> {
        self.keys.get(provider).map(|s| s.as_str())
    }

    /// Check if a key exists
    #[allow(dead_code)]
    pub fn has_key(&self, provider: &str) -> bool {
        self.keys.contains_key(provider)
    }

    /// Remove a key
    #[allow(dead_code)]
    pub fn remove_key(&mut self, provider: &str) -> Option<String> {
        self.keys.remove(provider)
    }

    /// Clear all keys
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.keys.clear();
    }

    /// List all providers with keys
    #[allow(dead_code)]
    pub fn providers(&self) -> Vec<&str> {
        self.keys.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ApiKeyManager {
    fn default() -> Self {
        Self::new()
    }
}
