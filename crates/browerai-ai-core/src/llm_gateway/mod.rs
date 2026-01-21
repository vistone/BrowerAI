/// Global LLM Gateway - Unified Interface for Global LLM APIs
///
/// This module provides a unified interface to access various global LLM services:
/// - OpenAI (GPT-4, GPT-3.5 Turbo)
/// - Anthropic (Claude 3 Opus, Sonnet, Haiku)
/// - Google (Gemini Pro, Flash)
/// - Azure OpenAI
/// - OpenAI-compatible APIs (TogetherAI, etc.)
///
/// # Usage
///
/// ```rust,no_run
/// use browerai_ai_core::llm_gateway::{LlmGateway, LlmProvider, ChatMessage};
///
/// let mut gateway = LlmGateway::new();
///
/// // Configure with API key
/// gateway.configure_provider(LlmProvider::OpenAI, "sk-...").unwrap();
///
/// // Use the gateway
/// // gateway.chat()
/// //     .user("What is Rust programming language?")
/// //     .model("gpt-4o")
/// //     .send();
/// ```
mod config;
mod providers;
mod types;

pub use config::{LlmConfig, LlmGatewayConfig, ProviderConfig};
pub use providers::{
    AnthropicProvider, AzureOpenaiProvider, DeepseekProvider, GithubProvider, GoogleProvider,
    KimiProvider, MinimaxProvider, OpenaiProvider,
};
pub use types::{
    ChatMessage, ChatRequest, ContentPart, LlmProvider, LlmResponse, MessageRole,
    ProviderCapability, StreamingChunk, UsageInfo,
};

use anyhow::{Context, Result};
use reqwest::{Client, ClientBuilder};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Unified LLM Gateway
#[derive(Debug)]
pub struct LlmGateway {
    client: Client,
    configs: RwLock<HashMap<LlmProvider, ProviderConfig>>,
    default_provider: LlmProvider,
}

impl Clone for LlmGateway {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            configs: RwLock::new(HashMap::new()), // Can't clone configs, start fresh
            default_provider: self.default_provider,
        }
    }
}

impl Default for LlmGateway {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmGateway {
    /// Create a new LLM Gateway instance
    pub fn new() -> Self {
        let client = ClientBuilder::new()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            configs: RwLock::new(HashMap::new()),
            default_provider: LlmProvider::OpenAI,
        }
    }

    /// Configure a provider with API key
    pub fn configure_provider(
        &mut self,
        provider: LlmProvider,
        api_key: impl Into<String>,
    ) -> Result<()> {
        let config = ProviderConfig {
            api_key: api_key.into(),
            base_url: None,
            model: None,
            temperature: None,
            max_tokens: None,
        };
        self.configs.write().unwrap().insert(provider, config);
        Ok(())
    }

    /// Configure a provider with full options
    pub fn configure_provider_full(
        &mut self,
        provider: LlmProvider,
        config: ProviderConfig,
    ) -> Result<()> {
        self.configs.write().unwrap().insert(provider, config);
        Ok(())
    }

    /// Configure provider from environment variable
    pub fn configure_provider_from_env(
        &mut self,
        provider: LlmProvider,
        env_var: &str,
    ) -> Result<()> {
        let api_key = std::env::var(env_var)
            .with_context(|| format!("Environment variable {} not set", env_var))?;
        self.configure_provider(provider, api_key)
    }

    /// Set the default provider
    pub fn set_default_provider(&mut self, provider: LlmProvider) {
        self.default_provider = provider;
    }

    /// Check if a provider is configured
    pub fn is_provider_configured(&self, provider: LlmProvider) -> bool {
        self.configs.read().unwrap().contains_key(&provider)
    }

    /// Get a chat request builder
    pub fn chat(&self) -> ChatBuilder<'_> {
        ChatBuilder::new(self)
    }

    /// Get the HTTP client
    pub fn client(&self) -> &Client {
        &self.client
    }
}

/// Chat completion builder
#[derive(Debug, Clone)]
pub struct ChatBuilder<'a> {
    gateway: &'a LlmGateway,
    provider: Option<LlmProvider>,
    messages: Vec<ChatMessage>,
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stream: bool,
}

impl<'a> ChatBuilder<'a> {
    fn new(gateway: &'a LlmGateway) -> Self {
        Self {
            gateway,
            provider: None,
            messages: Vec::new(),
            model: None,
            temperature: None,
            max_tokens: None,
            stream: false,
        }
    }

    /// Set the provider (defaults to gateway's default)
    pub fn with_provider(mut self, provider: LlmProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Add a user message
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages
            .push(ChatMessage::new(MessageRole::User, content.into()));
        self
    }

    /// Add a system message
    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages
            .push(ChatMessage::new(MessageRole::System, content.into()));
        self
    }

    /// Add an assistant message
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages
            .push(ChatMessage::new(MessageRole::Assistant, content.into()));
        self
    }

    /// Add multiple messages
    pub fn messages(mut self, messages: impl IntoIterator<Item = ChatMessage>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Set the model
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set temperature (0.0 - 2.0)
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature.clamp(0.0, 2.0));
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Enable streaming response
    pub fn stream(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Send the chat request
    pub async fn send(self) -> Result<LlmResponse> {
        let provider = self.provider.unwrap_or(self.gateway.default_provider);
        let config = self
            .gateway
            .configs
            .read()
            .unwrap()
            .get(&provider)
            .with_context(|| format!("Provider {} not configured", provider))?
            .clone();

        let model = config.model.clone();
        let request = ChatRequest {
            model: self
                .model
                .or(model)
                .unwrap_or_else(|| provider.default_model()),
            messages: self.messages,
            temperature: self.temperature.or(config.temperature).unwrap_or(0.7),
            max_tokens: self
                .max_tokens
                .or(config.max_tokens)
                .unwrap_or(DEFAULT_MAX_TOKENS),
            stream: self.stream,
        };

        match provider {
            LlmProvider::OpenAI => {
                let provider_impl = OpenaiProvider::new(&self.gateway.client, &config);
                provider_impl.chat(request).await
            }
            LlmProvider::Anthropic => {
                let provider_impl = AnthropicProvider::new(&self.gateway.client, &config);
                provider_impl.chat(request).await
            }
            LlmProvider::Google => {
                let provider_impl = GoogleProvider::new(&self.gateway.client, &config);
                provider_impl.chat(request).await
            }
            LlmProvider::AzureOpenAI => {
                let provider_impl = AzureOpenaiProvider::new(&self.gateway.client, &config);
                provider_impl.chat(request).await
            }
            LlmProvider::DeepSeek => {
                let provider_impl = DeepseekProvider::new(&self.gateway.client, &config);
                provider_impl.chat(request).await
            }
            LlmProvider::Kimi => {
                let provider_impl = KimiProvider::new(&self.gateway.client, &config);
                provider_impl.chat(request).await
            }
            LlmProvider::Minimax => {
                let provider_impl = MinimaxProvider::new(&self.gateway.client, &config);
                provider_impl.chat(request).await
            }
            LlmProvider::GitHub => {
                let provider_impl = GithubProvider::new(&self.gateway.client, &config);
                provider_impl.chat(request).await
            }
            LlmProvider::Custom => {
                anyhow::bail!("Custom provider requires manual implementation")
            }
        }
    }
}
