use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LlmProvider {
    #[serde(rename = "openai")]
    OpenAI,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "google")]
    Google,
    #[serde(rename = "azure")]
    AzureOpenAI,
    #[serde(rename = "deepseek")]
    DeepSeek,
    #[serde(rename = "kimi")]
    Kimi,
    #[serde(rename = "minimax")]
    Minimax,
    #[serde(rename = "github")]
    GitHub,
    #[serde(rename = "custom")]
    Custom,
}

impl fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmProvider::OpenAI => write!(f, "OpenAI"),
            LlmProvider::Anthropic => write!(f, "Anthropic"),
            LlmProvider::Google => write!(f, "Google"),
            LlmProvider::AzureOpenAI => write!(f, "Azure OpenAI"),
            LlmProvider::DeepSeek => write!(f, "DeepSeek"),
            LlmProvider::Kimi => write!(f, "Kimi (Moonshot AI)"),
            LlmProvider::Minimax => write!(f, "Minimax"),
            LlmProvider::GitHub => write!(f, "GitHub Models"),
            LlmProvider::Custom => write!(f, "Custom"),
        }
    }
}

impl LlmProvider {
    /// Get default model for this provider
    pub fn default_model(&self) -> String {
        match self {
            LlmProvider::OpenAI => "gpt-4o".to_string(),
            LlmProvider::Anthropic => "claude-sonnet-4-20250514".to_string(),
            LlmProvider::Google => "gemini-1.5-pro".to_string(),
            LlmProvider::AzureOpenAI => "gpt-4o".to_string(),
            LlmProvider::DeepSeek => "deepseek-chat".to_string(),
            LlmProvider::Kimi => "moonshot-v1-8k".to_string(),
            LlmProvider::Minimax => "abab6.5s-chat".to_string(),
            LlmProvider::GitHub => "openai/gpt-4o-mini".to_string(),
            LlmProvider::Custom => "unknown".to_string(),
        }
    }

    /// Get endpoint URL for this provider
    pub fn endpoint(&self) -> &'static str {
        match self {
            LlmProvider::OpenAI => "https://api.openai.com/v1",
            LlmProvider::Anthropic => "https://api.anthropic.com/v1",
            LlmProvider::Google => "https://generativelanguage.googleapis.com/v1beta",
            LlmProvider::AzureOpenAI => {
                "https://{resource}.openai.azure.com/openai/deployments/{deployment}"
            }
            LlmProvider::DeepSeek => "https://api.deepseek.com/chat/completions",
            LlmProvider::Kimi => "https://api.moonshot.cn/v1",
            LlmProvider::Minimax => "https://api.minimax.chat/v1/text/chatcompletion_v2",
            LlmProvider::GitHub => "https://models.inference.ai.azure.com",
            LlmProvider::Custom => "http://localhost:8000/v1",
        }
    }
}

/// Message role
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "function")]
    Function,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }
}

/// Chat request
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub stream: bool,
}

/// Chat response
#[derive(Debug, Clone, Deserialize)]
pub struct LlmResponse {
    pub content: String,
    pub provider: LlmProvider,
    pub model: String,
    pub usage: Option<UsageInfo>,
    pub raw_response: Option<serde_json::Value>,
}

/// Usage information from API response
#[derive(Debug, Clone, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Provider capabilities
#[derive(Debug, Clone)]
pub struct ProviderCapability {
    pub supports_streaming: bool,
    pub supports_functions: bool,
    pub max_context_tokens: u32,
    pub supports_vision: bool,
}

impl Default for ProviderCapability {
    fn default() -> Self {
        Self {
            supports_streaming: true,
            supports_functions: false,
            max_context_tokens: 128_000,
            supports_vision: false,
        }
    }
}

/// Content part for multimodal messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrl>,
}

/// Image URL for vision models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

/// Streaming chunk
#[derive(Debug)]
pub struct StreamingChunk {
    pub content: String,
    pub delta: String,
    pub done: bool,
}
