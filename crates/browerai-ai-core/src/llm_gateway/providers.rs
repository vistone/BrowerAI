use super::types::{ChatMessage, ChatRequest, LlmProvider, LlmResponse, MessageRole};
use super::ProviderConfig;
use crate::llm_gateway::types::UsageInfo;
use anyhow::{bail, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;

// ============================================================================
// OpenAI Provider
// ============================================================================

pub struct OpenaiProvider<'a> {
    client: &'a Client,
    config: &'a ProviderConfig,
}

impl<'a> OpenaiProvider<'a> {
    pub fn new(client: &'a Client, config: &'a ProviderConfig) -> Self {
        Self { client, config }
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<LlmResponse> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");

        let url = format!("{}/chat/completions", base_url);

        let payload = OpenaiChatRequest {
            model: &request.model,
            messages: &request.messages,
            temperature: Some(request.temperature),
            max_tokens: Some(request.max_tokens),
            stream: Some(false),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .context("Failed to send request to OpenAI")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read response body")?;

        if !status.is_success() {
            bail!("OpenAI API error ({}): {}", status.as_u16(), body);
        }

        let chat_response: OpenaiChatResponse =
            serde_json::from_str(&body).context("Failed to parse OpenAI response")?;

        let content = chat_response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("");

        let usage = chat_response.usage.as_ref().map(|u| super::UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content: content.to_string(),
            provider: LlmProvider::OpenAI,
            model: request.model,
            usage,
            raw_response: serde_json::to_value(chat_response).ok(),
        })
    }
}

#[derive(Debug, Serialize)]
struct OpenaiChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenaiChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    system_fingerprint: Option<String>,
    choices: Vec<OpenaiChoice>,
    usage: Option<OpenaiUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenaiChoice {
    index: u32,
    message: OpenaiMessage,
    finish_reason: Option<String>,
    logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenaiMessage {
    role: String,
    content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenaiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// ============================================================================
// Anthropic Provider
// ============================================================================

pub struct AnthropicProvider<'a> {
    client: &'a Client,
    config: &'a ProviderConfig,
}

impl<'a> AnthropicProvider<'a> {
    pub fn new(client: &'a Client, config: &'a ProviderConfig) -> Self {
        Self { client, config }
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<LlmResponse> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com");

        let url = format!("{}/messages", base_url);

        // Convert messages to Anthropic format
        let anthropic_messages: Vec<AnthropicMessage> = request
            .messages
            .iter()
            .map(|m| AnthropicMessage {
                role: match m.role {
                    super::MessageRole::User => "user".to_string(),
                    super::MessageRole::Assistant => "assistant".to_string(),
                    super::MessageRole::System => "user".to_string(), // Claude uses system in system prompt
                    super::MessageRole::Function => "assistant".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let system_prompt = request
            .messages
            .iter()
            .find(|m| m.role == super::MessageRole::System)
            .map(|m| m.content.clone());

        let payload = AnthropicChatRequest {
            model: &request.model,
            messages: anthropic_messages,
            max_tokens: request.max_tokens,
            temperature: Some(request.temperature),
            stream: Some(false),
        };

        let mut request_builder = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json");

        if let Some(ref system) = system_prompt {
            request_builder = request_builder.header("anthropic-system-prompt", system);
        }

        let response = request_builder
            .json(&payload)
            .send()
            .await
            .context("Failed to send request to Anthropic")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read response body")?;

        if !status.is_success() {
            bail!("Anthropic API error ({}): {}", status.as_u16(), body);
        }

        let chat_response: AnthropicChatResponse =
            serde_json::from_str(&body).context("Failed to parse Anthropic response")?;

        let content = chat_response
            .content
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        let usage = Some(UsageInfo {
            prompt_tokens: chat_response.usage.input_tokens,
            completion_tokens: chat_response.usage.output_tokens,
            total_tokens: chat_response.usage.input_tokens + chat_response.usage.output_tokens,
        });

        Ok(LlmResponse {
            content,
            provider: LlmProvider::Anthropic,
            model: request.model,
            usage,
            raw_response: serde_json::to_value(chat_response).ok(),
        })
    }
}

#[derive(Debug, Serialize)]
struct AnthropicChatRequest<'a> {
    model: &'a str,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    temperature: Option<f32>,
    stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicChatResponse {
    id: String,
    type_: String,
    role: String,
    content: Vec<AnthropicContentBlock>,
    model: String,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicContentBlock {
    type_: String,
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ============================================================================
// Google Provider
// ============================================================================

pub struct GoogleProvider<'a> {
    client: &'a Client,
    config: &'a ProviderConfig,
}

impl<'a> GoogleProvider<'a> {
    pub fn new(client: &'a Client, config: &'a ProviderConfig) -> Self {
        Self { client, config }
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<LlmResponse> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta");

        let model = request.model.replace("gemini-", "");
        let url = format!("{}/models/{}:generateContent", base_url, model);

        // Convert messages to Google format
        let contents = request
            .messages
            .iter()
            .map(|m| GoogleContent {
                role: match m.role {
                    super::MessageRole::User => Some("user".to_string()),
                    super::MessageRole::Assistant => Some("model".to_string()),
                    _ => None,
                },
                parts: vec![GooglePart {
                    text: m.content.clone(),
                }],
            })
            .collect();

        let payload = GoogleGenerateRequest {
            contents,
            generation_config: Some(GoogleGenerationConfig {
                temperature: Some(request.temperature),
                max_output_tokens: Some(request.max_tokens),
            }),
        };

        let url_with_key = format!("{}?key={}", url, self.config.api_key);

        let response = self
            .client
            .post(&url_with_key)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .context("Failed to send request to Google")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read response body")?;

        if !status.is_success() {
            bail!("Google API error ({}): {}", status.as_u16(), body);
        }

        let chat_response: GoogleGenerateResponse =
            serde_json::from_str(&body).context("Failed to parse Google response")?;

        let content = chat_response
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .and_then(|c| c.content.parts.first())
            .and_then(|p| p.text.clone())
            .unwrap_or_default();

        Ok(LlmResponse {
            content,
            provider: LlmProvider::Google,
            model: request.model,
            usage: None,
            raw_response: serde_json::to_value(chat_response).ok(),
        })
    }
}

#[derive(Debug, Serialize)]
struct GoogleGenerateRequest {
    contents: Vec<GoogleContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GoogleGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GoogleContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GooglePart>,
}

#[derive(Debug, Serialize)]
struct GooglePart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GoogleGenerationConfig {
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleGenerateResponse {
    candidates: Option<Vec<GoogleCandidate>>,
    #[serde(default)]
    usage_metadata: GoogleUsageMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleCandidate {
    content: GoogleContentResponse,
    finish_reason: Option<String>,
    index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleContentResponse {
    parts: Vec<GooglePartResponse>,
    role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GooglePartResponse {
    text: Option<String>,
    #[serde(flatten)]
    other: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GoogleUsageMetadata {
    prompt_token_count: Option<u32>,
    candidates_token_count: Option<u32>,
    total_token_count: Option<u32>,
}

// ============================================================================
// Azure OpenAI Provider
// ============================================================================

pub struct AzureOpenaiProvider<'a> {
    client: &'a Client,
    config: &'a ProviderConfig,
}

impl<'a> AzureOpenaiProvider<'a> {
    pub fn new(client: &'a Client, config: &'a ProviderConfig) -> Self {
        Self { client, config }
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<LlmResponse> {
        // Azure requires special configuration for base URL
        let base_url = self
            .config
            .base_url
            .as_ref()
            .with_context(|| {
                "Azure OpenAI requires base_url configuration with {resource} and {deployment}"
            })?
            .clone();

        // Extract deployment name from model or config
        let deployment = request.model.clone();

        let url = base_url
            .replace("{resource}", "unknown-resource")
            .replace("{deployment}", &deployment)
            + "/chat/completions?api-version=2024-02-15-preview";

        let payload = OpenaiChatRequest {
            model: &deployment,
            messages: &request.messages,
            temperature: Some(request.temperature),
            max_tokens: Some(request.max_tokens),
            stream: Some(false),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .context("Failed to send request to Azure OpenAI")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read response body")?;

        if !status.is_success() {
            bail!("Azure OpenAI API error ({}): {}", status.as_u16(), body);
        }

        let chat_response: OpenaiChatResponse =
            serde_json::from_str(&body).context("Failed to parse Azure response")?;

        let content = chat_response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("");

        let usage = chat_response.usage.as_ref().map(|u| super::UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content: content.to_string(),
            provider: LlmProvider::AzureOpenAI,
            model: deployment,
            usage,
            raw_response: serde_json::to_value(chat_response).ok(),
        })
    }
}

// ============================================================================
// DeepSeek Provider (深度求索)
// ============================================================================

pub struct DeepseekProvider<'a> {
    client: &'a Client,
    config: &'a ProviderConfig,
}

impl<'a> DeepseekProvider<'a> {
    pub fn new(client: &'a Client, config: &'a ProviderConfig) -> Self {
        Self { client, config }
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<LlmResponse> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.deepseek.com");

        let url = format!("{}/chat/completions", base_url);

        let payload = OpenaiChatRequest {
            model: &request.model,
            messages: &request.messages,
            temperature: Some(request.temperature),
            max_tokens: Some(request.max_tokens),
            stream: Some(false),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&payload)
            .send()
            .await
            .context("Failed to send request to DeepSeek")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read response body")?;

        if !status.is_success() {
            bail!("DeepSeek API error ({}): {}", status.as_u16(), body);
        }

        let chat_response: OpenaiChatResponse =
            serde_json::from_str(&body).context("Failed to parse DeepSeek response")?;

        let content = chat_response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("");

        let usage = chat_response.usage.as_ref().map(|u| UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content: content.to_string(),
            provider: LlmProvider::DeepSeek,
            model: request.model,
            usage,
            raw_response: serde_json::to_value(chat_response).ok(),
        })
    }
}

// ============================================================================
// Kimi Provider (Moonshot AI - 月之暗面)
// ============================================================================

pub struct KimiProvider<'a> {
    client: &'a Client,
    config: &'a ProviderConfig,
}

impl<'a> KimiProvider<'a> {
    pub fn new(client: &'a Client, config: &'a ProviderConfig) -> Self {
        Self { client, config }
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<LlmResponse> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.moonshot.cn/v1");

        let url = format!("{}/chat/completions", base_url);

        let payload = OpenaiChatRequest {
            model: &request.model,
            messages: &request.messages,
            temperature: Some(request.temperature),
            max_tokens: Some(request.max_tokens),
            stream: Some(false),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&payload)
            .send()
            .await
            .context("Failed to send request to Kimi (Moonshot AI)")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read response body")?;

        if !status.is_success() {
            bail!("Kimi API error ({}): {}", status.as_u16(), body);
        }

        let chat_response: OpenaiChatResponse =
            serde_json::from_str(&body).context("Failed to parse Kimi response")?;

        let content = chat_response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("");

        let usage = chat_response.usage.as_ref().map(|u| UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content: content.to_string(),
            provider: LlmProvider::Kimi,
            model: request.model,
            usage,
            raw_response: serde_json::to_value(chat_response).ok(),
        })
    }
}

// ============================================================================
// Minimax Provider (稀宇科技)
// ============================================================================

pub struct MinimaxProvider<'a> {
    client: &'a Client,
    config: &'a ProviderConfig,
}

impl<'a> MinimaxProvider<'a> {
    pub fn new(client: &'a Client, config: &'a ProviderConfig) -> Self {
        Self { client, config }
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<LlmResponse> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.minimax.chat");

        let url = format!("{}/v1/text/chatcompletion_v2", base_url);

        // Minimax uses a slightly different request format
        let messages: Vec<MinimaxMessage> = request
            .messages
            .iter()
            .map(|m| MinimaxMessage {
                role: match m.role {
                    MessageRole::User => "USER".to_string(),
                    MessageRole::Assistant => "ASSISTANT".to_string(),
                    MessageRole::System => "SYSTEM".to_string(),
                    _ => "USER".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let payload = MinimaxChatRequest {
            model: &request.model,
            messages,
            tokens_to_generate: Some(request.max_tokens),
            temperature: Some(request.temperature),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .context("Failed to send request to Minimax")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read response body")?;

        if !status.is_success() {
            bail!("Minimax API error ({}): {}", status.as_u16(), body);
        }

        let chat_response: MinimaxChatResponse =
            serde_json::from_str(&body).context("Failed to parse Minimax response")?;

        let content = chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = Some(UsageInfo {
            prompt_tokens: chat_response.usage.prompt_tokens,
            completion_tokens: chat_response.usage.completion_tokens,
            total_tokens: (chat_response.usage.prompt_tokens
                + chat_response.usage.completion_tokens) as u32,
        });

        Ok(LlmResponse {
            content,
            provider: LlmProvider::Minimax,
            model: request.model,
            usage,
            raw_response: serde_json::to_value(chat_response).ok(),
        })
    }
}

#[derive(Debug, Serialize)]
struct MinimaxChatRequest<'a> {
    model: &'a str,
    messages: Vec<MinimaxMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_to_generate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MinimaxMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MinimaxChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<MinimaxChoice>,
    usage: MinimaxUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MinimaxChoice {
    index: u32,
    message: MinimaxMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MinimaxMessageResponse {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MinimaxUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// ============================================================================
// GitHub Models Provider (Azure AI Inference)
// ============================================================================

pub struct GithubProvider<'a> {
    client: &'a Client,
    config: &'a ProviderConfig,
}

impl<'a> GithubProvider<'a> {
    pub fn new(client: &'a Client, config: &'a ProviderConfig) -> Self {
        Self { client, config }
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<LlmResponse> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://models.inference.ai.azure.com");

        let url = format!("{}/chat/completions", base_url);

        let payload = OpenaiChatRequest {
            model: &request.model,
            messages: &request.messages,
            temperature: Some(request.temperature),
            max_tokens: Some(request.max_tokens),
            stream: Some(false),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&payload)
            .send()
            .await
            .context("Failed to send request to GitHub Models")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read response body")?;

        if !status.is_success() {
            bail!("GitHub Models API error ({}): {}", status.as_u16(), body);
        }

        let chat_response: OpenaiChatResponse =
            serde_json::from_str(&body).context("Failed to parse GitHub Models response")?;

        let content = chat_response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("");

        let usage = chat_response.usage.as_ref().map(|u| UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content: content.to_string(),
            provider: super::LlmProvider::GitHub,
            model: request.model,
            usage,
            raw_response: serde_json::to_value(chat_response).ok(),
        })
    }
}
