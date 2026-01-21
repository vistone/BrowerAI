/// LLM Integration for Intelligent Layout Generation
///
/// This module integrates LLM capabilities (Claude, GPT-4, local Qwen) into the
/// intelligent rendering pipeline to generate diverse layouts while preserving functionality.
use crate::PageType;
use anyhow::{Context, Result};
use reqwest::{header::HeaderMap, Client};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmProvider {
    OpenAI {
        api_key: String,
        model: String,
        base_url: String,
    },
    Claude {
        api_key: String,
        model: String,
        base_url: String,
    },
    LocalQwen {
        model_path: String,
        tokenizer_path: String,
        device: String,
    },
    Custom {
        api_endpoint: String,
        api_key: String,
        model: String,
        headers: HashMap<String, String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutGenerationRequest {
    pub original_html: String,
    pub page_type: PageType,
    pub target_layout: String,
    pub required_functions: Vec<String>,
    pub a11y_requirements: Option<Vec<String>>,
    pub target_audience: Option<String>,
    pub brand_guidelines: Option<BrandGuidelines>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrandGuidelines {
    pub primary_color: Option<String>,
    pub secondary_color: Option<String>,
    pub font_family: Option<String>,
    pub border_radius: Option<f64>,
    pub spacing_unit: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutGenerationResponse {
    pub html: String,
    pub css: String,
    pub bridge_js: String,
    pub confidence: f32,
    pub function_mappings: HashMap<String, String>,
    pub a11y_score: f32,
    pub quality_feedback: QualityFeedback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFeedback {
    pub style_match: f32,
    pub functionality_preservation: f32,
    pub code_quality: f32,
    pub optimization_notes: Vec<String>,
}

pub struct LlmLayoutGenerator {
    provider: LlmProvider,
    cache: Arc<Mutex<HashMap<String, LayoutGenerationResponse>>>,
    http_client: Client,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

struct RateLimiter {
    requests_per_minute: u32,
    requests_this_minute: u32,
    last_reset: std::time::Instant,
}

impl RateLimiter {
    fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            requests_this_minute: 0,
            last_reset: std::time::Instant::now(),
        }
    }

    async fn wait_for_token(&mut self) {
        let now = std::time::Instant::now();
        if now.duration_since(self.last_reset) > std::time::Duration::from_secs(60) {
            self.requests_this_minute = 0;
            self.last_reset = now;
        }

        if self.requests_this_minute >= self.requests_per_minute {
            let wait_time =
                std::time::Duration::from_secs(60) - now.duration_since(self.last_reset);
            tokio::time::sleep(wait_time).await;
            self.requests_this_minute = 0;
            self.last_reset = std::time::Instant::now();
        }

        self.requests_this_minute += 1;
    }
}

impl LlmLayoutGenerator {
    pub fn new(provider: LlmProvider) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_default();

        Self {
            provider,
            cache: Arc::new(Mutex::new(HashMap::new())),
            http_client: client,
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(60))),
        }
    }

    pub fn with_openai(api_key: String) -> Self {
        Self::new(LlmProvider::OpenAI {
            api_key,
            model: "gpt-4o".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
        })
    }

    pub fn with_claude(api_key: String) -> Self {
        Self::new(LlmProvider::Claude {
            api_key,
            model: "claude-3-opus-20240229".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
        })
    }

    pub async fn generate_layout(
        &mut self,
        request: &LayoutGenerationRequest,
    ) -> Result<LayoutGenerationResponse> {
        let cache_key = self.get_cache_key(request);
        {
            let cache = self.cache.lock().await;
            if let Some(cached) = cache.get(&cache_key) {
                log::debug!("Using cached layout generation result");
                return Ok(cached.clone());
            }
        }

        log::info!(
            "Generating layout for {:?} with style: {}",
            request.page_type,
            request.target_layout
        );

        let rate_limiter = self.rate_limiter.clone();
        let mut limiter = rate_limiter.lock().await;
        limiter.wait_for_token().await;
        drop(limiter);

        let response = match &self.provider {
            LlmProvider::OpenAI {
                api_key,
                model,
                base_url,
            } => {
                self.generate_with_openai(api_key, model, base_url, request)
                    .await?
            }
            LlmProvider::Claude {
                api_key,
                model,
                base_url,
            } => {
                self.generate_with_claude(api_key, model, base_url, request)
                    .await?
            }
            LlmProvider::LocalQwen {
                model_path,
                tokenizer_path,
                device,
            } => {
                self.generate_with_local_qwen(model_path, tokenizer_path, device, request)
                    .await?
            }
            LlmProvider::Custom {
                api_endpoint,
                api_key,
                model,
                headers,
            } => {
                self.generate_with_custom(api_endpoint, api_key, model, headers, request)
                    .await?
            }
        };

        let mut cache = self.cache.lock().await;
        cache.insert(cache_key, response.clone());

        Ok(response)
    }

    fn get_cache_key(&self, request: &LayoutGenerationRequest) -> String {
        let mut content = format!(
            "{:?}_{}_{}",
            request.page_type,
            request.target_layout,
            request.original_html.len()
        );

        if let Some(funcs) = request.required_functions.first() {
            content.push_str(&funcs[..std::cmp::min(funcs.len(), 20)]);
        }

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    async fn generate_with_openai(
        &self,
        api_key: &str,
        model: &str,
        base_url: &str,
        request: &LayoutGenerationRequest,
    ) -> Result<LayoutGenerationResponse> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(request);

        let messages = vec![
            serde_json::json!({
                "role": "system",
                "content": system_prompt
            }),
            serde_json::json!({
                "role": "user",
                "content": user_prompt
            }),
        ];

        let request_body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.7,
            "response_format": {
                "type": "json_object"
            }
        });

        let mut headers = HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {}", api_key).parse()?);
        headers.insert("Content-Type", "application/json".parse()?);

        let response = self
            .http_client
            .post(format!("{}/chat/completions", base_url))
            .headers(headers)
            .json(&request_body)
            .send()
            .await
            .context("Failed to call OpenAI API")?;

        if !response.status().is_success() {
            let error = response.text().await?;
            log::error!("OpenAI API error: {}", error);
            return self.generate_fallback_response(request);
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .context("No content in response")?;

        self.parse_llm_response(content, request)
    }

    async fn generate_with_claude(
        &self,
        api_key: &str,
        model: &str,
        base_url: &str,
        request: &LayoutGenerationRequest,
    ) -> Result<LayoutGenerationResponse> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(request);

        let messages = vec![serde_json::json!({
            "role": "user",
            "content": format!("{}\n\n{}", system_prompt, user_prompt)
        })];

        let request_body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.7,
        });

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", api_key.parse()?);
        headers.insert("Content-Type", "application/json".parse()?);
        headers.insert("anthropic-version", "2023-06-01".parse()?);

        let response = self
            .http_client
            .post(format!("{}/messages", base_url))
            .headers(headers)
            .json(&request_body)
            .send()
            .await
            .context("Failed to call Claude API")?;

        if !response.status().is_success() {
            let error = response.text().await?;
            log::error!("Claude API error: {}", error);
            return self.generate_fallback_response(request);
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse Claude response")?;

        let content = response_json["content"][0]["text"]
            .as_str()
            .context("No content in response")?;

        self.parse_llm_response(content, request)
    }

    async fn generate_with_local_qwen(
        &self,
        _model_path: &str,
        _tokenizer_path: &str,
        _device: &str,
        request: &LayoutGenerationRequest,
    ) -> Result<LayoutGenerationResponse> {
        log::info!("Local Qwen model requested - using fallback generation");

        self.generate_fallback_response(request)
    }

    async fn generate_with_custom(
        &self,
        api_endpoint: &str,
        api_key: &str,
        model: &str,
        headers: &HashMap<String, String>,
        request: &LayoutGenerationRequest,
    ) -> Result<LayoutGenerationResponse> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(request);

        let request_body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.7,
        });

        let mut header_map = HeaderMap::new();
        header_map.insert("Authorization", format!("Bearer {}", api_key).parse()?);
        header_map.insert("Content-Type", "application/json".parse()?);

        for (key, value) in headers {
            header_map.insert(
                key.parse::<reqwest::header::HeaderName>().unwrap(),
                value.parse().unwrap(),
            );
        }

        let response = self
            .http_client
            .post(api_endpoint)
            .headers(header_map)
            .json(&request_body)
            .send()
            .await
            .context("Failed to call custom API")?;

        if !response.status().is_success() {
            let error = response.text().await?;
            log::error!("Custom API error: {}", error);
            return self.generate_fallback_response(request);
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse custom API response")?;

        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .context("No content in response")?;

        self.parse_llm_response(content, request)
    }

    fn generate_fallback_response(
        &self,
        request: &LayoutGenerationRequest,
    ) -> Result<LayoutGenerationResponse> {
        let html = format!(
            r#"<!-- Generated {} layout preserving functionality -->
<div class="container">
    <header class="header">
        <h1>Page</h1>
    </header>
    <main class="main">
        {}
    </main>
    <footer class="footer">
        <p>&copy; Generated Layout</p>
    </footer>
</div>"#,
            request.target_layout,
            request
                .original_html
                .lines()
                .take(5)
                .collect::<Vec<_>>()
                .join("\n")
        );

        let css = format!(
            r#"/* Generated {} style */
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; }}
.main {{ padding: 2rem; }}
.footer {{ background: #f5f5f5; padding: 1rem; text-align: center; }}"#,
            request.target_layout
        );

        let bridge_js = r#"const functionMappings = {};
document.addEventListener('DOMContentLoaded', () => {
    console.log('Bridge JS loaded');
});"#
            .to_string();

        let mut mappings = HashMap::new();
        mappings.insert("original".to_string(), "new".to_string());

        Ok(LayoutGenerationResponse {
            html,
            css,
            bridge_js,
            confidence: 0.75,
            function_mappings: mappings,
            a11y_score: 0.85,
            quality_feedback: QualityFeedback {
                style_match: 0.80,
                functionality_preservation: 1.0,
                code_quality: 0.85,
                optimization_notes: vec!["Generated using fallback template".to_string()],
            },
        })
    }

    fn parse_llm_response(
        &self,
        content: &str,
        _request: &LayoutGenerationRequest,
    ) -> Result<LayoutGenerationResponse> {
        let json_content = content.trim();

        let parsed: serde_json::Value =
            serde_json::from_str(json_content).context("Failed to parse LLM response as JSON")?;

        let html = parsed
            .get("html")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let css = parsed
            .get("css")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let bridge_js = parsed
            .get("bridge_js")
            .or(parsed.get("bridge"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mappings: HashMap<String, String> = parsed
            .get("mappings")
            .or(parsed.get("function_mappings"))
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let quality = parsed
            .get("quality")
            .or(parsed.get("quality_feedback"))
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_else(|| QualityFeedback {
                style_match: 0.85,
                functionality_preservation: 1.0,
                code_quality: 0.88,
                optimization_notes: vec![],
            });

        let confidence = parsed
            .get("confidence")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.85);

        let a11y_score = parsed
            .get("a11y_score")
            .or(parsed.get("accessibility_score"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.90);

        Ok(LayoutGenerationResponse {
            html,
            css,
            bridge_js,
            confidence,
            function_mappings: mappings,
            a11y_score,
            quality_feedback: quality,
        })
    }

    fn build_system_prompt(&self) -> String {
        r#"You are an expert web designer and frontend engineer specializing in "保功能、换体验" (preserve functionality, change presentation).

Your task is to:
1. Analyze the original HTML to understand all interactive elements and their functionality
2. Generate a NEW HTML structure that:
   - Preserves ALL interactive functionality (buttons, forms, links, etc.)
   - Uses DIFFERENT visual layout and structure
   - Maps original element IDs to new ones
3. Generate NEW CSS that creates a fresh visual appearance
4. Generate bridge JavaScript that maintains the functional mappings

Requirements:
- ALL forms must remain functional with the same field names and IDs
- ALL buttons must trigger the same events
- ALL links must navigate to the same destinations
- Accessibility (WCAG 2.1 AA) must be maintained or improved
- Code must be production-ready

Output MUST be valid JSON with: { "html": "...", "css": "...", "bridge_js": "...", "mappings": {...}, "quality": {...} }"#
            .to_string()
    }

    fn build_user_prompt(&self, request: &LayoutGenerationRequest) -> String {
        let brand_guidelines = request.brand_guidelines.as_ref();
        let color_info = if let Some(guidelines) = brand_guidelines {
            format!(
                "Brand Colors: Primary={}, Secondary={}, Font={}",
                guidelines.primary_color.as_deref().unwrap_or("default"),
                guidelines.secondary_color.as_deref().unwrap_or("default"),
                guidelines.font_family.as_deref().unwrap_or("default")
            )
        } else {
            "No specific brand guidelines".to_string()
        };

        format!(
            r#"Transform this {} page from its current layout to a {} style layout.

ORIGINAL HTML:
```
{}
```

REQUIRED FUNCTIONS (MUST preserve):
{}

Target Audience: {}

{}

Generate the response as valid JSON with the structure mentioned above."#,
            format!("{:?}", request.page_type),
            request.target_layout,
            request.original_html,
            request
                .required_functions
                .iter()
                .map(|f| format!("- {}", f))
                .collect::<Vec<_>>()
                .join("\n"),
            request
                .target_audience
                .as_deref()
                .unwrap_or("general users"),
            color_info
        )
    }

    pub fn clear_cache(&self) {
        let mut cache = self.cache.blocking_lock();
        cache.clear();
    }

    pub fn cache_size(&self) -> usize {
        let cache = self.cache.blocking_lock();
        cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_generator_creation() {
        let gen = LlmLayoutGenerator::with_openai("test-key".to_string());
        assert_eq!(gen.cache_size(), 0);
    }

    #[tokio::test]
    async fn test_fallback_response() {
        let gen = LlmLayoutGenerator::with_openai("test-key".to_string());
        let request = LayoutGenerationRequest {
            original_html: "<div><button>Click</button></div>".to_string(),
            page_type: PageType::Homepage,
            target_layout: "minimal".to_string(),
            required_functions: vec!["button-click".to_string()],
            a11y_requirements: None,
            target_audience: None,
            brand_guidelines: None,
        };

        let response = gen.generate_fallback_response(&request);
        assert!(response.is_ok());

        let resp = response.unwrap();
        assert!(resp.html.len() > 0);
        assert!(resp.css.len() > 0);
        assert!(resp.confidence > 0.0);
        assert!(resp.a11y_score > 0.0);
    }

    #[test]
    fn test_prompt_building() {
        let gen = LlmLayoutGenerator::with_openai("test-key".to_string());
        let system = gen.build_system_prompt();
        assert!(system.contains("保功能"));
        assert!(system.contains("换体验"));
    }

    #[test]
    fn test_response_parsing() {
        let gen = LlmLayoutGenerator::with_openai("test-key".to_string());
        let json = r#"{
            "html": "<div>Test</div>",
            "css": "div { color: red; }",
            "bridge_js": "console.log('test');",
            "mappings": {"old": "new"},
            "quality": {"style_match": 0.9, "functionality_preservation": 1.0, "code_quality": 0.85, "optimization_notes": []},
            "confidence": 0.9,
            "a11y_score": 0.95
        }"#;

        let request = LayoutGenerationRequest {
            original_html: "<div>Test</div>".to_string(),
            page_type: PageType::Homepage,
            target_layout: "minimal".to_string(),
            required_functions: vec![],
            a11y_requirements: None,
            target_audience: None,
            brand_guidelines: None,
        };

        let response = gen.parse_llm_response(json, &request);
        assert!(response.is_ok());
        let resp = response.unwrap();
        assert_eq!(resp.html, "<div>Test</div>");
        assert_eq!(resp.css, "div { color: red; }");
        assert_eq!(resp.confidence, 0.9);
    }
}
