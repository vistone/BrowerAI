//! 模型 API 客户端 - 调用 Python 模型库服务
//!
//! 与 training/api_server.py 通信，使用训练好的模型生成样式

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// API 客户端配置
#[derive(Debug, Clone)]
pub struct ModelApiConfig {
    /// API 服务器地址
    pub base_url: String,
    /// 请求超时
    pub timeout_secs: u64,
    /// 特征维度
    pub feature_dim: usize,
    /// 潜在空间维度
    pub latent_dim: usize,
}

impl Default for ModelApiConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:5000".to_string(),
            timeout_secs: 30,
            feature_dim: 48,
            latent_dim: 256,
        }
    }
}

/// 模型 API 客户端
pub struct ModelApiClient {
    client: Client,
    config: ModelApiConfig,
}

/// 特征提取请求
#[derive(Debug, Serialize)]
struct FeatureExtractionRequest {
    url: String,
    html: String,
    css: String,
    scripts: String,
}

/// 代码生成请求
#[derive(Debug, Serialize)]
struct CodeGenerationRequest {
    url: String,
    session_id: String,
    features: Vec<f32>,
    website_intent: String,
    design_style: String,
    timestamp: i64,
}

/// 代码生成响应
#[derive(Debug, Deserialize)]
struct CodeGenerationResponse {
    html: String,
    css: String,
    javascript: String,
    confidence: f32,
    should_use: bool,
    training_metrics: Option<HashMap<String, serde_json::Value>>,
}

/// 样式生成结果
#[derive(Debug, Clone)]
pub struct GeneratedStyle {
    /// 生成的 CSS
    pub css: String,
    /// 生成的 HTML 结构
    pub html_structure: String,
    /// 生成的 JavaScript
    pub javascript: String,
    /// 置信度
    pub confidence: f32,
    /// 主色
    pub primary_color: String,
    /// 背景色
    pub background_color: String,
    /// 文字色
    pub text_color: String,
}

impl ModelApiClient {
    /// 创建新的 API 客户端
    pub fn new(config: ModelApiConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self { client, config })
    }

    /// 检查 API 服务器是否可用
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/v1/health", self.config.base_url);
        
        match self.client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// 从网站内容生成样式
    /// 
    /// 流程：
    /// 1. 提取 48 维特征
    /// 2. 编码到 256 维潜在空间
    /// 3. 生成代码
    pub async fn generate_style_from_content(
        &self,
        url: &str,
        html: &str,
        css: &str,
        scripts: &str,
        design_style: &str,
    ) -> Result<GeneratedStyle> {
        // 步骤 1: 提取特征
        let features = self.extract_features(url, html, css, scripts).await?;
        
        // 步骤 2 & 3: 编码并生成代码
        let generated = self.generate_code(url, &features, design_style).await?;
        
        // 从生成的 CSS 中提取颜色
        let (primary, background, text) = self.extract_colors_from_css(&generated.css);
        
        Ok(GeneratedStyle {
            css: generated.css,
            html_structure: generated.html,
            javascript: generated.javascript,
            confidence: generated.confidence,
            primary_color: primary,
            background_color: background,
            text_color: text,
        })
    }

    /// 提取特征（48维）
    async fn extract_features(
        &self,
        _url: &str,
        html: &str,
        css: &str,
        scripts: &str,
    ) -> Result<Vec<f32>> {
        // 如果 API 服务器有特征提取端点，调用它
        // 否则在本地提取（简化版）
        Ok(self.extract_features_local(html, css, scripts))
    }

    /// 本地特征提取（48维，与 Python 版本对应）
    fn extract_features_local(&self, html: &str, css: &str, scripts: &str) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.config.feature_dim);

        // [0-9] HTML 指标
        let html_len = (html.len() as f32 / 1000000.0).min(1.0);
        let tag_count = (html.matches('<').count() as f32 / 1000.0).min(1.0);
        let div_count = (html.matches("<div").count() as f32 / 500.0).min(1.0);
        let link_count = (html.matches("<a ").count() as f32 / 100.0).min(1.0);
        let form_count = (html.matches("<form").count() as f32 / 50.0).min(1.0);
        let input_count = (html.matches("<input").count() as f32 / 100.0).min(1.0);
        let button_count = (html.matches("<button").count() as f32 / 50.0).min(1.0);
        let list_count = (html.matches("<li").count() as f32 / 100.0).min(1.0);
        let img_count = (html.matches("<img").count() as f32 / 100.0).min(1.0);
        let script_count = (html.matches("<script").count() as f32 / 50.0).min(1.0);

        features.extend_from_slice(&[
            html_len, tag_count, div_count, link_count, form_count,
            input_count, button_count, list_count, img_count, script_count,
        ]);

        // [10-17] CSS 指标
        let css_len = (css.len() as f32 / 1000000.0).min(1.0);
        let rule_count = (css.matches('{').count() as f32 / 500.0).min(1.0);
        let class_count = (css.matches('.').count() as f32 / 1000.0).min(1.0);
        let id_count = (css.matches('#').count() as f32 / 100.0).min(1.0);
        let selector_count = (css.matches(',').count() as f32 / 1000.0).min(1.0);
        let media_count = (css.matches("@media").count() as f32 / 50.0).min(1.0);
        let animation_count = (css.matches("@keyframes").count() as f32 / 30.0).min(1.0);
        let import_count = (css.matches("@import").count() as f32 / 20.0).min(1.0);

        features.extend_from_slice(&[
            css_len, rule_count, class_count, id_count, selector_count,
            media_count, animation_count, import_count,
        ]);

        // [18-27] JavaScript 指标
        let js_len = (scripts.len() as f32 / 1000000.0).min(1.0);
        let func_count = ((scripts.matches("function").count() + scripts.matches("=>") .count()) as f32 / 500.0).min(1.0);
        let var_count = ((scripts.matches("var ").count() + scripts.matches("let ").count() + scripts.matches("const ").count()) as f32 / 1000.0).min(1.0);
        let if_count = (scripts.matches("if ").count() as f32 / 500.0).min(1.0);
        let loop_count = ((scripts.matches("for ").count() + scripts.matches("while ").count()) as f32 / 200.0).min(1.0);
        let try_count = (scripts.matches("try ").count() as f32 / 100.0).min(1.0);
        let class_count = (scripts.matches("class ").count() as f32 / 100.0).min(1.0);
        let async_count = (scripts.matches("async").count() as f32 / 50.0).min(1.0);
        let import_count = ((scripts.matches("import ").count() + scripts.matches("require(").count()) as f32 / 100.0).min(1.0);
        let call_count = (scripts.matches('(').count() as f32 / 5000.0).min(1.0);

        features.extend_from_slice(&[
            js_len, func_count, var_count, if_count, loop_count,
            try_count, class_count, async_count, import_count, call_count,
        ]);

        // [28-35] 页面结构指标（简化）
        let has_nav = if html.contains("<nav") { 1.0 } else { 0.0 };
        let has_header = if html.contains("<header") { 1.0 } else { 0.0 };
        let has_footer = if html.contains("<footer") { 1.0 } else { 0.0 };
        let has_aside = if html.contains("<aside") { 1.0 } else { 0.0 };
        let has_main = if html.contains("<main") { 1.0 } else { 0.0 };
        let has_section = if html.contains("<section") { 1.0 } else { 0.0 };
        let has_article = if html.contains("<article") { 1.0 } else { 0.0 };
        let depth_estimate = (html.matches('<').count() as f32 / html.matches('>').count() as f32).min(1.0);

        features.extend_from_slice(&[
            has_nav, has_header, has_footer, has_aside,
            has_main, has_section, has_article, depth_estimate,
        ]);

        // [36-42] 设计风格指标（简化）
        let color_count = css.matches('#').count() as f32 / 50.0;
        let has_gradient = if css.contains("gradient") { 1.0 } else { 0.0 };
        let has_shadow = if css.contains("shadow") { 1.0 } else { 0.0 };
        let has_animation = if css.contains("animation") || css.contains("@keyframes") { 1.0 } else { 0.0 };
        let has_flex = if css.contains("flex") { 1.0 } else { 0.0 };
        let has_grid = if css.contains("grid") { 1.0 } else { 0.0 };
        let font_count = css.matches("font-family").count() as f32 / 10.0;

        features.extend_from_slice(&[
            color_count.min(1.0), has_gradient, has_shadow, has_animation,
            has_flex, has_grid, font_count.min(1.0),
        ]);

        // [43-47] 复杂度指标（简化）
        let total_len = (html.len() + css.len() + scripts.len()) as f32;
        let html_ratio = html.len() as f32 / total_len;
        let css_ratio = css.len() as f32 / total_len;
        let js_ratio = scripts.len() as f32 / total_len;
        let complexity_score = features.iter().sum::<f32>() / features.len() as f32;

        features.extend_from_slice(&[
            html_ratio, css_ratio, js_ratio, complexity_score, 0.5, // 最后一个占位
        ]);

        // 确保正好 48 维
        features.truncate(self.config.feature_dim);
        while features.len() < self.config.feature_dim {
            features.push(0.0);
        }

        features
    }

    /// 调用 API 生成代码
    async fn generate_code(
        &self,
        url: &str,
        features: &[f32],
        design_style: &str,
    ) -> Result<CodeGenerationResponse> {
        let api_url = format!("{}/api/v1/generate", self.config.base_url);
        
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
        let request = CodeGenerationRequest {
            url: url.to_string(),
            session_id: format!("session_{}", now.as_secs()),
            features: features.to_vec(),
            website_intent: "website".to_string(),
            design_style: design_style.to_string(),
            timestamp: now.as_secs() as i64,
        };

        let response = self
            .client
            .post(&api_url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request to model API")?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("API error: {}", error_text));
        }

        let result: CodeGenerationResponse = response
            .json()
            .await
            .context("Failed to parse API response")?;

        Ok(result)
    }

    /// 从 CSS 中提取颜色
    fn extract_colors_from_css(&self, css: &str) -> (String, String, String) {
        // 简单的颜色提取逻辑
        let primary = "#3B82F6".to_string();
        let mut background = "#FFFFFF".to_string();
        let mut text = "#111827".to_string();

        // 查找 background-color
        if let Some(pos) = css.find("background:") {
            let start = pos + "background:".len();
            if let Some(end) = css[start..].find(';') {
                let color = css[start..start+end].trim();
                if color.starts_with('#') || color.starts_with("rgb") {
                    background = color.to_string();
                }
            }
        }

        // 查找 color
        if let Some(pos) = css.find("color:") {
            let start = pos + "color:".len();
            if let Some(end) = css[start..].find(';') {
                let color = css[start..start+end].trim();
                if color.starts_with('#') || color.starts_with("rgb") {
                    text = color.to_string();
                }
            }
        }

        (primary, background, text)
    }
}

/// 回退样式生成器（当 API 不可用时）
pub struct FallbackStyleGenerator;

impl FallbackStyleGenerator {
    /// 生成回退样式
    pub fn generate(variant_index: usize) -> GeneratedStyle {
        let styles = vec![
            ("#3B82F6", "#F9FAFB", "#111827"), // 现代蓝
            ("#EA580C", "#FFFBEB", "#431407"), // 暖色调
            ("#0891B2", "#ECFEFF", "#164E63"), // 冷色调
            ("#000000", "#FFFFFF", "#000000"), // 高对比
        ];

        let (primary, bg, text) = styles[variant_index % styles.len()];

        let css = format!(
            r#"/* Fallback Style - Generated when API unavailable */
body {{
  font-family: system-ui, -apple-system, sans-serif !important;
  font-size: 16px !important;
  background: {} !important;
  color: {} !important;
  line-height: 1.6 !important;
}}

h1, h2, h3, h4, h5, h6 {{
  color: {} !important;
}}

a {{
  color: {} !important;
}}
"#,
            bg, text, primary, primary
        );

        GeneratedStyle {
            css,
            html_structure: String::new(),
            javascript: String::new(),
            confidence: 0.5,
            primary_color: primary.to_string(),
            background_color: bg.to_string(),
            text_color: text.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let config = ModelApiConfig::default();
        let client = ModelApiClient::new(config).unwrap();
        
        let html = "<html><body><div>Test</div></body></html>";
        let css = "body { color: #333; }";
        let scripts = "console.log('test');";

        let features = client.extract_features_local(html, css, scripts);
        
        assert_eq!(features.len(), 48);
        assert!(features.iter().all(|&f| f >= 0.0 && f <= 1.0));
    }

    #[test]
    fn test_fallback_generator() {
        let style = FallbackStyleGenerator::generate(0);
        assert!(!style.css.is_empty());
        assert!(style.confidence > 0.0);
    }
}
