//! Rust HTTP客户端 - 调用Python框架检测API

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// 框架检测请求
#[derive(Debug, Serialize)]
pub struct DetectRequest {
    pub html: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_ml: Option<bool>,
}

/// 框架检测响应
#[derive(Debug, Deserialize)]
pub struct DetectResponse {
    pub framework: String,
    pub confidence: f64,
    pub method: String,
    pub success: bool,
}

/// 批量检测网站
#[derive(Debug, Serialize)]
pub struct WebsiteInput {
    pub url: String,
    pub html: String,
}

#[derive(Debug, Serialize)]
pub struct BatchDetectRequest {
    pub websites: Vec<WebsiteInput>,
}

/// 批量检测结果
#[derive(Debug, Deserialize)]
pub struct BatchDetectResult {
    pub url: String,
    pub framework: String,
    pub confidence: f64,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BatchDetectResponse {
    pub results: Vec<BatchDetectResult>,
    pub total: usize,
    pub success: bool,
}

/// 框架检测HTTP客户端
pub struct FrameworkDetectorClient {
    api_url: String,
    client: reqwest::blocking::Client,
}

impl FrameworkDetectorClient {
    /// 创建新的客户端
    pub fn new(api_url: impl Into<String>) -> Self {
        Self {
            api_url: api_url.into(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }

    /// 使用默认本地地址创建
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new("http://localhost:5000")
    }

    /// 健康检查
    pub fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.api_url);
        let response = self.client.get(&url).send()?;
        Ok(response.status().is_success())
    }

    /// 检测单个网站框架
    pub fn detect(&self, html: &str) -> Result<DetectResponse> {
        let url = format!("{}/api/v1/detect", self.api_url);

        let request = DetectRequest {
            html: html.to_string(),
            use_ml: None,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .context("Failed to send request")?;

        if !response.status().is_success() {
            anyhow::bail!("API returned error: {}", response.status());
        }

        let result: DetectResponse = response.json().context("Failed to parse response")?;

        Ok(result)
    }

    /// 批量检测多个网站
    pub fn batch_detect(&self, websites: Vec<(String, String)>) -> Result<BatchDetectResponse> {
        let url = format!("{}/api/v1/batch_detect", self.api_url);

        let websites_input: Vec<WebsiteInput> = websites
            .into_iter()
            .map(|(url, html)| WebsiteInput { url, html })
            .collect();

        let request = BatchDetectRequest {
            websites: websites_input,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .context("Failed to send batch request")?;

        if !response.status().is_success() {
            anyhow::bail!("API returned error: {}", response.status());
        }

        let result: BatchDetectResponse =
            response.json().context("Failed to parse batch response")?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // 需要API服务器运行
    fn test_health_check() {
        let client = FrameworkDetectorClient::default();
        let result = client.health_check();
        assert!(result.is_ok());
    }

    #[test]
    #[ignore] // 需要API服务器运行
    fn test_detect_react() {
        let client = FrameworkDetectorClient::default();

        let react_html = r#"
            <!DOCTYPE html>
            <html>
            <head><title>React App</title></head>
            <body>
                <div id="root"></div>
                <script src="/_next/static/chunks/main.js"></script>
                <script>
                    const App = () => {
                        const [count, setCount] = React.useState(0);
                        return React.createElement('div', null, count);
                    };
                </script>
            </body>
            </html>
        "#;

        let result = client.detect(react_html).unwrap();
        println!(
            "Detected: {} (confidence: {:.2}%)",
            result.framework,
            result.confidence * 100.0
        );
        assert_eq!(result.framework, "React");
        assert!(result.confidence > 0.5);
    }

    #[test]
    #[ignore] // 需要API服务器运行
    fn test_batch_detect() {
        let client = FrameworkDetectorClient::default();

        let websites = vec![
            (
                "https://example-react.com".to_string(),
                r#"<html><script src="/_next/static/main.js"></script></html>"#.to_string(),
            ),
            (
                "https://example-vue.com".to_string(),
                r#"<html><div v-if="true" v-for="item in items"></div></html>"#.to_string(),
            ),
        ];

        let result = client.batch_detect(websites).unwrap();
        assert_eq!(result.total, 2);
        assert!(result.success);
    }
}
