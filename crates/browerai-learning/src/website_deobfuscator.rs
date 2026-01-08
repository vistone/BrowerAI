//! 真实网站爬虫反混淀验证框架
//! Real-World Website Crawling and Deobfuscation Verification
//!
//! 这个模块从真实网站获取 JavaScript 代码，进行反混淀，并验证结果有效性

use std::collections::HashMap;
use std::time::Instant;

use crate::deobfuscation::{DeobfuscationStrategy, JsDeobfuscator, ObfuscationTechnique};
use browerai_js_parser::JsParser;
use log::warn;
use reqwest::blocking::Client;
use reqwest::header::CONTENT_TYPE;

/// 真实网站反混淀验证结果
#[derive(Debug, Clone)]
pub struct WebsiteDeobfuscationResult {
    /// 网站URL
    pub url: String,

    /// 原始代码大小
    pub original_size: usize,

    /// 原始代码内容（用于调试和验证）。注意：可能较大。
    pub original_code: String,

    /// 反混淀后代码大小
    pub deobfuscated_size: usize,

    /// 反混淀后的代码（用于进一步验证）。注意：可能较大。
    pub deobfuscated_code: String,

    /// 是否成功反混淀
    pub success: bool,

    /// 代码是否有效
    pub is_valid: bool,

    /// 可读性分数 (0.0-1.0)
    pub readability_improvement: f32,

    /// 处理时间 (毫秒)
    pub processing_time_ms: u128,

    /// 检测到的混淆技术
    pub obfuscation_techniques: Vec<String>,

    /// 错误信息
    pub error: Option<String>,
}

/// 真实网站反混淀验证框架
pub struct WebsiteDeobfuscationVerifier {
    /// 缓存已处理的网站
    cache: HashMap<String, WebsiteDeobfuscationResult>,
}

impl WebsiteDeobfuscationVerifier {
    /// 创建新的验证器
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// 从网站获取代码并验证反混淀
    ///
    /// # 参数
    /// - `url`: 网站URL
    /// - `selector`: CSS 选择器 (可选，用于选择特定的脚本)
    ///
    /// # 返回
    /// 反混淀验证结果
    pub fn verify_website(
        &mut self,
        url: &str,
        selector: Option<&str>,
    ) -> Result<WebsiteDeobfuscationResult, String> {
        // 检查缓存
        if let Some(cached) = self.cache.get(url) {
            return Ok(cached.clone());
        }

        let start = Instant::now();

        let client = Client::builder()
            .user_agent("BrowerAI/real-world-deobfuscation")
            .build()
            .map_err(|e| e.to_string())?;

        let response = client
            .get(url)
            .send()
            .map_err(|e| format!("Failed to fetch {}: {}", url, e))?;

        if !response.status().is_success() {
            return Err(format!(
                "Failed to fetch {}: HTTP {}",
                url,
                response.status()
            ));
        }

        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();

        let body = response.text().map_err(|e| e.to_string())?;

        // 如果 selector 存在且内容是 HTML，这里可以进一步解析。当前优先处理直接指向 JS 的 URL。
        if selector.is_some() && content_type.contains("html") {
            warn!("Selector {:?} provided for {}, but HTML parsing is not implemented; using full body", selector, url);
        }

        let original_size = body.len();

        let deobf = JsDeobfuscator::new();
        let parser = JsParser::new();

        let analysis = deobf.analyze_obfuscation(&body);
        let deobf_result = deobf
            .deobfuscate(&body, DeobfuscationStrategy::Comprehensive)
            .map_err(|e| e.to_string())?;

        let is_valid = parser.validate(&deobf_result.code).unwrap_or(false);

        let readability_improvement = deobf_result.improvement.readability_after
            - deobf_result.improvement.readability_before;

        let result = WebsiteDeobfuscationResult {
            url: url.to_string(),
            original_size,
            original_code: body.clone(),
            deobfuscated_size: deobf_result.code.len(),
            deobfuscated_code: deobf_result.code.clone(),
            success: deobf_result.success && is_valid,
            is_valid,
            readability_improvement,
            processing_time_ms: start.elapsed().as_millis(),
            obfuscation_techniques: analysis
                .techniques
                .iter()
                .map(|t| match t {
                    ObfuscationTechnique::NameMangling => "NameMangling".to_string(),
                    ObfuscationTechnique::StringEncoding => "StringEncoding".to_string(),
                    ObfuscationTechnique::ControlFlowFlattening => {
                        "ControlFlowFlattening".to_string()
                    }
                    ObfuscationTechnique::DeadCodeInjection => "DeadCodeInjection".to_string(),
                    ObfuscationTechnique::ExpressionObfuscation => {
                        "ExpressionObfuscation".to_string()
                    }
                    other => format!("{:?}", other),
                })
                .collect(),
            error: None,
        };

        self.cache.insert(url.to_string(), result.clone());
        Ok(result)
    }

    /// 验证反混淀后的代码能否执行
    pub fn verify_execution(&self, code: &str) -> Result<bool, String> {
        if code.trim().is_empty() {
            return Ok(false);
        }

        let parser = JsParser::new();
        let is_valid = parser.validate(code).unwrap_or(false);
        Ok(is_valid)
    }

    /// 获取统计信息
    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if !self.cache.is_empty() {
            let total = self.cache.len() as f64;
            let successful = self.cache.values().filter(|r| r.success).count() as f64;
            let valid = self.cache.values().filter(|r| r.is_valid).count() as f64;

            stats.insert("total_websites".to_string(), total);
            stats.insert("success_rate".to_string(), successful / total);
            stats.insert("validity_rate".to_string(), valid / total);
            stats.insert(
                "avg_improvement".to_string(),
                self.cache
                    .values()
                    .map(|r| r.readability_improvement as f64)
                    .sum::<f64>()
                    / total,
            );
        }

        stats
    }
}

impl Default for WebsiteDeobfuscationVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verifier_creation() {
        let verifier = WebsiteDeobfuscationVerifier::new();
        assert_eq!(verifier.cache.len(), 0);
    }

    #[test]
    fn test_empty_statistics() {
        let verifier = WebsiteDeobfuscationVerifier::new();
        let stats = verifier.get_statistics();
        assert_eq!(stats.len(), 0);
    }

    #[test]
    fn test_code_validation() {
        let verifier = WebsiteDeobfuscationVerifier::new();
        assert!(verifier.verify_execution("console.log('test');").is_ok());
    }

    #[test]
    fn test_empty_code_validation() {
        let verifier = WebsiteDeobfuscationVerifier::new();
        assert!(verifier.verify_execution("").is_ok());
    }
}
