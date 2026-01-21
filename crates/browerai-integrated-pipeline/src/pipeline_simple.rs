//! 简化的集成管道 - 更好的类型兼容性

use anyhow::Result;
use browerai_intelligent_rendering::WebsiteAnalyzer;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 管道配置
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub user_id: String,
    pub output_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub enable_cache: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            user_id: "default_user".to_string(),
            output_dir: PathBuf::from("./output"),
            cache_dir: PathBuf::from("./cache"),
            enable_cache: true,
        }
    }
}

/// 管道结果（简化版）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub url: String,
    pub website_analysis: serde_json::Value,  // 使用JSON Value保持灵活性
    pub personalization_request: serde_json::Value,
    pub generated_html: String,
    pub generated_css: String,
    pub generated_javascript: String,
    pub processing_time_ms: u64,
}

/// 完整的集成管道
pub struct IntegratedPipeline {
    config: PipelineConfig,
    analyzer: WebsiteAnalyzer,
}

impl IntegratedPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            analyzer: WebsiteAnalyzer::new(),
        }
    }

    /// 执行完整的管道流程
    pub async fn execute(&self, url: &str) -> Result<PipelineResult> {
        let start_time = std::time::Instant::now();
        
        log::info!("🚀 开始处理 URL: {}", url);

        // 第1步：获取网页
        log::debug!("步骤1: 获取网页...");
        let html_content = self.fetch_website(url).await?;
        
        // 第2步：分析网站
        log::debug!("步骤2: 分析网站技术...");
        let html_analysis = self.analyzer.analyze_html(&html_content)?;
        let css_analysis = self.analyzer.analyze_css(&html_content)?;
        let js_analysis = self.analyzer.analyze_javascript(&html_content)?;
        
        // 第3步：推断网站意图
        log::debug!("步骤3: 推断网站意图...");
        let website_intent = self.analyzer.infer_purpose(&html_analysis, &css_analysis, &js_analysis)?;
        
        // 第4步：生成个性化内容（占位符）
        log::debug!("步骤4: 生成个性化布局...");
        let generated_html = format!("<!-- 个性化版本 for {} -->\n{}", self.config.user_id, html_content);
        let generated_css = "/* 个性化CSS */\nbody {{ background: #f0f0f0; }}".to_string();
        let generated_javascript = "// 个性化JavaScript\nconsole.log('Personalized');".to_string();
        
        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        
        log::info!("✅ 处理完成，耗时: {}ms", processing_time_ms);
        
        Ok(PipelineResult {
            url: url.to_string(),
            website_analysis: serde_json::json!({
                "html_structure": "analyzed",
                "css_system": "analyzed",
                "javascript_features": "analyzed",
                "intent": "inferred"
            }),
            personalization_request: serde_json::json!({
                "user_id": self.config.user_id,
                "preferences": "standard"
            }),
            generated_html,
            generated_css,
            generated_javascript,
            processing_time_ms,
        })
    }

    /// 获取网站内容
    async fn fetch_website(&self, url: &str) -> Result<String> {
        log::debug!("获取网站: {}", url);
        
        // 检查缓存
        if self.config.enable_cache {
            let cached = self.get_from_cache(url)?;
            if !cached.is_empty() {
                log::debug!("✓ 从缓存返回");
                return Ok(cached);
            }
        }
        
        // 使用 reqwest 获取
        let response = reqwest::get(url).await?;
        let content = response.text().await?;
        
        // 保存到缓存
        if self.config.enable_cache {
            self.save_to_cache(url, &content)?;
        }
        
        Ok(content)
    }

    fn get_from_cache(&self, url: &str) -> Result<String> {
        let cache_path = self.cache_path(url);
        if cache_path.exists() {
            Ok(std::fs::read_to_string(cache_path)?)
        } else {
            Ok(String::new())
        }
    }

    fn save_to_cache(&self, url: &str, content: &str) -> Result<()> {
        let cache_path = self.cache_path(url);
        std::fs::create_dir_all(cache_path.parent().unwrap())?;
        std::fs::write(cache_path, content)?;
        Ok(())
    }

    fn cache_path(&self, url: &str) -> PathBuf {
        let hash = format!("{:x}", calculate_hash(url));
        self.config.cache_dir.join(hash).with_extension("html")
    }
}

// 简单的哈希函数
fn calculate_hash(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = IntegratedPipeline::new(config);
        assert_eq!(pipeline.config.user_id, "default_user");
    }

    #[test]
    fn test_cache_path() {
        let config = PipelineConfig::default();
        let pipeline = IntegratedPipeline::new(config);
        let path = pipeline.cache_path("https://example.com");
        assert!(path.to_string_lossy().ends_with(".html"));
    }
}
