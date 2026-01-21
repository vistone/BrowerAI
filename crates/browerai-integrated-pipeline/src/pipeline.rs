//! 主集成管道 - 协调所有子模块的工作流程

use anyhow::Result;
use browerai_intelligent_rendering::{
    DualSandboxRenderer, PersonalizationRequest, UserPreferences, UserProfile,
    WebsiteLearningEngine, WebsiteTechAnalysis,
};
use browerai_learning::{
    BrowserTechDetector, ExternalResourceAnalyzer, ExternalResourceGraph, ResourceType,
    TechnologyDetectionResult, WasmAnalyzer, WasmModuleInfo, WebSocketAnalyzer, WebSocketInfo,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use url::Url;

/// 管道配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// 用户ID
    pub user_id: String,

    /// 用户偏好
    pub user_preferences: UserPreferences,

    /// 用户档案
    pub user_profile: UserProfile,

    /// 输出目录
    pub output_dir: PathBuf,

    /// 缓存目录
    pub cache_dir: PathBuf,

    /// 是否启用缓存
    pub enable_cache: bool,

    /// 是否启用JS分析
    pub analyze_javascript: bool,

    /// 是否启用反混淆
    pub enable_deobfuscation: bool,
}

/// 管道结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// 网址
    pub url: String,

    /// 网站分析结果
    pub website_analysis: WebsiteTechAnalysis,

    /// 个性化渲染请求
    pub personalization_request: PersonalizationRequest,

    /// 生成的HTML
    pub generated_html: String,

    /// 生成的CSS
    pub generated_css: String,

    /// 生成的JavaScript
    pub generated_javascript: String,

    /// 外部资源分析
    pub external_resources: ExternalResourceGraph,

    /// 技术栈检测结果
    pub tech_detection: TechnologyDetectionResult,

    /// WebSocket 连接信息
    pub websockets: Vec<WebSocketInfo>,

    /// WASM 模块摘要
    pub wasm_modules: Vec<WasmModuleInfo>,

    /// 处理耗时（毫秒）
    pub processing_time_ms: u128,
}

/// 完整的集成管道
pub struct IntegratedPipeline {
    config: PipelineConfig,
}

impl IntegratedPipeline {
    /// 创建新的管道
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// 执行完整的管道流程
    pub async fn execute(&self, url: &str) -> Result<PipelineResult> {
        let start_time = std::time::Instant::now();

        log::info!("🚀 开始处理 URL: {}", url);

        // 第1步：获取网页
        log::debug!("步骤1: 获取网页...");
        let client = reqwest::Client::new();
        let html_content = self.fetch_website(&client, url).await?;

        // 第2步：外部资源与技术检测
        log::debug!("步骤2: 外部资源与技术检测...");
        let external_resources =
            ExternalResourceAnalyzer::analyze_resources(&html_content, &html_content)?;
        let tech_detection =
            BrowserTechDetector::detect_technologies(&html_content, &html_content, &html_content)?;
        let websockets = WebSocketAnalyzer::default().extract_from_js(&html_content)?;

        // WASM 模块分析：解析外部资源并下载/缓存 WASM 二进制
        let wasm_analyzer = WasmAnalyzer::default();
        let wasm_urls: Vec<String> = external_resources
            .resources
            .iter()
            .filter_map(|(resource_url, dep)| {
                if dep.resource_type == ResourceType::WebAssembly {
                    self.resolve_url(url, resource_url)
                } else {
                    None
                }
            })
            .collect();

        let mut wasm_modules: Vec<WasmModuleInfo> = Vec::new();
        let mut seen_wasm = HashSet::new();
        for wasm_url in wasm_urls {
            if !seen_wasm.insert(wasm_url.clone()) {
                continue;
            }

            match self.fetch_binary(&client, &wasm_url, "wasm").await {
                Ok(bytes) => match wasm_analyzer.analyze(&bytes, &wasm_url) {
                    Ok(module) => wasm_modules.push(module),
                    Err(err) => {
                        log::warn!("WASM 分析失败: {} -> {}", wasm_url, err);
                    }
                },
                Err(err) => {
                    log::warn!("获取 WASM 失败: {} -> {}", wasm_url, err);
                }
            }
        }

        // 第3步：学习网站（使用 WebsiteLearningEngine 保持类型一致）
        log::debug!("步骤3: 学习网站技术...");
        let website_analysis =
            WebsiteLearningEngine::learn_website_with_html(url, &html_content).await?;

        // 第4步：创建个性化请求
        log::debug!("步骤4: 创建个性化请求...");
        let personalization_request = PersonalizationRequest {
            user_id: self.config.user_id.clone(),
            website_analysis: website_analysis.clone(),
            user_preferences: self.config.user_preferences.clone(),
            user_profile: self.config.user_profile.clone(),
        };

        // 第5步：生成个性化布局
        log::debug!("步骤5: 生成个性化布局...");
        let personalized =
            DualSandboxRenderer::render_personalized(personalization_request.clone()).await?;
        let generated_html = personalized.generated_html;
        let generated_css = personalized.generated_css;
        let generated_javascript = personalized.generated_javascript;

        let processing_time_ms = start_time.elapsed().as_millis();

        log::info!("✅ 处理完成，耗时: {}ms", processing_time_ms);

        Ok(PipelineResult {
            url: url.to_string(),
            website_analysis,
            personalization_request,
            generated_html,
            generated_css,
            generated_javascript,
            external_resources,
            tech_detection,
            websockets,
            wasm_modules,
            processing_time_ms,
        })
    }

    /// 获取网站内容
    async fn fetch_website(&self, client: &reqwest::Client, url: &str) -> Result<String> {
        log::debug!("获取网站: {}", url);

        // 检查缓存
        if self.config.enable_cache {
            if let Ok(cached) = self.get_from_cache(url) {
                log::debug!("从缓存读取: {}", url);
                return Ok(cached);
            }
        }

        // 获取网页
        let response = match client.get(url).send().await {
            Ok(resp) => resp,
            Err(err) => {
                if self.config.enable_cache {
                    if let Ok(cached) = self.get_from_cache(url) {
                        log::warn!("网络获取失败，回退到缓存: {} -> {}", url, err);
                        return Ok(cached);
                    }
                }
                return Err(err.into());
            }
        };

        let response = match response.error_for_status() {
            Ok(ok) => ok,
            Err(err) => {
                if self.config.enable_cache {
                    if let Ok(cached) = self.get_from_cache(url) {
                        log::warn!("HTTP 状态异常，回退到缓存: {} -> {}", url, err);
                        return Ok(cached);
                    }
                }
                return Err(err.into());
            }
        };

        let html = response.text().await?;

        // 保存到缓存
        if self.config.enable_cache {
            let _ = self.save_to_cache(url, &html);
        }

        Ok(html)
    }

    /// 获取二进制资源（用于 WASM 等）
    async fn fetch_binary(
        &self,
        client: &reqwest::Client,
        url: &str,
        extension: &str,
    ) -> Result<Vec<u8>> {
        log::debug!("获取二进制资源: {}", url);

        if self.config.enable_cache {
            if let Ok(bytes) = self.get_binary_from_cache(url, extension) {
                log::debug!("从缓存读取二进制: {}", url);
                return Ok(bytes);
            }
        }

        let response = match client.get(url).send().await {
            Ok(resp) => resp,
            Err(err) => {
                if self.config.enable_cache {
                    if let Ok(cached) = self.get_binary_from_cache(url, extension) {
                        log::warn!("网络获取二进制失败，回退到缓存: {} -> {}", url, err);
                        return Ok(cached);
                    }
                }
                return Err(err.into());
            }
        };

        let response = match response.error_for_status() {
            Ok(ok) => ok,
            Err(err) => {
                if self.config.enable_cache {
                    if let Ok(cached) = self.get_binary_from_cache(url, extension) {
                        log::warn!("二进制状态异常，回退到缓存: {} -> {}", url, err);
                        return Ok(cached);
                    }
                }
                return Err(err.into());
            }
        };

        let bytes = response.bytes().await?.to_vec();

        if self.config.enable_cache {
            let _ = self.save_binary_to_cache(url, extension, &bytes);
        }

        Ok(bytes)
    }

    /// 生成个性化布局
    #[allow(dead_code)]
    fn generate_personalized_layout(
        &self,
        _request: &PersonalizationRequest,
    ) -> Result<(String, String, String)> {
        // TODO: 实现布局生成算法
        // 现在返回占位符

        let html = r#"<!DOCTYPE html>
<html>
<head><title>个性化布局</title></head>
<body>
<h1>个性化布局生成中...</h1>
</body>
</html>"#
            .to_string();

        let css = r#"body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
}"#
        .to_string();

        let javascript = r#"console.log('Personalized layout loaded');"#.to_string();

        Ok((html, css, javascript))
    }

    /// 从缓存获取
    fn get_from_cache(&self, url: &str) -> Result<String> {
        let cache_file = self.get_cache_path(url);
        if cache_file.exists() {
            let content = std::fs::read_to_string(cache_file)?;
            Ok(content)
        } else {
            Err(anyhow::anyhow!("缓存不存在"))
        }
    }

    /// 保存到缓存
    fn save_to_cache(&self, url: &str, content: &str) -> Result<()> {
        let cache_file = self.get_cache_path(url);
        std::fs::create_dir_all(cache_file.parent().unwrap())?;
        std::fs::write(cache_file, content)?;
        Ok(())
    }

    /// 获取缓存文件路径
    fn get_cache_path(&self, url: &str) -> PathBuf {
        self.get_cache_path_with_extension(url, "html")
    }

    fn get_cache_path_with_extension(&self, url: &str, extension: &str) -> PathBuf {
        let filename = format!(
            "{}.{}",
            url.replace("https://", "")
                .replace("http://", "")
                .replace("/", "_"),
            extension
        );
        self.config.cache_dir.join(filename)
    }

    fn get_binary_from_cache(&self, url: &str, extension: &str) -> Result<Vec<u8>> {
        let cache_file = self.get_cache_path_with_extension(url, extension);
        if cache_file.exists() {
            let content = std::fs::read(cache_file)?;
            Ok(content)
        } else {
            Err(anyhow::anyhow!("缓存不存在"))
        }
    }

    fn save_binary_to_cache(&self, url: &str, extension: &str, bytes: &[u8]) -> Result<()> {
        let cache_file = self.get_cache_path_with_extension(url, extension);
        std::fs::create_dir_all(cache_file.parent().unwrap())?;
        std::fs::write(cache_file, bytes)?;
        Ok(())
    }

    fn resolve_url(&self, base_url: &str, resource_url: &str) -> Option<String> {
        if let Ok(absolute) = Url::parse(resource_url) {
            return Some(absolute.to_string());
        }

        if let Ok(base) = Url::parse(base_url) {
            if let Ok(joined) = base.join(resource_url) {
                return Some(joined.to_string());
            }
        }

        None
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            user_id: "default_user".to_string(),
            user_preferences: UserPreferences {
                layout_style: "modern".to_string(),
                color_scheme: "light".to_string(),
                font_preference: "sans-serif".to_string(),
                compactness: 5,
                information_density: 6,
                interaction_style: "interactive".to_string(),
                enable_animations: true,
            },
            user_profile: UserProfile {
                user_id_hash: 0,
                viewport_width: 1920,
                language: "zh-CN".to_string(),
                uses_screen_reader: false,
                interaction_history: vec![],
            },
            output_dir: PathBuf::from("./output"),
            cache_dir: PathBuf::from("./cache"),
            enable_cache: true,
            analyze_javascript: true,
            enable_deobfuscation: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_execute_real_website() {
        let config = PipelineConfig::default();
        let pipeline = IntegratedPipeline::new(config);

        let result = pipeline
            .execute("https://example.com")
            .await
            .expect("pipeline execution should succeed with real site");

        assert_eq!(result.url, "https://example.com");
        assert!(!result.generated_html.is_empty());
        assert!(!result.generated_css.is_empty());
        assert!(!result.generated_javascript.is_empty());

        // 结果字段应被填充
        assert!(result.processing_time_ms > 0);
        // 外部资源分析结构应可访问
        let _ = result.external_resources.total_size_bytes;
        // 技术检测结果结构应可访问
        let _ = result.tech_detection.detected_technologies.len();
    }

    #[tokio::test]
    async fn test_pipeline_handles_cached_wasm() {
        let temp = tempfile::tempdir().expect("create temp dir");
        let mut config = PipelineConfig::default();
        config.cache_dir = temp.path().to_path_buf();
        config.enable_cache = true;

        let pipeline = IntegratedPipeline::new(config);
        let url = "https://cache-wasm.test/entry";

        // 预写入 HTML 缓存，包含 WASM 引用
        let html_cache = pipeline.get_cache_path(url);
        std::fs::create_dir_all(html_cache.parent().unwrap()).unwrap();
        std::fs::write(
            &html_cache,
            r#"<html><script>WebAssembly.instantiate('module.wasm');</script></html>"#,
        )
        .unwrap();

        // 确认缓存可读
        assert!(html_cache.exists(), "HTML 缓存文件应存在");
        let cached_html = pipeline.get_from_cache(url).expect("应能从缓存读取 HTML");
        assert!(cached_html.contains("WebAssembly"));

        // 预写入对应的 WASM 缓存（二进制最小魔数 + 版本）
        let wasm_url = "https://cache-wasm.test/module.wasm";
        let wasm_cache = pipeline.get_cache_path_with_extension(wasm_url, "wasm");
        std::fs::create_dir_all(wasm_cache.parent().unwrap()).unwrap();
        let wasm_bytes: &[u8] = b"\0asm\x01\0\0\0more"; // 长度>=8 且包含魔数
        std::fs::write(&wasm_cache, wasm_bytes).unwrap();

        let result = pipeline
            .execute(url)
            .await
            .expect("pipeline should process cached wasm");

        assert_eq!(result.url, url);
        assert_eq!(result.wasm_modules.len(), 1);
        assert_eq!(result.wasm_modules[0].url, wasm_url);
    }

    #[test]
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = IntegratedPipeline::new(config);
        assert_eq!(pipeline.config.user_id, "default_user");
    }

    #[test]
    fn test_cache_path_generation() {
        let config = PipelineConfig::default();
        let pipeline = IntegratedPipeline::new(config);
        let path = pipeline.get_cache_path("https://example.com/page");
        assert!(path.to_string_lossy().contains("example.com_page"));
    }
}
