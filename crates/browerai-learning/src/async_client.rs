/// 异步客户端
/// 集成所有学习、推理、生成功能的完整客户端
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{
    CompleteInferencePipeline, CompleteInferenceResult, GeneratedModule, GeneratedWebsite,
    ImprovedCodeGenerator, LearningSession, RealWebsiteLearner, WebsiteConfig, WebsiteGenerator,
    WebsiteLearningTask,
};

/// 客户端配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientConfig {
    /// 是否自动保存生成的代码
    pub auto_save: bool,

    /// 输出目录
    pub output_directory: String,

    /// 最小学习质量阈值（0-1）
    pub min_quality_threshold: f64,

    /// 是否启用调试日志
    pub enable_debug_logging: bool,
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig {
            auto_save: true,
            output_directory: "output/browerai".to_string(),
            min_quality_threshold: 0.7,
            enable_debug_logging: false,
        }
    }
}

/// 客户端状态
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ClientState {
    Idle,
    Learning,
    Inferring,
    Generating,
    Completed,
    Error(String),
}

/// 完整的学习-推理-生成结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompleteResult {
    pub learning_session: LearningSession,
    pub inference_result: CompleteInferenceResult,
    pub generated_modules: Vec<GeneratedModule>,
    pub generated_website: Option<GeneratedWebsite>,
    pub total_execution_time_ms: u64,
}

/// 异步客户端
pub struct BrowserAIClient {
    config: ClientConfig,
    state: ClientState,
}

impl BrowserAIClient {
    /// 创建客户端
    pub fn new(config: ClientConfig) -> Self {
        log::info!("✓ 创建 BrowserAI 客户端");

        BrowserAIClient {
            config,
            state: ClientState::Idle,
        }
    }

    /// 获取当前状态
    pub fn state(&self) -> ClientState {
        self.state.clone()
    }

    /// 执行完整的学习-推理-生成流程
    pub async fn process_website(&mut self, task: WebsiteLearningTask) -> Result<CompleteResult> {
        let start_time = std::time::Instant::now();

        log::info!("🚀 开始处理网站: {}", task.name);

        // Step 1: 学习
        self.state = ClientState::Learning;
        log::info!("📚 步骤 1/3: 学习网站");

        let learner = RealWebsiteLearner::new()?;
        let learning_session = learner.learn_website(task).await?;

        // 检查学习质量
        if let Some(quality) = &learning_session.quality {
            if quality.overall_score < self.config.min_quality_threshold {
                log::warn!(
                    "⚠️  学习质量低于阈值: {:.1}% < {:.1}%",
                    quality.overall_score * 100.0,
                    self.config.min_quality_threshold * 100.0
                );
            }
        }

        // Step 2: 推理
        self.state = ClientState::Inferring;
        log::info!("🧠 步骤 2/3: 执行推理");

        let traces = learning_session
            .raw_traces
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No execution traces"))?;

        let workflows = learning_session
            .workflows
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No workflows extracted"))?;

        let inference_result = CompleteInferencePipeline::infer(traces, workflows)?;

        // Step 3: 生成
        self.state = ClientState::Generating;
        log::info!("💻 步骤 3/3: 生成代码");

        let generated_modules = ImprovedCodeGenerator::generate_code(&inference_result)?;

        // 保存生成的代码
        if self.config.auto_save {
            self.save_generated_code(&generated_modules)?;
        }

        let _elapsed = start_time.elapsed();

        // Step 4: 生成网站（新增！）
        log::info!("🌐 步骤 4/4: 生成现代网站");
        self.state = ClientState::Generating;

        let website_config = WebsiteConfig::default();
        let generator = WebsiteGenerator::new(website_config);
        let generated_website = generator.generate_website(&learning_session, &inference_result)?;

        // 保存网站文件
        if self.config.auto_save {
            self.save_website_files(&generated_website)?;
        }

        let elapsed = start_time.elapsed();

        self.state = ClientState::Completed;

        log::info!(
            "✅ 完成! 耗时: {:.2}s, 生成 {} 个代码模块 + 完整网站",
            elapsed.as_secs_f64(),
            generated_modules.len()
        );

        Ok(CompleteResult {
            learning_session,
            inference_result,
            generated_modules,
            generated_website: Some(generated_website),
            total_execution_time_ms: elapsed.as_millis() as u64,
        })
    }

    /// 保存生成的代码
    fn save_generated_code(&self, modules: &[GeneratedModule]) -> Result<()> {
        std::fs::create_dir_all(&self.config.output_directory)?;

        for module in modules {
            let file_path = format!("{}/{}", self.config.output_directory, module.module_name);

            std::fs::write(&file_path, &module.code)?;
            log::info!("  ✓ 保存 {}", file_path);
        }

        Ok(())
    }

    /// 保存生成的网站文件
    fn save_website_files(&self, website: &GeneratedWebsite) -> Result<()> {
        let site_dir = format!("{}/website", self.config.output_directory);
        std::fs::create_dir_all(&site_dir)?;

        // 保存 HTML
        let html_path = format!("{}/index.html", site_dir);
        std::fs::write(&html_path, &website.html)?;
        log::info!("  ✓ 保存网站 HTML: {}", html_path);

        // 保存 CSS
        let css_path = format!("{}/styles.css", site_dir);
        std::fs::write(&css_path, &website.css)?;
        log::info!("  ✓ 保存网站样式: {}", css_path);

        // 保存 JavaScript
        let js_path = format!("{}/app.js", site_dir);
        std::fs::write(&js_path, &website.javascript)?;
        log::info!("  ✓ 保存网站脚本: {}", js_path);

        Ok(())
    }

    /// 生成报告
    pub fn generate_report(&self, result: &CompleteResult) -> String {
        let mut report = String::new();

        report.push_str("# BrowserAI 处理报告\n\n");

        // 学习部分
        report.push_str("## 学习结果\n");
        if let Some(quality) = &result.learning_session.quality {
            report.push_str(&format!(
                "- 函数覆盖: {:.1}%\n",
                quality.function_coverage * 100.0
            ));
            report.push_str(&format!(
                "- 工作流完整性: {:.1}%\n",
                quality.workflow_completeness * 100.0
            ));
            report.push_str(&format!(
                "- 功能保留: {:.1}%\n",
                quality.functionality_preserved * 100.0
            ));
        }

        // 推理部分
        report.push_str("\n## 推理结果\n");
        report.push_str(&format!(
            "- 推理评分: {:.1}%\n",
            result.inference_result.overall_inference_score * 100.0
        ));
        report.push_str(&format!(
            "- 发现变量: {}\n",
            result.inference_result.variable_inference.variables.len()
        ));
        report.push_str(&format!(
            "- 数据结构: {}\n",
            result.inference_result.structure_inference.structures.len()
        ));

        // 生成部分
        report.push_str("\n## 代码生成\n");
        report.push_str(&format!("- 生成模块: {}\n", result.generated_modules.len()));
        for module in &result.generated_modules {
            report.push_str(&format!(
                "  - {} ({} 行)\n",
                module.module_name,
                module.code.lines().count()
            ));
        }

        // 网站生成部分
        if let Some(website) = &result.generated_website {
            report.push_str("\n## 网站生成 ✨\n");
            report.push_str(&format!("- 网站名称: {}\n", website.name));
            report.push_str(&format!("- HTML 大小: {} 字符\n", website.html.len()));
            report.push_str(&format!("- CSS 大小: {} 字符\n", website.css.len()));
            report.push_str(&format!("- JS 大小: {} 字符\n", website.javascript.len()));
            report.push_str(&format!(
                "- 功能保留: {}/{}\n",
                website.preserved_features.len(),
                website.preserved_features.len()
            ));
            report.push_str(&format!("- 主题色: {}\n", website.config.primary_color));
            report.push_str(&format!(
                "- 响应式设计: {}\n",
                if website.config.responsive_design {
                    "✓ 是"
                } else {
                    "✗ 否"
                }
            ));
            report.push_str(&format!(
                "- 深色模式: {}\n",
                if website.config.enable_dark_mode {
                    "✓ 支持"
                } else {
                    "✗ 不支持"
                }
            ));
        }

        report.push_str("\n## 执行时间\n");
        report.push_str(&format!(
            "- 总耗时: {:.2}秒\n",
            result.total_execution_time_ms as f64 / 1000.0
        ));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let config = ClientConfig::default();
        let client = BrowserAIClient::new(config);
        assert_eq!(client.state(), ClientState::Idle);
    }

    #[test]
    fn test_default_config() {
        let config = ClientConfig::default();
        assert!(config.auto_save);
        assert_eq!(config.min_quality_threshold, 0.7);
    }
}
