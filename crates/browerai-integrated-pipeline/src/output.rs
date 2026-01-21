//! 输出模块 - 结果输出和格式化

use crate::pipeline::PipelineResult;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 输出格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// JSON格式
    Json,
    /// HTML文件
    Html,
    /// 完整包（HTML + CSS + JS + 分析结果）
    Package,
}

/// 输出生成器
pub struct OutputGenerator {
    output_dir: PathBuf,
}

impl OutputGenerator {
    pub fn new(output_dir: PathBuf) -> Self {
        Self { output_dir }
    }

    /// 生成输出
    pub fn generate(&self, result: &PipelineResult, format: OutputFormat) -> Result<()> {
        std::fs::create_dir_all(&self.output_dir)?;

        match format {
            OutputFormat::Json => self.generate_json(result)?,
            OutputFormat::Html => self.generate_html(result)?,
            OutputFormat::Package => {
                self.generate_json(result)?;
                self.generate_html(result)?;
                self.generate_css(result)?;
                self.generate_javascript(result)?;
                self.generate_analysis(result)?;
            }
        }

        Ok(())
    }

    fn generate_json(&self, result: &PipelineResult) -> Result<()> {
        let json = serde_json::to_string_pretty(&result)?;
        let path = self.output_dir.join("result.json");
        std::fs::write(path, json)?;
        log::info!("✓ JSON生成完成");
        Ok(())
    }

    fn generate_html(&self, result: &PipelineResult) -> Result<()> {
        let path = self.output_dir.join("personalized.html");
        std::fs::write(path, &result.generated_html)?;
        log::info!("✓ HTML生成完成");
        Ok(())
    }

    fn generate_css(&self, result: &PipelineResult) -> Result<()> {
        let path = self.output_dir.join("personalized.css");
        std::fs::write(path, &result.generated_css)?;
        log::info!("✓ CSS生成完成");
        Ok(())
    }

    fn generate_javascript(&self, result: &PipelineResult) -> Result<()> {
        let path = self.output_dir.join("personalized.js");
        std::fs::write(path, &result.generated_javascript)?;
        log::info!("✓ JavaScript生成完成");
        Ok(())
    }

    fn generate_analysis(&self, result: &PipelineResult) -> Result<()> {
        let analysis = format!(
            r#"# 网站分析报告

## URL: {}

## 处理耗时: {}ms

## 分析详情:

### 网站分析结果:
{}

### 个性化请求:
{}
"#,
            result.url,
            result.processing_time_ms,
            serde_json::to_string_pretty(&result.website_analysis)?,
            serde_json::to_string_pretty(&result.personalization_request)?,
        );

        let path = self.output_dir.join("analysis.md");
        std::fs::write(path, analysis)?;
        log::info!("✓ 分析报告生成完成");
        Ok(())
    }
}
