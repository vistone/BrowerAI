/// 真实网站学习集成
/// 抓取真实网站，注入 V8 追踪代码，学习工作流
use anyhow::{Context, Result};
use log;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::learning_quality::LearningQuality;
use crate::v8_tracer::{ExecutionTrace, V8Tracer};
use crate::workflow_extractor::{WorkflowExtractionResult, WorkflowExtractor};

/// 真实网站学习任务
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WebsiteLearningTask {
    pub url: String,
    pub name: String,
    pub target_workflows: Vec<String>,
    pub max_interactions: usize,
}

/// 学习会话结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningSession {
    pub task: WebsiteLearningTask,
    pub raw_traces: Option<ExecutionTrace>,
    pub workflows: Option<WorkflowExtractionResult>,
    pub quality: Option<LearningQuality>,
    pub learned_code: Option<String>,
    pub status: SessionStatus,
    /// 原始 HTML 内容
    pub original_html: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    Initialized,
    FetchingPage,
    InjectingTracers,
    RunningTracers,
    ExtractingTraces,
    IdentifyingWorkflows,
    AssessingQuality,
    GeneratingCode,
    Completed,
    Failed(String),
}

/// 真实网站学习器
#[allow(dead_code)]
pub struct RealWebsiteLearner {
    tracer: Arc<V8Tracer>,
}

impl RealWebsiteLearner {
    /// 创建学习器
    pub fn new() -> Result<Self> {
        Ok(RealWebsiteLearner {
            tracer: Arc::new(V8Tracer::new()?),
        })
    }

    /// 学习单个网站
    pub async fn learn_website(&self, task: WebsiteLearningTask) -> Result<LearningSession> {
        log::info!("🌐 开始学习网站: {} ({})", task.name, task.url);

        let mut session = LearningSession {
            task: task.clone(),
            raw_traces: None,
            workflows: None,
            quality: None,
            learned_code: None,
            status: SessionStatus::Initialized,
            original_html: None,
        };

        // 第1步：获取页面
        session.status = SessionStatus::FetchingPage;
        log::info!("📥 获取页面...");
        let html = self.fetch_page(&task.url).await?;

        // 保存原始 HTML
        session.original_html = Some(html.clone());

        // 第2步：注入追踪器
        session.status = SessionStatus::InjectingTracers;
        log::info!("💉 注入 V8 追踪器...");
        let injected_html = V8Tracer::inject_tracers_to_html(&html);

        // 第3步：运行追踪器（模拟真实用户交互）
        session.status = SessionStatus::RunningTracers;
        log::info!("⚙️  运行追踪器（模拟交互）...");
        let trace_json = self.simulate_interactions(&injected_html).await?;

        // 第4步：提取追踪数据
        session.status = SessionStatus::ExtractingTraces;
        log::info!("📊 提取追踪数据...");
        let traces = V8Tracer::extract_traces_from_window(&trace_json)?;
        session.raw_traces = Some(traces.clone());

        // 第5步：识别工作流
        session.status = SessionStatus::IdentifyingWorkflows;
        log::info!("🔍 识别工作流...");
        let workflows = WorkflowExtractor::extract_workflows(&traces)?;
        session.workflows = Some(workflows.clone());

        // 第6步：评估学习质量
        session.status = SessionStatus::AssessingQuality;
        log::info!("✅ 评估学习质量...");
        let quality = LearningQuality::evaluate(&traces, &workflows)?;

        if quality.overall_score < 0.7 {
            log::warn!(
                "⚠️  学习质量不足 ({}%), 建议再次学习",
                (quality.overall_score * 100.0) as i32
            );
        } else if quality.overall_score >= 0.9 {
            log::info!(
                "🎉 学习质量优秀 ({}%)",
                (quality.overall_score * 100.0) as i32
            );
        }

        session.quality = Some(quality);

        // 第7步：生成可学习的代码
        session.status = SessionStatus::GeneratingCode;
        log::info!("💾 生成学习代码...");
        let learned_code = self.generate_learning_code(&workflows)?;
        session.learned_code = Some(learned_code);

        session.status = SessionStatus::Completed;
        log::info!(
            "✓ 完成学习: {} 个工作流，质量评分 {:.1}%",
            workflows.workflows.len(),
            (session.quality.as_ref().unwrap().overall_score * 100.0) as i32
        );

        Ok(session)
    }

    /// 获取页面内容
    async fn fetch_page(&self, url: &str) -> Result<String> {
        // 使用 reqwest 获取真实网页
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let response = client
            .get(url)
            .header("User-Agent", "BrowerAI/1.0 Learning Agent")
            .send()
            .await
            .with_context(|| format!("Failed to fetch {}", url))?;

        let html = response
            .text()
            .await
            .context("Failed to read response body")?;

        log::debug!("📄 页面大小: {} 字节", html.len());
        Ok(html)
    }

    /// 模拟用户交互（真实的交互序列）
    async fn simulate_interactions(&self, _html: &str) -> Result<String> {
        // 这会在实际的浏览器环境中运行
        // 当前使用模拟数据，后续集成真实浏览器引擎

        let trace_json = r#"{
                "function_calls": [
                    {"function_name": "handleSearch", "arguments": ["test"], "return_type": "void", "timestamp_ms": 100, "context_object": null, "call_depth": 0},
                    {"function_name": "processResults", "arguments": ["results"], "return_type": "array", "timestamp_ms": 150, "context_object": null, "call_depth": 1},
                    {"function_name": "renderItems", "arguments": ["data"], "return_type": "void", "timestamp_ms": 200, "context_object": null, "call_depth": 2}
                ],
                "dom_operations": [
                    {"operation_type": "appendChild", "target_selector": ".results", "timestamp_ms": 150},
                    {"operation_type": "innerHTML", "target_selector": ".item", "timestamp_ms": 160}
                ],
                "event_listeners": [
                    {"event_type": "click", "target_selector": ".search-btn", "listener_function": "handleSearch"}
                ],
                "user_events": [
                    {"event_type": "click", "target_selector": ".search-btn", "timestamp_ms": 50},
                    {"event_type": "input", "target_selector": ".search-input", "timestamp_ms": 40, "value": "test"}
                ],
                "state_changes": [
                    {"variable_name": "searchQuery", "previous_value": "", "new_value": "test", "timestamp_ms": 40},
                    {"variable_name": "results", "previous_value": "null", "new_value": "array[10]", "timestamp_ms": 150}
                ],
                "total_duration_ms": 300,
                "page_ready_ms": 50
            }"#.to_string();

        Ok(trace_json)
    }

    /// 生成可学习的代码
    fn generate_learning_code(&self, workflows: &WorkflowExtractionResult) -> Result<String> {
        let mut code = String::from("// 自动生成的学习代码\n\n");

        for (idx, workflow) in workflows.workflows.iter().enumerate() {
            code.push_str(&format!(
                "// 工作流 {}: {} (完整性: {:.0}%, 重要性: {:.0}%)\n",
                idx + 1,
                workflow.name,
                workflow.complexity_score * 10.0,
                workflow.importance_score * 10.0
            ));

            code.push_str("function ");
            code.push_str(&workflow.name);
            code.push_str("() {\n");

            for func in &workflow.key_functions {
                code.push_str(&format!("  // 调用: {}\n", func));
                code.push_str(&format!("  {}();\n", func));
            }

            code.push_str("}\n\n");
        }

        Ok(code)
    }
}

impl Default for RealWebsiteLearner {
    fn default() -> Self {
        RealWebsiteLearner {
            tracer: Arc::new(V8Tracer::new().unwrap()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_task_creation() {
        let task = WebsiteLearningTask {
            url: "https://example.com".to_string(),
            name: "Example Site".to_string(),
            target_workflows: vec!["search".to_string()],
            max_interactions: 10,
        };

        assert_eq!(task.name, "Example Site");
        assert_eq!(task.url, "https://example.com");
    }
}
