/// 双沙盒学习集成
///
/// 将V8追踪和工作流学习与DualSandboxRenderer整合
/// 支持从ExecutionTrace生成语义化的HTML/CSS/JS代码
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::code_verifier::{CodeVerificationResult, CodeVerifier};
use crate::learning_quality::LearningQuality;
use crate::semantic_comparator::{SemanticComparator, SemanticComparisonResult};
use crate::v8_tracer::ExecutionTrace;
use crate::workflow_extractor::{WorkflowExtractionResult, WorkflowExtractor};

/// 双沙盒学习任务结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DualSandboxLearningResult {
    /// 原始执行追踪
    pub traces: ExecutionTrace,

    /// 提取的工作流
    pub workflows: WorkflowExtractionResult,

    /// 学习质量评估
    pub quality: LearningQuality,

    /// 生成的语义化HTML
    pub generated_html: Option<String>,

    /// 生成的语义化CSS
    pub generated_css: Option<String>,

    /// 生成的语义化JavaScript
    pub generated_js: Option<String>,

    /// 学习总结
    pub summary: LearningSummary,

    /// 语义对比结果（可选）
    pub semantic_comparison: Option<SemanticComparisonResult>,

    /// 生成代码的验证结果（可选）
    pub verification: Option<CodeVerificationResult>,
}

/// 学习总结
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningSummary {
    /// 发现的工作流数量
    pub workflow_count: usize,

    /// 识别的关键函数
    pub key_functions: Vec<String>,

    /// 识别的DOM操作模式
    pub dom_patterns: Vec<String>,

    /// 识别的用户交互流程
    pub interaction_flows: Vec<String>,

    /// 整体学习评分（0-100）
    pub overall_score: i32,
}

/// 双沙盒学习器
///
/// 从V8执行追踪和工作流中提取语义信息，生成清晰的HTML/CSS/JavaScript代码
pub struct DualSandboxLearner {
    _marker: std::marker::PhantomData<()>,
}

impl DualSandboxLearner {
    /// 创建双沙盒学习器
    pub fn new() -> Result<Self> {
        Ok(DualSandboxLearner {
            _marker: std::marker::PhantomData,
        })
    }

    /// 从执行追踪学习并生成代码
    pub async fn learn_and_generate(
        &self,
        traces: ExecutionTrace,
    ) -> Result<DualSandboxLearningResult> {
        log::info!("🧠 双沙盒学习开始...");

        // 第1步：提取工作流
        log::info!("  Step 1: 从执行追踪提取工作流");
        let workflows = WorkflowExtractor::extract_workflows(&traces)?;

        // 第2步：评估学习质量
        log::info!("  Step 2: 评估学习质量");
        let quality = LearningQuality::evaluate(&traces, &workflows)?;

        // 第3步：生成语义化代码（通过理解工作流）
        log::info!("  Step 3: 生成语义化代码");
        let generated_html = self.generate_semantic_html(&workflows, &traces).ok();
        let generated_css = self.generate_semantic_css(&workflows, &traces).ok();
        let generated_js = self.generate_semantic_js(&workflows, &traces).ok();

        // 第4步：生成学习总结
        log::info!("  Step 4: 生成学习总结");
        let summary = self.generate_learning_summary(&workflows, &traces, &quality)?;

        log::info!(
            "✓ 双沙盒学习完成: {} 个工作流, 质量评分 {:.0}%",
            workflows.workflows.len(),
            quality.overall_score * 100.0
        );

        Ok(DualSandboxLearningResult {
            traces,
            workflows,
            quality,
            generated_html,
            generated_css,
            generated_js,
            summary,
            semantic_comparison: None,
            verification: None,
        })
    }

    /// 从执行追踪学习并生成代码，同时与原始代码进行语义对比
    #[allow(clippy::too_many_arguments)]
    pub async fn learn_and_generate_with_reference(
        &self,
        traces: ExecutionTrace,
        original_html: &str,
        original_css: &str,
        original_js: &str,
    ) -> Result<DualSandboxLearningResult> {
        log::info!("🧠 双沙盒对比学习开始...");

        // 第1步：提取工作流
        let workflows = WorkflowExtractor::extract_workflows(&traces)?;

        // 第2步：基础质量评估
        let mut quality = LearningQuality::evaluate(&traces, &workflows)?;

        // 第3步：生成语义化代码
        let generated_html = self.generate_semantic_html(&workflows, &traces).ok();
        let generated_css = self.generate_semantic_css(&workflows, &traces).ok();
        let generated_js = self.generate_semantic_js(&workflows, &traces).ok();

        // 第4步：生成学习总结（基础）
        let summary = self.generate_learning_summary(&workflows, &traces, &quality)?;

        // 第4.5步：语义对比 + 代码验证（如果生成代码存在）
        let mut semantic_comparison = None;
        let mut verification = None;
        if let (Some(ref html), Some(ref css), Some(ref js)) =
            (&generated_html, &generated_css, &generated_js)
        {
            semantic_comparison = Some(SemanticComparator::compare_all(
                original_html,
                original_css,
                original_js,
                html,
                css,
                js,
                &workflows
                    .workflows
                    .iter()
                    .flat_map(|w| w.key_functions.clone())
                    .collect::<Vec<_>>(),
            )?);

            // 更新质量评分（加入等价性）
            quality = LearningQuality::evaluate_with_comparison(
                &traces,
                &workflows,
                original_html,
                original_css,
                original_js,
                html,
                css,
                js,
            )?;

            verification = CodeVerifier::verify_all(html, css, js).ok();
        }

        log::info!(
            "✓ 对比学习完成: 工作流 {}, 语义相似度 {:.1}%",
            workflows.workflows.len(),
            semantic_comparison
                .as_ref()
                .map(|c| c.overall_similarity * 100.0)
                .unwrap_or(0.0)
        );

        Ok(DualSandboxLearningResult {
            traces,
            workflows,
            quality,
            generated_html,
            generated_css,
            generated_js,
            summary,
            semantic_comparison,
            verification,
        })
    }

    /// 生成语义化HTML
    fn generate_semantic_html(
        &self,
        workflows: &WorkflowExtractionResult,
        _traces: &ExecutionTrace,
    ) -> Result<String> {
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n  <meta charset=\"UTF-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n  <title>AI Generated Page</title>\n</head>\n<body>\n");

        // 根据识别的工作流生成语义化结构
        for (idx, workflow) in workflows.workflows.iter().enumerate() {
            html.push_str(&format!(
                "  <!-- 工作流 {}: {} -->\n",
                idx + 1,
                workflow.name
            ));
            html.push_str(&format!("  <section class=\"workflow-{}\">\n", idx + 1));
            html.push_str(&format!("    <h2>{}</h2>\n", workflow.name));

            // 基于关键函数生成HTML元素
            for (func_idx, func) in workflow.key_functions.iter().enumerate() {
                html.push_str(&format!(
                    "    <div class=\"function-{}\" data-handler=\"{}\">\n",
                    func_idx + 1,
                    func
                ));
                html.push_str(&format!(
                    "      <button onclick=\"{}\">Execute {}</button>\n",
                    func, func
                ));
                html.push_str("    </div>\n");
            }

            html.push_str("  </section>\n\n");
        }

        html.push_str("</body>\n</html>");
        Ok(html)
    }

    /// 生成语义化CSS
    fn generate_semantic_css(
        &self,
        workflows: &WorkflowExtractionResult,
        _traces: &ExecutionTrace,
    ) -> Result<String> {
        let mut css = String::from("/* AI Generated Semantic CSS */\n\n");

        // 为每个工作流生成CSS类
        for (idx, workflow) in workflows.workflows.iter().enumerate() {
            css.push_str(&format!("/* 工作流 {}: {} */\n", idx + 1, workflow.name));
            css.push_str(&format!(".workflow-{} {{\n", idx + 1));
            css.push_str("  padding: 20px;\n");
            css.push_str("  margin: 10px 0;\n");
            css.push_str("  border: 1px solid #ddd;\n");
            css.push_str("  border-radius: 4px;\n");
            css.push_str("}\n\n");

            // 为函数处理程序生成样式
            for func_idx in 0..workflow.key_functions.len() {
                css.push_str(&format!(".function-{} {{\n", func_idx + 1));
                css.push_str("  margin: 10px 0;\n");
                css.push_str("}\n\n");

                css.push_str(&format!(".function-{} button {{\n", func_idx + 1));
                css.push_str("  padding: 10px 20px;\n");
                css.push_str("  background-color: #007bff;\n");
                css.push_str("  color: white;\n");
                css.push_str("  border: none;\n");
                css.push_str("  border-radius: 4px;\n");
                css.push_str("  cursor: pointer;\n");
                css.push_str("}\n\n");
            }
        }

        Ok(css)
    }

    /// 生成语义化JavaScript
    fn generate_semantic_js(
        &self,
        workflows: &WorkflowExtractionResult,
        _traces: &ExecutionTrace,
    ) -> Result<String> {
        let mut js = String::from("// AI Generated Semantic JavaScript\n\n");

        // 为每个工作流生成函数框架
        for workflow in &workflows.workflows {
            js.push_str(&format!("/**\n * 工作流: {}\n", workflow.name));
            js.push_str(&format!(
                " * 重要性评分: {:.1}%\n */\n",
                workflow.importance_score * 100.0
            ));
            js.push_str(&format!("async function {}() {{\n", workflow.name));

            // 调用识别的关键函数
            for func in &workflow.key_functions {
                js.push_str(&format!("  // 调用关键函数: {}\n", func));
                js.push_str(&format!(
                    "  const result_{} = await {}();\n",
                    func.replace("-", "_"),
                    func
                ));
            }

            js.push_str("  return true;\n");
            js.push_str("}\n\n");
        }

        Ok(js)
    }

    /// 生成学习总结
    fn generate_learning_summary(
        &self,
        workflows: &WorkflowExtractionResult,
        traces: &ExecutionTrace,
        quality: &LearningQuality,
    ) -> Result<LearningSummary> {
        // 收集所有唯一的关键函数
        let mut key_functions = std::collections::HashSet::new();
        for workflow in &workflows.workflows {
            for func in &workflow.key_functions {
                key_functions.insert(func.clone());
            }
        }

        // 识别DOM操作模式
        let dom_patterns = traces
            .dom_operations
            .iter()
            .map(|op| format!("{:?}", op.operation_type))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // 识别交互流程
        let interaction_flows = traces
            .user_events
            .iter()
            .map(|ev| format!("{:?}", ev.event_type))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        Ok(LearningSummary {
            workflow_count: workflows.workflows.len(),
            key_functions: key_functions.into_iter().collect(),
            dom_patterns,
            interaction_flows,
            overall_score: (quality.overall_score * 100.0) as i32,
        })
    }
}

impl Default for DualSandboxLearner {
    fn default() -> Self {
        DualSandboxLearner {
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dual_sandbox_learner_creation() {
        let _learner = DualSandboxLearner::new().unwrap();
        // Learner created successfully
    }

    #[test]
    fn test_learning_summary_generation() {
        let workflow_result = WorkflowExtractionResult {
            workflows: vec![],
            total_user_interactions: 0,
            total_function_calls: 0,
            coverage_ratio: 0.0,
        };

        let trace = ExecutionTrace::new();
        let quality = LearningQuality {
            function_coverage: 0.0,
            workflow_completeness: 0.0,
            functionality_preserved: 0.0,
            overall_score: 0.0,
            issues: vec![],
            recommendations: vec![],
            semantic_comparison: None,
            code_equivalence_score: None,
        };

        let learner = DualSandboxLearner::new().unwrap();
        let summary = learner
            .generate_learning_summary(&workflow_result, &trace, &quality)
            .unwrap();

        assert_eq!(summary.workflow_count, 0);
        assert_eq!(summary.overall_score, 0);
    }

    #[tokio::test]
    async fn test_learn_and_generate_with_reference() {
        let learner = DualSandboxLearner::new().unwrap();
        let traces = ExecutionTrace::new();

        let result = learner
            .learn_and_generate_with_reference(
                traces,
                "<html><body><button onclick=\"hello()\">Hi</button></body></html>",
                "button { color: red; }",
                "function hello() { return true; }",
            )
            .await
            .unwrap();

        assert!(result.semantic_comparison.is_some());
    }
}
