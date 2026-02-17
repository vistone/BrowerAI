/// 比较学习主入口 - Phase D
///
/// 为用户提供简洁的"对比学习"高级API
/// 集成追踪 → 工作流 → 生成 → 验证 → 对比 → 评估 → 反馈的完整流程
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::code_verifier::CodeVerificationResult;
use crate::comparison_feedback::{ComparisonFeedback, ComparisonFeedbackGenerator};
use crate::dual_sandbox_learner::{DualSandboxLearner, LearningSummary};
use crate::learning_quality::LearningQuality;
use crate::semantic_comparator::SemanticComparisonResult;
use crate::v8_tracer::ExecutionTrace;
use crate::workflow_extractor::WorkflowExtractionResult;

/// 比较学习报告 - Phase D的输出
/// 包含完整的学习和对比流程的所有结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparativeLearningReport {
    /// 执行追踪信息
    pub execution_traces: ExecutionTrace,

    /// 提取的工作流
    pub workflows: WorkflowExtractionResult,

    /// 代码质量评估（包含对比）
    pub quality: LearningQuality,

    /// 生成的代码
    pub generated_html: Option<String>,
    pub generated_css: Option<String>,
    pub generated_js: Option<String>,

    /// 学习总结
    pub summary: LearningSummary,

    /// 代码验证结果（Phase A）
    pub verification: Option<CodeVerificationResult>,

    /// 语义对比结果（Phase B）
    pub comparison: Option<SemanticComparisonResult>,

    /// 反馈和改进建议（Phase C）
    pub feedback: Option<ComparisonFeedback>,

    /// 总体学习评分 (0-100)
    /// = quality (35%) + verification (25%) + comparison (25%) + confidence (15%)
    pub overall_learning_score: u32,
}

/// 比较学习器
///
/// 这是用户的主入口，提供简洁的API来执行完整的学习和对比流程
pub struct ComparativeLearner;

impl ComparativeLearner {
    /// 🎯 主要API：对比学习一个网站
    ///
    /// 执行完整的流程：
    /// 1. 执行追踪
    /// 2. 工作流提取
    /// 3. 代码生成
    /// 4. 代码验证（Phase A）
    /// 5. 语义对比（Phase B）
    /// 6. 质量评估
    /// 7. 反馈生成（Phase C）
    /// 8. 总体评分（Phase D）
    ///
    /// # 参数
    /// - `original_html`: 原始网站HTML
    /// - `original_css`: 原始网站CSS
    /// - `original_js`: 原始网站JavaScript
    /// - `url`: 网站URL（用于追踪和日志）
    ///
    /// # 返回
    /// 详细的对比学习报告，包含所有阶段的结果
    pub async fn learn_and_compare(
        original_html: String,
        original_css: String,
        original_js: String,
        url: &str,
    ) -> Result<ComparativeLearningReport> {
        log::info!("🔄 开始对比学习: {}", url);

        // Step 1: 创建学习器并生成执行追踪
        let learner = DualSandboxLearner::new()?;
        let traces = ExecutionTrace::new(); // 在实际应用中应从浏览器获取

        // Steps 2-5: 使用DualSandboxLearner执行完整的学习和对比流程
        let learning_result = learner
            .learn_and_generate_with_reference(traces, &original_html, &original_css, &original_js)
            .await?;

        // Step 6: 生成反馈（Phase C）
        let feedback = if let (Some(verification), Some(comparison)) = (
            &learning_result.verification,
            &learning_result.semantic_comparison,
        ) {
            Some(ComparisonFeedbackGenerator::generate(
                verification,
                comparison,
                &learning_result.workflows,
            )?)
        } else {
            None
        };

        // Step 7: 计算总体学习评分（Phase D）
        let overall_score = Self::calculate_overall_score(
            &learning_result.quality,
            learning_result.verification.as_ref(),
            learning_result.semantic_comparison.as_ref(),
            feedback.as_ref(),
        );

        log::info!(
            "✅ 对比学习完成: 总体评分 {}/100, 置信度 {:.1}%",
            overall_score,
            feedback
                .as_ref()
                .map(|f| f.learning_confidence * 100.0)
                .unwrap_or(0.0)
        );

        Ok(ComparativeLearningReport {
            execution_traces: learning_result.traces,
            workflows: learning_result.workflows,
            quality: learning_result.quality,
            generated_html: learning_result.generated_html,
            generated_css: learning_result.generated_css,
            generated_js: learning_result.generated_js,
            summary: learning_result.summary,
            verification: learning_result.verification,
            comparison: learning_result.semantic_comparison,
            feedback,
            overall_learning_score: overall_score,
        })
    }

    /// 批量对比学习多个网站
    pub async fn batch_learn_and_compare(
        websites: Vec<(String, String, String, String)>, // (html, css, js, url)
    ) -> Result<Vec<ComparativeLearningReport>> {
        let mut reports = Vec::new();

        for (html, css, js, url) in websites {
            match Self::learn_and_compare(html, css, js, &url).await {
                Ok(report) => {
                    log::info!(
                        "✓ {} 学习完成: 评分 {}/100",
                        url,
                        report.overall_learning_score
                    );
                    reports.push(report);
                }
                Err(e) => {
                    log::warn!("✗ {} 学习失败: {}", url, e);
                }
            }
        }

        Ok(reports)
    }

    /// 计算总体学习评分
    ///
    /// 加权公式：
    /// = quality (35%) + verification (25%) + comparison (25%) + confidence (15%)
    fn calculate_overall_score(
        quality: &LearningQuality,
        verification: Option<&CodeVerificationResult>,
        comparison: Option<&SemanticComparisonResult>,
        feedback: Option<&ComparisonFeedback>,
    ) -> u32 {
        let mut score = quality.overall_score * 0.35;

        if let Some(v) = verification {
            score += v.verification_score * 0.25;
        }

        if let Some(c) = comparison {
            score += c.overall_similarity * 0.25;
        }

        if let Some(f) = feedback {
            score += f.learning_confidence * 0.15;
        }

        (score * 100.0) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overall_score_calculation() {
        let quality = LearningQuality {
            function_coverage: 0.8,
            workflow_completeness: 0.75,
            functionality_preserved: 0.8,
            overall_score: 0.78,
            semantic_comparison: None,
            code_equivalence_score: None,
            issues: vec![],
            recommendations: vec![],
        };

        let score = ComparativeLearner::calculate_overall_score(&quality, None, None, None);

        // 0.78 * 0.35 * 100 = 27.3
        assert_eq!(score, 27);
    }

    #[test]
    fn test_comparative_learning_report_structure() {
        let report = ComparativeLearningReport {
            execution_traces: ExecutionTrace::new(),
            workflows: crate::workflow_extractor::WorkflowExtractionResult {
                workflows: vec![],
                total_user_interactions: 0,
                total_function_calls: 0,
                coverage_ratio: 0.0,
            },
            quality: LearningQuality {
                function_coverage: 0.8,
                workflow_completeness: 0.75,
                functionality_preserved: 0.8,
                overall_score: 0.78,
                semantic_comparison: None,
                code_equivalence_score: None,
                issues: vec![],
                recommendations: vec![],
            },
            generated_html: Some("<html></html>".to_string()),
            generated_css: Some("body {}".to_string()),
            generated_js: Some("console.log('test');".to_string()),
            summary: LearningSummary {
                workflow_count: 0,
                key_functions: vec![],
                dom_patterns: vec![],
                interaction_flows: vec![],
                overall_score: 78,
            },
            verification: None,
            comparison: None,
            feedback: None,
            overall_learning_score: 27,
        };

        assert_eq!(report.overall_learning_score, 27);
        assert!(report.generated_html.is_some());
    }
}
