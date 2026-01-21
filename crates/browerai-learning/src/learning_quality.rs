/// 学习质量评估模块
///
/// 评估从网站学到的工作流的质量：
/// - 函数覆盖率（学到了多少%的函数）
/// - 工作流完整性（工作流提取的完整程度）
/// - 功能保留度（能否完全重建功能）
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::v8_tracer::ExecutionTrace;
use crate::workflow_extractor::WorkflowExtractionResult;

/// 学习质量评估结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningQuality {
    /// 函数覆盖率（0-1）
    /// 计算：学到的唯一函数数 / 追踪到的唯一函数数
    /// 目标：95%+
    pub function_coverage: f64,

    /// 工作流完整性（0-1）
    /// 完整 = 有清晰的触发点和完整的操作链
    /// 目标：90%+
    pub workflow_completeness: f64,

    /// 功能保留度（0-1）
    /// 能完全重建的功能 / 总功能数
    /// 目标：99%+
    pub functionality_preserved: f64,

    /// 综合评分
    pub overall_score: f64,

    /// 发现的问题
    pub issues: Vec<QualityIssue>,

    /// 改进建议
    pub recommendations: Vec<String>,
}

/// 质量问题
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityIssue {
    pub issue_type: IssueType,
    pub description: String,
    pub severity: Severity,
    pub affected_elements: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum IssueType {
    UntracedFunction,
    IncompleteWorkflow,
    MissingFunctionality,
    IncompleteDOMTracking,
    MissedUserEvent,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Severity {
    Low,
    Medium,
    High,
}

impl LearningQuality {
    /// 评估学习质量
    pub fn evaluate(traces: &ExecutionTrace, workflows: &WorkflowExtractionResult) -> Result<Self> {
        log::info!("开始评估学习质量...");

        let func_coverage = Self::calc_function_coverage(traces, workflows)?;
        let workflow_completeness = Self::calc_workflow_completeness(workflows)?;
        let func_preserved = Self::calc_functionality_preserved(workflows)?;

        let overall_score = (func_coverage + workflow_completeness + func_preserved) / 3.0;

        let issues = Self::identify_issues(traces, workflows)?;
        let recommendations = Self::generate_recommendations(&issues);

        log::info!(
            "✓ 评估完成: 函数覆盖 {:.1}%, 工作流完整 {:.1}%, 功能保留 {:.1}%, 综合 {:.1}%",
            func_coverage * 100.0,
            workflow_completeness * 100.0,
            func_preserved * 100.0,
            overall_score * 100.0
        );

        Ok(LearningQuality {
            function_coverage: func_coverage,
            workflow_completeness,
            functionality_preserved: func_preserved,
            overall_score,
            issues,
            recommendations,
        })
    }

    /// 计算函数覆盖率
    fn calc_function_coverage(
        traces: &ExecutionTrace,
        workflows: &WorkflowExtractionResult,
    ) -> Result<f64> {
        let all_functions: HashSet<String> = traces
            .function_calls
            .iter()
            .map(|call| call.function_name.clone())
            .collect();

        let covered_functions: HashSet<String> = workflows
            .workflows
            .iter()
            .flat_map(|w| w.key_functions.iter().cloned())
            .collect();

        if all_functions.is_empty() {
            return Ok(1.0);
        }

        let coverage = covered_functions.len() as f64 / all_functions.len() as f64;
        Ok(coverage)
    }

    /// 计算工作流完整性
    fn calc_workflow_completeness(workflows: &WorkflowExtractionResult) -> Result<f64> {
        let total = workflows.workflows.len();
        if total == 0 {
            return Ok(1.0);
        }

        let complete = workflows
            .workflows
            .iter()
            .filter(|w| {
                !w.key_functions.is_empty()
                    && !w.operations.function_calls.is_empty()
                    && w.importance_score > 2.0
            })
            .count();

        Ok(complete as f64 / total as f64)
    }

    /// 计算功能保留度
    fn calc_functionality_preserved(workflows: &WorkflowExtractionResult) -> Result<f64> {
        let total = workflows.workflows.len();
        if total == 0 {
            return Ok(1.0);
        }

        let preservable = workflows
            .workflows
            .iter()
            .filter(|w| {
                w.complexity_score >= 2.0
                    && !w.operations.dom_operations.is_empty()
                    && w.importance_score >= 3.0
            })
            .count();

        Ok(preservable as f64 / total as f64)
    }

    /// 识别问题
    fn identify_issues(
        traces: &ExecutionTrace,
        workflows: &WorkflowExtractionResult,
    ) -> Result<Vec<QualityIssue>> {
        let mut issues = vec![];

        let uncovered = Self::find_uncovered_functions(traces, workflows);
        for func in uncovered.iter().take(5) {
            issues.push(QualityIssue {
                issue_type: IssueType::UntracedFunction,
                description: format!("函数 '{}' 未被任何工作流覆盖", func),
                severity: Severity::Medium,
                affected_elements: None,
            });
        }

        for workflow in &workflows.workflows {
            if workflow.operations.function_calls.is_empty() {
                issues.push(QualityIssue {
                    issue_type: IssueType::IncompleteWorkflow,
                    description: format!("工作流 '{}' 没有函数调用", workflow.name),
                    severity: Severity::High,
                    affected_elements: None,
                });
            }

            if workflow.operations.dom_operations.is_empty() {
                issues.push(QualityIssue {
                    issue_type: IssueType::IncompleteDOMTracking,
                    description: format!("工作流 '{}' 没有 DOM 操作", workflow.name),
                    severity: Severity::Medium,
                    affected_elements: None,
                });
            }
        }

        if traces.user_events.is_empty() && !traces.function_calls.is_empty() {
            issues.push(QualityIssue {
                issue_type: IssueType::MissedUserEvent,
                description: "没有识别到用户事件".to_string(),
                severity: Severity::High,
                affected_elements: None,
            });
        }

        Ok(issues)
    }

    fn find_uncovered_functions(
        traces: &ExecutionTrace,
        workflows: &WorkflowExtractionResult,
    ) -> Vec<String> {
        let traced_funcs: HashSet<_> = traces
            .function_calls
            .iter()
            .map(|c| c.function_name.clone())
            .collect();

        let covered_funcs: HashSet<_> = workflows
            .workflows
            .iter()
            .flat_map(|w| w.key_functions.iter().cloned())
            .collect();

        let mut uncovered: Vec<_> = traced_funcs.difference(&covered_funcs).cloned().collect();
        uncovered.sort();
        uncovered
    }

    fn generate_recommendations(issues: &[QualityIssue]) -> Vec<String> {
        let mut recommendations = vec![];

        let high_count = issues
            .iter()
            .filter(|i| i.severity == Severity::High)
            .count();
        if high_count > 0 {
            recommendations.push(format!("存在 {} 个严重问题，建议检查 V8 追踪", high_count));
        }

        let untraced = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::UntracedFunction)
            .count();
        if untraced > 0 {
            recommendations.push(format!(
                "{} 个函数未被覆盖，可能需要更长的执行时间或更多的用户交互",
                untraced
            ));
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v8_tracer::CallRecord;

    #[test]
    fn test_quality_evaluation() {
        let traces = ExecutionTrace::new();
        let result = WorkflowExtractionResult {
            workflows: vec![],
            total_user_interactions: 0,
            total_function_calls: 0,
            coverage_ratio: 0.0,
        };

        let quality = LearningQuality::evaluate(&traces, &result).unwrap();
        assert_eq!(quality.overall_score, 1.0);
    }

    #[test]
    fn test_function_coverage() {
        let mut traces = ExecutionTrace::new();
        traces.function_calls.push(CallRecord {
            function_name: "test".to_string(),
            arguments: vec![],
            return_type: "void".to_string(),
            timestamp_ms: 0,
            context_object: None,
            call_depth: 0,
        });

        let result = WorkflowExtractionResult {
            workflows: vec![],
            total_user_interactions: 0,
            total_function_calls: 1,
            coverage_ratio: 0.0,
        };

        let quality = LearningQuality::evaluate(&traces, &result).unwrap();
        assert!(quality.function_coverage < 1.0);
    }
}
