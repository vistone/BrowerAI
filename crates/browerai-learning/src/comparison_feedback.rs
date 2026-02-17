/// 比较反馈模块 - Phase C
///
/// 基于代码验证和语义对比结果，生成改进策略和反馈
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::code_verifier::CodeVerificationResult;
use crate::semantic_comparator::SemanticComparisonResult;
use crate::workflow_extractor::WorkflowExtractionResult;

/// 代码验证反馈
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeVerificationFeedback {
    /// 解析错误数量
    pub parse_error_count: usize,

    /// 警告数量
    pub warning_count: usize,

    /// 主要问题
    pub main_issues: Vec<String>,

    /// 建议修复
    pub suggested_fixes: Vec<String>,
}

/// 语义对比反馈
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticComparisonFeedback {
    /// 缺失函数数
    pub missing_function_count: usize,

    /// 缺失的函数列表
    pub missing_functions: Vec<String>,

    /// 额外函数数
    pub extra_function_count: usize,

    /// DOM差异程度
    pub dom_difference: String,

    /// 建议的改进方向
    pub improvement_areas: Vec<String>,
}

/// 改进策略
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ImprovementStrategy {
    /// 重新提取工作流（可能遗漏了重要交互）
    ReextractWorkflows,

    /// 增加函数追踪深度
    IncreaseTraceDepth,

    /// 改进代码生成策略
    ImproveCodeGeneration { rules: Vec<String> },

    /// 需要手动审查某些函数
    ManualReviewRequired { functions: Vec<String> },

    /// 使用更高级的模型重新学习
    UseAdvancedModel,
}

/// 完整的比较反馈（Phase C的输出）
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparisonFeedback {
    /// 代码验证反馈
    pub verification_feedback: CodeVerificationFeedback,

    /// 语义对比反馈
    pub comparison_feedback: SemanticComparisonFeedback,

    /// 推荐的改进方向
    pub improvement_strategies: Vec<ImprovementStrategy>,

    /// 学习置信度 (0-1)
    /// 基于: 验证评分 + 对比相似度 + 完整性
    pub learning_confidence: f64,
}

/// 比较反馈生成器
pub struct ComparisonFeedbackGenerator;

impl ComparisonFeedbackGenerator {
    /// 生成完整的比较反馈
    pub fn generate(
        verification: &CodeVerificationResult,
        comparison: &SemanticComparisonResult,
        workflows: &WorkflowExtractionResult,
    ) -> Result<ComparisonFeedback> {
        let verification_fb = Self::analyze_verification(verification);
        let comparison_fb = Self::analyze_comparison(comparison);
        let improvement_strategies =
            Self::recommend_strategies(&verification_fb, &comparison_fb, workflows);

        // 计算学习置信度
        let confidence =
            verification.verification_score * 0.4 + comparison.overall_similarity * 0.6;

        Ok(ComparisonFeedback {
            verification_feedback: verification_fb,
            comparison_feedback: comparison_fb,
            improvement_strategies,
            learning_confidence: confidence,
        })
    }

    /// 分析代码验证结果
    fn analyze_verification(verification: &CodeVerificationResult) -> CodeVerificationFeedback {
        let parse_error_count = verification.all_errors.len();
        let warning_count = verification.html.warnings.len()
            + verification.css.warnings.len()
            + verification.js.warnings.len();

        let mut main_issues = Vec::new();

        if !verification.html.valid {
            main_issues.push(format!(
                "HTML 解析失败: {} 个错误",
                verification.html.parse_errors.len()
            ));
        }
        if !verification.css.valid {
            main_issues.push(format!(
                "CSS 解析失败: {} 个错误",
                verification.css.parse_errors.len()
            ));
        }
        if !verification.js.syntax_valid {
            main_issues.push(format!(
                "JavaScript 语法错误: {} 个错误",
                verification.js.syntax_errors.len()
            ));
        }

        CodeVerificationFeedback {
            parse_error_count,
            warning_count,
            main_issues,
            suggested_fixes: verification
                .suggested_fixes
                .iter()
                .map(|(p, s)| format!("{}: {}", p, s))
                .collect(),
        }
    }

    /// 分析语义对比结果
    fn analyze_comparison(comparison: &SemanticComparisonResult) -> SemanticComparisonFeedback {
        let missing_function_count = comparison.function_similarity.missing_functions.len();
        let extra_function_count = comparison.extra_features.len();

        let mut dom_difference = "无显著差异".to_string();
        if comparison.dom_structure_similarity < 0.5 {
            dom_difference = "DOM结构差异很大".to_string();
        } else if comparison.dom_structure_similarity < 0.7 {
            dom_difference = "DOM结构有中等差异".to_string();
        } else if comparison.dom_structure_similarity < 0.9 {
            dom_difference = "DOM结构有轻微差异".to_string();
        }

        let mut improvement_areas = Vec::new();
        if comparison.event_handling_similarity < 0.8 {
            improvement_areas.push("改进事件处理的覆盖".to_string());
        }
        if comparison.style_similarity < 0.8 {
            improvement_areas.push("改进样式的完整性".to_string());
        }
        if !comparison.missing_features.is_empty() {
            improvement_areas.push("补充缺失的功能".to_string());
        }

        SemanticComparisonFeedback {
            missing_function_count,
            missing_functions: comparison.function_similarity.missing_functions.clone(),
            extra_function_count,
            dom_difference,
            improvement_areas,
        }
    }

    /// 推荐改进策略
    fn recommend_strategies(
        verification_fb: &CodeVerificationFeedback,
        comparison_fb: &SemanticComparisonFeedback,
        workflows: &WorkflowExtractionResult,
    ) -> Vec<ImprovementStrategy> {
        let mut strategies = Vec::new();

        // 1. 如果验证错误过多，建议改进代码生成
        if verification_fb.parse_error_count > 5 {
            strategies.push(ImprovementStrategy::ImproveCodeGeneration {
                rules: vec!["增加错误检查".to_string(), "改进语法正确性".to_string()],
            });
        }

        // 2. 如果缺失很多函数，建议重新提取工作流
        if comparison_fb.missing_function_count > workflows.workflows.len() / 3 {
            strategies.push(ImprovementStrategy::ReextractWorkflows);
        }

        // 3. 如果缺失函数过多，需要手动审查
        if comparison_fb.missing_function_count > 10 {
            strategies.push(ImprovementStrategy::ManualReviewRequired {
                functions: comparison_fb.missing_functions.clone(),
            });
        }

        // 4. 如果工作流覆盖率低，增加追踪深度
        if workflows.coverage_ratio < 0.7 {
            strategies.push(ImprovementStrategy::IncreaseTraceDepth);
        }

        // 5. 如果多个方面都很差，使用高级模型
        if verification_fb.parse_error_count > 10 && comparison_fb.missing_function_count > 5 {
            strategies.push(ImprovementStrategy::UseAdvancedModel);
        }

        strategies
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_improvement_strategy() {
        let strategy = ImprovementStrategy::ImproveCodeGeneration {
            rules: vec!["rule1".to_string(), "rule2".to_string()],
        };
        assert!(matches!(
            strategy,
            ImprovementStrategy::ImproveCodeGeneration { .. }
        ));
    }

    #[test]
    fn test_code_verification_feedback_generation() {
        use crate::code_verifier::VerificationError;
        use crate::code_verifier::{CssVerification, HtmlVerification, JsVerification};

        let verification = CodeVerificationResult {
            html: HtmlVerification {
                valid: true,
                parse_errors: vec![],
                warnings: vec!["warning1".to_string()],
                score: 0.9,
                detected_tags: vec![],
                event_handlers: vec![],
            },
            css: CssVerification {
                valid: true,
                parse_errors: vec![],
                warnings: vec![],
                score: 0.95,
                selectors: vec![],
                properties: vec![],
            },
            js: JsVerification {
                syntax_valid: true,
                syntax_errors: vec![],
                warnings: vec![],
                score: 0.88,
                functions: vec![],
                variables: vec![],
                async_operations: vec![],
                api_calls: vec![],
            },
            verification_score: 0.91,
            all_errors: vec![],
            suggested_fixes: vec![],
        };

        let fb = ComparisonFeedbackGenerator::analyze_verification(&verification);
        assert_eq!(fb.warning_count, 1);
        assert!(fb.main_issues.is_empty());
    }

    #[test]
    fn test_semantic_comparison_feedback() {
        use crate::semantic_comparator::FunctionSimilarity;

        let comparison = SemanticComparisonResult {
            function_similarity: FunctionSimilarity {
                function_scores: HashMap::new(),
                covered_functions: vec!["func1".to_string()],
                missing_functions: vec!["func2".to_string()],
            },
            dom_structure_similarity: 0.8,
            event_handling_similarity: 0.7,
            style_similarity: 0.9,
            overall_similarity: 0.8,
            missing_features: vec!["feature1".to_string()],
            extra_features: vec![],
        };

        let fb = ComparisonFeedbackGenerator::analyze_comparison(&comparison);
        assert_eq!(fb.missing_function_count, 1);
        assert_eq!(fb.extra_function_count, 0);
    }
}
