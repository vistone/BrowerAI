/// 完整推理管道
///
/// 整合所有推理模块，从学到的知识生成代码生成方案
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::data_structure_inference::DataStructureInferenceEngine;
use crate::learning_quality::LearningQuality;
use crate::v8_tracer::ExecutionTrace;
use crate::variable_semantics::VariableSemanticsAnalyzer;
use crate::workflow_extractor::WorkflowExtractionResult;

/// 完整推理结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompleteInferenceResult {
    /// 学习质量评估
    pub learning_quality: LearningQuality,

    /// 变量语义分析
    pub variable_inference: crate::variable_semantics::InferenceResult,

    /// 数据结构推断
    pub structure_inference: crate::data_structure_inference::StructureInferenceResult,

    /// 工作流信息
    pub workflows: WorkflowExtractionResult,

    /// 综合推理评分
    pub overall_inference_score: f64,

    /// 代码生成建议
    pub code_generation_hints: Vec<CodeGenerationHint>,
}

/// 代码生成建议
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeGenerationHint {
    pub hint_type: HintType,
    pub description: String,
    pub priority: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum HintType {
    DataStructure,
    FunctionSignature,
    ErrorHandling,
    Optimization,
    Refactoring,
}

/// 完整推理管道
pub struct CompleteInferencePipeline;

impl CompleteInferencePipeline {
    /// 执行完整推理
    pub fn infer(
        traces: &ExecutionTrace,
        workflows: &WorkflowExtractionResult,
    ) -> Result<CompleteInferenceResult> {
        log::info!("🧠 执行完整推理...");

        // 第1步：评估学习质量
        log::info!("  Step 1: 评估学习质量");
        let learning_quality = LearningQuality::evaluate(traces, workflows)?;

        // 第2步：分析变量语义
        log::info!("  Step 2: 分析变量语义");
        let variable_inference =
            VariableSemanticsAnalyzer::analyze_variables(traces, &workflows.workflows)?;

        // 第3步：推断数据结构
        log::info!("  Step 3: 推断数据结构");
        let structure_inference =
            DataStructureInferenceEngine::infer_structures(traces, &variable_inference.variables)?;

        // 第4步：生成代码生成建议
        log::info!("  Step 4: 生成代码生成建议");
        let code_generation_hints = Self::generate_code_generation_hints(
            &learning_quality,
            &variable_inference,
            &structure_inference,
            workflows,
        )?;

        // 第5步：计算综合推理评分
        log::info!("  Step 5: 计算综合评分");
        let overall_score = Self::calculate_overall_score(
            &learning_quality,
            &variable_inference,
            &structure_inference,
        );

        log::info!("✓ 推理完成: 综合评分 {:.1}%", overall_score * 100.0);

        Ok(CompleteInferenceResult {
            learning_quality,
            variable_inference,
            structure_inference,
            workflows: workflows.clone(),
            overall_inference_score: overall_score,
            code_generation_hints,
        })
    }

    /// 生成代码生成建议
    fn generate_code_generation_hints(
        quality: &LearningQuality,
        variables: &crate::variable_semantics::InferenceResult,
        structures: &crate::data_structure_inference::StructureInferenceResult,
        workflows: &WorkflowExtractionResult,
    ) -> Result<Vec<CodeGenerationHint>> {
        let mut hints = vec![];

        // 根据数据结构推荐
        if !structures.structures.is_empty() {
            hints.push(CodeGenerationHint {
                hint_type: HintType::DataStructure,
                description: format!("需要生成 {} 个数据类/接口定义", structures.structures.len()),
                priority: 10,
            });
        }

        // 根据工作流生成函数签名建议
        for workflow in &workflows.workflows {
            if workflow.importance_score > 5.0 {
                hints.push(CodeGenerationHint {
                    hint_type: HintType::FunctionSignature,
                    description: format!("工作流 '{}' 需要明确的函数签名", workflow.name),
                    priority: 8,
                });
            }
        }

        // 根据变量推荐类型检查
        let untyped_vars = variables
            .variables
            .iter()
            .filter(|v| v.data_type == crate::variable_semantics::DataType::Unknown)
            .count();

        if untyped_vars > 0 {
            hints.push(CodeGenerationHint {
                hint_type: HintType::Refactoring,
                description: format!("有 {} 个变量需要类型注解", untyped_vars),
                priority: 6,
            });
        }

        // 根据学习质量建议
        if quality.overall_score < 0.8 {
            hints.push(CodeGenerationHint {
                hint_type: HintType::Optimization,
                description: "学习质量不足，建议补充更多测试数据".to_string(),
                priority: 5,
            });
        }

        // 错误处理建议
        hints.push(CodeGenerationHint {
            hint_type: HintType::ErrorHandling,
            description: "添加 try-catch 块用于网络请求和 DOM 操作".to_string(),
            priority: 7,
        });

        hints.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(hints)
    }

    /// 计算综合推理评分
    fn calculate_overall_score(
        quality: &LearningQuality,
        variables: &crate::variable_semantics::InferenceResult,
        structures: &crate::data_structure_inference::StructureInferenceResult,
    ) -> f64 {
        let quality_weight = 0.4;
        let variable_weight = 0.3;
        let structure_weight = 0.3;

        (quality.overall_score * quality_weight)
            + (variables.accuracy * variable_weight)
            + (structures.accuracy * structure_weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overall_score_calculation() {
        let quality = LearningQuality {
            function_coverage: 0.9,
            workflow_completeness: 0.85,
            functionality_preserved: 0.95,
            overall_score: 0.9,
            issues: vec![],
            recommendations: vec![],
            semantic_comparison: None,
            code_equivalence_score: None,
        };

        let variables_result = crate::variable_semantics::InferenceResult {
            variables: vec![],
            dependencies: vec![],
            data_structures: vec![],
            accuracy: 0.85,
        };

        let structures_result = crate::data_structure_inference::StructureInferenceResult {
            structures: vec![],
            relationships: vec![],
            accuracy: 0.8,
        };

        let score = CompleteInferencePipeline::calculate_overall_score(
            &quality,
            &variables_result,
            &structures_result,
        );
        assert!(score > 0.8 && score < 0.95);
    }
}
