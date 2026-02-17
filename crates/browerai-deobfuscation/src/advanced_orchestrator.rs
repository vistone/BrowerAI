//! 高级反混淆编排器 (Advanced Deobfuscation Orchestrator)
//!
//! 整合多种最新技术的完整反混淆流程
//! 支持：符号执行、数据流分析、类型推断、多阶段处理

use crate::{
    DataFlowAnalyzer, EnhancedDeobfuscator, JSUnpackDeobfuscator, SymbolicExecutor, TypeInferencer,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// 高级反混淆管道
pub struct AdvancedDeobfuscationPipeline {
    stages: Vec<PipelineStage>,
    enable_symbolic_execution: bool,
    enable_data_flow_analysis: bool,
    enable_type_inference: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    Unpacking,     // Stage 1: JSUnpack 解包
    Symbolic,      // Stage 2: 符号执行
    DataFlow,      // Stage 3: 数据流分析
    TypeInference, // Stage 4: 类型推断
    EnhancedDeobf, // Stage 5: 增强反混淆
    Optimization,  // Stage 6: 代码优化
}

/// 管道分析结果
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineAnalysisResult {
    pub original_code: String,
    pub unpacked_code: String,
    pub final_code: String,
    pub total_obfuscation_layers: usize,
    pub analysis_summary: AnalysisSummary,
    pub insights: Vec<Insight>,
    pub recommendations: Vec<String>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisSummary {
    pub packer_detected: Option<String>,
    pub variables_count: usize,
    pub functions_count: usize,
    pub critical_variables: Vec<String>,
    pub data_flows_identified: usize,
    pub type_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    pub category: String,
    pub description: String,
    pub severity: String,
    pub evidence: Vec<String>,
}

impl AdvancedDeobfuscationPipeline {
    /// 创建新的管道
    pub fn new() -> Self {
        Self {
            stages: vec![
                PipelineStage::Unpacking,
                PipelineStage::Symbolic,
                PipelineStage::DataFlow,
                PipelineStage::TypeInference,
                PipelineStage::EnhancedDeobf,
                PipelineStage::Optimization,
            ],
            enable_symbolic_execution: true,
            enable_data_flow_analysis: true,
            enable_type_inference: true,
        }
    }

    /// 执行完整的反混淆管道
    pub fn process(&self, code: &str) -> Result<PipelineAnalysisResult> {
        let start = std::time::Instant::now();
        let mut result = PipelineAnalysisResult {
            original_code: code.to_string(),
            ..Default::default()
        };

        let mut current_code = code.to_string();

        // Stage 1: JSUnpack 解包
        log::info!("Stage 1: Unpacking");
        let mut jsunpack = JSUnpackDeobfuscator::new();
        let unpack_result = jsunpack.unpack(&current_code)?;

        result.unpacked_code = unpack_result.code.clone();
        result.total_obfuscation_layers = unpack_result.layers_unpacked;
        result.analysis_summary.packer_detected =
            unpack_result.packer_detected.map(|p| format!("{:?}", p));

        current_code = unpack_result.code;

        // Stage 2: 符号执行
        if self.enable_symbolic_execution {
            log::info!("Stage 2: Symbolic Execution");
            let mut executor = SymbolicExecutor::new();
            let sym_result = executor.analyze(&current_code)?;

            result.analysis_summary.variables_count = sym_result.assignments.len();
            result.analysis_summary.functions_count = sym_result.function_calls.len();

            for decoded in &sym_result.decoded_strings {
                result.insights.push(Insight {
                    category: "Decoded String".to_string(),
                    description: format!(
                        "Found decoded string: {}",
                        if decoded.len() > 50 {
                            format!("{}...", &decoded[..50])
                        } else {
                            decoded.clone()
                        }
                    ),
                    severity: "info".to_string(),
                    evidence: vec![],
                });
            }
        }

        // Stage 3: 数据流分析
        if self.enable_data_flow_analysis {
            log::info!("Stage 3: Data Flow Analysis");
            let mut dfa = DataFlowAnalyzer::new();
            let df_result = dfa.analyze(&current_code)?;

            result.analysis_summary.critical_variables = df_result.critical_variables.clone();
            result.analysis_summary.data_flows_identified = df_result.def_use_chains.len();

            // 检查污染变量
            for taint in &df_result.taints {
                if taint.is_tainted {
                    result.insights.push(Insight {
                        category: "Tainted Variable".to_string(),
                        description: format!(
                            "Variable '{}' is tainted by: {}",
                            taint.variable,
                            taint.taint_sources.join(", ")
                        ),
                        severity: "high".to_string(),
                        evidence: taint.taint_sources.clone(),
                    });
                }
            }
        }

        // Stage 4: 类型推断
        if self.enable_type_inference {
            log::info!("Stage 4: Type Inference");
            let mut type_infer = TypeInferencer::new();
            let type_result = type_infer.infer(&current_code)?;

            // 计算类型推断置信度
            let total_vars = type_result.type_info.len();
            if total_vars > 0 {
                let confident = type_result
                    .type_info
                    .iter()
                    .filter(|t| t.confidence >= 0.8)
                    .count();
                result.analysis_summary.type_confidence = confident as f32 / total_vars as f32;
            }

            for type_info in &type_result.type_info {
                if type_info.confidence >= 0.8 {
                    result.insights.push(Insight {
                        category: "Type Information".to_string(),
                        description: format!(
                            "Variable '{}' is inferred as type '{}' (confidence: {:.0}%)",
                            type_info.variable,
                            type_info.inferred_type,
                            type_info.confidence * 100.0
                        ),
                        severity: "info".to_string(),
                        evidence: type_info.evidence.clone(),
                    });
                }
            }
        }

        // Stage 5: 增强反混淆
        log::info!("Stage 5: Enhanced Deobfuscation");
        let mut enhanced = EnhancedDeobfuscator::new_aggressive();
        let enhanced_result = enhanced.deobfuscate(&current_code)?;
        current_code = enhanced_result.code;

        // Stage 6: 优化
        log::info!("Stage 6: Optimization");
        current_code = self.optimize_code(&current_code)?;

        result.final_code = current_code;

        // 生成建议
        result.recommendations = self.generate_recommendations(&result);

        result.processing_time_ms = start.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// 代码优化
    fn optimize_code(&self, code: &str) -> Result<String> {
        let mut optimized = code.to_string();

        // 移除多余的空格和换行
        optimized = optimized
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        // 移除注释
        let re = regex::Regex::new(r"//.*$")?;
        optimized = re.replace_all(&optimized, "").to_string();

        // 移除多行注释
        let re = regex::Regex::new(r"/\*[\s\S]*?\*/")?;
        optimized = re.replace_all(&optimized, "").to_string();

        Ok(optimized)
    }

    /// 生成建议
    fn generate_recommendations(&self, result: &PipelineAnalysisResult) -> Vec<String> {
        let mut recommendations = vec![];

        // 基于检测到的 packer
        if let Some(packer) = &result.analysis_summary.packer_detected {
            recommendations.push(format!(
                "Code was packed with {}, consider keeping the unpacked version",
                packer
            ));
        }

        // 基于混淆层数
        if result.total_obfuscation_layers > 3 {
            recommendations.push(
                "Multiple obfuscation layers detected. Apply multi-stage deobfuscation."
                    .to_string(),
            );
        }

        // 基于关键变量
        if !result.analysis_summary.critical_variables.is_empty() {
            recommendations.push(format!(
                "Found {} critical variables. Monitor their usage carefully: {}",
                result.analysis_summary.critical_variables.len(),
                result.analysis_summary.critical_variables
                    [..3.min(result.analysis_summary.critical_variables.len())]
                    .join(", ")
            ));
        }

        // 基于类型推断信心
        if result.analysis_summary.type_confidence < 0.5 {
            recommendations
                .push("Low type inference confidence. Code may use advanced patterns.".to_string());
        }

        recommendations
    }

    /// 获取管道信息
    pub fn get_pipeline_info(&self) -> String {
        format!(
            "Advanced Deobfuscation Pipeline\n\
             Stages: {}\n\
             Symbolic Execution: {}\n\
             Data Flow Analysis: {}\n\
             Type Inference: {}",
            self.stages.len(),
            self.enable_symbolic_execution,
            self.enable_data_flow_analysis,
            self.enable_type_inference
        )
    }
}

impl Default for AdvancedDeobfuscationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = AdvancedDeobfuscationPipeline::new();
        assert_eq!(pipeline.stages.len(), 6);
        assert!(pipeline.enable_symbolic_execution);
        assert!(pipeline.enable_data_flow_analysis);
        assert!(pipeline.enable_type_inference);
    }

    #[test]
    fn test_pipeline_info() {
        let pipeline = AdvancedDeobfuscationPipeline::new();
        let info = pipeline.get_pipeline_info();
        assert!(info.contains("Advanced Deobfuscation Pipeline"));
        assert!(info.contains("Stages:"));
    }

    #[test]
    fn test_simple_code_processing() {
        let pipeline = AdvancedDeobfuscationPipeline::new();
        let code = "var x = 'hello'; console.log(x);";

        let result = pipeline.process(code).unwrap();
        assert_eq!(result.original_code, code);
        assert!(!result.final_code.is_empty());
        assert!(result.processing_time_ms > 0);
    }
}
