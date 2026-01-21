/// 生成阶段：改进的代码生成
///
/// 根据推理结果生成高质量的代码：
/// - 生成数据结构定义
/// - 生成工作流函数
/// - 添加类型注解
/// - 添加错误处理
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::complete_inference_pipeline::CompleteInferenceResult;
use crate::data_structure_inference::{InferredStructure, StructureType};

/// 生成的代码模块
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratedModule {
    pub language: String,
    pub module_name: String,
    pub code: String,
    pub dependencies: Vec<String>,
    pub quality_score: f64,
}

/// 代码生成器
pub struct ImprovedCodeGenerator;

impl ImprovedCodeGenerator {
    /// 生成完整代码
    pub fn generate_code(
        inference_result: &CompleteInferenceResult,
    ) -> Result<Vec<GeneratedModule>> {
        log::info!("💻 生成代码模块...");

        let mut modules = vec![];

        // 第1步：生成数据结构定义
        log::info!("  生成数据结构...");
        if let Ok(structure_code) = Self::generate_data_structures(inference_result) {
            modules.push(structure_code);
        }

        // 第2步：生成工作流函数
        log::info!("  生成工作流函数...");
        if let Ok(workflow_code) = Self::generate_workflow_functions(inference_result) {
            modules.push(workflow_code);
        }

        // 第3步：生成主程序入口
        log::info!("  生成程序入口...");
        if let Ok(main_code) = Self::generate_main_module(inference_result) {
            modules.push(main_code);
        }

        log::info!("✓ 生成 {} 个代码模块", modules.len());

        Ok(modules)
    }

    /// 生成数据结构定义
    fn generate_data_structures(
        inference_result: &CompleteInferenceResult,
    ) -> Result<GeneratedModule> {
        let mut code = String::from("// 数据结构定义\n\n");

        for structure in &inference_result.structure_inference.structures {
            code.push_str(&Self::generate_structure_def(structure));
            code.push('\n');
        }

        Ok(GeneratedModule {
            language: "TypeScript".to_string(),
            module_name: "types.ts".to_string(),
            code,
            dependencies: vec![],
            quality_score: inference_result.structure_inference.accuracy,
        })
    }

    fn generate_structure_def(structure: &InferredStructure) -> String {
        let mut code = String::new();

        let keyword = match structure.structure_type {
            StructureType::Interface | StructureType::DTO => "interface",
            _ => "class",
        };

        code.push_str(&format!("export {} {} {{\n", keyword, structure.name));

        for field in &structure.fields {
            let required = if field.is_required { "" } else { "?" };
            code.push_str(&format!(
                "  {}{}: {};\n",
                field.name, required, field.field_type
            ));
        }

        code.push_str("}\n");

        code
    }

    /// 生成工作流函数
    fn generate_workflow_functions(
        inference_result: &CompleteInferenceResult,
    ) -> Result<GeneratedModule> {
        let mut code = String::from("// 工作流函数\n\n");

        for workflow in &inference_result.workflows.workflows {
            code.push_str("/**\n");
            code.push_str(&format!(" * 工作流: {}\n", workflow.name));
            code.push_str(&format!(" * 复杂度: {:.0}/10\n", workflow.complexity_score));
            code.push_str(&format!(" * 重要性: {:.0}/10\n", workflow.importance_score));
            code.push_str(" */\n");

            // 生成函数签名
            code.push_str(&format!(
                "export async function {}(): Promise<void> {{\n",
                workflow.name
            ));

            // 生成函数体
            for (idx, func) in workflow.key_functions.iter().enumerate() {
                if idx == 0 {
                    code.push_str("  try {\n");
                }
                code.push_str(&format!("    // 调用 {}\n", func));
                code.push_str(&format!("    await {}();\n", func));
            }

            code.push_str("  } catch (error) {\n");
            code.push_str(&format!(
                "    console.error('Error in {}:', error);\n",
                workflow.name
            ));
            code.push_str("    throw error;\n");
            code.push_str("  }\n");
            code.push_str("}\n\n");
        }

        Ok(GeneratedModule {
            language: "TypeScript".to_string(),
            module_name: "workflows.ts".to_string(),
            code,
            dependencies: vec!["types.ts".to_string()],
            quality_score: inference_result.learning_quality.overall_score,
        })
    }

    /// 生成主模块
    fn generate_main_module(inference_result: &CompleteInferenceResult) -> Result<GeneratedModule> {
        let mut code = String::from("// 主程序入口\n\n");
        code.push_str("import * as workflows from './workflows';\n");
        code.push_str("import type * as Types from './types';\n\n");

        code.push_str("export class BrowserAIAgent {\n");
        code.push_str("  /**\n");
        code.push_str("   * 初始化代理\n");
        code.push_str("   */\n");
        code.push_str("  constructor() {\n");
        code.push_str("    this.initialize();\n");
        code.push_str("  }\n\n");

        code.push_str("  private initialize(): void {\n");
        code.push_str("    // 初始化代理\n");
        code.push_str("  }\n\n");

        code.push_str("  /**\n");
        code.push_str("   * 执行一个工作流\n");
        code.push_str("   */\n");
        code.push_str("  async executeWorkflow(workflowName: string): Promise<void> {\n");
        code.push_str("    try {\n");
        code.push_str("      const workflow = (workflows as any)[workflowName];\n");
        code.push_str("      if (!workflow) {\n");
        code.push_str("        throw new Error(`Workflow ${workflowName} not found`);\n");
        code.push_str("      }\n");
        code.push_str("      await workflow();\n");
        code.push_str("    } catch (error) {\n");
        code.push_str("      console.error(`Error executing ${workflowName}:`, error);\n");
        code.push_str("      throw error;\n");
        code.push_str("    }\n");
        code.push_str("  }\n");
        code.push_str("}\n\n");

        code.push_str("// 导出代理实例\n");
        code.push_str("export const agent = new BrowserAIAgent();\n");

        Ok(GeneratedModule {
            language: "TypeScript".to_string(),
            module_name: "index.ts".to_string(),
            code,
            dependencies: vec!["workflows.ts".to_string(), "types.ts".to_string()],
            quality_score: inference_result.overall_inference_score,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_def_generation() {
        let structure = InferredStructure {
            name: "User".to_string(),
            structure_type: StructureType::Class,
            fields: vec![],
            occurrences: 1,
            confidence: 0.9,
        };

        let code = ImprovedCodeGenerator::generate_structure_def(&structure);
        assert!(code.contains("export class User"));
    }
}
