use anyhow::Result;
use log::debug;

/// JS 分析管线中的 AST 提取提供器
///
/// 注意：此模块提供纯启发式实现，以避免循环依赖。
///
/// 依赖关系：
/// - `browerai-ai-integration` depends on `browerai-js-analyzer`（获取 SwcAstExtractor）
/// - `browerai-js-analyzer` 不应该反向依赖 `browerai-ai-integration`
///
/// 为了支持混合编排器的 AST 分析，真正的集成应该在上层（如 browerai/unified_js.rs）进行。
pub struct AnalysisJsAstProvider;

impl AnalysisJsAstProvider {
    /// 创建 AST 提供器
    pub fn new() -> Self {
        debug!("AnalysisJsAstProvider initialized (using heuristic implementation)");
        Self
    }

    /// 解析 JS 并获取统一 AST 信息（包括模块判定）
    /// 使用启发式方法判定模块类型和 TypeScript/JSX 特征
    pub fn parse_and_analyze(&self, js: &str) -> Result<JsAnalysisResult> {
        debug!("Analyzing JS with heuristic method");

        let is_module = js.contains("import ") || js.contains("export ");
        let is_typescript_jsx = js.contains(": ") || (js.contains("</") && js.contains("<"));
        let statement_count = js.matches(';').count();

        Ok(JsAnalysisResult {
            is_valid: !js.is_empty(),
            statement_count,
            is_module,
            is_typescript_jsx,
            engine_source: "Heuristic".to_string(),
        })
    }

    /// 验证 JS 语法
    pub fn validate(&self, js: &str) -> Result<bool> {
        debug!("Validating JS with heuristic method");
        Ok(!js.is_empty())
    }
}

impl Default for AnalysisJsAstProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// JS 分析结果（统一格式）
#[derive(Debug, Clone)]
pub struct JsAnalysisResult {
    /// 代码是否有效
    pub is_valid: bool,

    /// 语句数
    pub statement_count: usize,

    /// 是否为 ES 模块
    pub is_module: bool,

    /// 是否包含 TypeScript 或 JSX
    pub is_typescript_jsx: bool,

    /// 使用的解析引擎来源（Swc/Boa/Heuristic）
    pub engine_source: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_ast_provider_creation() {
        let _provider = AnalysisJsAstProvider::new();
    }

    #[test]
    fn test_parse_and_analyze() {
        let provider = AnalysisJsAstProvider::new();
        let result = provider.parse_and_analyze("import x from 'y'; export const a = 1;");
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert!(analysis.is_module);
    }

    #[test]
    fn test_validate_js() {
        let provider = AnalysisJsAstProvider::new();
        let valid = provider.validate("const x = 1;").unwrap();
        assert!(valid);
    }
}
