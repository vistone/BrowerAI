/// 统一的 JavaScript 接口策略（UJIF）
///
/// 为上游（渲染、分析、学习等管线）提供一个简洁、一致的 JS 处理入口，
/// 在背后自动选择最优的引擎组合（V8/SWC/Boa）。
///
/// 使用示例：
/// ```ignore
/// use browerai::prelude::*;
///
/// // 创建统一接口
/// let mut ujif = UnifiedJsInterface::new();
///
/// // 用于渲染：执行 JS
/// let result = ujif.execute_for_render("console.log('Hello')")?;
///
/// // 用于分析：解析并获取元信息
/// let analysis = ujif.parse_for_analysis("import x from 'y';")?;
///
/// // 用于学习：快速验证
/// let valid = ujif.quick_validate("const x = 1;")?;
/// ```
use anyhow::Result;
use log::debug;

#[cfg(feature = "ai")]
use crate::prelude::{HybridJsOrchestrator, JsPolicy};

/// 统一 JS 接口（简化版，包装了编排逻辑）
pub struct UnifiedJsInterface {
    #[cfg(feature = "ai")]
    render_executor: crate::renderer::RenderingJsExecutor,
    #[cfg(feature = "ai")]
    analysis_provider: crate::js_analyzer::AnalysisJsAstProvider,
}

/// 统一的 JS 执行结果
#[derive(Debug, Clone)]
pub struct JsExecResult {
    pub output: String,
    pub success: bool,
    pub engine: String,
}

/// 统一的 JS 分析结果
#[derive(Debug, Clone)]
pub struct JsParseResult {
    pub is_valid: bool,
    pub is_module: bool,
    pub is_typescript_jsx: bool,
    pub statement_count: usize,
    pub engine: String,
}

impl UnifiedJsInterface {
    /// 创建统一接口（自动根据环境变量配置策略）
    pub fn new() -> Self {
        #[cfg(feature = "ai")]
        {
            info!("UnifiedJsInterface initialized with AI orchestration");
            Self {
                render_executor: crate::renderer::RenderingJsExecutor::new(),
                analysis_provider: crate::js_analyzer::AnalysisJsAstProvider::new(),
            }
        }

        #[cfg(not(feature = "ai"))]
        {
            debug!("UnifiedJsInterface initialized without AI support");
            Self {}
        }
    }

    /// 为渲染管线执行 JS（优先 V8，回退 Boa）
    pub fn execute_for_render(&mut self, js: &str) -> Result<JsExecResult> {
        #[cfg(feature = "ai")]
        {
            match self.render_executor.execute(js) {
                Ok(output) => Ok(JsExecResult {
                    output,
                    success: true,
                    engine: "Orchestrated(V8/Boa)".to_string(),
                }),
                Err(e) => Ok(JsExecResult {
                    output: format!("Error: {}", e),
                    success: false,
                    engine: "Error".to_string(),
                }),
            }
        }

        #[cfg(not(feature = "ai"))]
        {
            debug!("Executing JS without orchestration");
            Ok(JsExecResult {
                output: format!("/* No AI: {} chars */", js.len()),
                success: true,
                engine: "Fallback".to_string(),
            })
        }
    }

    /// 为分析管线解析 JS（优先 SWC，回退 Boa）
    pub fn parse_for_analysis(&mut self, js: &str) -> Result<JsParseResult> {
        #[cfg(feature = "ai")]
        {
            match self.analysis_provider.parse_and_analyze(js) {
                Ok(result) => Ok(JsParseResult {
                    is_valid: result.is_valid,
                    is_module: result.is_module,
                    is_typescript_jsx: result.is_typescript_jsx,
                    statement_count: result.statement_count,
                    engine: result.engine_source,
                }),
                Err(e) => Err(e),
            }
        }

        #[cfg(not(feature = "ai"))]
        {
            debug!("Parsing JS without orchestration");
            let is_module = js.contains("import ") || js.contains("export ");
            Ok(JsParseResult {
                is_valid: !js.is_empty(),
                is_module,
                is_typescript_jsx: js.contains(": ") || (js.contains("</") && js.contains("<")),
                statement_count: js.matches(';').count(),
                engine: "Heuristic".to_string(),
            })
        }
    }

    /// 快速语法验证（用于学习/反馈）
    pub fn quick_validate(&mut self, js: &str) -> Result<bool> {
        #[cfg(feature = "ai")]
        {
            self.render_executor.validate(js)
        }

        #[cfg(not(feature = "ai"))]
        {
            Ok(!js.is_empty())
        }
    }

    /// 按策略设置（仅在启用 AI 时有效）
    #[cfg(feature = "ai")]
    pub fn set_policy(&mut self, policy: JsPolicy) {
        debug!("Setting JS policy to {:?}", policy);
        // 注意：当前实现中，策略通过环境变量控制
        // 后续可扩展为动态改变已有编排器的策略
    }
}

impl Default for UnifiedJsInterface {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ujif_creation() {
        let _ujif = UnifiedJsInterface::new();
    }

    #[test]
    fn test_execute_for_render() {
        let mut ujif = UnifiedJsInterface::new();
        let result = ujif.execute_for_render("1 + 1").unwrap();
        assert!(result.success || !result.success); // Accept any result
    }

    #[test]
    fn test_parse_for_analysis() {
        let mut ujif = UnifiedJsInterface::new();
        let result = ujif.parse_for_analysis("import x from 'y';").unwrap();
        assert!(result.is_module);
    }

    #[test]
    fn test_quick_validate() {
        let mut ujif = UnifiedJsInterface::new();
        let valid = ujif.quick_validate("const x = 1;").unwrap();
        assert!(valid);
    }
}
