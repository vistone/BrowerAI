use anyhow::{Context, Result};
use log::{debug, info, warn};

use crate::JsParser as BoaJsParser;
use browerai_js_v8::V8JsParser;
use browerai_js_analyzer::{EnhancedAst, SwcAstExtractor};

/// 统一的JS运行时门面，按策略选择性调用V8、SWC和Boa
///
/// 策略目标：
/// - 解析（AST）：优先SWC（支持TS/JSX/模块），回退Boa（脚本模式）
/// - 执行：优先V8（性能与兼容性），回退Boa（更安全的Rust沙箱）
/// - 验证：按解析器选择
pub struct HybridJsOrchestrator {
    policy: OrchestrationPolicy,
    boa: BoaJsParser,
    v8: Option<V8JsParser>,
    swc: SwcAstExtractor,
}

/// 选择策略（可通过环境变量覆盖）
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum OrchestrationPolicy {
    /// 性能优先：尽量走V8+SWC，失败再回退Boa
    Performance,
    /// 安全优先：默认Boa执行，SWC解析；仅在明确需要时使用V8
    Secure,
    /// 平衡模式：根据代码特征动态选择
    Balanced,
}

impl Default for OrchestrationPolicy {
    fn default() -> Self {
        OrchestrationPolicy::Balanced
    }
}

impl HybridJsOrchestrator {
    /// 创建带默认策略的编排器
    pub fn new() -> Self {
        let policy = std::env::var("BROWERAI_JS_POLICY")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "performance" => Some(OrchestrationPolicy::Performance),
                "secure" => Some(OrchestrationPolicy::Secure),
                "balanced" => Some(OrchestrationPolicy::Balanced),
                _ => None,
            })
            .unwrap_or_default();

        let v8 = match V8JsParser::new() {
            Ok(p) => Some(p),
            Err(e) => {
                warn!("V8 init failed, will fallback to Boa for exec: {}", e);
                None
            }
        };

        Self {
            policy,
            boa: BoaJsParser::new(),
            v8,
            swc: SwcAstExtractor::new(),
        }
    }

    /// 使用指定策略创建
    pub fn with_policy(policy: OrchestrationPolicy) -> Self {
        let v8 = match V8JsParser::new() {
            Ok(p) => Some(p),
            Err(e) => {
                warn!("V8 init failed, will fallback to Boa for exec: {}", e);
                None
            }
        };

        Self {
            policy,
            boa: BoaJsParser::new(),
            v8,
            swc: SwcAstExtractor::new(),
        }
    }

    /// 解析并返回统一的轻量AST元信息（来源可能是SWC或Boa）
    pub fn parse(&mut self, js: &str) -> Result<UnifiedAst> {
        // 首先检测特征
        let features = JsFeatures::detect(js);
        debug!("JS features: {:?}", features);

        // 选择解析器
        if self.should_use_swc(&features) {
            match self.swc.extract_from_source(js) {
                Ok(extracted) => {
                    info!("Parsed via SWC (module/tsx/ts supported)");
                    return Ok(UnifiedAst::from_enhanced(&extracted));
                }
                Err(e) => {
                    warn!("SWC parse failed: {}. Fallback to Boa.", e);
                }
            }
        }

        // Boa回退（脚本模式）
        let ast = self.boa.parse(js)?;
        Ok(UnifiedAst {
            is_valid: ast.is_valid,
            statement_count: ast.statement_count,
            source_kind: SourceKind::Script,
            engine: AstEngine::Boa,
        })
    }

    /// 语法验证（按解析器选择）
    pub fn validate(&mut self, js: &str) -> Result<bool> {
        match self.parse(js) {
            Ok(ast) => Ok(ast.is_valid),
            Err(_) => Ok(false),
        }
    }

    /// 执行代码（优先V8，必要时回退Boa）
    pub fn execute(&mut self, js: &str) -> Result<String> {
        let features = JsFeatures::detect(js);

        // 根据策略选择执行引擎
        match self.policy {
            OrchestrationPolicy::Secure => {
                // 安全优先，尽量在Rust的Boa沙箱中执行
                match self.execute_with_boa(js) {
                    Ok(v) => return Ok(v),
                    Err(e) => {
                        warn!("Boa exec failed: {}. Trying V8 as fallback.", e);
                        return self.execute_with_v8(js);
                    }
                }
            }
            OrchestrationPolicy::Performance | OrchestrationPolicy::Balanced => {
                // 先尝试V8
                match self.execute_with_v8(js) {
                    Ok(v) => return Ok(v),
                    Err(e) => {
                        warn!("V8 exec failed: {}. Fallback to Boa.", e);
                        return self.execute_with_boa(js);
                    }
                }
            }
        }
    }

    fn execute_with_v8(&mut self, js: &str) -> Result<String> {
        if let Some(v8) = self.v8.as_mut() {
            v8.execute(js)
        } else {
            Err(anyhow::anyhow!("V8 not available"))
        }
    }

    fn execute_with_boa(&mut self, js: &str) -> Result<String> {
        // 简化：Boa执行返回固定字符串，未来可扩展到boa_engine执行
        // 这里保留安全沙箱的占位实现
        if self.boa.validate(js)? {
            Ok("<ok>".to_string())
        } else {
            Err(anyhow::anyhow!("Boa validation failed"))
        }
    }

    fn should_use_swc(&self, features: &JsFeatures) -> bool {
        match self.policy {
            OrchestrationPolicy::Performance => true,
            OrchestrationPolicy::Secure => features.has_modules || features.has_ts || features.has_jsx,
            OrchestrationPolicy::Balanced => {
                features.has_modules || features.has_ts || features.has_jsx
            }
        }
    }
}

/// 统一的AST元信息
#[derive(Debug, Clone)]
pub struct UnifiedAst {
    pub is_valid: bool,
    pub statement_count: usize,
    pub source_kind: SourceKind,
    pub engine: AstEngine,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AstEngine {
    Swc,
    Boa,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SourceKind {
    Script,
    Module,
    TsOrTsx,
}

impl UnifiedAst {
    fn from_enhanced(extracted: &EnhancedAst) -> Self {
        let mut kind = SourceKind::Script;
        if extracted.metadata.is_module {
            kind = SourceKind::Module;
        }
        if extracted.has_typescript || extracted.has_jsx {
            kind = SourceKind::TsOrTsx;
        }

        Self {
            is_valid: extracted.metadata.is_valid,
            statement_count: extracted.metadata.statement_count,
            source_kind: kind,
            engine: AstEngine::Swc,
        }
    }
}

/// 代码特征检测（轻量启发式）
#[derive(Debug, Default, Clone)]
pub struct JsFeatures {
    pub has_modules: bool,
    pub has_dynamic_import: bool,
    pub has_ts: bool,
    pub has_jsx: bool,
}

impl JsFeatures {
    pub fn detect(src: &str) -> Self {
        let has_modules = src.contains("import ") || src.contains("export ");
        let has_dynamic_import = src.contains("import(");
        let has_ts = src.contains(": ") || src.contains("interface ") || src.contains("type ");
        let has_jsx = src.contains("</") && src.contains("<") && src.contains(">");

        Self {
            has_modules,
            has_dynamic_import,
            has_ts,
            has_jsx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_parse_swc_module() {
        let mut orch = HybridJsOrchestrator::with_policy(OrchestrationPolicy::Balanced);
        let code = "import x from 'y'; export const a = 1;";
        let ast = orch.parse(code).unwrap();
        assert!(ast.is_valid);
        assert_eq!(ast.engine, AstEngine::Swc);
        assert!(matches!(ast.source_kind, SourceKind::Module));
    }

    #[test]
    fn test_orchestrator_parse_boa_script() {
        let mut orch = HybridJsOrchestrator::with_policy(OrchestrationPolicy::Balanced);
        let code = "function a() { return 1 }";
        let ast = orch.parse(code).unwrap();
        assert!(ast.is_valid);
        assert_eq!(ast.engine, AstEngine::Boa);
        assert!(matches!(ast.source_kind, SourceKind::Script));
    }

    #[test]
    fn test_orchestrator_execute_v8_first() {
        let mut orch = HybridJsOrchestrator::with_policy(OrchestrationPolicy::Performance);
        let result = orch.execute("1 + 2").unwrap();
        assert_eq!(result, "3");
    }
}
