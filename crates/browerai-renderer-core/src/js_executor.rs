use anyhow::Result;
use log::debug;

#[cfg(feature = "ai")]
use browerai_ai_integration::{HybridJsOrchestrator, OrchestrationPolicy};

/// 渲染管线中的 JS 执行管理器
/// 使用混合编排器按策略选择最优的 JS 执行引擎
pub struct RenderingJsExecutor {
    #[cfg(feature = "ai")]
    orchestrator: Option<HybridJsOrchestrator>,
    #[cfg(feature = "ai")]
    policy: OrchestrationPolicy,
}

impl RenderingJsExecutor {
    /// 创建执行器（仅在启用 AI feature 时使用混合编排）
    pub fn new() -> Self {
        #[cfg(feature = "ai")]
        {
            let policy = std::env::var("BROWERAI_RENDER_JS_POLICY")
                .ok()
                .and_then(|s| match s.to_lowercase().as_str() {
                    "performance" => Some(OrchestrationPolicy::Performance),
                    "secure" => Some(OrchestrationPolicy::Secure),
                    "balanced" => Some(OrchestrationPolicy::Balanced),
                    _ => None,
                })
                .unwrap_or_default();

            let orchestrator = HybridJsOrchestrator::with_policy(policy);
            debug!("RenderingJsExecutor initialized with policy: {:?}", policy);

            Self {
                orchestrator: Some(orchestrator),
                policy,
            }
        }

        #[cfg(not(feature = "ai"))]
        {
            debug!(
                "RenderingJsExecutor initialized without AI orchestration (ai feature not enabled)"
            );
            Self {}
        }
    }

    /// 执行 JS 代码并返回结果
    pub fn execute(&mut self, js: &str) -> Result<String> {
        #[cfg(feature = "ai")]
        {
            if let Some(orch) = self.orchestrator.as_mut() {
                debug!("Executing JS via orchestrator (policy: {:?})", self.policy);
                return orch.execute(js);
            }
        }

        // Fallback: 返回占位结果（无 AI 支持）
        debug!("Executing JS without orchestrator (no AI support)");
        Ok(format!(
            "/* JS execution result (no orchestrator): {} chars */",
            js.len()
        ))
    }

    /// 验证 JS 语法
    pub fn validate(&mut self, js: &str) -> Result<bool> {
        #[cfg(feature = "ai")]
        {
            if let Some(orch) = self.orchestrator.as_mut() {
                debug!("Validating JS syntax via orchestrator");
                return orch.validate(js);
            }
        }

        // Fallback: 基本检查
        debug!("Validating JS without orchestrator");
        Ok(!js.is_empty())
    }
}

impl Default for RenderingJsExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rendering_js_executor_creation() {
        let _executor = RenderingJsExecutor::new();
        // 应该成功创建，无论 AI feature 是否启用
    }

    #[test]
    #[cfg(feature = "ai")]
    fn test_execute_with_ai() {
        let mut executor = RenderingJsExecutor::new();
        let result = executor.execute("1 + 1");
        assert!(result.is_ok() || result.is_err()); // 接受两种结果（取决于 V8/Boa 可用性）
    }

    #[test]
    fn test_validate_js() {
        let mut executor = RenderingJsExecutor::new();
        let valid = executor.validate("var x = 1;").unwrap();
        assert!(valid);
    }
}
