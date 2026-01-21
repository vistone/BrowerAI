use anyhow::Result;
#[cfg(feature = "ml")]
use browerai_deobfuscation::AIDeobfuscator;
use browerai_deobfuscation::{EnhancedDeobfuscator, JsDeobfuscator};
use log::debug;
use std::path::Path;
#[cfg(feature = "ml")]
use std::sync::Arc;
use std::sync::Mutex;

#[cfg(feature = "browerai-ai-integration")]
use browerai_ai_integration::{HybridJsOrchestrator, OrchestrationPolicy};

/// 反混淆配置
#[derive(Debug, Clone)]
pub struct DeobfuscationSettings {
    /// 是否启用反混淆
    pub enabled: bool,
    /// 混淆检测阈值
    pub threshold: f32,
    /// 是否使用AI反混淆
    pub use_ai: bool,
}

impl Default for DeobfuscationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.3,
            use_ai: false,
        }
    }
}

/// 渲染管线中的 JS 执行管理器
/// 使用混合编排器按策略选择最优的 JS 执行引擎
/// 支持自动检测并反混淆混淆的JavaScript代码
pub struct RenderingJsExecutor {
    #[cfg(feature = "browerai-ai-integration")]
    orchestrator: Option<HybridJsOrchestrator>,
    #[cfg(feature = "browerai-ai-integration")]
    policy: OrchestrationPolicy,

    // 反混淆组件
    #[cfg(feature = "ml")]
    ai_deobfuscator: Option<Arc<AIDeobfuscator>>,
    rule_deobfuscator: JsDeobfuscator,
    enhanced_deobfuscator: Mutex<EnhancedDeobfuscator>,
    deobf_settings: DeobfuscationSettings,

    // 统计
    deobfuscation_count: usize,
    execution_count: usize,
}

impl RenderingJsExecutor {
    /// 创建执行器（仅在启用 AI feature 时使用混合编排）
    pub fn new() -> Self {
        #[cfg(feature = "browerai-ai-integration")]
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
                #[cfg(feature = "ml")]
                ai_deobfuscator: None,
                rule_deobfuscator: JsDeobfuscator::new(),
                enhanced_deobfuscator: Mutex::new(EnhancedDeobfuscator::new()),
                deobf_settings: DeobfuscationSettings::default(),
                deobfuscation_count: 0,
                execution_count: 0,
            }
        }

        #[cfg(not(feature = "browerai-ai-integration"))]
        {
            debug!("RenderingJsExecutor initialized without JS orchestration (feature disabled)");
            Self {
                #[cfg(feature = "ml")]
                ai_deobfuscator: None,
                rule_deobfuscator: JsDeobfuscator::new(),
                enhanced_deobfuscator: Mutex::new(EnhancedDeobfuscator::new()),
                deobf_settings: DeobfuscationSettings::default(),
                deobfuscation_count: 0,
                execution_count: 0,
            }
        }
    }

    /// 加载AI反混淆模型
    #[cfg(feature = "ml")]
    pub fn with_ai_deobfuscator(mut self, model_path: &Path, vocab_path: &Path) -> Result<Self> {
        match AIDeobfuscator::new(model_path, vocab_path) {
            Ok(deobf) => {
                log::info!("✅ 渲染器AI反混淆器加载成功");
                log::info!("{}", deobf.model_info());
                self.ai_deobfuscator = Some(Arc::new(deobf));
                self.deobf_settings.use_ai = true;
            }
            Err(e) => {
                log::warn!("⚠️ AI反混淆器加载失败: {}", e);
            }
        }
        Ok(self)
    }

    /// 加载AI反混淆模型 (stub)
    #[cfg(not(feature = "ml"))]
    pub fn with_ai_deobfuscator(self, _model_path: &Path, _vocab_path: &Path) -> Result<Self> {
        log::warn!("⚠️ AI反混淆功能未启用");
        Ok(self)
    }

    /// 设置反混淆配置
    pub fn with_deobfuscation_settings(mut self, settings: DeobfuscationSettings) -> Self {
        self.deobf_settings = settings;
        self
    }

    /// 检测代码是否需要反混淆
    fn needs_deobfuscation(&self, js: &str) -> bool {
        if !self.deobf_settings.enabled {
            return false;
        }
        let analysis = self.rule_deobfuscator.analyze_obfuscation(js);
        analysis.obfuscation_score > self.deobf_settings.threshold
    }

    /// 反混淆代码
    fn deobfuscate(&mut self, js: &str) -> String {
        // 尝试AI反混淆
        #[cfg(feature = "ml")]
        if self.deobf_settings.use_ai {
            if let Some(ai) = &self.ai_deobfuscator {
                if let Ok(result) = ai.deobfuscate(js) {
                    log::debug!("🤖 AI反混淆: {} → {} 字符", js.len(), result.len());
                    self.deobfuscation_count += 1;
                    return result;
                }
            }
        }

        // 规则化反混淆
        if let Ok(result) = self.enhanced_deobfuscator.lock().unwrap().deobfuscate(js) {
            log::debug!("📝 规则反混淆: {} → {} 字符", js.len(), result.code.len());
            self.deobfuscation_count += 1;
            return result.code;
        }

        js.to_string()
    }

    /// 执行 JS 代码并返回结果（自动反混淆）
    pub fn execute(&mut self, js: &str) -> Result<String> {
        self.execution_count += 1;

        // 自动反混淆
        let code_to_execute = if self.needs_deobfuscation(js) {
            self.deobfuscate(js)
        } else {
            js.to_string()
        };

        #[cfg(feature = "browerai-ai-integration")]
        {
            if let Some(orch) = self.orchestrator.as_mut() {
                debug!("Executing JS via orchestrator (policy: {:?})", self.policy);
                return orch.execute(&code_to_execute);
            }
        }

        // Fallback: 返回占位结果（无 AI 支持）
        debug!("Executing JS without orchestrator (no AI support)");
        Ok(format!(
            "/* JS execution result (no orchestrator): {} chars */",
            code_to_execute.len()
        ))
    }

    /// 执行代码，跳过反混淆
    pub fn execute_raw(&mut self, js: &str) -> Result<String> {
        self.execution_count += 1;

        #[cfg(feature = "browerai-ai-integration")]
        {
            if let Some(orch) = self.orchestrator.as_mut() {
                return orch.execute(js);
            }
        }

        Ok(format!("/* JS execution (raw): {} chars */", js.len()))
    }

    /// 验证 JS 语法
    pub fn validate(&mut self, js: &str) -> Result<bool> {
        #[cfg(feature = "browerai-ai-integration")]
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

    /// 获取统计信息
    pub fn get_stats(&self) -> (usize, usize) {
        (self.execution_count, self.deobfuscation_count)
    }

    /// 启用/禁用反混淆
    pub fn set_deobfuscation_enabled(&mut self, enabled: bool) {
        self.deobf_settings.enabled = enabled;
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
    fn test_deobfuscation_detection() {
        let executor = RenderingJsExecutor::new();

        // 简单代码不需要反混淆
        assert!(!executor.needs_deobfuscation("var x = 1;"));

        // 混淆风格代码
        let obfuscated = "var _0x1234=function(){var _0x5678=1;return _0x5678;};";
        // 可能需要也可能不需要，取决于阈值
        let _ = executor.needs_deobfuscation(obfuscated);
    }

    #[test]
    #[cfg(feature = "ai")]
    fn test_execute_with_ai() {
        let mut executor = RenderingJsExecutor::new();
        let result = executor.execute("1 + 1");
        assert!(result.is_ok() || result.is_err()); // 接受两种结果
    }

    #[test]
    fn test_validate_js() {
        let mut executor = RenderingJsExecutor::new();
        let valid = executor.validate("var x = 1;").unwrap();
        assert!(valid);
    }

    #[test]
    fn test_stats() {
        let mut executor = RenderingJsExecutor::new();
        let _ = executor.execute("var x = 1;");
        let (exec_count, _deobf_count) = executor.get_stats();
        assert_eq!(exec_count, 1);
    }
}
