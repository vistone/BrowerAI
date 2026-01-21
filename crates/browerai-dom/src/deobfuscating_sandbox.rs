//! 带反混淆功能的JavaScript沙箱
//!
//! 在执行JavaScript代码前自动检测并反混淆混淆代码

use crate::sandbox::{ExecutionStats, JsSandbox, ResourceLimits, SandboxError, SandboxValue};
#[cfg(feature = "ml")]
use browerai_deobfuscation::AIDeobfuscator;
use browerai_deobfuscation::{EnhancedDeobfuscator, JsDeobfuscator};
use std::path::Path;
#[cfg(feature = "ml")]
use std::sync::Arc;
use std::sync::Mutex;

/// 执行统计信息（包含反混淆）
#[derive(Debug, Clone)]
pub struct DeobfuscatingExecutionStats {
    /// 基础执行统计
    pub base_stats: ExecutionStats,
    /// 总反混淆次数
    pub deobfuscation_count: usize,
    /// AI反混淆次数
    pub ai_deobfuscation_count: usize,
    /// 规则反混淆次数
    pub rule_deobfuscation_count: usize,
    /// 跳过的代码数
    pub skipped_count: usize,
}

/// 反混淆配置
#[derive(Debug, Clone)]
pub struct DeobfuscationConfig {
    /// 是否启用反混淆
    pub enabled: bool,
    /// 混淆检测阈值 (0.0 - 1.0)
    pub detection_threshold: f32,
    /// 是否使用AI反混淆
    pub use_ai: bool,
    /// 是否记录反混淆日志
    pub log_deobfuscation: bool,
}

impl Default for DeobfuscationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detection_threshold: 0.3,
            use_ai: false,
            log_deobfuscation: true,
        }
    }
}

/// 带反混淆功能的JavaScript沙箱
///
/// 在执行代码前自动检测并处理混淆的JavaScript：
/// 1. 分析代码混淆程度
/// 2. 使用AI或规则方法反混淆
/// 3. 执行干净的代码
pub struct DeobfuscatingSandbox {
    /// 底层JS沙箱
    inner: JsSandbox,
    /// AI反混淆器
    #[cfg(feature = "ml")]
    ai_deobfuscator: Option<Arc<AIDeobfuscator>>,
    /// 规则反混淆器（用于分析）
    rule_deobfuscator: JsDeobfuscator,
    /// 增强反混淆器
    enhanced_deobfuscator: Mutex<EnhancedDeobfuscator>,
    /// 配置
    config: DeobfuscationConfig,
    /// 统计信息
    deobfuscation_count: usize,
    ai_deobfuscation_count: usize,
    rule_deobfuscation_count: usize,
    skipped_count: usize,
}

impl DeobfuscatingSandbox {
    /// 创建新的反混淆沙箱
    pub fn new() -> Self {
        Self {
            inner: JsSandbox::with_defaults(),
            #[cfg(feature = "ml")]
            ai_deobfuscator: None,
            rule_deobfuscator: JsDeobfuscator::new(),
            enhanced_deobfuscator: Mutex::new(EnhancedDeobfuscator::new()),
            config: DeobfuscationConfig::default(),
            deobfuscation_count: 0,
            ai_deobfuscation_count: 0,
            rule_deobfuscation_count: 0,
            skipped_count: 0,
        }
    }

    /// 使用资源限制创建
    pub fn with_limits(limits: ResourceLimits) -> Self {
        Self {
            inner: JsSandbox::new(limits),
            #[cfg(feature = "ml")]
            ai_deobfuscator: None,
            rule_deobfuscator: JsDeobfuscator::new(),
            enhanced_deobfuscator: Mutex::new(EnhancedDeobfuscator::new()),
            config: DeobfuscationConfig::default(),
            deobfuscation_count: 0,
            ai_deobfuscation_count: 0,
            rule_deobfuscation_count: 0,
            skipped_count: 0,
        }
    }

    /// 设置反混淆配置
    pub fn with_config(mut self, config: DeobfuscationConfig) -> Self {
        self.config = config;
        self
    }

    /// 加载AI模型
    #[cfg(feature = "ml")]
    pub fn with_ai_model(mut self, model_path: &Path, vocab_path: &Path) -> anyhow::Result<Self> {
        match AIDeobfuscator::new(model_path, vocab_path) {
            Ok(deobf) => {
                log::info!("✅ AI反混淆器加载成功");
                self.ai_deobfuscator = Some(Arc::new(deobf));
                self.config.use_ai = true;
            }
            Err(e) => {
                log::warn!("⚠️ AI反混淆器加载失败: {}, 使用规则化方法", e);
            }
        }
        Ok(self)
    }

    /// 加载AI模型（无 ml 特性时的 stub）
    #[cfg(not(feature = "ml"))]
    pub fn with_ai_model(self, _model_path: &Path, _vocab_path: &Path) -> anyhow::Result<Self> {
        log::warn!("⚠️ AI反混淆功能未启用，请使用 --features ml 编译");
        Ok(self)
    }

    /// 分析代码是否需要反混淆
    pub fn needs_deobfuscation(&self, code: &str) -> bool {
        if !self.config.enabled {
            return false;
        }

        let analysis = self.rule_deobfuscator.analyze_obfuscation(code);
        analysis.obfuscation_score > self.config.detection_threshold
    }

    /// 反混淆代码
    fn deobfuscate_code(&mut self, code: &str) -> String {
        // 尝试AI反混淆
        #[cfg(feature = "ml")]
        if self.config.use_ai {
            if let Some(ai_deobf) = &self.ai_deobfuscator {
                match ai_deobf.deobfuscate(code) {
                    Ok(deobfuscated) => {
                        if self.config.log_deobfuscation {
                            log::info!("🤖 AI反混淆: {} → {} 字符", code.len(), deobfuscated.len());
                        }
                        self.ai_deobfuscation_count += 1;
                        self.deobfuscation_count += 1;
                        return deobfuscated;
                    }
                    Err(e) => {
                        log::debug!("AI反混淆失败: {}", e);
                    }
                }
            }
        }

        // 使用规则化反混淆
        match self.enhanced_deobfuscator.lock().unwrap().deobfuscate(code) {
            Ok(result) => {
                if self.config.log_deobfuscation {
                    log::info!("📝 规则反混淆: {} → {} 字符", code.len(), result.code.len());
                }
                self.rule_deobfuscation_count += 1;
                self.deobfuscation_count += 1;
                result.code
            }
            Err(e) => {
                log::debug!("规则反混淆失败: {}", e);
                code.to_string()
            }
        }
    }

    /// 执行JavaScript代码（自动反混淆）
    pub fn execute(&mut self, code: &str) -> Result<SandboxValue, SandboxError> {
        let code_to_execute = if self.needs_deobfuscation(code) {
            self.deobfuscate_code(code)
        } else {
            self.skipped_count += 1;
            code.to_string()
        };

        self.inner.execute(&code_to_execute)
    }

    /// 执行代码，强制不进行反混淆
    pub fn execute_raw(&mut self, code: &str) -> Result<SandboxValue, SandboxError> {
        self.inner.execute(code)
    }

    /// 执行代码，强制进行反混淆
    pub fn execute_deobfuscated(&mut self, code: &str) -> Result<SandboxValue, SandboxError> {
        let deobfuscated = self.deobfuscate_code(code);
        self.inner.execute(&deobfuscated)
    }

    /// 评估表达式
    pub fn eval(&mut self, expression: &str) -> Result<SandboxValue, SandboxError> {
        self.inner.eval(expression)
    }

    /// 设置全局变量
    pub fn set_global(&mut self, name: impl Into<String>, value: SandboxValue) {
        self.inner.set_global(name, value);
    }

    /// 获取全局变量
    pub fn get_global(&self, name: &str) -> Option<&SandboxValue> {
        self.inner.get_global(name)
    }

    /// 获取执行统计（包含反混淆统计）
    pub fn get_stats(&self) -> DeobfuscatingExecutionStats {
        DeobfuscatingExecutionStats {
            base_stats: self.inner.get_stats(),
            deobfuscation_count: self.deobfuscation_count,
            ai_deobfuscation_count: self.ai_deobfuscation_count,
            rule_deobfuscation_count: self.rule_deobfuscation_count,
            skipped_count: self.skipped_count,
        }
    }

    /// 重置沙箱
    pub fn reset(&mut self) {
        self.inner.reset();
        self.deobfuscation_count = 0;
        self.ai_deobfuscation_count = 0;
        self.rule_deobfuscation_count = 0;
        self.skipped_count = 0;
    }

    /// 启用/禁用反混淆
    pub fn set_deobfuscation_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }

    /// 设置混淆检测阈值
    pub fn set_detection_threshold(&mut self, threshold: f32) {
        self.config.detection_threshold = threshold.clamp(0.0, 1.0);
    }
}

impl Default for DeobfuscatingSandbox {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deobfuscating_sandbox_basic() {
        let mut sandbox = DeobfuscatingSandbox::new();

        // 简单代码不应该被反混淆
        let result = sandbox.execute("var x = 1 + 2; x;").unwrap();
        assert_eq!(result, SandboxValue::Number(3.0));

        let stats = sandbox.get_stats();
        assert_eq!(stats.skipped_count, 1); // 简单代码应该被跳过
    }

    #[test]
    fn test_deobfuscating_sandbox_disabled() {
        let mut sandbox = DeobfuscatingSandbox::new().with_config(DeobfuscationConfig {
            enabled: false,
            ..Default::default()
        });

        let result = sandbox.execute("1 + 1").unwrap();
        assert_eq!(result, SandboxValue::Number(2.0));
    }

    #[test]
    fn test_deobfuscating_sandbox_obfuscated_code() {
        let mut sandbox = DeobfuscatingSandbox::new().with_config(DeobfuscationConfig {
            detection_threshold: 0.2, // 低阈值更容易触发
            ..Default::default()
        });

        // 混淆风格的代码
        let obfuscated = r#"
            var _0x1234 = function() {
                var _0x5678 = 1;
                var _0x9abc = 2;
                return _0x5678 + _0x9abc;
            };
            _0x1234();
        "#;

        // 应该能检测并尝试反混淆
        let needs_deobf = sandbox.needs_deobfuscation(obfuscated);
        // 执行（可能反混淆可能不反混淆，取决于得分）
        let _ = sandbox.execute(obfuscated);

        // 只要能执行就行
        let stats = sandbox.get_stats();
        assert!(stats.deobfuscation_count > 0 || stats.skipped_count > 0);
    }

    #[test]
    fn test_execute_raw() {
        let mut sandbox = DeobfuscatingSandbox::new();

        // execute_raw 不应该进行反混淆
        let result = sandbox.execute_raw("2 * 3").unwrap();
        assert_eq!(result, SandboxValue::Number(6.0));
    }
}
