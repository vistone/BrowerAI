//! JavaScript反混淆处理模块
//!
//! 集成AI反混淆器到HTTP网络层，自动检测并处理混淆的JavaScript代码

use anyhow::Result;
#[cfg(feature = "ml")]
use browerai_deobfuscation::AIDeobfuscator;
use browerai_deobfuscation::{EnhancedDeobfuscator, JsDeobfuscator};
use std::path::Path;
#[cfg(feature = "ml")]
use std::sync::Arc;
use std::sync::Mutex;

/// JavaScript反混淆处理器
///
/// 集成多种反混淆策略：
/// 1. AI反混淆器 (PyTorch Transformer模型)
/// 2. 规则化反混淆器 (JsDeobfuscator)
/// 3. 增强反混淆器 (EnhancedDeobfuscator)
pub struct JsDeobfuscationProcessor {
    #[cfg(feature = "ml")]
    ai_deobfuscator: Option<Arc<AIDeobfuscator>>,
    rule_deobfuscator: JsDeobfuscator,
    enhanced_deobfuscator: Mutex<EnhancedDeobfuscator>,
    enabled: bool,
    #[allow(dead_code)]
    use_ai: bool,
}

impl JsDeobfuscationProcessor {
    /// 创建新的反混淆处理器
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "ml")]
            ai_deobfuscator: None,
            rule_deobfuscator: JsDeobfuscator::new(),
            enhanced_deobfuscator: Mutex::new(EnhancedDeobfuscator::new()),
            enabled: true,
            use_ai: false,
        }
    }

    /// 启用AI反混淆器
    #[cfg(feature = "ml")]
    pub fn with_ai_model(mut self, model_path: &Path, vocab_path: &Path) -> Result<Self> {
        match AIDeobfuscator::new(model_path, vocab_path) {
            Ok(deobf) => {
                log::info!("✅ AI反混淆器加载成功");
                log::info!("{}", deobf.model_info());
                self.ai_deobfuscator = Some(Arc::new(deobf));
                self.use_ai = true;
            }
            Err(e) => {
                log::warn!("⚠️ AI反混淆器加载失败: {}，使用规则化方法", e);
            }
        }
        Ok(self)
    }

    /// 启用AI反混淆器（无 ml 特性时的 stub）
    #[cfg(not(feature = "ml"))]
    pub fn with_ai_model(self, _model_path: &Path, _vocab_path: &Path) -> Result<Self> {
        log::warn!("⚠️ AI反混淆功能未启用，请使用 --features ml 编译");
        Ok(self)
    }

    /// 启用/禁用反混淆
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// 检查内容是否为JavaScript
    pub fn is_javascript(content_type: &str) -> bool {
        content_type.contains("javascript")
            || content_type.contains("application/json")
            || content_type.contains("text/js")
    }

    /// 检查代码是否需要反混淆
    pub fn needs_deobfuscation(&self, code: &str) -> bool {
        if code.len() < 100 {
            return false;
        }

        // 使用规则反混淆器分析
        let analysis = self.rule_deobfuscator.analyze_obfuscation(code);

        // 如果混淆得分超过阈值，需要反混淆
        analysis.obfuscation_score > 0.3
    }

    /// 处理JavaScript代码
    ///
    /// 自动选择最佳反混淆策略
    pub fn process(&self, code: &str) -> Result<ProcessedJs> {
        if !self.enabled {
            return Ok(ProcessedJs {
                original: code.to_string(),
                deobfuscated: code.to_string(),
                was_processed: false,
                method: "disabled".to_string(),
            });
        }

        // 分析混淆程度
        let analysis = self.rule_deobfuscator.analyze_obfuscation(code);

        if analysis.obfuscation_score < 0.3 {
            log::debug!(
                "代码混淆程度低 (score={}), 跳过反混淆",
                analysis.obfuscation_score
            );
            return Ok(ProcessedJs {
                original: code.to_string(),
                deobfuscated: code.to_string(),
                was_processed: false,
                method: "skipped".to_string(),
            });
        }

        log::info!(
            "📝 检测到混淆代码 (score={}), 开始反混淆",
            analysis.obfuscation_score
        );

        // 尝试AI反混淆
        #[cfg(feature = "ml")]
        if self.use_ai {
            if let Some(ai_deobf) = &self.ai_deobfuscator {
                match ai_deobf.deobfuscate(code) {
                    Ok(deobfuscated) => {
                        log::info!(
                            "✅ AI反混淆完成: {} → {} 字符",
                            code.len(),
                            deobfuscated.len()
                        );
                        return Ok(ProcessedJs {
                            original: code.to_string(),
                            deobfuscated,
                            was_processed: true,
                            method: "ai_transformer".to_string(),
                        });
                    }
                    Err(e) => {
                        log::warn!("⚠️ AI反混淆失败: {}, 使用规则化方法", e);
                    }
                }
            }
        }

        // 使用增强反混淆器
        match self.enhanced_deobfuscator.lock().unwrap().deobfuscate(code) {
            Ok(result) => {
                log::info!(
                    "✅ 规则化反混淆完成: {} → {} 字符",
                    code.len(),
                    result.code.len()
                );
                Ok(ProcessedJs {
                    original: code.to_string(),
                    deobfuscated: result.code,
                    was_processed: true,
                    method: "enhanced_rules".to_string(),
                })
            }
            Err(e) => {
                log::warn!("⚠️ 反混淆失败: {}, 返回原始代码", e);
                Ok(ProcessedJs {
                    original: code.to_string(),
                    deobfuscated: code.to_string(),
                    was_processed: false,
                    method: "failed".to_string(),
                })
            }
        }
    }

    /// 批量处理JavaScript代码
    pub fn process_batch(&self, codes: &[&str]) -> Result<Vec<ProcessedJs>> {
        codes.iter().map(|code| self.process(code)).collect()
    }

    /// 获取AI反混淆器的引用
    #[cfg(feature = "ml")]
    pub fn ai_deobfuscator(&self) -> Option<Arc<AIDeobfuscator>> {
        self.ai_deobfuscator.clone()
    }

    /// 获取AI反混淆器的引用（无 ml 特性时）
    #[cfg(not(feature = "ml"))]
    pub fn ai_deobfuscator(&self) -> Option<()> {
        None
    }

    /// 检查是否启用了AI
    #[cfg(feature = "ml")]
    pub fn has_ai(&self) -> bool {
        self.use_ai && self.ai_deobfuscator.is_some()
    }

    /// 检查是否启用了AI（无 ml 特性时）
    #[cfg(not(feature = "ml"))]
    pub fn has_ai(&self) -> bool {
        false
    }
}

impl Default for JsDeobfuscationProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// 处理后的JavaScript代码
#[derive(Debug, Clone)]
pub struct ProcessedJs {
    /// 原始代码
    pub original: String,
    /// 反混淆后的代码
    pub deobfuscated: String,
    /// 是否进行了处理
    pub was_processed: bool,
    /// 使用的方法
    pub method: String,
}

impl ProcessedJs {
    /// 获取最终代码（优先使用反混淆后的）
    pub fn code(&self) -> &str {
        &self.deobfuscated
    }

    /// 计算改进率
    pub fn improvement_ratio(&self) -> f32 {
        if self.original.is_empty() {
            return 0.0;
        }
        let original_len = self.original.len() as f32;
        let deobf_len = self.deobfuscated.len() as f32;

        // 反混淆后代码通常会变长（因为添加了格式化）
        if deobf_len > original_len {
            (deobf_len - original_len) / original_len
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let processor = JsDeobfuscationProcessor::new();
        assert!(processor.enabled);
        assert!(!processor.use_ai);
    }

    #[test]
    fn test_is_javascript() {
        assert!(JsDeobfuscationProcessor::is_javascript(
            "application/javascript"
        ));
        assert!(JsDeobfuscationProcessor::is_javascript("text/javascript"));
        assert!(JsDeobfuscationProcessor::is_javascript("application/json"));
        assert!(!JsDeobfuscationProcessor::is_javascript("text/html"));
    }

    #[test]
    fn test_process_simple_code() {
        let processor = JsDeobfuscationProcessor::new();
        let code = "var a = 1; var b = 2;";

        let result = processor.process(code).unwrap();
        assert!(!result.was_processed); // 简单代码不需要反混淆
    }

    #[test]
    fn test_process_obfuscated_code() {
        let processor = JsDeobfuscationProcessor::new();
        // 模拟混淆代码（短变量名、压缩格式）
        let code = "var a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8,i=9,j=10;function k(l,m){return l+m}var n=k(a,b);var o=k(c,d);var p=k(e,f);console.log(n,o,p)";

        let result = processor.process(code).unwrap();
        // 验证处理完成
        assert!(result.deobfuscated.len() > 0);
    }

    #[test]
    fn test_batch_processing() {
        let processor = JsDeobfuscationProcessor::new();
        let codes = vec!["var x = 1;", "function f() { return 42; }"];

        let results = processor.process_batch(&codes).unwrap();
        assert_eq!(results.len(), 2);
    }
}
