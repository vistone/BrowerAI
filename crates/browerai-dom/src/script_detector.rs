/// DOM 脚本元素框架检测集成
/// 
/// 在DOM处理脚本元素时，自动检测所用框架

use anyhow::Result;
use log::{debug, warn};
use std::collections::HashMap;
use regex::Regex;

/// DOM脚本收集器 - 框架检测
#[derive(Clone)]
pub struct DomScriptCollector {
    /// 框架规则库
    rules: HashMap<String, Vec<Regex>>,
}

/// 脚本元素信息
#[derive(Debug, Clone)]
pub struct ScriptElement {
    pub id: String,
    pub source: ScriptSource,
    pub detected_framework: String,
    pub confidence: f32,
    pub content_hash: u64,
}

/// 脚本源类型
#[derive(Debug, Clone)]
pub enum ScriptSource {
    /// 内联脚本
    Inline { content: String },
    /// 外部脚本
    External { src: String, content: Option<String> },
    /// 模块脚本
    Module { src: String, content: Option<String> },
}

/// 框架统计
#[derive(Debug, Clone, Default)]
pub struct FrameworkStats {
    pub total_scripts: usize,
    pub inline_scripts: usize,
    pub external_scripts: usize,
    pub module_scripts: usize,
    pub framework_distribution: HashMap<String, usize>,
    pub avg_confidence: f32,
}

impl DomScriptCollector {
    /// 创建新的DOM脚本收集器
    pub fn new() -> Result<Self> {
        let rules = Self::build_rules();
        debug!("✅ DOM脚本收集器初始化完成");
        
        Ok(Self { rules })
    }
    
    /// 构建框架检测规则
    fn build_rules() -> HashMap<String, Vec<Regex>> {
        let mut rules = HashMap::new();
        
        // React 规则
        rules.insert("react".to_string(), vec![
            Regex::new(r"import\s+React\s+from").unwrap(),
            Regex::new(r#"from\s+['"]react['"]"#).unwrap(),
            Regex::new(r"React\.createElement").unwrap(),
            Regex::new(r"useState\(").unwrap(),
            Regex::new(r"useEffect\(").unwrap(),
        ]);
        
        // Vue 规则
        rules.insert("vue".to_string(), vec![
            Regex::new(r#"from\s+['"]vue['"]"#).unwrap(),
            Regex::new(r"createApp\(").unwrap(),
            Regex::new(r"ref\(").unwrap(),
            Regex::new(r"reactive\(").unwrap(),
        ]);
        
        // Angular 规则
        rules.insert("angular".to_string(), vec![
            Regex::new(r"@angular/core").unwrap(),
            Regex::new(r"@Component").unwrap(),
            Regex::new(r"@NgModule").unwrap(),
        ]);
        
        // Express 规则
        rules.insert("express".to_string(), vec![
            Regex::new(r"const\s+express\s*=").unwrap(),
            Regex::new(r"app\.get\(").unwrap(),
            Regex::new(r"app\.listen\(").unwrap(),
        ]);
        
        rules
    }
    
    /// 检测脚本框架
    fn detect_framework(&self, code: &str) -> (String, f32) {
        let mut best_match = ("unknown".to_string(), 0.0);
        
        for (framework, patterns) in &self.rules {
            let mut matched = 0;
            for pattern in patterns {
                if pattern.is_match(code) {
                    matched += 1;
                }
            }
            
            if matched > 0 {
                let confidence = matched as f32 / patterns.len() as f32;
                if confidence > best_match.1 {
                    best_match = (framework.clone(), confidence);
                }
            }
        }
        
        best_match
    }
    
    /// 处理脚本元素
    pub fn process_script(&self, id: String, source: ScriptSource) -> Result<ScriptElement> {
        let content = match &source {
            ScriptSource::Inline { content } => content.clone(),
            ScriptSource::External { content, .. } => {
                content.clone().unwrap_or_else(|| {
                    debug!("⚠️  外部脚本无可用内容，跳过框架检测: {}", id);
                    String::new()
                })
            }
            ScriptSource::Module { content, .. } => {
                content.clone().unwrap_or_else(|| {
                    debug!("⚠️  模块脚本无可用内容，跳过框架检测: {}", id);
                    String::new()
                })
            }
        };
        
        if content.is_empty() {
            return Ok(ScriptElement {
                id: id.clone(),
                source,
                detected_framework: "unknown".to_string(),
                confidence: 0.0,
                content_hash: 0,
            });
        }
        
        // 计算内容哈希
        let content_hash = self.hash_content(&content);
        
        // 检测框架
        let (framework, confidence) = self.detect_framework(&content);
        
        debug!("🔍 DOM脚本框架检测: {} -> {} (置信度: {:.0}%)",
               id, framework, confidence * 100.0);
        
        Ok(ScriptElement {
            id,
            source,
            detected_framework: framework,
            confidence,
            content_hash,
        })
    }
    
    /// 处理多个脚本元素
    pub fn process_scripts(&self, scripts: Vec<(String, ScriptSource)>) -> Result<Vec<ScriptElement>> {
        let mut results = Vec::new();
        
        for (id, source) in scripts {
            match self.process_script(id, source) {
                Ok(element) => results.push(element),
                Err(e) => warn!("处理脚本失败: {}", e),
            }
        }
        
        Ok(results)
    }
    
    /// 计算内容哈希
    fn hash_content(&self, content: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }
    
    /// 生成框架统计
    pub fn generate_stats(&self, scripts: &[ScriptElement]) -> FrameworkStats {
        let mut stats = FrameworkStats::default();
        let mut confidence_sum = 0.0;
        
        for script in scripts {
            stats.total_scripts += 1;
            
            match &script.source {
                ScriptSource::Inline { .. } => stats.inline_scripts += 1,
                ScriptSource::External { .. } => stats.external_scripts += 1,
                ScriptSource::Module { .. } => stats.module_scripts += 1,
            }
            
            *stats.framework_distribution
                .entry(script.detected_framework.clone())
                .or_insert(0) += 1;
            
            confidence_sum += script.confidence;
        }
        
        if !scripts.is_empty() {
            stats.avg_confidence = confidence_sum / scripts.len() as f32;
        }
        
        stats
    }
    
    /// 按框架过滤脚本
    pub fn filter_by_framework(&self, scripts: &[ScriptElement], framework: &str) -> Vec<ScriptElement> {
        scripts
            .iter()
            .filter(|s| s.detected_framework == framework)
            .cloned()
            .collect()
    }
}

impl Default for DomScriptCollector {
    fn default() -> Self {
        Self::new().expect("Failed to create DOM script collector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_collector_creation() {
        let collector = DomScriptCollector::new().unwrap();
        assert!(true); // Just verify creation succeeds
    }
    
    #[test]
    fn test_inline_script_detection() {
        let collector = DomScriptCollector::new().unwrap();
        let react_code = "import React from 'react'; export default App;".to_string();
        
        let element = collector.process_script(
            "script-1".to_string(),
            ScriptSource::Inline { content: react_code },
        ).unwrap();
        
        assert_eq!(element.detected_framework, "react");
        assert!(element.confidence > 0.0);
    }
    
    #[test]
    fn test_external_script_with_content() {
        let collector = DomScriptCollector::new().unwrap();
        let vue_code = "import { createApp } from 'vue';".to_string();
        
        let element = collector.process_script(
            "script-2".to_string(),
            ScriptSource::External {
                src: "app.js".to_string(),
                content: Some(vue_code),
            },
        ).unwrap();
        
        assert_eq!(element.detected_framework, "vue");
    }
    
    #[test]
    fn test_module_script_detection() {
        let collector = DomScriptCollector::new().unwrap();
        let angular_code = "import { Component } from '@angular/core';".to_string();
        
        let element = collector.process_script(
            "script-3".to_string(),
            ScriptSource::Module {
                src: "module.js".to_string(),
                content: Some(angular_code),
            },
        ).unwrap();
        
        assert_eq!(element.detected_framework, "angular");
    }
    
    #[test]
    fn test_multiple_scripts_processing() {
        let collector = DomScriptCollector::new().unwrap();
        
        let scripts = vec![
            ("script-1".to_string(), ScriptSource::Inline {
                content: "import React from 'react';".to_string(),
            }),
            ("script-2".to_string(), ScriptSource::External {
                src: "app.js".to_string(),
                content: Some("import { createApp } from 'vue';".to_string()),
            }),
            ("script-3".to_string(), ScriptSource::Module {
                src: "module.js".to_string(),
                content: Some("import { Component } from '@angular/core';".to_string()),
            }),
        ];
        
        let results = collector.process_scripts(scripts).unwrap();
        assert_eq!(results.len(), 3);
        
        let stats = collector.generate_stats(&results);
        assert_eq!(stats.total_scripts, 3);
        assert_eq!(stats.inline_scripts, 1);
        assert_eq!(stats.external_scripts, 1);
        assert_eq!(stats.module_scripts, 1);
        assert_eq!(stats.framework_distribution.len(), 3);
    }
    
    #[test]
    fn test_framework_filtering() {
        let collector = DomScriptCollector::new().unwrap();
        
        let scripts = vec![
            collector.process_script(
                "r1".to_string(),
                ScriptSource::Inline { content: "import React from 'react';".to_string() },
            ).unwrap(),
            collector.process_script(
                "r2".to_string(),
                ScriptSource::Inline { content: "import React from 'react';".to_string() },
            ).unwrap(),
            collector.process_script(
                "v1".to_string(),
                ScriptSource::Inline { content: "import { createApp } from 'vue';".to_string() },
            ).unwrap(),
        ];
        
        let react_scripts = collector.filter_by_framework(&scripts, "react");
        assert_eq!(react_scripts.len(), 2);
        
        let vue_scripts = collector.filter_by_framework(&scripts, "vue");
        assert_eq!(vue_scripts.len(), 1);
    }
}
