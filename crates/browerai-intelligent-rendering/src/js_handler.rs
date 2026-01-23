/// JavaScript 框架检测集成 - 用于Renderer和DOM处理
///
/// 在JS加载时自动检测所用框架，用于优化渲染和分析

use anyhow::Result;
use log::{debug, info};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use regex::Regex;

/// JS处理器 - 集成框架检测
pub struct JsHandler {
    /// 框架规则库
    rules: Arc<HashMap<String, Vec<Regex>>>,
    /// 已检测的框架缓存: 代码哈希 -> 框架名
    framework_cache: Arc<Mutex<HashMap<u64, String>>>,
    /// 统计信息
    stats: Arc<Mutex<JsHandlerStats>>,
}

/// 统计信息
#[derive(Debug, Clone, Default)]
pub struct JsHandlerStats {
    pub total_scripts: usize,
    pub detected_frameworks: HashMap<String, usize>,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub detection_time_ms: f64,
}

impl JsHandler {
    /// 创建新的JS处理器
    pub fn new() -> Result<Self> {
        let rules = Self::build_rules();
        debug!("✅ JS处理器初始化完成");
        
        Ok(Self {
            rules: Arc::new(rules),
            framework_cache: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(JsHandlerStats::default())),
        })
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
            Regex::new(r"ReactDOM\.render").unwrap(),
        ]);
        
        // Vue 规则
        rules.insert("vue".to_string(), vec![
            Regex::new(r#"from\s+['"]vue['"]"#).unwrap(),
            Regex::new(r"import\s+\{\s*createApp").unwrap(),
            Regex::new(r"createApp\(").unwrap(),
            Regex::new(r"ref\(").unwrap(),
            Regex::new(r"reactive\(").unwrap(),
            Regex::new(r"<script\s+setup").unwrap(),
        ]);
        
        // Angular 规则
        rules.insert("angular".to_string(), vec![
            Regex::new(r"@angular/core").unwrap(),
            Regex::new(r"@Component").unwrap(),
            Regex::new(r"@NgModule").unwrap(),
            Regex::new(r"@Injectable").unwrap(),
            Regex::new(r"OnInit").unwrap(),
            Regex::new(r"\*ngIf\s*=").unwrap(),
        ]);
        
        // Express 规则
        rules.insert("express".to_string(), vec![
            Regex::new(r#"const\s+express\s*=\s*require\(['"]express['"]\)"#).unwrap(),
            Regex::new(r"app\.get\(").unwrap(),
            Regex::new(r"app\.post\(").unwrap(),
            Regex::new(r"app\.listen\(").unwrap(),
            Regex::new(r"express\.json\(\)").unwrap(),
        ]);
        
        // Lodash 规则
        rules.insert("lodash".to_string(), vec![
            Regex::new(r#"import\s+_\s+from\s+['"]lodash"#).unwrap(),
            Regex::new(r#"_\.map"#).unwrap(),
            Regex::new(r#"_\.filter"#).unwrap(),
            Regex::new(r#"_\.reduce"#).unwrap(),
            Regex::new(r#"_\.debounce"#).unwrap(),
        ]);
        
        rules
    }
    
    /// 检测脚本框架
    fn detect_framework(&self, code: &str) -> (String, f32) {
        let mut best_match = ("unknown".to_string(), 0.0);
        
        for (framework, patterns) in self.rules.iter() {
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
    
    /// 处理加载的JS脚本
    pub fn on_script_loaded(&self, script_id: u64, code: &str) -> Result<String> {
        let start = std::time::Instant::now();
        
        // 第一步: 检查缓存
        {
            let cache = self.framework_cache.lock().unwrap();
            if let Some(framework) = cache.get(&script_id) {
                debug!("📦 框架检测缓存命中: {} -> {}", script_id, framework);
                
                let mut stats = self.stats.lock().unwrap();
                stats.cache_hits += 1;
                return Ok(framework.clone());
            }
        }
        
        // 第二步: 检测框架
        let (framework, confidence) = self.detect_framework(code);
        
        debug!("🔍 检测到框架: {} (置信度: {:.0}%)", 
               framework, confidence * 100.0);
        
        // 第三步: 更新缓存
        {
            let mut cache = self.framework_cache.lock().unwrap();
            cache.insert(script_id, framework.clone());
        }
        
        // 第四步: 更新统计
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_scripts += 1;
            stats.cache_misses += 1;
            *stats.detected_frameworks.entry(framework.clone()).or_insert(0) += 1;
            stats.detection_time_ms += start.elapsed().as_secs_f64() * 1000.0;
        }
        
        Ok(framework)
    }
    
    /// 检测脚本框架（简化版）
    pub fn detect_framework_simple(&self, code: &str) -> String {
        let (framework, _) = self.detect_framework(code);
        framework
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> JsHandlerStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// 清除缓存
    pub fn clear_cache(&self) {
        self.framework_cache.lock().unwrap().clear();
        debug!("Clear cache");
    }
    
    /// 打印统计信息
    pub fn print_stats(&self) {
        let stats = self.get_stats();
        info!("JS handler stats:");
        info!("   total scripts: {}", stats.total_scripts);
        info!("   cache hits: {}", stats.cache_hits);
        info!("   cache misses: {}", stats.cache_misses);
        info!("   total detection time: {:.2}ms", stats.detection_time_ms);
        
        if stats.total_scripts > 0 {
            let hit_rate = stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64 * 100.0;
            info!("   cache hit rate: {:.1}%", hit_rate);
        }
        
        if !stats.detected_frameworks.is_empty() {
            info!("   detected frameworks:");
            for (framework, count) in &stats.detected_frameworks {
                info!("     - {}: {} scripts", framework, count);
            }
        }
    }
}

impl Default for JsHandler {
    fn default() -> Self {
        Self::new().expect("Failed to create JS handler")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_js_handler_creation() {
        let handler = JsHandler::new().unwrap();
        assert_eq!(handler.get_stats().total_scripts, 0);
    }
    
    #[test]
    fn test_script_detection() {
        let handler = JsHandler::new().unwrap();
        let react_code = r#"import React from 'react';"#;
        
        let framework = handler.detect_framework_simple(react_code);
        assert_eq!(framework, "react");
    }
    
    #[test]
    fn test_cache_functionality() {
        let handler = JsHandler::new().unwrap();
        let vue_code = r#"import { createApp } from 'vue';"#;
        
        // 第一次检测 (缓存失败)
        let fw1 = handler.on_script_loaded(1, vue_code).unwrap();
        assert_eq!(fw1, "vue");
        
        let stats1 = handler.get_stats();
        assert_eq!(stats1.cache_misses, 1);
        assert_eq!(stats1.cache_hits, 0);
        
        // 第二次检测 (缓存命中)
        let fw2 = handler.on_script_loaded(1, vue_code).unwrap();
        assert_eq!(fw2, "vue");
        
        let stats2 = handler.get_stats();
        assert_eq!(stats2.cache_hits, 1);
    }
    
    #[test]
    fn test_multiple_scripts() {
        let handler = JsHandler::new().unwrap();
        
        let codes = vec![
            (r#"import React from 'react';"#, "react"),
            (r#"import { createApp } from 'vue';"#, "vue"),
            (r#"import { Component } from '@angular/core';"#, "angular"),
            (r#"const express = require('express');"#, "express"),
        ];
        
        for (i, (code, expected)) in codes.iter().enumerate() {
            let framework = handler.on_script_loaded(i as u64, code).unwrap();
            assert_eq!(framework, *expected);
        }
        
        let stats = handler.get_stats();
        assert_eq!(stats.total_scripts, 4);
        assert_eq!(stats.detected_frameworks.len(), 4);
    }
}
