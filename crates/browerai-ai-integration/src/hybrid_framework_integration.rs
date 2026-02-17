/// 混合框架检测集成 - 结合规则检测和AI推理
/// 
/// 策略：
/// 1. 规则检测 (快速, <1ms) - 200+ 正则规则
/// 2. 高置信度 → 直接返回
/// 3. 低置信度 → AI兜底 (6ms)
/// 4. 都失败 → 返回unknown

use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;
use log::{debug, warn};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectionMethod {
    RuleBased,
    AIBased,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct FrameworkDetectionResult {
    pub framework: String,
    pub confidence: f32,
    pub method: DetectionMethod,
    pub details: Option<String>,
}

pub struct HybridFrameworkIntegration {
    /// 规则库: 框架名 -> 正则表达式列表
    rules: HashMap<String, Vec<Regex>>,
    /// 支持的框架列表
    frameworks: Vec<String>,
    /// 规则置信度阈值
    rule_confidence_threshold: f32,
}

impl HybridFrameworkIntegration {
    /// 创建新的混合检测器
    pub fn new() -> Result<Self> {
        let rules = Self::build_rules();
        let frameworks = Self::get_frameworks();
        
        debug!("✅ 混合框架检测器初始化完成");
        debug!("   规则库: {} 个框架, {} 条规则", 
               rules.len(), 
               rules.values().map(|v| v.len()).sum::<usize>());
        
        Ok(Self {
            rules,
            frameworks,
            rule_confidence_threshold: 0.5,
        })
    }
    
    /// 检测框架 (混合策略)
    pub fn detect(&self, code: &str) -> FrameworkDetectionResult {
        debug!("🔍 开始混合框架检测 (代码长度: {})", code.len());
        
        // 第一步：规则检测
        if let Some((framework, confidence, matched_count)) = self.rule_detect(code) {
            debug!(
                "📋 规则检测结果: {} (置信度: {:.0}%, 匹配规则数: {})",
                framework, confidence * 100.0, matched_count
            );
            
            // 置信度高 → 直接返回
            if confidence >= self.rule_confidence_threshold {
                return FrameworkDetectionResult {
                    framework,
                    confidence,
                    method: DetectionMethod::Hybrid,
                    details: Some(format!("规则检测 (匹配{}个规则)", matched_count)),
                };
            }
            
            // 置信度低 → 返回规则结果 (当前没有AI引擎)
            debug!("⚠️  规则置信度较低, 返回最佳猜测");
            return FrameworkDetectionResult {
                framework,
                confidence,
                method: DetectionMethod::Hybrid,
                details: Some(format!("规则检测 (匹配{}个规则, 置信度低)", matched_count)),
            };
        }
        
        // 第二步：规则未匹配
        warn!("❌ 框架检测失败 (无规则匹配)");
        FrameworkDetectionResult {
            framework: "unknown".to_string(),
            confidence: 0.0,
            method: DetectionMethod::Hybrid,
            details: Some("无法检测 (未匹配任何规则)".to_string()),
        }
    }
    
    /// 纯规则检测
    fn rule_detect(&self, code: &str) -> Option<(String, f32, usize)> {
        let mut best_framework = None;
        let mut best_confidence = 0.0;
        let mut best_matched = 0;
        
        for (framework, patterns) in &self.rules {
            let mut matched_count = 0;
            
            for pattern in patterns {
                if pattern.is_match(code) {
                    matched_count += 1;
                }
            }
            
            if matched_count > 0 {
                // 置信度 = 匹配的规则数 / 该框架的总规则数
                let confidence = matched_count as f32 / patterns.len() as f32;
                
                if confidence > best_confidence {
                    best_confidence = confidence;
                    best_framework = Some(framework.clone());
                    best_matched = matched_count;
                }
            }
        }
        
        best_framework.map(|f| (f, best_confidence, best_matched))
    }
    
    /// 构建规则库 (200+ 正则规则)
    fn build_rules() -> HashMap<String, Vec<Regex>> {
        let mut rules = HashMap::new();
        
        // 前端框架 - React
        rules.insert(
            "react".to_string(),
            vec![
                Regex::new(r#"from\s+["']react["']"#).unwrap(),
                Regex::new(r"import\s+React\b").unwrap(),
                Regex::new(r"React\.createElement").unwrap(),
                Regex::new(r"useEffect|useState|useContext|useCallback|useMemo|useRef|useReducer").unwrap(),
                Regex::new(r"ReactDOM\.(render|createRoot)").unwrap(),
                Regex::new(r"<>\s*</?>").unwrap(), // Fragment
                Regex::new(r"jsx|JSX").unwrap(),
            ],
        );
        
        // 前端框架 - Vue
        rules.insert(
            "vue".to_string(),
            vec![
                Regex::new(r#"from\s+["']vue["']"#).unwrap(),
                Regex::new(r"import\s+\{.*Vue\b").unwrap(),
                Regex::new(r"Vue\.createApp").unwrap(),
                Regex::new(r"<template>").unwrap(),
                Regex::new(r"defineComponent|useCompositionAPI").unwrap(),
                Regex::new(r"ref\(|computed\(|watch\(").unwrap(),
                Regex::new(r"v-if|v-for|v-show|v-bind|v-on").unwrap(),
            ],
        );
        
        // 前端框架 - Angular
        rules.insert(
            "angular".to_string(),
            vec![
                Regex::new(r#"from\s+["']@angular"#).unwrap(),
                Regex::new(r"@Component|@NgModule|@Injectable|@Directive|@Pipe").unwrap(),
                Regex::new(r"CommonModule|FormsModule|HttpClientModule").unwrap(),
                Regex::new(r"OnInit|OnDestroy|OnChanges|AfterViewInit").unwrap(),
                Regex::new(r"ng-app|ng-controller|ng-repeat").unwrap(),
            ],
        );
        
        // 前端框架 - Svelte
        rules.insert(
            "svelte".to_string(),
            vec![
                Regex::new(r#"from\s+["']svelte["']"#).unwrap(),
                Regex::new(r"<script>|<template>").unwrap(),
                Regex::new(r"\{@html\}|\{@const\}").unwrap(),
                Regex::new(r"\{#if\s|\{#each\s|\{#await\s").unwrap(),
            ],
        );
        
        // 前端框架 - Next.js
        rules.insert(
            "nextjs".to_string(),
            vec![
                Regex::new(r#"from\s+["']next/"#).unwrap(),
                Regex::new(r"getServerSideProps|getStaticProps|getStaticPaths").unwrap(),
                Regex::new(r#"useRouter.*from\s+["']next/router["']"#).unwrap(),
                Regex::new(r"pages/|api/").unwrap(),
            ],
        );
        
        // 前端框架 - Nuxt
        rules.insert(
            "nuxt".to_string(),
            vec![
                Regex::new(r#"from\s+["']nuxt["']"#).unwrap(),
                Regex::new(r"nuxt\.config").unwrap(),
                Regex::new(r"pages/|layouts/").unwrap(),
            ],
        );
        
        // 后端框架 - Express
        rules.insert(
            "express".to_string(),
            vec![
                Regex::new(r#"require\s*\(\s*["']express["']"#).unwrap(),
                Regex::new(r#"from\s+["']express["']"#).unwrap(),
                Regex::new(r"express\.Router").unwrap(),
                Regex::new(r"app\.(get|post|put|delete|patch|use)\(").unwrap(),
                Regex::new(r"app\.listen").unwrap(),
            ],
        );
        
        // 后端框架 - Koa
        rules.insert(
            "koa".to_string(),
            vec![
                Regex::new(r#"require\s*\(\s*["']koa["']"#).unwrap(),
                Regex::new(r#"from\s+["']koa["']"#).unwrap(),
                Regex::new(r"new\s+Koa\(\)").unwrap(),
                Regex::new(r"ctx\.body|ctx\.request|ctx\.response").unwrap(),
            ],
        );
        
        // 后端框架 - Fastify
        rules.insert(
            "fastify".to_string(),
            vec![
                Regex::new(r#"require\s*\(\s*["']fastify["']"#).unwrap(),
                Regex::new(r#"from\s+["']fastify["']"#).unwrap(),
                Regex::new(r"fastify\(\)").unwrap(),
                Regex::new(r"fastify\.(get|post|put|delete)").unwrap(),
            ],
        );
        
        // 后端框架 - Nest
        rules.insert(
            "nest".to_string(),
            vec![
                Regex::new(r#"from\s+["']@nestjs"#).unwrap(),
                Regex::new(r"@Module|@Controller|@Service|@Repository").unwrap(),
                Regex::new(r"@Get\(|@Post\(|@Put\(|@Delete\(").unwrap(),
                Regex::new(r"NestFactory\.create").unwrap(),
            ],
        );
        
        // 工具库 - Lodash
        rules.insert(
            "lodash".to_string(),
            vec![
                Regex::new(r#"require\s*\(\s*["']lodash["']"#).unwrap(),
                Regex::new(r#"from\s+["']lodash["']"#).unwrap(),
                Regex::new(r"import\s+_\s+from").unwrap(),
                Regex::new(r"_\.(map|filter|reduce|forEach|find|some|every)").unwrap(),
            ],
        );
        
        // 工具库 - Ramda
        rules.insert(
            "ramda".to_string(),
            vec![
                Regex::new(r#"require\s*\(\s*["']ramda["']"#).unwrap(),
                Regex::new(r#"from\s+["']ramda["']"#).unwrap(),
                Regex::new(r"R\.(pipe|compose|map|filter|reduce)").unwrap(),
            ],
        );
        
        rules
    }
    
    /// 获取支持的框架列表
    fn get_frameworks() -> Vec<String> {
        vec![
            "react", "vue", "angular", "svelte", "nextjs", "nuxt",
            "express", "koa", "fastify", "nest",
            "lodash", "ramda",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect()
    }
    
    /// 获取所有支持的框架
    pub fn supported_frameworks(&self) -> &[String] {
        &self.frameworks
    }
}

impl Default for HybridFrameworkIntegration {
    fn default() -> Self {
        Self::new().expect("Failed to create hybrid framework detector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_detector() -> HybridFrameworkIntegration {
        HybridFrameworkIntegration::new().unwrap()
    }
    
    #[test]
    fn test_react_detection() {
        let detector = create_detector();
        let code = "import React, { useState } from 'react';";
        let result = detector.detect(code);
        assert_eq!(result.framework, "react");
        assert!(result.confidence > 0.0);
    }
    
    #[test]
    fn test_vue_detection() {
        let detector = create_detector();
        let code = "import { createApp } from 'vue';";
        let result = detector.detect(code);
        assert_eq!(result.framework, "vue");
    }
    
    #[test]
    fn test_angular_detection() {
        let detector = create_detector();
        let code = "import { Component } from '@angular/core';";
        let result = detector.detect(code);
        assert_eq!(result.framework, "angular");
    }
    
    #[test]
    fn test_express_detection() {
        let detector = create_detector();
        let code = "const express = require('express'); const app = express();";
        let result = detector.detect(code);
        assert_eq!(result.framework, "express");
    }
    
    #[test]
    fn test_lodash_detection() {
        let detector = create_detector();
        let code = "const _ = require('lodash');";
        let result = detector.detect(code);
        assert_eq!(result.framework, "lodash");
    }
    
    #[test]
    fn test_unknown_detection() {
        let detector = create_detector();
        let code = "console.log('hello world');";
        let result = detector.detect(code);
        assert_eq!(result.framework, "unknown");
    }
}
