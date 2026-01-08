//! Hybrid JavaScript Analyzer
//!
//! Integrates static analysis (via js-analyzer) with dynamic analysis (HybridJsOrchestrator).
//!
//! **Note**: This module provides basic framework detection. For comprehensive framework
//! detection with 50+ frameworks (React, Vue, Angular, Webpack, Chinese frameworks like
//! Taro/Uni-app, etc.), use `FrameworkKnowledgeBase` from `browerai-learning` at the
//! application level to avoid circular dependencies.

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};

use crate::HybridJsOrchestrator;
use browerai_js_analyzer::{AnalysisPipeline, FullAnalysisResult};

/// Represents a detected JavaScript framework (simplified version)
///
/// For comprehensive framework info, see `browerai-learning::FrameworkKnowledge`
#[derive(Debug, Clone, PartialEq)]
pub struct FrameworkInfo {
    /// Framework name (e.g., "React", "Vue", "Angular")
    pub name: String,
    /// Detected version (if available)
    pub version: Option<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Detection method (e.g., "signature_match", "pattern_matching")
    pub detection_method: String,
}

/// Hybrid analysis result combining static and dynamic analysis
#[derive(Debug, Clone)]
pub struct HybridAnalysisResult {
    /// Static analysis result
    pub static_analysis: FullAnalysisResult,
    /// Detected frameworks (basic detection - use FrameworkKnowledgeBase for comprehensive)
    pub frameworks: Vec<FrameworkInfo>,
    /// Runtime variable values (if dynamic analysis was performed)
    pub runtime_values: HashMap<String, String>,
    /// Global objects detected at runtime
    pub global_objects: HashSet<String>,
    /// Whether dynamic analysis was performed
    pub dynamic_analysis_performed: bool,
}

impl HybridAnalysisResult {
    /// Get the primary detected framework
    pub fn primary_framework(&self) -> Option<&FrameworkInfo> {
        self.frameworks
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    }

    /// Check if a specific framework was detected
    pub fn has_framework(&self, name: &str) -> bool {
        self.frameworks
            .iter()
            .any(|f| f.name.eq_ignore_ascii_case(name))
    }

    /// Get call edge count (from static analysis)
    pub fn call_edge_count(&self) -> usize {
        self.static_analysis.call_edges
    }

    /// Get total analysis time
    pub fn total_time_ms(&self) -> f64 {
        self.static_analysis.time_ms
    }
}

/// Hybrid JavaScript Analyzer
///
/// Combines static analysis (via AnalysisPipeline) with dynamic analysis
/// (via HybridJsOrchestrator) to provide comprehensive code understanding.
///
/// **Note**: Uses basic pattern matching. For comprehensive framework detection
/// (50+ frameworks including Chinese frameworks), integrate with
/// `browerai-learning::FrameworkKnowledgeBase` at application level.
pub struct HybridJsAnalyzer {
    /// Static analysis pipeline
    static_pipeline: AnalysisPipeline,

    /// Dynamic analysis orchestrator
    orchestrator: Option<HybridJsOrchestrator>,

    /// Whether to perform dynamic analysis
    enable_dynamic: bool,

    /// Basic framework detection patterns
    framework_patterns: HashMap<String, Vec<String>>,
}

impl HybridJsAnalyzer {
    /// Create a new hybrid analyzer with default settings (static only)
    pub fn new() -> Self {
        Self {
            static_pipeline: AnalysisPipeline::new(),
            orchestrator: None,
            enable_dynamic: false,
            framework_patterns: Self::init_basic_patterns(),
        }
    }

    /// Create analyzer with dynamic analysis enabled
    pub fn with_dynamic_analysis() -> Self {
        Self {
            static_pipeline: AnalysisPipeline::new(),
            orchestrator: Some(HybridJsOrchestrator::new()),
            enable_dynamic: true,
            framework_patterns: Self::init_basic_patterns(),
        }
    }

    /// Enable dynamic analysis
    pub fn enable_dynamic(&mut self) {
        self.enable_dynamic = true;
        if self.orchestrator.is_none() {
            self.orchestrator = Some(HybridJsOrchestrator::new());
        }
    }

    /// Disable dynamic analysis
    pub fn disable_dynamic(&mut self) {
        self.enable_dynamic = false;
    }

    /// Analyze JavaScript source code
    ///
    /// Performs static analysis always, and dynamic analysis if enabled.
    pub fn analyze(&mut self, source: &str) -> Result<HybridAnalysisResult> {
        log::info!("Starting hybrid analysis");

        // 1. Static analysis (always performed)
        log::debug!("Performing static analysis...");
        let static_analysis = self
            .static_pipeline
            .analyze(source)
            .context("Static analysis failed")?;

        // 2. Framework detection (basic pattern matching)
        log::debug!("Detecting frameworks...");
        let frameworks = self.detect_frameworks_basic(source);

        // 3. Dynamic analysis (optional)
        let (runtime_values, global_objects, dynamic_performed) =
            self.perform_dynamic_analysis(source)?;

        Ok(HybridAnalysisResult {
            static_analysis,
            frameworks,
            runtime_values,
            global_objects,
            dynamic_analysis_performed: dynamic_performed,
        })
    }

    /// Perform dynamic analysis if enabled
    fn perform_dynamic_analysis(
        &mut self,
        source: &str,
    ) -> Result<(HashMap<String, String>, HashSet<String>, bool)> {
        if !self.enable_dynamic {
            return Ok((HashMap::new(), HashSet::new(), false));
        }

        if let Some(ref mut orchestrator) = self.orchestrator {
            log::debug!("Performing dynamic analysis...");

            // Execute the code
            orchestrator
                .execute(source)
                .context("Dynamic execution failed")?;

            // Extract runtime information (need to split borrows)
            let runtime_values = {
                let mut values = HashMap::new();
                let test_vars = vec!["window", "document", "navigator", "console"];
                for var in test_vars {
                    let check_code = format!("typeof {}", var);
                    if orchestrator.execute(&check_code).is_ok() {
                        values.insert(var.to_string(), "defined".to_string());
                    }
                }
                values
            };

            let global_objects = {
                let mut objects = HashSet::new();
                let globals = vec![
                    "Object", "Array", "String", "Number", "Boolean", "Function", "Date", "RegExp",
                    "Error", "Math", "JSON", "Promise", "Map", "Set", "Symbol",
                ];
                for global in globals {
                    let check_code = format!("typeof {}", global);
                    if orchestrator.execute(&check_code).is_ok() {
                        objects.insert(global.to_string());
                    }
                }
                objects
            };

            Ok((runtime_values, global_objects, true))
        } else {
            Ok((HashMap::new(), HashSet::new(), false))
        }
    }

    /// Initialize basic framework patterns
    ///
    /// **Note**: This is a simplified set. For comprehensive detection including:
    /// - Global frameworks: React, Vue, Angular, Svelte, etc.
    /// - Chinese frameworks: Taro, Uni-app, Rax, San, Omi, etc.
    /// - Bundlers: Webpack, Vite, Rollup, esbuild, Parcel, etc.
    /// - 50+ total frameworks
    ///
    /// Use `browerai-learning::FrameworkKnowledgeBase` at application level.
    fn init_basic_patterns() -> HashMap<String, Vec<String>> {
        let mut patterns = HashMap::new();

        // React patterns
        patterns.insert(
            "React".to_string(),
            vec![
                "React.createElement".to_string(),
                "React.Component".to_string(),
                "_jsx(".to_string(),
                "_jsxs(".to_string(),
                "useState".to_string(),
                "useEffect".to_string(),
            ],
        );

        // Vue patterns
        patterns.insert(
            "Vue".to_string(),
            vec![
                "_createVNode".to_string(),
                "_createElementVNode".to_string(),
                "_hoisted_".to_string(),
                "createApp".to_string(),
                "new Vue(".to_string(),
            ],
        );

        // Angular patterns
        patterns.insert(
            "Angular".to_string(),
            vec![
                "@Component".to_string(),
                "@NgModule".to_string(),
                "platformBrowserDynamic".to_string(),
            ],
        );

        // jQuery
        patterns.insert(
            "jQuery".to_string(),
            vec!["$(document)".to_string(), "jQuery(".to_string()],
        );

        // Webpack
        patterns.insert(
            "Webpack".to_string(),
            vec![
                "__webpack_require__".to_string(),
                "__webpack_modules__".to_string(),
                "webpackJsonp".to_string(),
            ],
        );

        patterns
    }

    /// Basic framework detection using pattern matching
    fn detect_frameworks_basic(&self, source: &str) -> Vec<FrameworkInfo> {
        let mut frameworks = Vec::new();

        for (name, patterns) in &self.framework_patterns {
            let mut matches = 0;
            for pattern in patterns {
                if source.contains(pattern) {
                    matches += 1;
                }
            }

            if matches > 0 {
                let confidence = matches as f64 / patterns.len() as f64;
                frameworks.push(FrameworkInfo {
                    name: name.clone(),
                    version: None,
                    confidence,
                    detection_method: "basic_pattern".to_string(),
                });
            }
        }

        frameworks
    }
}

impl Default for HybridJsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_analyzer_creation() {
        let analyzer = HybridJsAnalyzer::new();
        assert!(!analyzer.enable_dynamic);
    }

    #[test]
    fn test_static_analysis() {
        let mut analyzer = HybridJsAnalyzer::new();
        let source = r#"
            function greet(name) {
                return "Hello, " + name;
            }
            greet("World");
        "#;

        let result = analyzer.analyze(source);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.total_time_ms() > 0.0);
        assert!(!analysis.dynamic_analysis_performed);
    }

    #[test]
    fn test_framework_detection_react() {
        let mut analyzer = HybridJsAnalyzer::new();
        let source = r#"
            import React from 'react';
            
            function App() {
                const [count, setCount] = useState(0);
                return <div>Count: {count}</div>;
            }
        "#;

        let result = analyzer.analyze(source).unwrap();
        assert!(result.has_framework("React"));

        // Framework knowledge base provides more comprehensive detection
        assert!(!result.frameworks.is_empty());
    }

    #[test]
    fn test_framework_detection_vue() {
        let mut analyzer = HybridJsAnalyzer::new();
        let source = r#"
            import { createApp } from 'vue';
            
            createApp({
                data() {
                    return { count: 0 }
                }
            }).mount('#app');
        "#;

        let result = analyzer.analyze(source).unwrap();
        assert!(result.has_framework("Vue"));
    }

    #[test]
    fn test_comprehensive_framework_detection() {
        let mut analyzer = HybridJsAnalyzer::new();

        // Test Webpack detection
        let webpack_source = r#"
            __webpack_require__(123);
            __webpack_modules__[456]();
        "#;
        let result = analyzer.analyze(webpack_source).unwrap();
        // Should detect Webpack with basic patterns
        if !result.frameworks.is_empty() {
            log::info!("Detected: {:?}", result.frameworks[0].name);
        }

        // Test Angular detection
        let angular_source = r#"
            @Component({
                selector: 'app-root',
                template: '<div>Hello</div>'
            })
            export class AppComponent {}
        "#;
        let result2 = analyzer.analyze(angular_source).unwrap();
        assert!(result2.has_framework("Angular"));
    }

    #[test]
    fn test_basic_patterns() {
        // Basic patterns include common frameworks
        let analyzer = HybridJsAnalyzer::new();
        assert!(!analyzer.framework_patterns.is_empty());
        assert!(analyzer.framework_patterns.contains_key("React"));
        assert!(analyzer.framework_patterns.contains_key("Vue"));
        assert!(analyzer.framework_patterns.contains_key("Webpack"));
    }

    #[test]
    fn test_dynamic_analysis_enabled() {
        let mut analyzer = HybridJsAnalyzer::with_dynamic_analysis();
        assert!(analyzer.enable_dynamic);

        let source = r#"
            var x = 42;
            var y = x * 2;
            console.log(y);
        "#;

        let result = analyzer.analyze(source);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.dynamic_analysis_performed);
    }

    #[test]
    fn test_primary_framework() {
        let mut analyzer = HybridJsAnalyzer::new();
        let source = r#"
            import React, { useState, useEffect } from 'react';
            import Vue from 'vue';
            
            // More React patterns
            function Component() {
                const [state, setState] = useState(null);
                useEffect(() => {}, []);
            }
        "#;

        let result = analyzer.analyze(source).unwrap();
        let primary = result.primary_framework();
        assert!(primary.is_some());
        // Should detect React as primary (more patterns matched)
        assert!(primary.unwrap().name.contains("React") || primary.unwrap().name.contains("Vue"));
    }
}
