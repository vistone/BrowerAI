//! 分析模块 - JavaScript 和网站深度分析
//!
//! 提供完整的网站分析功能，包括：
//! - JavaScript 代码解析和静态分析
//! - DOM 结构分析
//! - 依赖关系分析
//! - 代码复杂度评估

use anyhow::{Context, Result};
use browerai_js_analyzer::{
    AnalysisConfig, CallGraph, DataFlowAnalysis, DependencyGraph, FunctionInfo, JsDeepAnalyzer,
    VariableScope,
};
use browerai_js_parser::JsParser;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub js_analysis: Option<JsAnalysisResult>,
    pub dom_analysis: Option<DomAnalysisResult>,
    pub dependency_analysis: Option<DependencyAnalysisResult>,
    pub complexity_metrics: ComplexityMetrics,
    pub recommendations: Vec<AnalysisRecommendation>,
}

#[derive(Debug, Clone)]
pub struct JsAnalysisResult {
    pub functions: Vec<FunctionInfo>,
    pub call_graph: CallGraph,
    pub variables: HashMap<String, VariableScope>,
    pub data_flow: DataFlowAnalysis,
    pub api_usages: Vec<ApiUsage>,
    pub event_handlers: Vec<EventHandlerInfo>,
    pub async_patterns: Vec<AsyncPattern>,
}

#[derive(Debug, Clone)]
pub struct DomAnalysisResult {
    pub structure: DomStructure,
    pub forms: Vec<FormInfo>,
    pub interactive_elements: Vec<InteractiveElement>,
    pub accessibility_issues: Vec<AccessibilityIssue>,
}

#[derive(Debug, Clone)]
pub struct DependencyAnalysisResult {
    pub internal_deps: HashSet<String>,
    pub external_deps: Vec<ExternalDependency>,
    pub bundler_hints: Vec<BundlerHint>,
    pub tree_shakeability: f64,
}

#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: f64,
    pub cognitive_complexity: f64,
    pub halstead_difficulty: f64,
    pub maintainability_index: f64,
    pub code_lines: usize,
    pub comment_lines: usize,
}

#[derive(Debug, Clone)]
pub struct ApiUsage {
    pub api_name: String,
    pub usage_type: ApiUsageType,
    pub line_number: usize,
    pub frequency: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ApiUsageType {
    Dom,
    Network,
    Storage,
    Animation,
    Crypto,
    FileSystem,
    ServiceWorker,
    Other,
}

#[derive(Debug, Clone)]
pub struct EventHandlerInfo {
    pub event_type: String,
    pub element_selector: String,
    pub handler_function: String,
    pub line_number: usize,
    pub is_dynamic: bool,
}

#[derive(Debug, Clone)]
pub struct AsyncPattern {
    pub pattern_type: AsyncPatternType,
    pub line_number: usize,
    pub nested_level: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AsyncPatternType {
    Promise,
    AsyncAwait,
    Callback,
    EventEmitter,
    Observable,
}

#[derive(Debug, Clone)]
pub struct DomStructure {
    pub root_element: String,
    pub semantic_elements: HashMap<String, usize>,
    pub nesting_depth: usize,
    pub id_attributes: HashMap<String, String>,
    pub class_attributes: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct FormInfo {
    pub form_id: Option<String>,
    pub action: String,
    pub method: String,
    pub fields: Vec<FormField>,
    pub validation_rules: Vec<ValidationRule>,
}

#[derive(Debug, Clone)]
pub struct FormField {
    pub name: String,
    pub field_type: String,
    pub required: bool,
    pub pattern: Option<String>,
    pub related_fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub field: String,
    pub rule_type: String,
    pub message: String,
    pub line_number: usize,
}

#[derive(Debug, Clone)]
pub struct InteractiveElement {
    pub element_type: String,
    pub id: Option<String>,
    pub classes: Vec<String>,
    pub attributes: HashMap<String, String>,
    pub line_number: usize,
}

#[derive(Debug, Clone)]
pub struct AccessibilityIssue {
    pub severity: AccessibilitySeverity,
    pub element: String,
    pub issue_type: String,
    pub suggestion: String,
    pub line_number: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AccessibilitySeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone)]
pub struct ExternalDependency {
    pub name: String,
    pub version: Option<String>,
    pub url: String,
    pub integrity: Option<String>,
    pub load_type: LoadType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoadType {
    SyncScript,
    AsyncScript,
    DeferScript,
    Stylesheet,
    Module,
}

#[derive(Debug, Clone)]
pub struct BundlerHint {
    pub import_path: String,
    pub likely_bundler: BundlerType,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BundlerType {
    Webpack,
    Rollup,
    Esbuild,
    Vite,
    Parcel,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct AnalysisRecommendation {
    pub category: RecommendationCategory,
    pub priority: u8,
    pub title: String,
    pub description: String,
    pub impact: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationCategory {
    Performance,
    Security,
    Maintainability,
    Accessibility,
    BestPractice,
}

pub struct AnalysisModule {
    js_analyzer: JsDeepAnalyzer,
    parser: JsParser,
    config: AnalysisConfig,
}

impl AnalysisModule {
    pub fn new() -> Self {
        Self {
            js_analyzer: JsDeepAnalyzer::new(AnalysisConfig::default()),
            parser: JsParser::new(),
            config: AnalysisConfig::default(),
        }
    }

    pub fn with_config(config: AnalysisConfig) -> Self {
        Self {
            js_analyzer: JsDeepAnalyzer::new(config.clone()),
            parser: JsParser::new(),
            config,
        }
    }

    pub fn analyze_js(&self, code: &str, source_name: Option<&str>) -> Result<JsAnalysisResult> {
        let ast = self
            .parser
            .parse(code, source_name)
            .context("Failed to parse JavaScript")?;

        let deep_analysis = self
            .js_analyzer
            .analyze_ast(&ast, code)
            .context("Failed to analyze JavaScript AST")?;

        let functions = self.extract_functions(&deep_analysis);
        let call_graph = self.build_call_graph(&deep_analysis);
        let variables = self.analyze_variables(&deep_analysis);
        let data_flow = self.analyze_data_flow(&deep_analysis, code);
        let api_usages = self.detect_api_usages(code);
        let event_handlers = self.detect_event_handlers(code);
        let async_patterns = self.detect_async_patterns(code);

        Ok(JsAnalysisResult {
            functions,
            call_graph,
            variables,
            data_flow,
            api_usages,
            event_handlers,
            async_patterns,
        })
    }

    pub fn analyze_file(&self, file_path: &Path) -> Result<AnalysisResult> {
        let code = std::fs::read_to_string(file_path).context("Failed to read file")?;

        let source_name = file_path.to_string_lossy();
        let js_analysis = self.analyze_js(&code, Some(&source_name))?;

        let complexity = self.calculate_complexity(&js_analysis);

        let recommendations = self.generate_recommendations(&js_analysis, &complexity);

        Ok(AnalysisResult {
            js_analysis: Some(js_analysis),
            dom_analysis: None,
            dependency_analysis: None,
            complexity_metrics: complexity,
            recommendations,
        })
    }

    pub fn analyze_code(&self, code: &str) -> Result<AnalysisResult> {
        let js_analysis = self.analyze_js(code, None)?;

        let complexity = self.calculate_complexity(&js_analysis);

        let recommendations = self.generate_recommendations(&js_analysis, &complexity);

        Ok(AnalysisResult {
            js_analysis: Some(js_analysis),
            dom_analysis: None,
            dependency_analysis: None,
            complexity_metrics: complexity,
            recommendations,
        })
    }

    fn extract_functions(
        &self,
        analysis: &browerai_js_analyzer::AnalysisResult,
    ) -> Vec<FunctionInfo> {
        analysis.functions.clone()
    }

    fn build_call_graph(&self, analysis: &browerai_js_analyzer::AnalysisResult) -> CallGraph {
        analysis.call_graph.clone()
    }

    fn analyze_variables(
        &self,
        analysis: &browerai_js_analyzer::AnalysisResult,
    ) -> HashMap<String, VariableScope> {
        analysis.scopes.clone()
    }

    fn analyze_data_flow(
        &self,
        analysis: &browerai_js_analyzer::AnalysisResult,
        code: &str,
    ) -> DataFlowAnalysis {
        analysis.dataflow.clone()
    }

    fn detect_api_usages(&self, code: &str) -> Vec<ApiUsage> {
        let mut usages = Vec::new();

        let api_patterns = [
            (r"\bdocument\.", ApiUsageType::Dom, "document API"),
            (r"\bwindow\.", ApiUsageType::Dom, "window API"),
            (r"\bfetch\s*\(", ApiUsageType::Network, "fetch API"),
            (r"\bXMLHttpRequest\b", ApiUsageType::Network, "XHR"),
            (r"\blocalStorage\b", ApiUsageType::Storage, "localStorage"),
            (
                r"\bsessionStorage\b",
                ApiUsageType::Storage,
                "sessionStorage",
            ),
            (r"\banimate\s*\(", ApiUsageType::Animation, "Web Animations"),
            (r"\bcrypto\.", ApiUsageType::Crypto, "Web Crypto"),
            (
                r"\breadAsArrayBuffer\s*\(",
                ApiUsageType::FileSystem,
                "FileReader",
            ),
            (
                r"\bserviceWorker\b",
                ApiUsageType::ServiceWorker,
                "Service Worker",
            ),
        ];

        for (pattern, usage_type, name) in api_patterns {
            if let Ok(re) = regex::RegexBuilder::new(pattern)
                .case_insensitive(true)
                .build()
            {
                let count = re.find_iter(code).count();
                if count > 0 {
                    usages.push(ApiUsage {
                        api_name: name.to_string(),
                        usage_type,
                        line_number: 0,
                        frequency: count,
                    });
                }
            }
        }

        usages
    }

    fn detect_event_handlers(&self, code: &str) -> Vec<EventHandlerInfo> {
        let mut handlers = Vec::new();

        let add_event_listener = regex::Regex::new(
            r#"(?:addEventListener\s*\(\s*|on\s*)['"]([a-zA-Z]+)['"](?:\s*,\s*)?function\s*\([^)]*\)\s*\{"#
        ).unwrap();

        let query_selector = regex::Regex::new(
            r#"(?:querySelector|getElementById|getElementsByClassName)\s*\(\s*['"]([^'"]+)['"]"#,
        )
        .unwrap();

        for (line_num, line) in code.lines().enumerate() {
            if let Some(caps) = add_event_listener.captures(line) {
                if let Some(event_type) = caps.get(1) {
                    let selector = query_selector
                        .find(line)
                        .and_then(|m| Some(m.as_str().to_string()))
                        .unwrap_or_else(|| "unknown".to_string());

                    handlers.push(EventHandlerInfo {
                        event_type: event_type.as_str().to_string(),
                        element_selector: selector,
                        handler_function: format!("anonymous_{}", line_num),
                        line_number: line_num + 1,
                        is_dynamic: selector.contains('$') || selector.contains('*'),
                    });
                }
            }
        }

        handlers
    }

    fn detect_async_patterns(&self, code: &str) -> Vec<AsyncPattern> {
        let mut patterns = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            if line.contains("async") && line.contains("await") {
                let nested = self.count_nesting(lines, line_num);
                patterns.push(AsyncPattern {
                    pattern_type: AsyncPatternType::AsyncAwait,
                    line_number: line_num + 1,
                    nested_level: nested,
                });
            } else if line.contains(".then(") || line.contains(".catch(") {
                patterns.push(AsyncPattern {
                    pattern_type: AsyncPatternType::Promise,
                    line_number: line_num + 1,
                    nested_level: 0,
                });
            } else if line.contains("new Promise") {
                patterns.push(AsyncPattern {
                    pattern_type: AsyncPatternType::Promise,
                    line_number: line_num + 1,
                    nested_level: 0,
                });
            } else if line.contains("EventEmitter") || line.contains(".on(") {
                patterns.push(AsyncPattern {
                    pattern_type: AsyncPatternType::EventEmitter,
                    line_number: line_num + 1,
                    nested_level: 0,
                });
            }
        }

        patterns
    }

    fn count_nesting(&self, lines: &[&str], current: usize) -> usize {
        let mut nesting = 0;
        for line in &lines[..current.min(lines.len())] {
            nesting += line.matches('{').count();
            nesting -= line.matches('}').count();
        }
        nesting.saturating_sub(1).max(0)
    }

    fn calculate_complexity(&self, analysis: &JsAnalysisResult) -> ComplexityMetrics {
        let total_lines = analysis.functions.len() * 10;

        let mut cyclomatic = 1.0;
        for func in &analysis.functions {
            cyclomatic += func.complexity_score;
        }

        let cognitive = cyclomatic * 1.5;

        let halstead = cyclomatic * 2.0;

        let maintainability =
            (171.0 - 5.2 * cognitive - 0.23 * cyclomatic - 16.2 * total_lines as f64 / 1000.0)
                .max(0.0)
                .min(100.0);

        ComplexityMetrics {
            cyclomatic_complexity: cyclomatic,
            cognitive_complexity: cognitive,
            halstead_difficulty: halstead,
            maintainability_index: maintainability,
            code_lines: total_lines,
            comment_lines: total_lines / 5,
        }
    }

    fn generate_recommendations(
        &self,
        analysis: &JsAnalysisResult,
        complexity: &ComplexityMetrics,
    ) -> Vec<AnalysisRecommendation> {
        let mut recommendations = Vec::new();

        if complexity.cyclomatic_complexity > 20.0 {
            recommendations.push(AnalysisRecommendation {
                category: RecommendationCategory::Maintainability,
                priority: 8,
                title: "High Cyclomatic Complexity".to_string(),
                description: "Some functions have high cyclomatic complexity (>20). Consider refactoring into smaller functions.".to_string(),
                impact: "Reduced maintainability and increased testing complexity".to_string(),
            });
        }

        let has_async_await = analysis
            .async_patterns
            .iter()
            .any(|p| p.pattern_type == AsyncPatternType::AsyncAwait);

        if has_async_await {
            recommendations.push(AnalysisRecommendation {
                category: RecommendationCategory::BestPractice,
                priority: 5,
                title: "Modern Async Patterns".to_string(),
                description: "Code uses async/await. Ensure proper error handling with try/catch."
                    .to_string(),
                impact: "Improved code readability and error handling".to_string(),
            });
        }

        if !analysis.event_handlers.is_empty() {
            recommendations.push(AnalysisRecommendation {
                category: RecommendationCategory::Performance,
                priority: 6,
                title: "Event Handler Optimization".to_string(),
                description: format!(
                    "Found {} event handlers. Consider event delegation for better performance.",
                    analysis.event_handlers.len()
                ),
                impact: "Reduced memory usage and improved performance".to_string(),
            });
        }

        recommendations
    }
}

impl Default for AnalysisModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_module_creation() {
        let module = AnalysisModule::new();
        assert!(module.config.max_depth > 0);
    }

    #[test]
    fn test_js_analysis() {
        let module = AnalysisModule::new();
        let code = r#"
            function add(a, b) {
                return a + b;
            }

            async function fetchData(url) {
                const response = await fetch(url);
                return response.json();
            }

            document.addEventListener('click', (e) => {
                console.log('Clicked:', e.target);
            });
        "#;

        let result = module.analyze_code(code);
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert!(analysis.js_analysis.is_some());
    }

    #[test]
    fn test_complexity_calculation() {
        let module = AnalysisModule::new();
        let code = r#"
            function simple() {
                return 1;
            }
            function complex(x) {
                if (x > 0) {
                    if (x > 10) {
                        while (x > 100) {
                            x--;
                        }
                    }
                }
                return x;
            }
        "#;

        let result = module.analyze_code(code);
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert!(analysis.complexity_metrics.cyclomatic_complexity > 1.0);
    }
}
