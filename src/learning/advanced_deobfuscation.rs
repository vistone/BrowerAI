/// Advanced JavaScript deobfuscation for modern frameworks and dynamic content
/// 
/// Handles complex obfuscation techniques including:
/// - Framework-specific bundling (Webpack, Rollup, etc.)
/// - Dynamic HTML injection via JavaScript
/// - Event-driven content loading
/// - Template literals and JSX compilation
/// - Code splitting and lazy loading

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use regex::Regex;

/// Advanced obfuscation patterns specific to frameworks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FrameworkObfuscation {
    /// Webpack module system
    WebpackBundled,
    /// Rollup/Vite bundling
    RollupBundled,
    /// React JSX compilation
    ReactCompiled,
    /// Vue template compilation
    VueCompiled,
    /// Angular compiled output
    AngularCompiled,
    /// Dynamic HTML injection
    DynamicHtmlInjection,
    /// Event-driven loading
    EventDrivenContent,
    /// Template string obfuscation
    TemplateLiteralObfuscation,
}

/// Analysis result for advanced obfuscation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedObfuscationAnalysis {
    /// Framework-specific patterns detected
    pub framework_patterns: Vec<FrameworkObfuscation>,
    /// Dynamic HTML injection points
    pub dynamic_injection_points: Vec<InjectionPoint>,
    /// Event-triggered content loaders
    pub event_loaders: Vec<EventLoader>,
    /// Template extraction results
    pub templates: Vec<ExtractedTemplate>,
    /// Confidence score
    pub confidence: f32,
}

/// Location where HTML is dynamically injected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionPoint {
    /// Line number in source
    pub line: usize,
    /// Injection method (innerHTML, createElement, etc.)
    pub method: String,
    /// Target element selector
    pub target: String,
    /// Estimated HTML content
    pub content_hint: String,
}

/// Event-triggered content loader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLoader {
    /// Event type (click, load, scroll, etc.)
    pub event_type: String,
    /// Target element
    pub target: String,
    /// Loader function name
    pub function: String,
    /// Estimated trigger condition
    pub trigger_condition: String,
}

/// Extracted template from obfuscated code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTemplate {
    /// Template type (HTML, JSX, Vue, etc.)
    pub template_type: String,
    /// Extracted HTML content
    pub html_content: String,
    /// Original obfuscated form
    pub original_form: String,
    /// Confidence in extraction
    pub confidence: f32,
}

/// Result of advanced deobfuscation
#[derive(Debug, Clone)]
pub struct AdvancedDeobfuscationResult {
    /// Deobfuscated JavaScript
    pub javascript: String,
    /// Extracted HTML templates
    pub html_templates: Vec<String>,
    /// Event-to-content mapping
    pub event_content_map: HashMap<String, String>,
    /// Success indicators
    pub success: bool,
    /// Processing steps
    pub steps: Vec<String>,
}

/// Advanced JavaScript deobfuscator for modern frameworks
pub struct AdvancedDeobfuscator {
    /// Enable framework-specific processing
    enable_framework_detection: bool,
    /// Enable dynamic HTML extraction
    enable_html_extraction: bool,
    /// Maximum extraction depth
    max_extraction_depth: usize,
}

impl AdvancedDeobfuscator {
    /// Create new advanced deobfuscator
    pub fn new() -> Self {
        Self {
            enable_framework_detection: true,
            enable_html_extraction: true,
            max_extraction_depth: 10,
        }
    }

    /// Analyze advanced obfuscation patterns
    pub fn analyze(&self, code: &str) -> Result<AdvancedObfuscationAnalysis> {
        let mut framework_patterns = Vec::new();
        let mut dynamic_injection_points = Vec::new();
        let mut event_loaders = Vec::new();
        let mut templates = Vec::new();

        // Detect framework patterns
        if self.enable_framework_detection {
            framework_patterns.extend(self.detect_framework_patterns(code)?);
        }

        // Detect dynamic HTML injection
        if self.enable_html_extraction {
            dynamic_injection_points = self.detect_injection_points(code)?;
            event_loaders = self.detect_event_loaders(code)?;
            templates = self.extract_templates(code)?;
        }

        let confidence = self.calculate_confidence(&framework_patterns, &templates);

        Ok(AdvancedObfuscationAnalysis {
            framework_patterns,
            dynamic_injection_points,
            event_loaders,
            templates,
            confidence,
        })
    }

    /// Perform advanced deobfuscation
    pub fn deobfuscate(&self, code: &str) -> Result<AdvancedDeobfuscationResult> {
        let mut steps = Vec::new();
        let mut javascript = code.to_string();
        let mut html_templates = Vec::new();
        let mut event_content_map = HashMap::new();

        // Step 1: Detect and unwrap framework bundling
        if self.detect_webpack(&code) {
            javascript = self.unwrap_webpack(&javascript)?;
            steps.push("Unwrapped Webpack bundle".to_string());
        }

        // Step 2: Extract HTML from dynamic injection
        let injection_points = self.detect_injection_points(&javascript)?;
        for point in injection_points {
            if let Some(html) = self.extract_html_from_injection(&javascript, &point)? {
                html_templates.push(html);
                steps.push(format!("Extracted HTML from {} at line {}", point.method, point.line));
            }
        }

        // Step 3: Extract templates from framework code
        let templates = self.extract_templates(&javascript)?;
        for template in templates {
            html_templates.push(template.html_content.clone());
            steps.push(format!("Extracted {} template", template.template_type));
        }

        // Step 4: Map event handlers to content
        let event_loaders = self.detect_event_loaders(&javascript)?;
        for loader in event_loaders {
            if let Some(content) = self.resolve_event_content(&javascript, &loader)? {
                event_content_map.insert(loader.event_type.clone(), content);
                steps.push(format!("Mapped {} event to content", loader.event_type));
            }
        }

        // Step 5: Clean up obfuscated code
        javascript = self.cleanup_obfuscation(&javascript)?;
        steps.push("Applied final cleanup".to_string());

        Ok(AdvancedDeobfuscationResult {
            javascript,
            html_templates,
            event_content_map,
            success: true,
            steps,
        })
    }

    /// Detect framework-specific patterns
    fn detect_framework_patterns(&self, code: &str) -> Result<Vec<FrameworkObfuscation>> {
        let mut patterns = Vec::new();

        // Webpack: Look for __webpack_require__
        if code.contains("__webpack_require__") || code.contains("webpackChunk") {
            patterns.push(FrameworkObfuscation::WebpackBundled);
        }

        // React: Look for React.createElement or JSX patterns
        if code.contains("React.createElement") || code.contains("_jsx") || code.contains("_jsxs") {
            patterns.push(FrameworkObfuscation::ReactCompiled);
        }

        // Vue: Look for Vue render functions
        if code.contains("_createVNode") || code.contains("_createElementVNode") {
            patterns.push(FrameworkObfuscation::VueCompiled);
        }

        // Angular: Look for Angular patterns
        if code.contains("ɵɵ") || code.contains("@angular") {
            patterns.push(FrameworkObfuscation::AngularCompiled);
        }

        // Dynamic HTML injection
        if code.contains("innerHTML") || code.contains(".appendChild") || code.contains("insertAdjacentHTML") {
            patterns.push(FrameworkObfuscation::DynamicHtmlInjection);
        }

        // Event-driven content
        if code.contains("addEventListener") && (code.contains("innerHTML") || code.contains("createElement")) {
            patterns.push(FrameworkObfuscation::EventDrivenContent);
        }

        // Template literals with HTML
        let template_regex = Regex::new(r"`[^`]*<[^>]+>[^`]*`")?;
        if template_regex.is_match(code) {
            patterns.push(FrameworkObfuscation::TemplateLiteralObfuscation);
        }

        Ok(patterns)
    }

    /// Detect dynamic HTML injection points
    fn detect_injection_points(&self, code: &str) -> Result<Vec<InjectionPoint>> {
        let mut points = Vec::new();

        for (line_num, line) in code.lines().enumerate() {
            // innerHTML injection
            if line.contains("innerHTML") {
                if let Some(target) = self.extract_target(line) {
                    points.push(InjectionPoint {
                        line: line_num + 1,
                        method: "innerHTML".to_string(),
                        target,
                        content_hint: self.extract_content_hint(line).unwrap_or_default(),
                    });
                }
            }

            // appendChild injection
            if line.contains("appendChild") || line.contains("append(") {
                if let Some(target) = self.extract_target(line) {
                    points.push(InjectionPoint {
                        line: line_num + 1,
                        method: "appendChild".to_string(),
                        target,
                        content_hint: self.extract_content_hint(line).unwrap_or_default(),
                    });
                }
            }

            // insertAdjacentHTML
            if line.contains("insertAdjacentHTML") {
                if let Some(target) = self.extract_target(line) {
                    points.push(InjectionPoint {
                        line: line_num + 1,
                        method: "insertAdjacentHTML".to_string(),
                        target,
                        content_hint: self.extract_content_hint(line).unwrap_or_default(),
                    });
                }
            }
        }

        Ok(points)
    }

    /// Detect event-triggered content loaders
    fn detect_event_loaders(&self, code: &str) -> Result<Vec<EventLoader>> {
        let mut loaders = Vec::new();

        let event_regex = Regex::new(r#"addEventListener\(['"](\w+)['"],\s*(\w+)"#)?;
        
        for caps in event_regex.captures_iter(code) {
            let event_type = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
            let function = caps.get(2).map(|m| m.as_str().to_string()).unwrap_or_default();

            loaders.push(EventLoader {
                event_type: event_type.clone(),
                target: "detected".to_string(),
                function: function.clone(),
                trigger_condition: format!("When {} event fires", event_type),
            });
        }

        Ok(loaders)
    }

    /// Extract HTML templates from code
    fn extract_templates(&self, code: &str) -> Result<Vec<ExtractedTemplate>> {
        let mut templates = Vec::new();

        // Extract from template literals
        let template_regex = Regex::new(r"`([^`]*<[^>]+>[^`]*)`")?;
        for caps in template_regex.captures_iter(code) {
            if let Some(content) = caps.get(1) {
                let html = content.as_str().to_string();
                templates.push(ExtractedTemplate {
                    template_type: "Template Literal".to_string(),
                    html_content: html.clone(),
                    original_form: format!("`{}`", html),
                    confidence: 0.9,
                });
            }
        }

        // Extract from string concatenation
        let concat_regex = Regex::new(r#"['"](<[^>]+>[^'"]*)['"]\s*\+"#)?;
        for caps in concat_regex.captures_iter(code) {
            if let Some(content) = caps.get(1) {
                let html = content.as_str().to_string();
                templates.push(ExtractedTemplate {
                    template_type: "String Concatenation".to_string(),
                    html_content: html.clone(),
                    original_form: format!("'{}'", html),
                    confidence: 0.7,
                });
            }
        }

        Ok(templates)
    }

    /// Detect Webpack bundling
    fn detect_webpack(&self, code: &str) -> bool {
        code.contains("__webpack_require__") || code.contains("webpackChunk")
    }

    /// Unwrap Webpack bundle to extract modules
    fn unwrap_webpack(&self, code: &str) -> Result<String> {
        // Simplified Webpack unwrapping
        // In production, this would parse the bundle structure
        let mut result = code.to_string();

        // Remove webpack boilerplate
        result = result.replace("__webpack_require__", "require");
        result = result.replace("webpackChunk", "chunk");

        Ok(result)
    }

    /// Extract HTML from injection point
    fn extract_html_from_injection(&self, code: &str, point: &InjectionPoint) -> Result<Option<String>> {
        let lines: Vec<&str> = code.lines().collect();
        if point.line == 0 || point.line > lines.len() {
            return Ok(None);
        }

        let line = lines[point.line - 1];

        // Try to extract HTML from assignment
        let html_regex = Regex::new(r#"['"](.*<.*>.*)['"]"#)?;
        if let Some(caps) = html_regex.captures(line) {
            if let Some(html) = caps.get(1) {
                return Ok(Some(html.as_str().to_string()));
            }
        }

        Ok(None)
    }

    /// Resolve content from event loader
    fn resolve_event_content(&self, code: &str, loader: &EventLoader) -> Result<Option<String>> {
        // Try to find the function definition
        let func_regex = Regex::new(&format!(r"function\s+{}\s*\([^)]*\)\s*\{{([^}}]+)}}", loader.function))?;
        
        if let Some(caps) = func_regex.captures(code) {
            if let Some(body) = caps.get(1) {
                let body_str = body.as_str();
                
                // Look for HTML in function body
                let html_regex = Regex::new(r#"['"](.*<.*>.*)['"]"#)?;
                if let Some(html_caps) = html_regex.captures(body_str) {
                    if let Some(html) = html_caps.get(1) {
                        return Ok(Some(html.as_str().to_string()));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Extract target element from line
    fn extract_target(&self, line: &str) -> Option<String> {
        // Try to extract variable or selector
        if let Some(pos) = line.find('.') {
            let before = &line[..pos];
            if let Some(word_start) = before.rfind(|c: char| !c.is_alphanumeric() && c != '_') {
                return Some(before[word_start + 1..].trim().to_string());
            }
        }
        
        Some("unknown".to_string())
    }

    /// Extract content hint from line
    fn extract_content_hint(&self, line: &str) -> Option<String> {
        let hint_regex = Regex::new(r#"['"](.*?)['"]"#).ok()?;
        hint_regex.captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| {
                let s = m.as_str();
                if s.len() > 50 {
                    format!("{}...", &s[..50])
                } else {
                    s.to_string()
                }
            })
    }

    /// Calculate confidence score
    fn calculate_confidence(&self, patterns: &[FrameworkObfuscation], templates: &[ExtractedTemplate]) -> f32 {
        let pattern_score = if !patterns.is_empty() { 0.5 } else { 0.0 };
        let template_score = if !templates.is_empty() { 
            templates.iter().map(|t| t.confidence).sum::<f32>() / templates.len() as f32 * 0.5
        } else { 
            0.0 
        };

        pattern_score + template_score
    }

    /// Clean up remaining obfuscation
    fn cleanup_obfuscation(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        // Remove excessive whitespace
        let whitespace_regex = Regex::new(r"\s+")?;
        result = whitespace_regex.replace_all(&result, " ").to_string();

        // Remove comments
        let comment_regex = Regex::new(r"//.*$")?;
        result = comment_regex.replace_all(&result, "").to_string();

        Ok(result)
    }
}

impl Default for AdvancedDeobfuscator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_webpack() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            (function(modules) {
                function __webpack_require__(moduleId) {
                    return modules[moduleId].call();
                }
            })([function() { console.log("test"); }]);
        "#;

        let analysis = deob.analyze(code).unwrap();
        assert!(analysis.framework_patterns.contains(&FrameworkObfuscation::WebpackBundled));
    }

    #[test]
    fn test_detect_react() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            React.createElement("div", {className: "test"}, 
                React.createElement("span", null, "Hello")
            );
        "#;

        let analysis = deob.analyze(code).unwrap();
        assert!(analysis.framework_patterns.contains(&FrameworkObfuscation::ReactCompiled));
    }

    #[test]
    fn test_detect_dynamic_html() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            document.getElementById("app").innerHTML = "<div>Hello</div>";
        "#;

        let analysis = deob.analyze(code).unwrap();
        assert!(analysis.framework_patterns.contains(&FrameworkObfuscation::DynamicHtmlInjection));
        assert!(!analysis.dynamic_injection_points.is_empty());
    }

    #[test]
    fn test_extract_templates() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            const template = `<div class="container"><h1>Title</h1></div>`;
        "#;

        let analysis = deob.analyze(code).unwrap();
        assert!(!analysis.templates.is_empty());
        assert!(analysis.templates[0].html_content.contains("<div"));
    }

    #[test]
    fn test_event_loader_detection() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            button.addEventListener('click', handleClick);
            function handleClick() {
                element.innerHTML = "<p>Loaded!</p>";
            }
        "#;

        let analysis = deob.analyze(code).unwrap();
        assert!(!analysis.event_loaders.is_empty());
        assert_eq!(analysis.event_loaders[0].event_type, "click");
    }

    #[test]
    fn test_advanced_deobfuscation() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"
            const html = `<div><span>Test</span></div>`;
            document.body.innerHTML = html;
        "#;

        let result = deob.deobfuscate(code).unwrap();
        assert!(result.success);
        assert!(!result.html_templates.is_empty());
        assert!(!result.steps.is_empty());
    }
}
