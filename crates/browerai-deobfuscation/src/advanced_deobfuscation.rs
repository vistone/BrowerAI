/// Advanced JavaScript deobfuscation for modern frameworks and dynamic content
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkInfo {
    pub name: String,
    pub category: String,
    pub patterns: Vec<&'static str>,
    pub deobfuscation_strategy: &'static str,
    pub origin: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FrameworkObfuscation {
    WebpackBundled,
    RollupBundled,
    ParcelBundled,
    EsbuildMinified,
    ReactCompiled,
    VueCompiled,
    AngularCompiled,
    SvelteCompiled,
    SolidJSCompiled,
    NextJSFramework,
    NuxtJSFramework,
    TaroFramework,
    UniAppFramework,
    MaterialUICompiled,
    AntDesignCompiled,
    TailwindJIT,
    ServerSideRendered,
    DynamicHtmlInjection,
    ESModules,
    CommonJS,
    UnknownFramework,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedObfuscationAnalysis {
    pub framework_patterns: Vec<FrameworkObfuscation>,
    pub dynamic_injection_points: Vec<InjectionPoint>,
    pub event_loaders: Vec<EventLoader>,
    pub templates: Vec<ExtractedTemplate>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionPoint {
    pub line: usize,
    pub method: String,
    pub target: String,
    pub content_hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLoader {
    pub event_type: String,
    pub target: String,
    pub function: String,
    pub trigger_condition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTemplate {
    pub template_type: String,
    pub html_content: String,
    pub original_form: String,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct AdvancedDeobfuscationResult {
    pub javascript: String,
    pub html_templates: Vec<String>,
    pub event_content_map: HashMap<String, String>,
    pub success: bool,
    pub steps: Vec<String>,
}

pub struct AdvancedDeobfuscator {
    enable_framework_detection: bool,
    enable_html_extraction: bool,
    _max_extraction_depth: usize,
}

impl AdvancedDeobfuscator {
    pub fn new() -> Self {
        Self {
            enable_framework_detection: true,
            enable_html_extraction: true,
            _max_extraction_depth: 10,
        }
    }

    pub fn analyze(&self, code: &str) -> Result<AdvancedObfuscationAnalysis> {
        let mut framework_patterns = Vec::new();
        let mut dynamic_injection_points = Vec::new();
        let mut event_loaders = Vec::new();
        let mut templates = Vec::new();

        if self.enable_framework_detection {
            framework_patterns.extend(self.detect_framework_patterns(code)?);
        }

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

    pub fn deobfuscate(&self, code: &str) -> Result<AdvancedDeobfuscationResult> {
        let mut steps = Vec::new();
        let mut javascript = code.to_string();
        let mut html_templates = Vec::new();
        let event_content_map = HashMap::new();

        if self.detect_webpack(code) {
            javascript = self.unwrap_webpack(&javascript)?;
            steps.push("Unwrapped Webpack bundle".to_string());
        }

        let templates = self.extract_templates(&javascript)?;
        for template in templates {
            html_templates.push(template.html_content.clone());
            steps.push(format!("Extracted {} template", template.template_type));
        }

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

    fn detect_framework_patterns(&self, code: &str) -> Result<Vec<FrameworkObfuscation>> {
        let mut patterns = Vec::new();

        if code.contains("__webpack_require__") || code.contains("webpackChunk") {
            patterns.push(FrameworkObfuscation::WebpackBundled);
        }
        if code.contains("import.meta") || code.contains("__vite") {
            patterns.push(FrameworkObfuscation::RollupBundled);
        }
        if code.contains("$parcel$") || code.contains("parcelRequire") {
            patterns.push(FrameworkObfuscation::ParcelBundled);
        }
        if code.contains("React.createElement") || code.contains("_jsx") {
            patterns.push(FrameworkObfuscation::ReactCompiled);
        }
        if code.contains("_createVNode") || code.contains("_createElementVNode") {
            patterns.push(FrameworkObfuscation::VueCompiled);
        }
        if code.contains("ɵɵ") || code.contains("@angular") {
            patterns.push(FrameworkObfuscation::AngularCompiled);
        }
        if code.contains("__next") || code.contains("next/router") {
            patterns.push(FrameworkObfuscation::NextJSFramework);
        }
        if code.contains("$nuxt") || code.contains("asyncData") {
            patterns.push(FrameworkObfuscation::NuxtJSFramework);
        }
        if code.contains("innerHTML") || code.contains(".appendChild") {
            patterns.push(FrameworkObfuscation::DynamicHtmlInjection);
        }

        Ok(patterns)
    }

    pub fn get_framework_info(&self, framework: &FrameworkObfuscation) -> FrameworkInfo {
        match framework {
            FrameworkObfuscation::WebpackBundled => FrameworkInfo {
                name: "Webpack".to_string(),
                category: "Bundler".to_string(),
                patterns: vec!["__webpack_require__", "webpackChunk"],
                deobfuscation_strategy: "Unwrap module system",
                origin: "Global".to_string(),
            },
            FrameworkObfuscation::ReactCompiled => FrameworkInfo {
                name: "React".to_string(),
                category: "Frontend Framework".to_string(),
                patterns: vec!["React.createElement", "_jsx"],
                deobfuscation_strategy: "Convert to JSX",
                origin: "USA (Meta)".to_string(),
            },
            FrameworkObfuscation::VueCompiled => FrameworkInfo {
                name: "Vue".to_string(),
                category: "Frontend Framework".to_string(),
                patterns: vec!["_createVNode", "_hoisted_"],
                deobfuscation_strategy: "Extract templates",
                origin: "China (Evan You)".to_string(),
            },
            _ => FrameworkInfo {
                name: "Unknown".to_string(),
                category: "Unknown".to_string(),
                patterns: vec![],
                deobfuscation_strategy: "Generic deobfuscation",
                origin: "Unknown".to_string(),
            },
        }
    }

    pub fn unwrap_webpack(&self, code: &str) -> Result<String> {
        let chunk_regex = Regex::new(
            r#"\(self\["webpackChunk[^"]*"\][^)]*\)\.push\(\[\[.*?\],\s*\{([^}]+)\}\]\)"#,
        )?;

        let mut result = String::new();
        result.push_str("// Unwrapped from Webpack bundle\n\n");

        if let Some(caps) = chunk_regex.captures(code) {
            if let Some(module_map) = caps.get(1) {
                let module_text = module_map.as_str();
                let module_regex = Regex::new(r#"(\d+):\s*function\([^)]*\)\s*\{([^}]*)\}"#)?;
                for module_cap in module_regex.captures_iter(module_text) {
                    if let (Some(id), Some(body)) = (module_cap.get(1), module_cap.get(2)) {
                        result.push_str(&format!("// Module {}\n", id.as_str()));
                        result.push_str(body.as_str());
                        result.push_str("\n\n");
                    }
                }
            }
        }

        if !result.contains("// Module") {
            result.push_str("// Warning: Could not extract modules\n");
            result.push_str(code);
        }

        Ok(result)
    }

    fn detect_injection_points(&self, code: &str) -> Result<Vec<InjectionPoint>> {
        let mut points = Vec::new();

        for (line_num, line) in code.lines().enumerate() {
            if line.contains("innerHTML") {
                points.push(InjectionPoint {
                    line: line_num + 1,
                    method: "innerHTML".to_string(),
                    target: "detected".to_string(),
                    content_hint: "".to_string(),
                });
            }
        }

        Ok(points)
    }

    fn detect_event_loaders(&self, code: &str) -> Result<Vec<EventLoader>> {
        let mut loaders = Vec::new();
        let event_regex = Regex::new(r#"addEventListener\(['"](\w+)['"],\s*(\w+)"#)?;

        for caps in event_regex.captures_iter(code) {
            let event_type = caps
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            let function = caps
                .get(2)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();

            loaders.push(EventLoader {
                event_type,
                target: "detected".to_string(),
                function,
                trigger_condition: "Event-driven".to_string(),
            });
        }

        Ok(loaders)
    }

    fn extract_templates(&self, code: &str) -> Result<Vec<ExtractedTemplate>> {
        let mut templates = Vec::new();
        let template_regex = Regex::new(r"`([^`]*<[^>]+>[^`]*)`")?;

        for caps in template_regex.captures_iter(code) {
            if let Some(content) = caps.get(1) {
                templates.push(ExtractedTemplate {
                    template_type: "Template Literal".to_string(),
                    html_content: content.as_str().to_string(),
                    original_form: format!("`{}`", content.as_str()),
                    confidence: 0.9,
                });
            }
        }

        Ok(templates)
    }

    fn detect_webpack(&self, code: &str) -> bool {
        code.contains("__webpack_require__") || code.contains("webpackChunk")
    }

    fn cleanup_obfuscation(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();
        let whitespace_regex = Regex::new(r"\s+")?;
        result = whitespace_regex.replace_all(&result, " ").to_string();
        Ok(result)
    }

    fn calculate_confidence(
        &self,
        patterns: &[FrameworkObfuscation],
        _templates: &[ExtractedTemplate],
    ) -> f32 {
        if !patterns.is_empty() {
            0.7
        } else {
            0.3
        }
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
        let code = r#"(function(modules) { function __webpack_require__(m) { return modules[m](); } })({});"#;
        assert!(deob.detect_webpack(code));
    }

    #[test]
    fn test_detect_react() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"React.createElement("div", null, "Hello");"#;
        let analysis = deob.analyze(code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::ReactCompiled));
    }

    #[test]
    fn test_extract_templates() {
        let deob = AdvancedDeobfuscator::new();
        let code = r#"const html = `<div><span>Test</span></div>`;""#;
        let analysis = deob.analyze(code).unwrap();
        assert!(!analysis.templates.is_empty());
    }
}
