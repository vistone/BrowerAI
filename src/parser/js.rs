use anyhow::{Context, Result};
use boa_interner::Interner;
use boa_parser::Source;

use std::path::PathBuf;

use crate::ai::integration::JsModelIntegration;
use crate::ai::model_manager::ModelType;
use crate::ai::{AiRuntime, InferenceEngine};

/// JavaScript parser with AI enhancement capabilities
/// Uses Boa Parser - a pure Rust ECMAScript parser (part of boa JavaScript engine)
pub struct JsParser {
    inference_engine: Option<InferenceEngine>,
    ai_runtime: Option<AiRuntime>,
    model_path: Option<PathBuf>,
    model_name: Option<String>,
    enable_ai: bool,
    enforce_compatibility: bool,
}

/// Compatibility warning for JS features we do not fully support yet
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityWarning {
    pub feature: String,
    pub detail: String,
}

impl JsParser {
    /// Create a new JavaScript parser with Boa (native Rust)
    pub fn new() -> Self {
        Self {
            inference_engine: None,
            ai_runtime: None,
            model_path: None,
            model_name: None,
            enable_ai: false,
            enforce_compatibility: false,
        }
    }

    /// Create a new JavaScript parser with AI capabilities
    #[allow(dead_code)]
    pub fn with_ai(inference_engine: InferenceEngine) -> Self {
        Self {
            inference_engine: Some(inference_engine),
            ai_runtime: None,
            model_path: None,
            model_name: None,
            enable_ai: true,
            enforce_compatibility: false,
        }
    }

    /// Create a new JavaScript parser with AI runtime (engine + model catalog + monitor)
    #[allow(dead_code)]
    pub fn with_ai_runtime(ai_runtime: AiRuntime) -> Self {
        let (model_name, model_path) = ai_runtime
            .best_model(ModelType::JsParser)
            .map(|(cfg, path)| (Some(cfg.name), Some(path)))
            .unwrap_or((None, None));

        Self {
            inference_engine: Some(ai_runtime.engine()),
            ai_runtime: Some(ai_runtime),
            model_path,
            model_name,
            enable_ai: true,
            enforce_compatibility: false,
        }
    }

    /// Enable or disable strict compatibility enforcement
    pub fn set_enforce_compatibility(&mut self, enforce: bool) {
        self.enforce_compatibility = enforce;
    }

    /// Get compatibility enforcement status
    pub fn is_enforcing_compatibility(&self) -> bool {
        self.enforce_compatibility
    }

    /// Parse JavaScript content using Boa native Rust parser
    pub fn parse(&self, js: &str) -> Result<JsAst> {
        let compatibility_warnings = Self::detect_compatibility_issues(js);
        for warn in &compatibility_warnings {
            log::warn!(
                "JS compatibility warning: {} - {}",
                warn.feature,
                warn.detail
            );
        }

        if self.enforce_compatibility && !compatibility_warnings.is_empty() {
            let details: Vec<String> = compatibility_warnings
                .iter()
                .map(|w| format!("{}: {}", w.feature, w.detail))
                .collect();
            return Err(anyhow::anyhow!(
                "JS compatibility enforcement failed: {}",
                details.join(" | ")
            ));
        }

        // Create interner for string interning
        let mut interner = Interner::new();

        // Parse the JavaScript source
        let result = boa_parser::Parser::new(Source::from_bytes(js))
            .parse_script(
                &boa_ast::scope::Scope::new_global(),
                &mut interner,
            )
            .map_err(|e| anyhow::anyhow!("Parse error: {}", e))
            .context("Failed to parse JavaScript with Boa")?;

        let statement_count = result.statements().len();

        log::info!(
            "Successfully parsed JavaScript with {} statements using Boa native Rust parser",
            statement_count
        );

        // Check if AI runtime is available and AI is enabled
        let runtime = self.ai_runtime.as_ref();
        let ai_enabled = runtime.map_or(false, |r| r.is_ai_enabled()) && self.enable_ai;

        if !ai_enabled {
            if self.enable_ai && runtime.is_none() {
                log::debug!("AI enhancement disabled for JS parsing; no runtime available");
            } else {
                log::debug!("AI enhancement disabled for JS parsing; baseline path in use");
            }
            return Ok(JsAst {
                statement_count,
                is_valid: true,
            });
        }

        let runtime = runtime.unwrap();
        let tracker = runtime.fallback_tracker();
        tracker.record_attempt();

        // Try AI enhancement
        let ai_active = self.enable_ai && self.inference_engine.is_some();
        if !ai_active {
            use crate::ai::config::FallbackReason;
            tracker.record_fallback(FallbackReason::NoModelAvailable);
            log::warn!(
                "AI was requested for JavaScript parsing but no inference engine is available; falling back to baseline parser"
            );
            return Ok(JsAst {
                statement_count,
                is_valid: true,
            });
        }

        use std::time::Instant;
        use crate::ai::config::FallbackReason;

        let monitor = runtime.monitor();
        let model_path = self.model_path.as_deref();
        let model_name = self.model_name.as_deref().unwrap_or("js_model");
        let start_time = Instant::now();

        match JsModelIntegration::new(
            self.inference_engine.as_ref().unwrap(),
            model_path,
            monitor,
        ) {
            Ok(integration) => match integration.analyze_patterns(js) {
                Ok(patterns) => {
                    let elapsed_ms = start_time.elapsed().as_millis() as u64;
                    tracker.record_success(elapsed_ms);
                    log::info!(
                        "AI JS analysis (model={}, {}ms): detected {} patterns",
                        model_name, elapsed_ms,
                        patterns.len()
                    );
                }
                Err(err) => {
                    tracker.record_fallback(FallbackReason::InferenceFailed(err.to_string()));
                    log::warn!(
                        "AI JS analysis failed (model={}): {}; continuing with baseline AST",
                        model_name,
                        err
                    );
                }
            },
            Err(err) => {
                tracker.record_fallback(FallbackReason::ModelLoadFailed(err.to_string()));
                log::warn!(
                    "AI JS integration could not start (model={}): {}; continuing without AI",
                    model_name,
                    err
                );
            }
        }

        Ok(JsAst {
            statement_count,
            is_valid: true,
        })
    }

    /// Detect basic compatibility issues for known unsupported features
    fn detect_compatibility_issues(js: &str) -> Vec<CompatibilityWarning> {
        let mut warnings = Vec::new();

        // Boa parser here is used in script mode; ES modules are not supported in this path
        if js.contains("import ") || js.contains("export ") {
            warnings.push(CompatibilityWarning {
                feature: "ES modules".to_string(),
                detail: "Module syntax is not supported in script mode; use inline scripts or transpile.".to_string(),
            });
        }

        // Dynamic import not supported in this execution path
        if js.contains("import(") {
            warnings.push(CompatibilityWarning {
                feature: "dynamic import".to_string(),
                detail: "Dynamic import is not supported; consider bundling or static imports.".to_string(),
            });
        }

        // Top-level await not supported in script mode
        if js.contains("await ") && !js.contains("function") {
            warnings.push(CompatibilityWarning {
                feature: "top-level await".to_string(),
                detail: "Top-level await is not supported in script mode.".to_string(),
            });
        }

        warnings
    }

    /// Tokenize JavaScript for compatibility with old API
    #[allow(dead_code)]
    fn tokenize(&self, js: &str) -> Result<Vec<String>> {
        // Simple tokenization for backwards compatibility
        let tokens: Vec<String> = js
            .split(|c: char| c.is_whitespace() || "(){}[];,".contains(c))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        Ok(tokens)
    }

    /// Validate JavaScript syntax using Boa parser
    #[allow(dead_code)]
    pub fn validate(&self, js: &str) -> Result<bool> {
        // Use Boa to validate - if it parses successfully, it's valid
        match self.parse(js) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Enable or disable AI enhancement
    #[allow(dead_code)]
    pub fn set_ai_enabled(&mut self, enabled: bool) {
        self.enable_ai = enabled && self.inference_engine.is_some();
    }

    /// Check if AI enhancement is enabled
    #[allow(dead_code)]
    pub fn is_ai_enabled(&self) -> bool {
        self.enable_ai
    }
}

impl Default for JsParser {
    fn default() -> Self {
        Self::new()
    }
}

/// JavaScript AST representation using Boa
/// This is a simplified representation - the full Boa AST is available internally
#[derive(Debug, Clone)]
pub struct JsAst {
    pub statement_count: usize,
    pub is_valid: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_js() {
        let parser = JsParser::new();
        let js = "function hello() { return 'world'; }";
        let result = parser.parse(js);
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert!(ast.is_valid);
        assert!(ast.statement_count > 0);
    }

    #[test]
    fn test_detects_module_syntax_warning() {
        let warnings = JsParser::detect_compatibility_issues("import { a } from 'x';");
        assert!(!warnings.is_empty());
        assert!(warnings
            .iter()
            .any(|w| w.feature == "ES modules"));
    }

    #[test]
    fn test_validate_valid_js() {
        let parser = JsParser::new();
        let js = "if (true) { console.log('test'); }";
        let result = parser.validate(js);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_validate_invalid_js() {
        let parser = JsParser::new();
        let js = "if (true) { console.log('test';"; // Missing closing brace
        let result = parser.validate(js);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_parse_modern_js() {
        let parser = JsParser::new();
        let js = "const x = 10; const y = () => x + 5;";
        let result = parser.parse(js);
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert!(ast.is_valid);
        assert_eq!(ast.statement_count, 2);
    }
}
