use anyhow::{Context, Result};
use boa_interner::Interner;
use boa_parser::Source;

/// JavaScript parser with AI enhancement capabilities
/// Uses Boa Parser - a pure Rust ECMAScript parser (part of boa JavaScript engine)
pub struct JsParser {
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
            .parse_script(&boa_ast::scope::Scope::new_global(), &mut interner)
            .map_err(|e| anyhow::anyhow!("Parse error: {}", e))
            .context("Failed to parse JavaScript with Boa")?;

        let statement_count = result.statements().len();

        log::info!(
            "Successfully parsed JavaScript with {} statements using Boa native Rust parser",
            statement_count
        );


            // AI enhancement removed; returning baseline AST
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
                detail: "Dynamic import is not supported; consider bundling or static imports."
                    .to_string(),
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
        assert!(warnings.iter().any(|w| w.feature == "ES modules"));
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
