use anyhow::{Context, Result};
use boa_interner::Interner;
use boa_parser::Source;

use crate::ai::InferenceEngine;

/// JavaScript parser with AI enhancement capabilities
/// Uses Boa Parser - a pure Rust ECMAScript parser (part of boa JavaScript engine)
pub struct JsParser {
    inference_engine: Option<InferenceEngine>,
    enable_ai: bool,
}

impl JsParser {
    /// Create a new JavaScript parser with Boa (native Rust)
    pub fn new() -> Self {
        Self {
            inference_engine: None,
            enable_ai: false,
        }
    }

    /// Create a new JavaScript parser with AI capabilities
    #[allow(dead_code)]
    pub fn with_ai(inference_engine: InferenceEngine) -> Self {
        Self {
            inference_engine: Some(inference_engine),
            enable_ai: true,
        }
    }

    /// Parse JavaScript content using Boa native Rust parser
    pub fn parse(&self, js: &str) -> Result<JsAst> {
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

        // Apply AI-based optimizations
        if self.enable_ai && self.inference_engine.is_some() {
            log::debug!("AI enhancement enabled for JavaScript parsing");
            // Future: Use AI model to optimize JS, detect patterns,
            // suggest improvements, etc.
        }

        Ok(JsAst {
            statement_count,
            is_valid: true,
        })
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
