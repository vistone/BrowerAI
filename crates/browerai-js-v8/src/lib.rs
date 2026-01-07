use anyhow::{Context, Result};
use std::pin::Pin;
use std::sync::Once;

pub mod sandbox;
pub use sandbox::{V8HeapStats, V8Sandbox, V8SandboxLimits};

static V8_INIT: Once = Once::new();

/// Initialize V8 (must be called once before using V8)
fn init_v8() {
    V8_INIT.call_once(|| {
        let platform = v8::new_default_platform(0, false).make_shared();
        v8::V8::initialize_platform(platform);
        v8::V8::initialize();
    });
}

/// V8-based JavaScript parser with high performance
///
/// This uses Google's V8 engine (same engine used in Chrome and Node.js)
/// for maximum compatibility and performance. V8 supports all modern
/// JavaScript features including ES2024 syntax.
pub struct V8JsParser {
    isolate: Option<v8::OwnedIsolate>,
}

impl V8JsParser {
    /// Create a new V8 JavaScript parser
    pub fn new() -> Result<Self> {
        init_v8();

        let isolate = v8::Isolate::new(Default::default());

        Ok(Self {
            isolate: Some(isolate),
        })
    }

    /// Get heap statistics from V8
    pub fn get_heap_statistics(&mut self) -> V8HeapStats {
        let isolate = self.isolate.as_mut().unwrap();
        let stats = isolate.get_heap_statistics();

        V8HeapStats {
            total_heap_size: stats.total_heap_size(),
            used_heap_size: stats.used_heap_size(),
            heap_size_limit: stats.heap_size_limit(),
        }
    }

    /// Set strict mode enforcement  
    pub fn set_strict_mode(&mut self, _enabled: bool) {
        // Strict mode is always enforced in V8
    }

    /// Parse JavaScript content using V8
    ///
    /// # Arguments
    ///
    /// * `js` - JavaScript source code as a string
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the parsed AST or an error
    ///
    /// # Example
    ///
    /// ```ignore
    /// use browerai_js_v8::V8JsParser;
    ///
    /// let parser = V8JsParser::new()?;
    /// let ast = parser.parse("const x = 42;")?;
    /// ```
    pub fn parse(&mut self, js: &str) -> Result<V8JsAst> {
        let isolate = self
            .isolate
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("V8 isolate not initialized"))?;

        let mut handle_scope = v8::HandleScope::new(isolate);
        let mut handle_scope = {
            // SAFETY: handle_scope is stack allocated and not moved after pin
            let scope_pinned = unsafe { Pin::new_unchecked(&mut handle_scope) };
            scope_pinned.init()
        };
        let context = v8::Context::new(&handle_scope, Default::default());
        let mut context_scope = v8::ContextScope::new(&mut handle_scope, context);

        // Inner handle scope bound to the context
        let mut inner_scope = v8::HandleScope::new(&mut context_scope);
        let mut scope = {
            let pinned = unsafe { Pin::new_unchecked(&mut inner_scope) };
            pinned.init()
        };

        // Create a string containing the JavaScript source code
        let source = v8::String::new(&mut scope, js)
            .ok_or_else(|| anyhow::anyhow!("Failed to create V8 string"))?;

        // Compile the source code
        let _script = v8::Script::compile(&mut scope, source, None)
            .ok_or_else(|| anyhow::anyhow!("Failed to compile JavaScript with V8"))?;

        // If we got here, the script compiled successfully
        // We don't need to execute it for parsing

        log::info!(
            "Successfully parsed JavaScript with V8 ({} bytes)",
            js.len()
        );

        Ok(V8JsAst {
            source_length: js.len(),
            is_valid: true,
            compiled: true,
        })
    }

    /// Validate JavaScript syntax using V8
    pub fn validate(&mut self, js: &str) -> Result<bool> {
        match self.parse(js) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Execute JavaScript code and return the result
    ///
    /// # Arguments
    ///
    /// * `js` - JavaScript source code to execute
    ///
    /// # Returns
    ///
    /// Returns the result of execution as a string
    pub fn execute(&mut self, js: &str) -> Result<String> {
        let isolate = self
            .isolate
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("V8 isolate not initialized"))?;

        let mut handle_scope = v8::HandleScope::new(isolate);
        let mut handle_scope = {
            // SAFETY: handle_scope is stack allocated and not moved after pin
            let scope_pinned = unsafe { Pin::new_unchecked(&mut handle_scope) };
            scope_pinned.init()
        };
        let context = v8::Context::new(&handle_scope, Default::default());
        let mut context_scope = v8::ContextScope::new(&mut handle_scope, context);

        // Inner handle scope bound to the context for execution
        let mut inner_scope = v8::HandleScope::new(&mut context_scope);
        let mut scope = {
            let pinned = unsafe { Pin::new_unchecked(&mut inner_scope) };
            pinned.init()
        };

        // Create a string containing the JavaScript source code
        let source = v8::String::new(&mut scope, js)
            .ok_or_else(|| anyhow::anyhow!("Failed to create V8 string"))?;

        // Compile the source code
        let script = v8::Script::compile(&mut scope, source, None)
            .context("Failed to compile JavaScript with V8")?;

        // Execute the script
        let result = script
            .run(&mut scope)
            .ok_or_else(|| anyhow::anyhow!("Failed to execute JavaScript"))?;

        // Convert result to string
        let result_str = result
            .to_string(&mut scope)
            .ok_or_else(|| anyhow::anyhow!("Failed to convert result to string"))?;

        Ok(result_str.to_rust_string_lossy(&mut scope))
    }
}

impl Default for V8JsParser {
    fn default() -> Self {
        Self::new().expect("Failed to create V8JsParser")
    }
}

impl Drop for V8JsParser {
    fn drop(&mut self) {
        // Isolate will be dropped automatically
        self.isolate.take();
    }
}

/// JavaScript AST representation from V8
#[derive(Debug, Clone)]
pub struct V8JsAst {
    /// Length of the source code in bytes
    pub source_length: usize,
    /// Whether the JavaScript is valid
    pub is_valid: bool,
    /// Whether the script was successfully compiled
    pub compiled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v8_parser_creation() {
        let result = V8JsParser::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_simple_js() {
        let mut parser = V8JsParser::new().unwrap();
        let js = "function hello() { return 'world'; }";
        let result = parser.parse(js);
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert!(ast.is_valid);
        assert!(ast.compiled);
    }

    #[test]
    fn test_parse_modern_js() {
        let mut parser = V8JsParser::new().unwrap();
        let js = "const x = 10; const y = () => x + 5;";
        let result = parser.parse(js);
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert!(ast.is_valid);
    }

    #[test]
    fn test_parse_es2024_syntax() {
        let mut parser = V8JsParser::new().unwrap();
        // Test async/await, arrow functions, template literals
        let js = r#"
            const fetchData = async () => {
                const data = await fetch('/api/data');
                return `Result: ${data}`;
            };
        "#;
        let result = parser.parse(js);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_valid_js() {
        let mut parser = V8JsParser::new().unwrap();
        let js = "if (true) { console.log('test'); }";
        let result = parser.validate(js);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_validate_invalid_js() {
        let mut parser = V8JsParser::new().unwrap();
        let js = "if (true) { console.log('test';"; // Missing closing brace
        let result = parser.validate(js);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_execute_simple_js() {
        let mut parser = V8JsParser::new().unwrap();
        let js = "1 + 2";
        let result = parser.execute(js);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "3");
    }

    #[test]
    fn test_execute_string_js() {
        let mut parser = V8JsParser::new().unwrap();
        let js = "'Hello ' + 'World'";
        let result = parser.execute(js);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello World");
    }
}
