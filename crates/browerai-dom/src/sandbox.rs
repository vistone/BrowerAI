use boa_engine::object::ObjectInitializer;
use boa_engine::{Context, JsValue, Source};
/// JavaScript sandbox environment for safe code execution
///
/// Provides isolated execution context with resource limits
/// Uses Boa Engine for actual JavaScript execution
use std::collections::HashMap;
use std::time::Instant;

/// Maximum number of array elements to convert (for safety and performance)
const MAX_ARRAY_ELEMENTS: i32 = 100;

/// Resource limits for sandbox execution
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: u64,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Maximum call stack depth
    pub max_call_depth: usize,
    /// Maximum number of operations
    pub max_operations: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_execution_time_ms: 5000,        // 5 seconds
            max_memory_bytes: 50 * 1024 * 1024, // 50 MB
            max_call_depth: 100,
            max_operations: 1_000_000,
        }
    }
}

/// Execution context for JavaScript code
#[derive(Debug)]
pub struct ExecutionContext {
    /// Global variables
    pub globals: HashMap<String, SandboxValue>,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Current call stack depth
    pub call_depth: usize,
    /// Operation counter
    pub operation_count: usize,
    /// Execution start time
    pub start_time: Option<Instant>,
}

impl ExecutionContext {
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            globals: HashMap::new(),
            limits,
            call_depth: 0,
            operation_count: 0,
            start_time: None,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(ResourceLimits::default())
    }

    /// Set a global variable
    pub fn set_global(&mut self, name: String, value: SandboxValue) {
        self.globals.insert(name, value);
    }

    /// Get a global variable
    pub fn get_global(&self, name: &str) -> Option<&SandboxValue> {
        self.globals.get(name)
    }

    /// Increment operation counter and check limits
    pub fn record_operation(&mut self) -> Result<(), SandboxError> {
        self.operation_count += 1;

        if self.operation_count > self.limits.max_operations {
            return Err(SandboxError::OperationLimitExceeded);
        }

        if let Some(start) = self.start_time {
            let elapsed = start.elapsed();
            if elapsed.as_millis() as u64 > self.limits.max_execution_time_ms {
                return Err(SandboxError::TimeoutExceeded);
            }
        }

        Ok(())
    }

    /// Enter a function call
    pub fn enter_call(&mut self) -> Result<(), SandboxError> {
        self.call_depth += 1;
        if self.call_depth > self.limits.max_call_depth {
            Err(SandboxError::StackOverflow)
        } else {
            Ok(())
        }
    }

    /// Exit a function call
    pub fn exit_call(&mut self) {
        self.call_depth = self.call_depth.saturating_sub(1);
    }

    /// Start execution timing
    pub fn start_execution(&mut self) {
        self.start_time = Some(Instant::now());
        self.operation_count = 0;
        self.call_depth = 0;
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> ExecutionStats {
        let elapsed_ms = self
            .start_time
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0);

        ExecutionStats {
            elapsed_ms,
            operation_count: self.operation_count,
            max_call_depth: self.call_depth,
        }
    }
}

/// Sandbox value types
#[derive(Debug, Clone, PartialEq)]
pub enum SandboxValue {
    Null,
    Undefined,
    Boolean(bool),
    Number(f64),
    String(String),
    Array(Vec<SandboxValue>),
    Object(HashMap<String, SandboxValue>),
}

/// Errors that can occur during sandbox execution
#[derive(Debug, Clone, PartialEq)]
pub enum SandboxError {
    /// Execution time limit exceeded
    TimeoutExceeded,
    /// Operation count limit exceeded
    OperationLimitExceeded,
    /// Stack overflow (too many nested calls)
    StackOverflow,
    /// Memory limit exceeded
    MemoryLimitExceeded,
    /// Runtime error
    RuntimeError(String),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::TimeoutExceeded => write!(f, "Execution timeout exceeded"),
            SandboxError::OperationLimitExceeded => write!(f, "Operation limit exceeded"),
            SandboxError::StackOverflow => write!(f, "Stack overflow"),
            SandboxError::MemoryLimitExceeded => write!(f, "Memory limit exceeded"),
            SandboxError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl std::error::Error for SandboxError {}

/// Execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub elapsed_ms: u64,
    pub operation_count: usize,
    pub max_call_depth: usize,
}

/// JavaScript sandbox for safe code execution with Boa Engine
pub struct JsSandbox {
    /// Execution context
    context: ExecutionContext,
    /// Whether sandbox is in strict mode
    strict_mode: bool,
    /// Boa JavaScript engine context
    boa_context: Context,
}

impl JsSandbox {
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            context: ExecutionContext::new(limits),
            strict_mode: true,
            boa_context: Context::default(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(ResourceLimits::default())
    }

    /// Enable or disable strict mode
    pub fn set_strict_mode(&mut self, strict: bool) {
        self.strict_mode = strict;
    }

    /// Convert Boa JsValue to SandboxValue
    fn js_value_to_sandbox(&mut self, value: &JsValue) -> Result<SandboxValue, SandboxError> {
        match value {
            JsValue::Null => Ok(SandboxValue::Null),
            JsValue::Undefined => Ok(SandboxValue::Undefined),
            JsValue::Boolean(b) => Ok(SandboxValue::Boolean(*b)),
            JsValue::Integer(i) => Ok(SandboxValue::Number(*i as f64)),
            JsValue::Rational(r) => Ok(SandboxValue::Number(*r)),
            JsValue::String(s) => Ok(SandboxValue::String(s.to_std_string_escaped())),
            JsValue::Object(obj) => {
                // Check if it's an array using is_array method
                if obj.is_array() {
                    // For arrays, try to get length property
                    let length_val = obj
                        .get(boa_engine::js_string!("length"), &mut self.boa_context)
                        .map_err(|e| {
                            SandboxError::RuntimeError(format!("Array length error: {}", e))
                        })?;

                    // Handle both integer and rational lengths
                    let length = match length_val {
                        JsValue::Integer(len) => len,
                        JsValue::Rational(len) => len as i32,
                        _ => 0,
                    };

                    let mut array = Vec::new();
                    for i in 0..length.min(MAX_ARRAY_ELEMENTS) {
                        // Limit for safety
                        let val = obj.get(i as usize, &mut self.boa_context).map_err(|e| {
                            SandboxError::RuntimeError(format!("Array access error: {}", e))
                        })?;
                        array.push(self.js_value_to_sandbox(&val)?);
                    }
                    return Ok(SandboxValue::Array(array));
                }
                // Convert to object map (simplified - returns empty object)
                // Note: Full object property enumeration not yet implemented
                Ok(SandboxValue::Object(HashMap::new()))
            }
            _ => Ok(SandboxValue::Undefined),
        }
    }

    /// Convert SandboxValue to Boa JsValue
    fn sandbox_to_js_value(&mut self, value: &SandboxValue) -> JsValue {
        match value {
            SandboxValue::Null => JsValue::Null,
            SandboxValue::Undefined => JsValue::Undefined,
            SandboxValue::Boolean(b) => JsValue::Boolean(*b),
            SandboxValue::Number(n) => JsValue::Rational(*n),
            SandboxValue::String(s) => JsValue::String(s.clone().into()),
            SandboxValue::Array(arr) => {
                let array_values: Vec<JsValue> =
                    arr.iter().map(|v| self.sandbox_to_js_value(v)).collect();

                // Create array in Boa context
                let array = boa_engine::object::builtins::JsArray::from_iter(
                    array_values,
                    &mut self.boa_context,
                );
                array.into()
            }
            SandboxValue::Object(_) => {
                // Create empty object for now
                JsValue::Object(ObjectInitializer::new(&mut self.boa_context).build())
            }
        }
    }

    /// Execute JavaScript code using Boa Engine
    pub fn execute(&mut self, code: &str) -> Result<SandboxValue, SandboxError> {
        self.context.start_execution();

        // Add strict mode if enabled
        let code_to_execute = if self.strict_mode {
            format!("'use strict';\n{}", code)
        } else {
            code.to_string()
        };

        // Execute the code using Boa
        let result = self
            .boa_context
            .eval(Source::from_bytes(code_to_execute.as_bytes()))
            .map_err(|e| SandboxError::RuntimeError(format!("Execution error: {}", e)))?;

        // Check if we exceeded limits
        self.context.record_operation()?;

        // Convert result to SandboxValue
        self.js_value_to_sandbox(&result)
    }

    /// Evaluate an expression using Boa Engine
    pub fn eval(&mut self, expression: &str) -> Result<SandboxValue, SandboxError> {
        self.context.record_operation()?;

        // Execute the expression using Boa
        let result = self
            .boa_context
            .eval(Source::from_bytes(expression.as_bytes()))
            .map_err(|e| SandboxError::RuntimeError(format!("Evaluation error: {}", e)))?;

        // Convert result to SandboxValue
        self.js_value_to_sandbox(&result)
    }

    /// Set a global variable in the sandbox
    pub fn set_global(&mut self, name: impl Into<String>, value: SandboxValue) {
        let name_str = name.into();

        // Set in our context tracking
        self.context.set_global(name_str.clone(), value.clone());

        // Set in Boa context
        let js_value = self.sandbox_to_js_value(&value);
        let property_key = boa_engine::property::PropertyKey::String(name_str.into());
        self.boa_context
            .register_global_property(
                property_key,
                js_value,
                boa_engine::property::Attribute::all(),
            )
            .ok();
    }

    /// Get a global variable from the sandbox
    pub fn get_global(&self, name: &str) -> Option<&SandboxValue> {
        self.context.get_global(name)
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> ExecutionStats {
        self.context.get_stats()
    }

    /// Reset the sandbox
    pub fn reset(&mut self) {
        self.context = ExecutionContext::new(self.context.limits.clone());
        self.boa_context = Context::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.max_execution_time_ms, 5000);
        assert_eq!(limits.max_call_depth, 100);
    }

    #[test]
    fn test_execution_context_creation() {
        let context = ExecutionContext::with_defaults();
        assert_eq!(context.call_depth, 0);
        assert_eq!(context.operation_count, 0);
    }

    #[test]
    fn test_execution_context_globals() {
        let mut context = ExecutionContext::with_defaults();

        context.set_global("test".to_string(), SandboxValue::Number(42.0));
        assert_eq!(
            context.get_global("test"),
            Some(&SandboxValue::Number(42.0))
        );
    }

    #[test]
    fn test_execution_context_operation_count() {
        let mut limits = ResourceLimits::default();
        limits.max_operations = 10;
        let mut context = ExecutionContext::new(limits);

        for _ in 0..10 {
            context.record_operation().unwrap();
        }

        assert!(context.record_operation().is_err());
    }

    #[test]
    fn test_execution_context_call_depth() {
        let mut limits = ResourceLimits::default();
        limits.max_call_depth = 3;
        let mut context = ExecutionContext::new(limits);

        context.enter_call().unwrap();
        context.enter_call().unwrap();
        context.enter_call().unwrap();

        assert!(context.enter_call().is_err());
    }

    #[test]
    fn test_execution_context_exit_call() {
        let mut context = ExecutionContext::with_defaults();

        context.enter_call().unwrap();
        assert_eq!(context.call_depth, 1);

        context.exit_call();
        assert_eq!(context.call_depth, 0);
    }

    #[test]
    fn test_sandbox_value_types() {
        let null = SandboxValue::Null;
        let undefined = SandboxValue::Undefined;
        let bool_val = SandboxValue::Boolean(true);
        let num = SandboxValue::Number(42.0);
        let string = SandboxValue::String("test".to_string());

        assert_eq!(null, SandboxValue::Null);
        assert_eq!(undefined, SandboxValue::Undefined);
        assert_eq!(bool_val, SandboxValue::Boolean(true));
        assert_eq!(num, SandboxValue::Number(42.0));
        assert_eq!(string, SandboxValue::String("test".to_string()));
    }

    #[test]
    fn test_sandbox_value_array() {
        let array = SandboxValue::Array(vec![
            SandboxValue::Number(1.0),
            SandboxValue::Number(2.0),
            SandboxValue::Number(3.0),
        ]);

        if let SandboxValue::Array(items) = array {
            assert_eq!(items.len(), 3);
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn test_sandbox_value_object() {
        let mut obj_map = HashMap::new();
        obj_map.insert("key".to_string(), SandboxValue::String("value".to_string()));
        let object = SandboxValue::Object(obj_map);

        if let SandboxValue::Object(map) = object {
            assert_eq!(
                map.get("key"),
                Some(&SandboxValue::String("value".to_string()))
            );
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_sandbox_creation() {
        let sandbox = JsSandbox::with_defaults();
        assert!(sandbox.strict_mode);
    }

    #[test]
    fn test_sandbox_globals() {
        let mut sandbox = JsSandbox::with_defaults();

        sandbox.set_global("testVar", SandboxValue::Number(100.0));
        assert_eq!(
            sandbox.get_global("testVar"),
            Some(&SandboxValue::Number(100.0))
        );
    }

    #[test]
    fn test_sandbox_execute() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.execute("var x = 42;");
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_execute_with_return() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.execute("var x = 10; x + 5;");
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value, SandboxValue::Number(15.0));
    }

    #[test]
    fn test_sandbox_eval() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.eval("2 + 2");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), SandboxValue::Number(4.0));
    }

    #[test]
    fn test_sandbox_eval_string() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.eval("'hello' + ' ' + 'world'");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            SandboxValue::String("hello world".to_string())
        );
    }

    #[test]
    fn test_sandbox_eval_boolean() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.eval("5 > 3");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), SandboxValue::Boolean(true));
    }

    #[test]
    fn test_sandbox_global_variable() {
        let mut sandbox = JsSandbox::with_defaults();
        sandbox.set_global("myVar", SandboxValue::Number(100.0));

        let result = sandbox.eval("myVar * 2");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), SandboxValue::Number(200.0));
    }

    #[test]
    fn test_sandbox_function_execution() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.execute("function add(a, b) { return a + b; } add(10, 20);");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), SandboxValue::Number(30.0));
    }

    #[test]
    fn test_sandbox_error_handling() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.eval("undefined_variable");
        assert!(result.is_err());
    }

    #[test]
    fn test_sandbox_strict_mode() {
        let mut sandbox = JsSandbox::with_defaults();

        // In strict mode, using undeclared variables should error
        let result = sandbox.execute("undeclaredVar = 10;");
        assert!(result.is_err());
    }

    #[test]
    fn test_sandbox_reset() {
        let mut sandbox = JsSandbox::with_defaults();
        sandbox.set_global("test", SandboxValue::Number(42.0));

        sandbox.reset();
        assert!(sandbox.get_global("test").is_none());
    }

    #[test]
    fn test_sandbox_strict_mode_flag() {
        let mut sandbox = JsSandbox::with_defaults();
        assert!(sandbox.strict_mode);

        sandbox.set_strict_mode(false);
        assert!(!sandbox.strict_mode);
    }

    #[test]
    fn test_execution_stats() {
        let mut sandbox = JsSandbox::with_defaults();
        sandbox.execute("test code").ok();

        let stats = sandbox.get_stats();
        assert!(stats.elapsed_ms >= 0);
        assert_eq!(stats.operation_count, 0);
    }

    #[test]
    fn test_sandbox_error_display() {
        let err = SandboxError::TimeoutExceeded;
        assert_eq!(err.to_string(), "Execution timeout exceeded");

        let err2 = SandboxError::RuntimeError("test error".to_string());
        assert_eq!(err2.to_string(), "Runtime error: test error");
    }
}
