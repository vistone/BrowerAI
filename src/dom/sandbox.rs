/// JavaScript sandbox environment for safe code execution
/// 
/// Provides isolated execution context with resource limits

use std::collections::HashMap;
use std::time::{Duration, Instant};

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
            max_execution_time_ms: 5000, // 5 seconds
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
        let elapsed_ms = self.start_time
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

/// JavaScript sandbox for safe code execution
pub struct JsSandbox {
    /// Execution context
    context: ExecutionContext,
    /// Whether sandbox is in strict mode
    strict_mode: bool,
}

impl JsSandbox {
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            context: ExecutionContext::new(limits),
            strict_mode: true,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(ResourceLimits::default())
    }

    /// Enable or disable strict mode
    pub fn set_strict_mode(&mut self, strict: bool) {
        self.strict_mode = strict;
    }

    /// Execute JavaScript code (stub implementation)
    pub fn execute(&mut self, _code: &str) -> Result<SandboxValue, SandboxError> {
        self.context.start_execution();

        // In a real implementation, this would:
        // 1. Parse the JavaScript code
        // 2. Execute in isolated context
        // 3. Enforce resource limits
        // 4. Return the result

        // For now, return a stub value
        Ok(SandboxValue::Undefined)
    }

    /// Evaluate an expression (stub implementation)
    pub fn eval(&mut self, _expression: &str) -> Result<SandboxValue, SandboxError> {
        self.context.record_operation()?;
        
        // Stub implementation
        Ok(SandboxValue::Undefined)
    }

    /// Set a global variable in the sandbox
    pub fn set_global(&mut self, name: impl Into<String>, value: SandboxValue) {
        self.context.set_global(name.into(), value);
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
        assert_eq!(context.get_global("test"), Some(&SandboxValue::Number(42.0)));
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
            assert_eq!(map.get("key"), Some(&SandboxValue::String("value".to_string())));
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
        assert_eq!(sandbox.get_global("testVar"), Some(&SandboxValue::Number(100.0)));
    }

    #[test]
    fn test_sandbox_execute() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.execute("var x = 42;");
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_eval() {
        let mut sandbox = JsSandbox::with_defaults();
        let result = sandbox.eval("2 + 2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_reset() {
        let mut sandbox = JsSandbox::with_defaults();
        sandbox.set_global("test", SandboxValue::Number(42.0));

        sandbox.reset();
        assert!(sandbox.get_global("test").is_none());
    }

    #[test]
    fn test_sandbox_strict_mode() {
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
