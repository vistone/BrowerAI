# JavaScript Compatibility Guide

## Overview

BrowerAI uses **Boa Parser** (pure Rust ECMAScript parser) for JavaScript parsing. This document outlines supported and unsupported features, pre-run compatibility checks, and execution quotas.

## Boa Parser Capabilities

### Supported Features ✅

**ECMAScript Core (ES5-ES2020)**:
- Variables: `var`, `let`, `const`
- Functions: regular functions, arrow functions
- Objects and arrays
- Control flow: `if`, `switch`, `for`, `while`, `do-while`
- Operators: arithmetic, logical, bitwise, comparison
- Template literals
- Destructuring assignment
- Spread operator
- Default parameters
- Rest parameters
- Classes and inheritance
- Async/await (within functions)
- Promises
- Generators
- Symbol
- Map, Set, WeakMap, WeakSet
- Proxy and Reflect
- Regular expressions

### Unsupported Features ⚠️

**ES Modules in Script Mode**:
- ❌ `import` statements
- ❌ `export` statements
- ❌ Dynamic `import()`
- **Workaround**: Use inline scripts or transpile with tools like Babel/esbuild

**Top-Level Await**:
- ❌ `await` at module top level (outside functions)
- **Workaround**: Wrap in async IIFE: `(async () => { await ... })()`

**Web APIs** (not JavaScript language features):
- ❌ `fetch`, `XMLHttpRequest`
- ❌ `setTimeout`, `setInterval`
- ❌ DOM APIs (provided separately via BrowerAI's DOM module)
- ❌ `localStorage`, `sessionStorage`
- ❌ `WebSocket`, `WebRTC`
- **Note**: Some APIs are sandboxed out for security

**Experimental Features**:
- ❌ Stage 3 and below TC39 proposals
- ❌ TypeScript syntax (needs transpilation)
- ❌ JSX (needs transpilation)

## Pre-Run Compatibility Checks

BrowerAI automatically detects compatibility issues before execution:

### Automatic Detection

```rust
use browerai::JsParser;

let parser = JsParser::new();
let result = parser.parse(r#"
    import { foo } from 'bar';  // Will trigger warning
    export default foo;          // Will trigger warning
"#);
```

**Console Output**:
```
WARN: JS compatibility warning: ES modules - Module syntax is not supported in script mode; use inline scripts or transpile.
```

### Compatibility Warnings

The parser returns `CompatibilityWarning` for each detected issue:

```rust
pub struct CompatibilityWarning {
    pub feature: String,  // e.g., "ES modules"
    pub detail: String,   // Detailed explanation
}
```

### Detected Patterns

1. **ES Modules**: Detects `import ` and `export ` keywords
2. **Dynamic Import**: Detects `import(` pattern
3. **Top-Level Await**: Detects `await ` outside function contexts

### Enforcement Mode

Enable strict compatibility checking:

```rust
let mut parser = JsParser::new();
parser.set_enforce_compatibility(true);  // Fails on incompatible code

let result = parser.parse("import foo from 'bar';");
// Returns Err with compatibility failure details
```

## Execution Quotas and Limits

BrowerAI's JavaScript sandbox enforces resource limits to prevent:
- Infinite loops
- Memory exhaustion
- Stack overflow
- Excessive computation

### Default Resource Limits

```rust
pub struct ResourceLimits {
    max_execution_time_ms: 5000,      // 5 seconds
    max_memory_bytes: 50 * 1024 * 1024, // 50 MB
    max_call_depth: 100,              // Stack depth
    max_operations: 1_000_000,        // Operation count
}
```

### Customizing Limits

```rust
use browerai::{JsSandbox, ResourceLimits};

let limits = ResourceLimits {
    max_execution_time_ms: 1000,  // 1 second
    max_memory_bytes: 10 * 1024 * 1024, // 10 MB
    max_call_depth: 50,
    max_operations: 100_000,
};

let sandbox = JsSandbox::new(limits);
```

### Enforced Restrictions

**Timeout Enforcement**:
- Tracks execution time from start
- Returns `SandboxError::TimeoutExceeded` if exceeded
- Prevents infinite loops and long-running scripts

**Operation Counting**:
- Increments counter on each operation
- Returns `SandboxError::OperationLimitExceeded` if exceeded
- Prevents computational DoS

**Stack Depth**:
- Tracks function call depth
- Returns `SandboxError::StackOverflow` if exceeded
- Prevents stack exhaustion attacks

**IO/Network Denial**:
- ❌ No file system access
- ❌ No network requests (`fetch`, `XMLHttpRequest` not available)
- ❌ No subprocess spawning
- ✅ Only in-memory computation allowed

**Sandboxed Globals**:
- No access to Node.js `process`, `require`, `__dirname`
- No access to browser `window.location.href` manipulation
- Controlled access to `console` for logging only

## Compatibility Testing

### Test Suite Structure

```rust
#[test]
fn test_es_modules_detection() {
    let parser = JsParser::new();
    let result = parser.parse("import foo from 'bar';");
    // Should parse but log compatibility warning
    assert!(result.is_ok());
}

#[test]
fn test_sandbox_timeout() {
    let sandbox = JsSandbox::with_defaults();
    let result = sandbox.execute("while(true) {}");
    // Should timeout and return error
    assert!(result.is_err());
}

#[test]
fn test_sandbox_stack_overflow() {
    let sandbox = JsSandbox::with_defaults();
    let result = sandbox.execute(r#"
        function recurse() { recurse(); }
        recurse();
    "#);
    // Should detect stack overflow
    assert!(result.is_err());
}
```

### Running Tests

```bash
# Run JS compatibility tests
cargo test js_compatibility

# Run sandbox quota tests
cargo test sandbox_limits
```

## Migration Guide

### From ES Modules to Script Mode

**Before** (unsupported):
```javascript
import { feature } from './module.js';
export default feature;
```

**After** (supported):
```javascript
// Option 1: Inline everything
const feature = { /* ... */ };

// Option 2: Use IIFE
(function() {
    const feature = { /* ... */ };
    window.myFeature = feature;  // Export to global
})();
```

### Transpilation Workflow

For modern JavaScript with unsupported features:

1. **Use Babel or esbuild**:
   ```bash
   npx esbuild input.js --bundle --format=iife --outfile=output.js
   ```

2. **Target ES2020**:
   ```json
   {
     "presets": [["@babel/preset-env", { "targets": "es2020" }]]
   }
   ```

3. **Parse transpiled output**:
   ```rust
   let transpiled = std::fs::read_to_string("output.js")?;
   let parser = JsParser::new();
   parser.parse(&transpiled)?;
   ```

## Future Engine Support

BrowerAI is designed with a sandbox interface to allow future integration of:
- **V8** (Google Chrome's engine)
- **QuickJS** (lightweight ES2020 engine)
- **WASM** (WebAssembly-based execution)

The sandbox abstraction ensures compatibility regardless of underlying engine.

## Best Practices

### ✅ DO:
- Use ES2020 features within functions
- Wrap async code in functions
- Test with pre-run compatibility checks
- Set appropriate resource limits
- Use transpilation for modern syntax

### ❌ DON'T:
- Use ES modules directly
- Rely on browser-specific APIs
- Use top-level await
- Create infinite loops
- Perform network requests
- Access file system

## Error Handling

### Compatibility Errors

```rust
match parser.parse(js_code) {
    Ok(ast) => {
        // Check for warnings
        for warning in parser.get_warnings() {
            log::warn!("Compatibility: {} - {}", warning.feature, warning.detail);
        }
    }
    Err(e) => {
        log::error!("Parse failed: {}", e);
    }
}
```

### Sandbox Errors

```rust
match sandbox.execute(js_code) {
    Ok(result) => println!("Result: {:?}", result),
    Err(SandboxError::TimeoutExceeded) => {
        log::error!("Execution timeout - possible infinite loop");
    }
    Err(SandboxError::StackOverflow) => {
        log::error!("Stack overflow - excessive recursion");
    }
    Err(SandboxError::OperationLimitExceeded) => {
        log::error!("Operation limit exceeded - too many operations");
    }
    Err(e) => log::error!("Sandbox error: {}", e),
}
```

## References

- [Boa Engine Documentation](https://github.com/boa-dev/boa)
- [ECMAScript Specification](https://tc39.es/ecma262/)
- [TC39 Proposals](https://github.com/tc39/proposals)
- [BrowerAI Sandbox Implementation](../../src/dom/sandbox.rs)
- [BrowerAI JS Parser](../../src/parser/js.rs)

## Version History

- **v1.0** (January 2026): Initial compatibility documentation
  - Boa parser integration
  - Compatibility detection system
  - Resource limits and quotas
  - Sandbox enforcement

---

**Last Updated**: January 2026  
**Status**: M2 Complete - JS Sandbox & Compatibility Documentation
