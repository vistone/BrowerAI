# JavaScript Deobfuscation Techniques - GitHub Research Summary

This document summarizes JavaScript deobfuscation techniques learned from popular GitHub projects and implemented in BrowerAI.

## Research Sources

### 1. [webcrack](https://github.com/j4k0xb/webcrack) by j4k0xb ⭐ 2.3k
**Key Techniques Learned:**
- String array detection and unpacking
- Array rotation reversal
- Decoder wrapper function inlining
- Control flow object detection
- Control flow switch unflattening
- Dead code elimination
- Webpack/Browserify bundle unwrapping
- VM-based safe string decoder evaluation

**Implementation Status:**
- ✅ String array detection and unpacking
- ✅ Decoder wrapper concepts (simplified)
- ✅ Dead code elimination
- ✅ Control flow detection patterns
- ⚠️ VM-based evaluation (architecture only, disabled by default for security)

### 2. [synchrony](https://github.com/relative/synchrony) by relative ⭐ 500+
**Key Techniques Learned:**
- javascript-obfuscator/obfuscator.io specific patterns
- AST-based transformations
- Automatic string decoder detection
- Control flow unflattening
- Multi-pass deobfuscation

**Implementation Status:**
- ✅ Multi-pass iterative deobfuscation
- ✅ Convergence detection
- ✅ AST-inspired transformations
- ✅ Control flow pattern detection

### 3. [decode-js](https://github.com/echo094/decode-js) by echo094
**Key Techniques Learned:**
- StringArray with rotation, wrappers, and chained calls
- Control flow flattening (switch-based)
- Transformer patterns (object expressions, split strings)
- Custom code removal (self-defending, debug protection, console output)
- Comprehensive Chinese documentation and examples

**Implementation Status:**
- ✅ String array handling
- ✅ Self-defending code removal
- ✅ Debug protection removal
- ✅ Console hijacking removal
- ✅ Control flow flattening detection

### 4. [javascript-deobfuscator](https://github.com/ben-sb/javascript-deobfuscator) by ben-sb
**Key Techniques Learned:**
- Array unpacking and reference replacement
- Proxy function removal (simple, arithmetic, array access)
- Expression simplification
- String concatenation merging
- Hexadecimal identifier renaming
- Computed to static member expression conversion

**Implementation Status:**
- ✅ Array unpacking
- ✅ Proxy function detection and removal
- ✅ Expression simplification
- ✅ Member expression conversion
- ✅ Hex to decimal conversion

## Implemented Techniques

### Core Modules

#### 1. Enhanced Deobfuscation (`enhanced_deobfuscation.rs`)
Primary techniques from all sources:

**String Array Techniques:**
- Detection: `detect_string_array()` - Pattern matching for obfuscated arrays
- Unpacking: `unpack_string_array()` - Replace all array references with literals
- Supports: Hex indexing (0x0, 0x1, etc.)

**Proxy Function Techniques:**
- Detection: `detect_proxy_functions()` - Identifies wrapper functions
- Types supported:
  - Simple (direct call forwarding)
  - Arithmetic (expression wrappers)
  - Array access proxies
  - Chained proxies
- Removal: `remove_proxy_functions()` - Safe inlining

**Self-Defending Code Removal:**
- Anti-debugging (debugger statements)
- Console hijacking
- DevTools detection
- Domain locks
- Function toString checks
- Pattern: `detect_self_defending()`, `remove_self_defending()`

**Control Flow Simplification:**
- Switch-based state machine detection
- Object-based control flow detection
- Opaque predicate simplification (!![], ![], etc.)
- Pattern: `detect_control_flow_flattening()`, `unflatten_control_flow()`

**AST Transformations:**
- Constant folding (hex to decimal, arithmetic)
- Member expression simplification (obj["prop"] → obj.prop)
- String decoding (hex, unicode escapes)

**Statistics Tracking:**
- String arrays unpacked
- Proxy functions removed
- Control flow nodes simplified
- Constants folded
- Self-defending patterns removed
- Size reduction
- Readability improvement score

#### 2. AST Deobfuscation (`ast_deobfuscation.rs`)
Advanced techniques requiring deeper analysis:

**Variable Analysis:**
- Usage tracking: `track_variable_usage()`
- Constant identification
- Safe inlining decisions
- Single-use constant propagation

**Dead Code Elimination:**
- Unreachable code after return/throw
- False branch removal
- Unused function detection
- Unused variable removal

**Function Inlining:**
- Simple function call inlining
- Zero-parameter function optimization
- Return value propagation

**Array Rotation:**
- Detection: `detect_array_rotation()`
- Reversal: `reverse_array_rotation()`
- Restores original array order from rotated arrays

**Sequence Expression Simplification:**
- Detects comma-separated expressions
- Simplification opportunities

#### 3. Advanced Deobfuscation (`advanced_deobfuscation.rs`)
Framework-specific patterns (75+ frameworks detected):

**Bundler Detection:**
- Webpack (all versions)
- Rollup/Vite
- Parcel
- esbuild
- Turbopack
- Browserify
- SystemJS
- RequireJS/AMD

**Frontend Frameworks:**
- React (createElement → JSX)
- Vue (template compilation)
- Angular (Ivy)
- Svelte
- Solid.js
- Preact
- Ember.js
- Alpine.js
- And 40+ more...

**Chinese Frameworks:**
- Taro (京东)
- Uni-app (DCloud)
- mpvue (美团)
- Rax (阿里)
- Remax (阿里)
- Kbone (微信)
- Omi (腾讯)
- San (百度)
- Qiankun (阿里乾坤)
- And more...

**Dynamic HTML Extraction:**
- innerHTML injection detection
- appendChild tracking
- Template literal extraction
- Event-driven content mapping

## Architecture

### Multi-Pass Deobfuscation Flow

```rust
┌─────────────────────────────────────┐
│   Input: Obfuscated JavaScript      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Iteration Loop (Max 5-10 passes)  │
│                                      │
│  1. String Array Unpacking           │
│  2. Proxy Function Removal           │
│  3. Control Flow Unflattening        │
│  4. Constant Folding                 │
│  5. Self-Defending Removal           │
│  6. Member Expression Simplification │
│                                      │
│  Check convergence (code unchanged?) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  AST-based Post-Processing           │
│  - Variable inlining                 │
│  - Dead code elimination             │
│  - Function inlining                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Output: Deobfuscated JavaScript     │
│  + Statistics                        │
└─────────────────────────────────────┘
```

### Safety Considerations

**Enabled by Default:**
- Pattern-based transformations
- Static analysis
- Constant folding
- String literal decoding
- Dead code removal

**Disabled by Default (Security):**
- VM-based code evaluation
- Dynamic code execution
- Arbitrary code running

## Usage Examples

### Basic Usage

```rust
use browerai::learning::EnhancedDeobfuscator;

let mut deobfuscator = EnhancedDeobfuscator::new();
let result = deobfuscator.deobfuscate(obfuscated_code)?;

println!("Deobfuscated code: {}", result.code);
println!("Transformations: {}", result.transformations.len());
println!("Statistics: {:?}", result.stats);
```

### AST-based Analysis

```rust
use browerai::learning::ASTDeobfuscator;

let mut ast_deob = ASTDeobfuscator::new();
let result = ast_deob.deobfuscate(code)?;
let stats = ast_deob.get_stats();

println!("Variables inlined: {}", stats.variables_inlined);
println!("Dead code removed: {}", stats.dead_code_removed);
```

### Framework Detection

```rust
use browerai::learning::AdvancedDeobfuscator;

let deob = AdvancedDeobfuscator::new();
let analysis = deob.analyze(code)?;

for framework in &analysis.framework_patterns {
    let info = deob.get_framework_info(framework);
    println!("Detected: {} ({})", info.name, info.origin);
    println!("Strategy: {}", info.deobfuscation_strategy);
}
```

## Test Coverage

### Enhanced Deobfuscation
- ✅ String array detection
- ✅ String array unpacking
- ✅ Proxy function detection
- ✅ Self-defending detection
- ✅ Opaque predicate simplification
- ✅ Hex constant folding
- ✅ Member expression simplification
- ✅ Comprehensive deobfuscation

### AST Deobfuscation
- ✅ Constant inlining
- ✅ Dead code removal
- ✅ Simple function inlining
- ✅ Array rotation detection
- ✅ Variable usage tracking
- ✅ Simple constant identification
- ✅ Comprehensive AST deobfuscation

### Framework Detection
- ✅ Webpack detection
- ✅ React detection
- ✅ Dynamic HTML detection
- ✅ Template extraction
- ✅ Event loader detection
- ✅ Advanced deobfuscation flow

## Performance

**Multi-pass Convergence:**
- Typically converges in 2-4 iterations
- Maximum 5-10 iterations (configurable)
- Early exit on convergence
- No infinite loops

**Memory Usage:**
- Maintains single copy of code
- Incremental transformations
- Statistics tracking overhead minimal

## Future Enhancements

### Planned Features
1. **Full VM-based String Decoding**
   - Sandboxed execution environment
   - Safe decoder evaluation
   - Timeout protection

2. **Control Flow Graph Analysis**
   - Advanced CFG construction
   - State machine detection
   - Dispatcher identification

3. **Semantic Analysis**
   - Data flow tracking
   - Taint analysis
   - Side effect detection

4. **Machine Learning Integration**
   - Pattern learning from samples
   - Obfuscation technique classification
   - Automatic strategy selection

## Comparison with Other Tools

| Feature | BrowerAI | webcrack | synchrony | decode-js | ben-sb |
|---------|----------|----------|-----------|-----------|--------|
| String Arrays | ✅ | ✅ | ✅ | ✅ | ✅ |
| Proxy Functions | ✅ | ✅ | ✅ | ❌ | ✅ |
| Control Flow | ⚠️ | ✅ | ✅ | ✅ | ❌ |
| Self-Defending | ✅ | ⚠️ | ⚠️ | ✅ | ❌ |
| Framework Detection | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multi-language | Rust | TypeScript | TypeScript | JavaScript | TypeScript |
| AST Analysis | ✅ | ✅ | ✅ | ✅ | ✅ |
| VM Evaluation | ⚠️ | ✅ | ❌ | ✅ | ❌ |
| Statistics | ✅ | ⚠️ | ❌ | ❌ | ❌ |

Legend: ✅ Full support, ⚠️ Partial support, ❌ Not supported

## References

1. [webcrack Documentation](https://webcrack.netlify.app/docs)
2. [synchrony GitHub](https://github.com/relative/synchrony)
3. [decode-js Examples](https://github.com/echo094/decode-js)
4. [javascript-deobfuscator README](https://github.com/ben-sb/javascript-deobfuscator)
5. [Obfuscator.io](https://obfuscator.io) - Reference obfuscation tool
6. [JavaScript Obfuscator](https://github.com/javascript-obfuscator/javascript-obfuscator)

## Contributing

To add new deobfuscation techniques:

1. Research the obfuscation pattern
2. Add detection in `enhanced_deobfuscation.rs` or `ast_deobfuscation.rs`
3. Implement transformation
4. Add comprehensive tests
5. Update this documentation
6. Run demo: `cargo run --example enhanced_js_deobfuscation_demo`

## License

See main project LICENSE file.
