# Advanced JavaScript Deobfuscation

## Overview

BrowerAI's advanced deobfuscation system handles modern JavaScript frameworks and complex obfuscation techniques that go far beyond basic name mangling and string encoding.

## Key Challenges Addressed

### 1. Framework Bundling

Modern web applications use bundlers like Webpack, Rollup, and Vite that wrap code in module loaders:

```javascript
// Webpack bundled code
(function(modules) {
    function __webpack_require__(moduleId) {
        // Module loading logic
    }
    __webpack_require__(0);
})([
    function(module, exports) {
        // Actual application code buried inside
    }
]);
```

**BrowerAI Solution**: Detects bundle patterns and unwraps modules to access the actual application code.

### 2. Dynamic HTML Injection

Many frameworks don't include HTML in the initial page load. Instead, they inject it dynamically via JavaScript:

```javascript
// HTML hidden in JavaScript
document.getElementById('app').innerHTML = `
    <div class="container">
        <h1>Welcome</h1>
        <p>Content loaded dynamically</p>
    </div>
`;
```

**BrowerAI Solution**: Detects injection points (`innerHTML`, `appendChild`, `insertAdjacentHTML`) and extracts the HTML content.

### 3. Event-Driven Content

Content may only load when specific events fire:

```javascript
button.addEventListener('click', function() {
    // HTML only appears after user clicks
    element.innerHTML = '<div>Clicked content!</div>';
});

window.addEventListener('scroll', function() {
    // More content loads on scroll
    if (window.scrollY > 500) {
        loadMoreContent();
    }
});
```

**BrowerAI Solution**: Maps event handlers to their content, allowing us to understand what will appear and when.

### 4. Framework-Compiled Templates

React, Vue, and Angular compile templates into JavaScript:

```javascript
// React JSX compiled to:
React.createElement('div', {className: 'app'},
    React.createElement('h1', null, 'Title')
);

// Vue template compiled to:
_createElementVNode("div", {class: "vue-app"},
    _createElementVNode("h1", null, "Title")
);
```

**BrowerAI Solution**: Recognizes framework patterns and reconstructs the original template structure.

### 5. Template Literals with HTML

ES6 template literals often contain HTML:

```javascript
const template = `
    <article>
        <h2>${title}</h2>
        <p>${content}</p>
    </article>
`;
```

**BrowerAI Solution**: Extracts HTML from template literals, handling interpolations.

## Architecture

### AdvancedDeobfuscator

The main class that orchestrates advanced deobfuscation:

```rust
pub struct AdvancedDeobfuscator {
    enable_framework_detection: bool,
    enable_html_extraction: bool,
    max_extraction_depth: usize,
}
```

### Key Methods

#### `analyze()` - Detect Patterns

```rust
pub fn analyze(&self, code: &str) -> Result<AdvancedObfuscationAnalysis>
```

Detects:
- Framework patterns (Webpack, React, Vue, Angular)
- Dynamic HTML injection points
- Event-driven content loaders
- Templates and their type

#### `deobfuscate()` - Extract Content

```rust
pub fn deobfuscate(&self, code: &str) -> Result<AdvancedDeobfuscationResult>
```

Returns:
- Cleaned JavaScript
- Extracted HTML templates
- Event-to-content mappings
- Processing steps taken

## Detection Algorithms

### Framework Pattern Detection

```rust
// Webpack: Look for __webpack_require__
if code.contains("__webpack_require__") || code.contains("webpackChunk") {
    patterns.push(FrameworkObfuscation::WebpackBundled);
}

// React: Look for React.createElement or JSX patterns  
if code.contains("React.createElement") || code.contains("_jsx") {
    patterns.push(FrameworkObfuscation::ReactCompiled);
}

// Vue: Look for Vue render functions
if code.contains("_createVNode") || code.contains("_createElementVNode") {
    patterns.push(FrameworkObfuscation::VueCompiled);
}
```

### Injection Point Detection

Uses regex patterns to find:
- `element.innerHTML = ...`
- `element.appendChild(...)`
- `element.insertAdjacentHTML(...)`
- `document.write(...)`

For each injection point, we record:
- Line number
- Method used
- Target element
- Content hint (preview of the HTML)

### Event Loader Detection

Finds event listeners and traces them to content generation:

```rust
let event_regex = Regex::new(r#"addEventListener\(['"](\w+)['"],\s*(\w+)"#)?;

for caps in event_regex.captures_iter(code) {
    let event_type = caps.get(1); // "click", "scroll", etc.
    let function = caps.get(2);   // handler function name
    
    // Then find the function and extract its HTML content
}
```

### Template Extraction

Multiple extraction methods:

1. **Template Literals**: `` `<div>...</div>` ``
2. **String Concatenation**: `"<div>" + content + "</div>"`
3. **Framework Calls**: `createElement("div", ...)`

## Usage Examples

### Basic Analysis

```rust
use browerai::learning::AdvancedDeobfuscator;

let deob = AdvancedDeobfuscator::new();

let code = r#"
    document.getElementById('app').innerHTML = `
        <div class="container">
            <h1>My App</h1>
        </div>
    `;
"#;

let analysis = deob.analyze(code)?;
println!("Patterns: {:?}", analysis.framework_patterns);
println!("Templates: {}", analysis.templates.len());
```

### Full Deobfuscation

```rust
let result = deob.deobfuscate(code)?;

// Access extracted HTML
for template in result.html_templates {
    println!("Extracted HTML: {}", template);
}

// Access event mappings
for (event, content) in result.event_content_map {
    println!("Event '{}' loads: {}", event, content);
}
```

### Framework-Specific Processing

```rust
let analysis = deob.analyze(code)?;

if analysis.framework_patterns.contains(&FrameworkObfuscation::WebpackBundled) {
    println!("This is a Webpack bundle");
    // Special processing for Webpack
}

if analysis.framework_patterns.contains(&FrameworkObfuscation::ReactCompiled) {
    println!("This is compiled React code");
    // Reconstruct JSX from createElement calls
}
```

## Supported Frameworks

### Webpack
- Detects module wrapper
- Unwraps modules
- Extracts module code

### React
- Detects `React.createElement` calls
- Detects JSX runtime (`_jsx`, `_jsxs`)
- Reconstructs component structure

### Vue
- Detects `_createVNode` and `_createElementVNode`
- Extracts template structure
- Identifies reactive data

### Angular
- Detects Angular-specific patterns (`ɵɵ` prefix)
- Identifies components
- Extracts templates

### Generic Bundlers
- Rollup/Vite detection
- Parcel detection
- ESBuild output detection

## Real-World Examples

### Example 1: Single Page Application

```javascript
// Original obfuscated SPA code
(function() {
    const routes = {
        '/': `<div class="home"><h1>Home</h1></div>`,
        '/about': `<div class="about"><h1>About</h1></div>`
    };
    
    window.addEventListener('popstate', function() {
        const route = window.location.pathname;
        document.getElementById('app').innerHTML = routes[route];
    });
})();
```

**BrowerAI extracts**:
- 2 HTML templates (home and about pages)
- 1 event loader (popstate for routing)
- Route-to-content mapping

### Example 2: Lazy Loading

```javascript
document.querySelectorAll('.lazy-load').forEach(el => {
    const observer = new IntersectionObserver(entries => {
        if (entries[0].isIntersecting) {
            el.innerHTML = `<img src="${el.dataset.src}" />`;
        }
    });
    observer.observe(el);
});
```

**BrowerAI extracts**:
- Intersection Observer pattern
- Lazy-load trigger condition
- Content to be loaded (image template)

### Example 3: Form Validation

```javascript
form.addEventListener('submit', function(e) {
    e.preventDefault();
    const error = document.getElementById('error');
    error.innerHTML = `
        <div class="error-message">
            <p>Please fix the following errors:</p>
            <ul>${errors.map(e => `<li>${e}</li>`).join('')}</ul>
        </div>
    `;
});
```

**BrowerAI extracts**:
- Submit event handler
- Error message template
- Dynamic list generation pattern

## Performance

All operations are designed to be efficient:

- Pattern detection: O(n) single pass through code
- Regex matching: Compiled and cached
- Template extraction: Incremental parsing
- Typical processing time: <50ms for 100KB of JavaScript

## Testing

Comprehensive test suite covers:
- Webpack bundle detection
- React JSX compilation
- Vue template compilation
- Dynamic HTML injection
- Event-driven content
- Template literal extraction

Run tests:
```bash
cargo test advanced_deobfuscation
```

Run demo:
```bash
cargo run --example advanced_deobfuscation_demo
```

## Limitations and Future Work

### Current Limitations

1. **Encrypted Strings**: Cannot decode encrypted content (requires runtime execution)
2. **Async Loading**: Doesn't execute network requests to fetch remote content
3. **Computed Properties**: Limited support for dynamically computed HTML
4. **Minification**: Better results with non-minified code

### Planned Enhancements

1. **JavaScript Execution**: Optional V8 integration for runtime evaluation
2. **Network Tracing**: Intercept and record AJAX/fetch requests
3. **Source Maps**: Use source maps to reconstruct original code
4. **AST Transformation**: Full AST-based code transformation
5. **Machine Learning**: Train models to recognize obfuscation patterns

## Integration with BrowerAI

The advanced deobfuscator integrates with other BrowerAI components:

### Intelligent Rendering System

```rust
// In site understanding phase
let deob = AdvancedDeobfuscator::new();
let analysis = deob.analyze(&javascript_code)?;

// Use extracted templates in reasoning
for template in analysis.templates {
    // Analyze structure
    // Identify functionality
    // Generate alternatives
}
```

### Learning Loop

```rust
// Continuously improve detection
loop {
    let code = get_new_website_code();
    let analysis = deob.analyze(&code)?;
    
    // Learn from patterns
    update_detection_models(&analysis);
}
```

### Code Generation

```rust
// Use extracted patterns to generate similar code
let templates = extract_templates(obfuscated_code);
let generated = code_generator.generate_from_templates(templates);
```

## Best Practices

### 1. Always Analyze First

```rust
// Don't skip analysis
let analysis = deob.analyze(code)?;
if analysis.confidence < 0.5 {
    // Low confidence - might be simple code
    // Use basic deobfuscation instead
}
```

### 2. Handle Failures Gracefully

```rust
match deob.deobfuscate(code) {
    Ok(result) => {
        // Process result
    }
    Err(e) => {
        // Fall back to basic processing
        log::warn!("Advanced deobfuscation failed: {}", e);
    }
}
```

### 3. Validate Extracted HTML

```rust
for template in result.html_templates {
    // Validate HTML before using
    if is_valid_html(&template) {
        use_template(&template);
    }
}
```

### 4. Consider Performance

```rust
// For large files, set limits
let mut deob = AdvancedDeobfuscator::new();
deob.max_extraction_depth = 5; // Limit recursion

// Or process in chunks
for chunk in code.chunks(10_000) {
    process_chunk(chunk);
}
```

## Conclusion

BrowerAI's advanced deobfuscation goes far beyond traditional techniques. It understands modern JavaScript frameworks, extracts dynamically-generated HTML, and maps event-driven content loading.

This enables BrowerAI to:
- ✅ Process modern SPAs (React, Vue, Angular)
- ✅ Extract HTML hidden in JavaScript
- ✅ Understand event-driven interactions
- ✅ Handle framework-specific bundling
- ✅ Map content to user actions

All while maintaining the original functionality - the core promise of BrowerAI's intelligent rendering system.
