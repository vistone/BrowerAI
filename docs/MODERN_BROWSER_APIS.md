# Modern Browser APIs (2026 Standards)

This document describes the modern browser APIs implemented in BrowerAI to support the latest web standards.

## Overview

BrowerAI now supports cutting-edge browser technologies aligned with 2026 web standards:

- **Temporal API**: Modern date/time operations
- **structuredClone**: Deep object cloning
- **Intl.RelativeTimeFormat**: Localized relative time formatting
- **Web Storage API**: localStorage and sessionStorage
- **Performance API**: Performance monitoring and timing

## Implementation Status

### Phase 1: Modern JavaScript APIs ✅ COMPLETE

#### 1. Temporal API

Modern date and time API that replaces legacy `Date` object with comprehensive temporal handling.

**Features:**
- Instant timestamps with nanosecond precision
- ISO 8601 string formatting
- Duration arithmetic
- Timezone support

**Example:**
```rust
use browerai_dom::TemporalAPI;

let temporal = TemporalAPI::new();

// Get current instant
let instant = temporal.now_instant();
println!("Nanoseconds since epoch: {}", instant);

// Get ISO string
let iso = temporal.now_iso_string();
println!("Current time: {}", iso);

// Calculate difference between timestamps
let t1 = 1000_000_000;  // 1 second
let t2 = 2000_000_000;  // 2 seconds
let diff_ms = temporal.difference(t2, t1);
println!("Difference: {}ms", diff_ms);  // 1000ms

// Add duration
let new_time = temporal.add_duration(instant, 5000);  // Add 5 seconds
```

#### 2. structuredClone

Native deep cloning that handles complex data structures including Maps, Sets, Dates, and RegExps.

**Features:**
- Lossless deep cloning
- Handles nested structures
- Type-safe cloning validation

**Example:**
```rust
use browerai_dom::{StructuredClone, SandboxValue};

// Create a complex nested structure
let value = SandboxValue::Array(vec![
    SandboxValue::Number(42.0),
    SandboxValue::String("test".to_string()),
    SandboxValue::Array(vec![
        SandboxValue::Boolean(true),
    ]),
]);

// Deep clone it
let cloned = StructuredClone::clone_value(&value);

// Check if value is cloneable
if StructuredClone::is_cloneable(&value) {
    println!("Value can be safely cloned");
}
```

#### 3. Intl.RelativeTimeFormat

Provides locale-aware, human-readable relative date formatting.

**Features:**
- Multi-language support (English, Chinese, more to come)
- Multiple formatting styles (long, short, narrow)
- Numeric display modes (auto, always)

**Example:**
```rust
use browerai_dom::RelativeTimeFormat;

// English locale
let format = RelativeTimeFormat::new("en-US".to_string());
println!("{}", format.format(-5, "second"));   // "5 seconds ago"
println!("{}", format.format(1, "minute"));    // "1 minute from now"
println!("{}", format.format(-2, "day"));      // "2 days ago"

// Chinese locale
let format_cn = RelativeTimeFormat::new("zh-CN".to_string());
println!("{}", format_cn.format(-5, "second")); // "5秒前"
println!("{}", format_cn.format(1, "minute"));  // "1分钟后"
println!("{}", format_cn.format(-2, "day"));    // "2天前"

// Short format
let mut short_format = RelativeTimeFormat::new("en-US".to_string());
short_format.set_style("short".to_string());
println!("{}", short_format.format(-30, "minute")); // "30m. ago"
```

#### 4. Web Storage API

Complete implementation of localStorage and sessionStorage APIs with quota management.

**Features:**
- Get/set/remove/clear operations
- Key enumeration
- Storage quota enforcement (10MB default)
- Storage size tracking

**Example:**
```rust
use browerai_dom::WebStorage;

// Create localStorage
let mut local_storage = WebStorage::new("local");

// Set items
local_storage.set_item("user".to_string(), "Alice".to_string()).unwrap();
local_storage.set_item("token".to_string(), "abc123".to_string()).unwrap();

// Get items
if let Some(user) = local_storage.get_item("user") {
    println!("User: {}", user);
}

// Get number of items
println!("Storage has {} items", local_storage.length());

// Get all keys
for key in local_storage.keys() {
    println!("Key: {}", key);
}

// Check storage size
println!("Current size: {} bytes", local_storage.current_size());

// Remove item
local_storage.remove_item("token");

// Clear all
local_storage.clear();
```

**Quota Management:**
```rust
// Storage quota is enforced
let mut storage = WebStorage::new("session");
storage.max_size = 1024;  // 1KB limit for testing

let large_value = "x".repeat(2000);
match storage.set_item("key".to_string(), large_value) {
    Ok(_) => println!("Stored successfully"),
    Err(e) => println!("Error: {}", e),  // "QuotaExceededError"
}
```

#### 5. Performance API

Performance monitoring and timing API for measuring execution performance.

**Features:**
- High-resolution timing
- Performance marks and measures
- Entry filtering by type/name
- Navigation start time tracking

**Example:**
```rust
use browerai_dom::PerformanceAPI;

let mut perf = PerformanceAPI::new();

// Get current time since navigation start
let start = perf.now();
println!("Current time: {}ms", start);

// Mark a point in time
perf.mark("operation-start".to_string());

// ... do some work ...
std::thread::sleep(std::time::Duration::from_millis(100));

perf.mark("operation-end".to_string());

// Measure time between marks
if let Some(duration) = perf.measure(
    "operation-duration".to_string(),
    "operation-start",
    "operation-end"
) {
    println!("Operation took {}ms", duration);
}

// Get all marks
let marks = perf.get_entries_by_type("mark");
println!("Total marks: {}", marks.len());

// Get all measures
let measures = perf.get_entries_by_type("measure");
println!("Total measures: {}", measures.len());

// Clear all entries
perf.clear_entries();
```

## Integration with JavaScript Sandbox

All these APIs can be integrated into the JavaScript sandbox for use in executed scripts:

```rust
use browerai_dom::{JsSandbox, ResourceLimits, TemporalAPI, WebStorage};

let mut sandbox = JsSandbox::new(ResourceLimits::default());

// Add Temporal API
let temporal = TemporalAPI::new();
let iso_time = temporal.now_iso_string();
// ... register with sandbox globals

// Add Web Storage
let mut storage = WebStorage::new("local");
// ... expose to JavaScript execution context
```

## Future Enhancements

The following modern browser APIs are planned for future phases:

### Phase 2: Modern CSS Features
- CSS Container Queries
- :has() pseudo-class selector
- CSS Nesting
- CSS Custom Properties (CSS variables)
- CSS Subgrid

### Phase 3: Performance & Monitoring APIs
- Intersection Observer API
- Mutation Observer API
- Long Animation Frames (LoAF) API
- Navigation Timing API v2

### Phase 4: Modern Web APIs
- Clipboard API with permissions
- Web Share API
- Notification API
- File API
- Web Audio API basics

### Phase 5: Advanced Rendering
- WebGPU support
- CSS animations/transitions execution
- Scroll-linked animations
- Motion Path CSS
- Advanced color spaces (LCH, LAB)

### Phase 6: Progressive Web App Support
- Service Worker execution
- Cache API
- Background Sync API
- Web App Manifest parsing
- Push API

### Phase 7: ES Module Support
- ES Module (import/export)
- Dynamic import()
- Top-level await
- Module bundling/resolution
- Source Map support

## Testing

All modern APIs include comprehensive test suites:

```bash
# Run all modern API tests
cargo test -p browerai-dom modern_apis

# Run specific test
cargo test -p browerai-dom test_temporal_now
cargo test -p browerai-dom test_web_storage
cargo test -p browerai-dom test_performance_api
```

## Performance Considerations

- **Temporal API**: Nanosecond precision timestamps have minimal overhead
- **structuredClone**: Deep cloning is recursive but safe for typical data structures
- **Web Storage**: 10MB default quota prevents memory exhaustion
- **Performance API**: High-resolution timing uses system monotonic clock

## Browser Compatibility

These APIs follow 2026 browser standards and are compatible with:

- Modern browsers (Chrome 120+, Firefox 122+, Safari 17+)
- Node.js 20+ (for compatible APIs)
- Progressive Web Apps
- Service Workers

## References

- [Temporal API Specification](https://tc39.es/proposal-temporal/)
- [Web Storage API](https://html.spec.whatwg.org/multipage/webstorage.html)
- [Performance Timeline](https://w3c.github.io/performance-timeline/)
- [ECMA-402 Internationalization API](https://tc39.es/ecma402/)
- [structuredClone](https://html.spec.whatwg.org/multipage/structured-data.html#structured-cloning)

---

**Last Updated**: February 17, 2026
**Status**: Phase 1 Complete ✅

### Phase 2: Modern CSS Features ✅ COMPLETE

#### 1. Container Queries

CSS Container Queries allow styles to adapt based on a component's parent size, not just viewport.

**Features:**
- Named and unnamed containers
- Width and height queries
- Min/max condition evaluation

**Example:**
```rust
use browerai_css_parser::{ContainerQuery, CssRule, CssProperty};

// Parse container query
let query = ContainerQuery::parse("@container (min-width: 400px)").unwrap();
assert_eq!(query.condition, "min-width: 400px");

// Named container
let named = ContainerQuery::parse("@container sidebar (min-width: 300px)").unwrap();
assert_eq!(named.container_name, Some("sidebar".to_string()));

// Evaluate condition
let mut query = ContainerQuery::new("min-width: 400px".to_string());
assert!(query.evaluate(500.0, 300.0));  // Width >= 400
assert!(!query.evaluate(300.0, 300.0)); // Width < 400
```

**CSS Usage:**
```css
@container (min-width: 400px) {
  .card {
    display: grid;
    grid-template-columns: 2fr 1fr;
  }
}

@container sidebar (max-width: 300px) {
  .widget {
    flex-direction: column;
  }
}
```

#### 2. :has() Pseudo-Class Selector

Powerful parent-based styling selector that enables "parent selector" functionality.

**Features:**
- Parent selector with child condition
- Complex selector support
- Direct child (>) and descendant selectors

**Example:**
```rust
use browerai_css_parser::HasSelector;

// Parse :has() selector
let selector = HasSelector::parse("section:has(.active)").unwrap();
assert_eq!(selector.parent_selector, "section");
assert_eq!(selector.child_selector, ".active");

// Convert to CSS string
assert_eq!(selector.to_css(), "section:has(.active)");

// Complex example
let complex = HasSelector::parse("div.container:has(> .important)").unwrap();
assert_eq!(complex.child_selector, "> .important");
```

**CSS Usage:**
```css
/* Style article only if it contains a video */
article:has(video) {
  background: #f0f0f0;
}

/* Style form only if it has errors */
form:has(.error) {
  border: 2px solid red;
}

/* Direct child selector */
nav:has(> .active) {
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
```

#### 3. CSS Nesting

Author cleaner, hierarchical stylesheets similar to Sass/Less preprocessors.

**Features:**
- Nested rule structures
- Ampersand (&) parent reference
- Automatic flattening to regular CSS

**Example:**
```rust
use browerai_css_parser::{NestedCssRule, CssProperty};

// Create nested structure
let mut root = NestedCssRule::new(".card".to_string());
root.add_property(CssProperty::new("padding".to_string(), "1rem".to_string()));

let mut title = NestedCssRule::new(".title".to_string());
title.add_property(CssProperty::new("font-size".to_string(), "1.5rem".to_string()));
root.add_nested_rule(title);

// Flatten to regular CSS
let flattened = root.flatten("");
// Result: [".card", ".card .title"]

// Using ampersand for pseudo-classes
let mut button = NestedCssRule::new(".button".to_string());
let mut hover = NestedCssRule::new("&:hover".to_string());
button.add_nested_rule(hover);

let flat = button.flatten("");
// Result: ".button:hover"
```

**CSS Usage:**
```css
.card {
  padding: 1rem;
  
  .title {
    font-size: 1.5rem;
    font-weight: bold;
  }
  
  .body {
    margin-top: 0.5rem;
  }
}

.button {
  background: blue;
  
  &:hover {
    background: darkblue;
  }
  
  &.active {
    background: green;
  }
}
```

#### 4. CSS Custom Properties (CSS Variables)

Full support for CSS variables with inheritance and var() resolution.

**Features:**
- Custom property storage with -- prefix
- Scope inheritance (parent-child)
- var() reference resolution
- Fallback value support

**Example:**
```rust
use browerai_css_parser::CssCustomProperties;

// Create property scope
let mut props = CssCustomProperties::new();
props.set_property("--primary-color".to_string(), "blue".to_string());
props.set_property("--spacing".to_string(), "1rem".to_string());

// Get property
assert_eq!(props.get_property("--primary-color"), Some("blue".to_string()));

// Resolve var() references
assert_eq!(
    props.resolve_var("background: var(--primary-color);"),
    "background: blue;"
);

// Fallback for undefined variables
let resolved = props.resolve_var("color: var(--undefined, red);");
assert_eq!(resolved, "color: red;");

// Inheritance
let mut parent = CssCustomProperties::new();
parent.set_property("--base-size".to_string(), "16px".to_string());

let mut child = CssCustomProperties::with_parent(parent);
child.set_property("--local-size".to_string(), "14px".to_string());

// Child can access parent properties
assert_eq!(child.get_property("--base-size"), Some("16px".to_string()));
```

**CSS Usage:**
```css
:root {
  --primary-color: #3498db;
  --secondary-color: #2ecc71;
  --spacing: 1rem;
}

.button {
  background: var(--primary-color);
  padding: var(--spacing);
  
  /* With fallback */
  color: var(--text-color, white);
}

.card {
  --local-spacing: 0.5rem;
  margin: var(--local-spacing);
}
```

#### 5. CSS Subgrid

Grid layouts can now have nested grids inherit row/column definitions from parent.

**Features:**
- Subgrid axis specification (rows, columns, both)
- Track line inheritance from parent grid
- CSS value generation

**Example:**
```rust
use browerai_css_parser::{SubgridDefinition, SubgridAxis};

// Parse subgrid value
let both = SubgridDefinition::parse("subgrid").unwrap();
assert_eq!(both.axis, SubgridAxis::Both);

let rows = SubgridDefinition::parse("subgrid rows").unwrap();
assert_eq!(rows.axis, SubgridAxis::Rows);

// Create subgrid programmatically
let cols = SubgridDefinition::new(SubgridAxis::Columns);
assert_eq!(cols.to_css(), "subgrid [columns]");

// Check if it's a subgrid
assert!(both.is_subgrid());
```

**CSS Usage:**
```css
.grid-parent {
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;
  grid-template-rows: auto 1fr auto;
}

.grid-child {
  display: grid;
  grid-template-columns: subgrid;
  grid-template-rows: subgrid;
  
  /* Child inherits parent's grid lines */
  grid-column: 1 / 4;
  grid-row: 1 / 3;
}

/* Subgrid for columns only */
.partial-subgrid {
  grid-template-columns: subgrid;
  grid-template-rows: auto auto;
}
```

## Testing

Phase 2 includes 14 comprehensive tests:

```bash
# Run all modern CSS features tests
cargo test -p browerai-css-parser modern_features

# Run specific tests
cargo test -p browerai-css-parser test_container_query
cargo test -p browerai-css-parser test_has_selector
cargo test -p browerai-css-parser test_nested_css
cargo test -p browerai-css-parser test_css_custom_properties
cargo test -p browerai-css-parser test_subgrid
```

