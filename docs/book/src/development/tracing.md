# Tracing and Logging Guide

BrowerAI uses structured logging with the `tracing` crate for comprehensive observability.

## Overview

The `tracing` crate provides:
- Structured, contextual logging
- Performance profiling
- Distributed tracing
- Multiple output formats

## Basic Usage

### Enabling Tracing

Add to your `Cargo.toml`:
```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = "0.3"
```

### Initialize in Your Application

```rust
use tracing_subscriber;

fn main() {
    // Initialize with default settings
    tracing_subscriber::fmt::init();
    
    // Your code here
}
```

### Advanced Configuration

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "browerai=debug,info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();
}
```

## Instrumentation

### Functions

Use the `#[tracing::instrument]` attribute:

```rust
use tracing::instrument;

#[instrument]
fn parse_html(html: &str) -> Result<Dom> {
    tracing::info!("Starting HTML parsing");
    // ... parsing logic ...
    tracing::debug!("Parsed {} bytes", html.len());
    Ok(dom)
}
```

### Events

Log events with different levels:

```rust
use tracing::{trace, debug, info, warn, error};

debug!("Processing request");
info!(url = %url, "Fetching page");
warn!("Cache miss for {}", url);
error!(error = ?err, "Failed to parse");
```

### Spans

Create spans for logical operations:

```rust
use tracing::span;

let span = span!(tracing::Level::INFO, "render");
let _enter = span.enter();

// All logs within this scope are associated with the span
render_page();
```

## Performance Profiling

### Timing Operations

```rust
use tracing::instrument;

#[instrument]
fn expensive_operation() {
    // Automatically records execution time
}
```

### Custom Metrics

```rust
use tracing::field;

tracing::info!(
    bytes_parsed = bytes.len(),
    duration_ms = duration.as_millis(),
    "Parsing complete"
);
```

## Output Formats

### JSON Format

```rust
tracing_subscriber::fmt()
    .json()
    .init();
```

### Pretty Format

```rust
tracing_subscriber::fmt()
    .pretty()
    .init();
```

### Compact Format

```rust
tracing_subscriber::fmt()
    .compact()
    .init();
```

## Filtering

### Environment Variable

```bash
RUST_LOG=browerai=debug,browerai_html_parser=trace cargo run
```

### Programmatic Filtering

```rust
use tracing_subscriber::EnvFilter;

let filter = EnvFilter::new("browerai=debug")
    .add_directive("browerai_html_parser=trace".parse()?);
```

## Best Practices

1. **Use appropriate levels**:
   - `error`: Failures that require attention
   - `warn`: Unusual but recoverable situations
   - `info`: High-level progress information
   - `debug`: Detailed debugging information
   - `trace`: Very detailed tracing information

2. **Add context**: Use structured fields for searchability

3. **Instrument critical paths**: Focus on performance-critical code

4. **Avoid sensitive data**: Don't log passwords, tokens, or PII

5. **Use spans for operations**: Group related log entries

## Integration with BrowerAI

### HTML Parser

```rust
use tracing::instrument;

#[instrument(skip(html), fields(html_len = html.len()))]
pub fn parse(&self, html: &str) -> Result<Dom> {
    tracing::debug!("Starting HTML parse");
    // ... parsing ...
    tracing::info!("Parse complete");
    Ok(dom)
}
```

### CSS Parser

```rust
#[instrument]
pub fn parse_stylesheet(&mut self, css: &str) -> Result<Vec<Rule>> {
    tracing::span!(tracing::Level::DEBUG, "parse_css").in_scope(|| {
        // ... parsing ...
    })
}
```

### Renderer

```rust
#[instrument(skip(dom))]
pub fn render(&mut self, dom: &Dom) -> Result<String> {
    tracing::info!("Rendering DOM");
    let start = Instant::now();
    
    let result = self.render_internal(dom)?;
    
    tracing::info!(
        duration_ms = start.elapsed().as_millis(),
        "Rendering complete"
    );
    
    Ok(result)
}
```

## Distributed Tracing

For distributed systems, use OpenTelemetry:

```toml
[dependencies]
tracing-opentelemetry = "0.18"
opentelemetry = "0.18"
```

```rust
use tracing_subscriber::layer::SubscriberExt;
use tracing_opentelemetry::OpenTelemetryLayer;

let tracer = opentelemetry_jaeger::new_pipeline()
    .with_service_name("browerai")
    .install_simple()?;

tracing_subscriber::registry()
    .with(OpenTelemetryLayer::new(tracer))
    .init();
```

## Resources

- [tracing documentation](https://docs.rs/tracing)
- [tracing-subscriber](https://docs.rs/tracing-subscriber)
- [OpenTelemetry](https://opentelemetry.io/)
