# browerai-metrics

Comprehensive metrics collection and monitoring for BrowerAI using Prometheus.

## Features

- **Parser Metrics**: Track HTML, CSS, and JavaScript parsing performance
- **V8 Metrics**: Monitor V8 heap usage, execution time, and compilations
- **Rendering Metrics**: Measure rendering performance
- **AI Metrics**: Track AI inference duration and model loads
- **Prometheus Export**: Standard Prometheus text format
- **Optional OpenTelemetry**: Distributed tracing support (feature: `otel`)

## Usage

```rust
use browerai_metrics::MetricsRegistry;

// Create global metrics registry
let metrics = MetricsRegistry::new()?;

// Track HTML parsing
let timer = metrics.html_parse_duration.start_timer();
let result = parse_html(html);
timer.observe_duration();
metrics.html_parse_total.inc();

// Export metrics for Prometheus scraping
let prometheus_text = metrics.export()?;
println!("{}", prometheus_text);
```

## Metrics

### Parser Metrics
- `html_parse_duration_seconds` - HTML parsing duration histogram
- `html_parse_total` - Total HTML parses counter
- `html_parse_errors` - HTML parse errors counter
- `css_parse_duration_seconds` - CSS parsing duration histogram
- `css_parse_total` - Total CSS parses counter
- `css_parse_errors` - CSS parse errors counter
- `js_parse_duration_seconds` - JavaScript parsing duration histogram
- `js_parse_total` - Total JS parses counter
- `js_parse_errors` - JS parse errors counter

### V8 Metrics
- `v8_heap_used_bytes` - V8 heap memory used gauge
- `v8_execution_duration_seconds` - V8 execution duration histogram
- `v8_compilations_total` - V8 compilations counter

### Rendering Metrics
- `render_duration_seconds` - Rendering duration histogram
- `render_total` - Total renders counter
- `render_errors` - Render errors counter

### AI Metrics
- `ai_inference_duration_seconds` - AI inference duration histogram
- `ai_inference_total` - Total AI inferences counter
- `ai_model_loads_total` - AI model loads counter

## Integration

### With HTTP Server

```rust
use axum::{Router, routing::get};

async fn metrics_handler() -> String {
    METRICS.export().unwrap_or_default()
}

let app = Router::new()
    .route("/metrics", get(metrics_handler));
```

### With OpenTelemetry (optional)

Enable the `otel` feature:
```toml
browerai-metrics = { version = "0.1", features = ["otel"] }
```

## License

MIT OR Apache-2.0
