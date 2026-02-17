use lazy_static::lazy_static;
use prometheus::{
    register_counter_vec, register_gauge, register_histogram_vec, CounterVec, Encoder, Gauge,
    HistogramVec, TextEncoder,
};

lazy_static! {
    /// HTTP request counter by endpoint and status
    pub static ref HTTP_REQUESTS_TOTAL: CounterVec = register_counter_vec!(
        "browerai_http_requests_total",
        "Total number of HTTP requests",
        &["endpoint", "method", "status"]
    )
    .expect("Failed to create HTTP_REQUESTS_TOTAL metric");

    /// HTTP request duration histogram
    pub static ref HTTP_REQUEST_DURATION_SECONDS: HistogramVec = register_histogram_vec!(
        "browerai_http_request_duration_seconds",
        "HTTP request duration in seconds",
        &["endpoint", "method"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    .expect("Failed to create HTTP_REQUEST_DURATION_SECONDS metric");

    /// CSS parser cache hit counter
    pub static ref CSS_CACHE_HITS: CounterVec = register_counter_vec!(
        "browerai_css_cache_hits_total",
        "Total number of CSS cache hits",
        &["cache_type"]
    )
    .expect("Failed to create CSS_CACHE_HITS metric");

    /// CSS parser cache miss counter
    pub static ref CSS_CACHE_MISSES: CounterVec = register_counter_vec!(
        "browerai_css_cache_misses_total",
        "Total number of CSS cache misses",
        &["cache_type"]
    )
    .expect("Failed to create CSS_CACHE_MISSES metric");

    /// CSS parser cache size gauge
    pub static ref CSS_CACHE_SIZE: Gauge = register_gauge!(
        "browerai_css_cache_size",
        "Current size of CSS parser cache"
    )
    .expect("Failed to create CSS_CACHE_SIZE metric");

    /// AI inference duration histogram
    pub static ref AI_INFERENCE_DURATION_SECONDS: HistogramVec = register_histogram_vec!(
        "browerai_ai_inference_duration_seconds",
        "AI inference duration in seconds",
        &["model_type"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    .expect("Failed to create AI_INFERENCE_DURATION_SECONDS metric");

    /// AI inference counter
    pub static ref AI_INFERENCE_TOTAL: CounterVec = register_counter_vec!(
        "browerai_ai_inference_total",
        "Total number of AI inferences",
        &["model_type", "status"]
    )
    .expect("Failed to create AI_INFERENCE_TOTAL metric");

    /// CSS rules parsed counter
    pub static ref CSS_RULES_PARSED: CounterVec = register_counter_vec!(
        "browerai_css_rules_parsed_total",
        "Total number of CSS rules parsed",
        &["ai_enhanced"]
    )
    .expect("Failed to create CSS_RULES_PARSED metric");

    /// HTML elements parsed counter
    pub static ref HTML_ELEMENTS_PARSED: CounterVec = register_counter_vec!(
        "browerai_html_elements_parsed_total",
        "Total number of HTML elements parsed",
        &["ai_enhanced"]
    )
    .expect("Failed to create HTML_ELEMENTS_PARSED metric");
}

/// Export all metrics in Prometheus format
pub fn export_metrics() -> Result<String, Box<dyn std::error::Error>> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    Ok(String::from_utf8(buffer)?)
}

/// Helper function to record HTTP request
pub fn record_http_request(endpoint: &str, method: &str, status: u16) {
    HTTP_REQUESTS_TOTAL
        .with_label_values(&[endpoint, method, &status.to_string()])
        .inc();
}

/// Helper function to record HTTP request duration
pub fn record_http_duration(endpoint: &str, method: &str, duration_secs: f64) {
    HTTP_REQUEST_DURATION_SECONDS
        .with_label_values(&[endpoint, method])
        .observe(duration_secs);
}

/// Helper function to record cache hit
pub fn record_cache_hit(cache_type: &str) {
    CSS_CACHE_HITS.with_label_values(&[cache_type]).inc();
}

/// Helper function to record cache miss
pub fn record_cache_miss(cache_type: &str) {
    CSS_CACHE_MISSES.with_label_values(&[cache_type]).inc();
}

/// Helper function to update cache size
pub fn update_cache_size(size: usize) {
    CSS_CACHE_SIZE.set(size as f64);
}

/// Helper function to record AI inference
pub fn record_ai_inference(model_type: &str, success: bool, duration_secs: f64) {
    let status = if success { "success" } else { "error" };
    AI_INFERENCE_TOTAL
        .with_label_values(&[model_type, status])
        .inc();
    AI_INFERENCE_DURATION_SECONDS
        .with_label_values(&[model_type])
        .observe(duration_secs);
}

/// Helper function to record CSS rules parsed
pub fn record_css_rules_parsed(count: usize, ai_enhanced: bool) {
    let enhanced = if ai_enhanced { "true" } else { "false" };
    CSS_RULES_PARSED
        .with_label_values(&[enhanced])
        .inc_by(count as f64);
}

/// Helper function to record HTML elements parsed
pub fn record_html_elements_parsed(count: usize, ai_enhanced: bool) {
    let enhanced = if ai_enhanced { "true" } else { "false" };
    HTML_ELEMENTS_PARSED
        .with_label_values(&[enhanced])
        .inc_by(count as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        // Just verify metrics can be initialized
        assert!(export_metrics().is_ok());
    }

    #[test]
    fn test_record_http_request() {
        record_http_request("/api/health", "GET", 200);
        // Should not panic
    }

    #[test]
    fn test_record_cache_operations() {
        record_cache_hit("selector_embedding");
        record_cache_miss("selector_embedding");
        update_cache_size(42);
        // Should not panic
    }

    #[test]
    fn test_record_ai_inference() {
        record_ai_inference("selector_embedding", true, 0.123);
        record_ai_inference("property_predictor", false, 0.456);
        // Should not panic
    }
}
