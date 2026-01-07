//! Metrics collection and monitoring for BrowerAI
//!
//! This crate provides comprehensive metrics collection using Prometheus,
//! with optional OpenTelemetry support for distributed systems.

use anyhow::Result;
use prometheus::{
    Histogram, HistogramOpts, IntCounter, IntGauge, Registry,
};
use std::sync::Arc;

/// Global metrics registry for the entire application
pub struct MetricsRegistry {
    registry: Arc<Registry>,
    
    // Parser metrics
    pub html_parse_duration: Histogram,
    pub html_parse_total: IntCounter,
    pub html_parse_errors: IntCounter,
    
    pub css_parse_duration: Histogram,
    pub css_parse_total: IntCounter,
    pub css_parse_errors: IntCounter,
    
    pub js_parse_duration: Histogram,
    pub js_parse_total: IntCounter,
    pub js_parse_errors: IntCounter,
    
    // V8 metrics
    pub v8_heap_used: IntGauge,
    pub v8_execution_duration: Histogram,
    pub v8_compilations: IntCounter,
    
    // Rendering metrics
    pub render_duration: Histogram,
    pub render_total: IntCounter,
    pub render_errors: IntCounter,
    
    // AI metrics
    pub ai_inference_duration: Histogram,
    pub ai_inference_total: IntCounter,
    pub ai_model_loads: IntCounter,
}

impl MetricsRegistry {
    /// Create a new metrics registry with all counters and histograms
    pub fn new() -> Result<Self> {
        let registry = Registry::new();
        
        // HTML Parser metrics
        let html_parse_duration = Histogram::with_opts(
            HistogramOpts::new("html_parse_duration_seconds", "HTML parsing duration")
                .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]),
        )?;
        registry.register(Box::new(html_parse_duration.clone()))?;
        
        let html_parse_total = IntCounter::new("html_parse_total", "Total HTML parses")?;
        registry.register(Box::new(html_parse_total.clone()))?;
        
        let html_parse_errors = IntCounter::new("html_parse_errors", "HTML parse errors")?;
        registry.register(Box::new(html_parse_errors.clone()))?;
        
        // CSS Parser metrics
        let css_parse_duration = Histogram::with_opts(
            HistogramOpts::new("css_parse_duration_seconds", "CSS parsing duration")
                .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
        )?;
        registry.register(Box::new(css_parse_duration.clone()))?;
        
        let css_parse_total = IntCounter::new("css_parse_total", "Total CSS parses")?;
        registry.register(Box::new(css_parse_total.clone()))?;
        
        let css_parse_errors = IntCounter::new("css_parse_errors", "CSS parse errors")?;
        registry.register(Box::new(css_parse_errors.clone()))?;
        
        // JS Parser metrics
        let js_parse_duration = Histogram::with_opts(
            HistogramOpts::new("js_parse_duration_seconds", "JavaScript parsing duration")
                .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]),
        )?;
        registry.register(Box::new(js_parse_duration.clone()))?;
        
        let js_parse_total = IntCounter::new("js_parse_total", "Total JS parses")?;
        registry.register(Box::new(js_parse_total.clone()))?;
        
        let js_parse_errors = IntCounter::new("js_parse_errors", "JS parse errors")?;
        registry.register(Box::new(js_parse_errors.clone()))?;
        
        // V8 metrics
        let v8_heap_used = IntGauge::new("v8_heap_used_bytes", "V8 heap memory used")?;
        registry.register(Box::new(v8_heap_used.clone()))?;
        
        let v8_execution_duration = Histogram::with_opts(
            HistogramOpts::new("v8_execution_duration_seconds", "V8 execution duration")
                .buckets(vec![0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
        )?;
        registry.register(Box::new(v8_execution_duration.clone()))?;
        
        let v8_compilations = IntCounter::new("v8_compilations_total", "V8 compilations")?;
        registry.register(Box::new(v8_compilations.clone()))?;
        
        // Rendering metrics
        let render_duration = Histogram::with_opts(
            HistogramOpts::new("render_duration_seconds", "Rendering duration")
                .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]),
        )?;
        registry.register(Box::new(render_duration.clone()))?;
        
        let render_total = IntCounter::new("render_total", "Total renders")?;
        registry.register(Box::new(render_total.clone()))?;
        
        let render_errors = IntCounter::new("render_errors", "Render errors")?;
        registry.register(Box::new(render_errors.clone()))?;
        
        // AI metrics
        let ai_inference_duration = Histogram::with_opts(
            HistogramOpts::new("ai_inference_duration_seconds", "AI inference duration")
                .buckets(vec![0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0]),
        )?;
        registry.register(Box::new(ai_inference_duration.clone()))?;
        
        let ai_inference_total = IntCounter::new("ai_inference_total", "Total AI inferences")?;
        registry.register(Box::new(ai_inference_total.clone()))?;
        
        let ai_model_loads = IntCounter::new("ai_model_loads_total", "AI model loads")?;
        registry.register(Box::new(ai_model_loads.clone()))?;
        
        Ok(Self {
            registry: Arc::new(registry),
            html_parse_duration,
            html_parse_total,
            html_parse_errors,
            css_parse_duration,
            css_parse_total,
            css_parse_errors,
            js_parse_duration,
            js_parse_total,
            js_parse_errors,
            v8_heap_used,
            v8_execution_duration,
            v8_compilations,
            render_duration,
            render_total,
            render_errors,
            ai_inference_duration,
            ai_inference_total,
            ai_model_loads,
        })
    }
    
    /// Get the Prometheus registry for exporting metrics
    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
    
    /// Export metrics in Prometheus text format
    pub fn export(&self) -> Result<String> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new().expect("Failed to create metrics registry")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_creation() {
        let metrics = MetricsRegistry::new().unwrap();
        assert!(metrics.export().is_ok());
    }
    
    #[test]
    fn test_html_metrics() {
        let metrics = MetricsRegistry::new().unwrap();
        metrics.html_parse_total.inc();
        metrics.html_parse_duration.observe(0.05);
        
        let export = metrics.export().unwrap();
        assert!(export.contains("html_parse_total"));
        assert!(export.contains("html_parse_duration"));
    }
    
    #[test]
    fn test_v8_metrics() {
        let metrics = MetricsRegistry::new().unwrap();
        metrics.v8_heap_used.set(1024 * 1024); // 1MB
        metrics.v8_execution_duration.observe(0.1);
        metrics.v8_compilations.inc();
        
        let export = metrics.export().unwrap();
        assert!(export.contains("v8_heap_used_bytes"));
        assert!(export.contains("v8_execution_duration"));
    }
}
