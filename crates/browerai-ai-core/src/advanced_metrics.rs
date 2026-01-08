/// Advanced Metrics Collection and Observability
///
/// This module provides comprehensive metrics collection for model inference,
/// including latency, throughput, memory usage, and hardware utilization.
/// It supports multiple collection backends and export formats.
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Detailed inference metrics
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Name of the model used
    pub model_name: String,
    /// Total inference duration
    pub inference_time: Duration,
    /// Input size in bytes
    pub input_size: usize,
    /// Output size in bytes
    pub output_size: usize,
    /// Peak memory used during inference in MB
    pub memory_peak_mb: u64,
    /// Whether cache was hit
    pub cache_hit: bool,
    /// Whether inference succeeded
    pub success: bool,
    /// Timestamp of the inference
    pub timestamp: Instant,
}

impl InferenceMetrics {
    /// Calculate throughput (items/second)
    pub fn throughput(&self) -> f64 {
        if self.inference_time.is_zero() {
            0.0
        } else {
            1.0 / self.inference_time.as_secs_f64()
        }
    }

    /// Get latency in milliseconds
    pub fn latency_ms(&self) -> f64 {
        self.inference_time.as_secs_f64() * 1000.0
    }

    /// Get effective throughput (output bytes per second)
    pub fn effective_throughput_mbps(&self) -> f64 {
        let elapsed_secs = self.inference_time.as_secs_f64();
        if elapsed_secs == 0.0 {
            0.0
        } else {
            (self.output_size as f64 / (1024.0 * 1024.0)) / elapsed_secs
        }
    }
}

/// Histogram bucket for tracking latency distribution
#[derive(Debug, Clone, Default)]
pub struct HistogramBucket {
    /// Measurements
    measurements: Vec<f64>,
}

impl HistogramBucket {
    /// Add a measurement
    pub fn record(&mut self, value: f64) {
        self.measurements.push(value);
    }

    /// Get percentile (e.g., 0.95 for 95th percentile)
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.measurements.is_empty() {
            return None;
        }

        let mut sorted = self.measurements.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((sorted.len() as f64 - 1.0) * p) as usize;
        sorted.get(idx).copied()
    }

    /// Get average
    pub fn average(&self) -> Option<f64> {
        if self.measurements.is_empty() {
            None
        } else {
            Some(self.measurements.iter().sum::<f64>() / self.measurements.len() as f64)
        }
    }

    /// Get min
    pub fn min(&self) -> Option<f64> {
        self.measurements
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }

    /// Get max
    pub fn max(&self) -> Option<f64> {
        self.measurements
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }

    /// Clear measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }

    /// Get count
    pub fn count(&self) -> usize {
        self.measurements.len()
    }
}

/// Snapshot of collected metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Count of successful inferences
    pub successful_inferences: usize,
    /// Count of failed inferences
    pub failed_inferences: usize,
    /// Total inference time in seconds
    pub total_inference_time_s: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// P50 latency in milliseconds
    pub p50_latency_ms: Option<f64>,
    /// P95 latency in milliseconds
    pub p95_latency_ms: Option<f64>,
    /// P99 latency in milliseconds
    pub p99_latency_ms: Option<f64>,
    /// Maximum latency in milliseconds
    pub max_latency_ms: Option<f64>,
    /// Average memory usage in MB
    pub avg_memory_mb: f64,
    /// Peak memory usage in MB
    pub peak_memory_mb: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
}

/// Metrics aggregator for computing statistics
#[derive(Debug)]
pub struct MetricsAggregator {
    metrics: Arc<RwLock<VecDeque<InferenceMetrics>>>,
    max_history: usize,
    latency_histogram: Arc<RwLock<HistogramBucket>>,
}

impl MetricsAggregator {
    /// Create a new aggregator with maximum history size
    pub fn new(max_history: usize) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(VecDeque::with_capacity(max_history))),
            max_history,
            latency_histogram: Arc::new(RwLock::new(HistogramBucket::default())),
        }
    }

    /// Add a metric
    pub fn record(&self, metric: InferenceMetrics) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.push_back(metric.clone());
            if metrics.len() > self.max_history {
                metrics.pop_front();
            }
        }

        if let Ok(mut histogram) = self.latency_histogram.write() {
            histogram.record(metric.latency_ms());
        }
    }

    /// Get current snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        if let Ok(metrics) = self.metrics.read() {
            let total = metrics.len();
            let successful = metrics.iter().filter(|m| m.success).count();
            let failed = total - successful;

            let total_time: f64 = metrics.iter().map(|m| m.inference_time.as_secs_f64()).sum();
            let avg_latency = if total > 0 {
                (total_time / total as f64) * 1000.0
            } else {
                0.0
            };

            let avg_memory = if total > 0 {
                metrics.iter().map(|m| m.memory_peak_mb as f64).sum::<f64>() / total as f64
            } else {
                0.0
            };

            let peak_memory = metrics.iter().map(|m| m.memory_peak_mb).max().unwrap_or(0);

            let cache_hits = metrics.iter().filter(|m| m.cache_hit).count();
            let cache_hit_rate = if total > 0 {
                cache_hits as f64 / total as f64
            } else {
                0.0
            };

            let success_rate = if total > 0 {
                successful as f64 / total as f64
            } else {
                0.0
            };

            let histogram = self.latency_histogram.read().ok();
            let p50 = histogram.as_ref().and_then(|h| h.percentile(0.5));
            let p95 = histogram.as_ref().and_then(|h| h.percentile(0.95));
            let p99 = histogram.as_ref().and_then(|h| h.percentile(0.99));
            let max_latency = histogram.as_ref().and_then(|h| h.max());

            MetricsSnapshot {
                successful_inferences: successful,
                failed_inferences: failed,
                total_inference_time_s: total_time,
                avg_latency_ms: avg_latency,
                p50_latency_ms: p50,
                p95_latency_ms: p95,
                p99_latency_ms: p99,
                max_latency_ms: max_latency,
                avg_memory_mb: avg_memory,
                peak_memory_mb: peak_memory,
                cache_hit_rate,
                success_rate,
            }
        } else {
            MetricsSnapshot {
                successful_inferences: 0,
                failed_inferences: 0,
                total_inference_time_s: 0.0,
                avg_latency_ms: 0.0,
                p50_latency_ms: None,
                p95_latency_ms: None,
                p99_latency_ms: None,
                max_latency_ms: None,
                avg_memory_mb: 0.0,
                peak_memory_mb: 0,
                cache_hit_rate: 0.0,
                success_rate: 0.0,
            }
        }
    }

    /// Clear all metrics
    pub fn clear(&self) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.clear();
        }
        if let Ok(mut histogram) = self.latency_histogram.write() {
            histogram.clear();
        }
    }
}

impl Clone for MetricsAggregator {
    fn clone(&self) -> Self {
        Self {
            metrics: Arc::clone(&self.metrics),
            max_history: self.max_history,
            latency_histogram: Arc::clone(&self.latency_histogram),
        }
    }
}

/// Callback interface for inference events
pub trait InferenceCallback: Send + Sync {
    /// Called before inference starts
    fn on_pre_inference(&self, _model_name: &str) {}

    /// Called after inference completes successfully
    fn on_post_inference(&self, metrics: &InferenceMetrics);

    /// Called when inference fails
    fn on_inference_failed(&self, model_name: &str, error: &str);
}

/// Default callback that does nothing
#[derive(Debug, Clone, Default)]
pub struct NoOpCallback;

impl InferenceCallback for NoOpCallback {
    fn on_post_inference(&self, _metrics: &InferenceMetrics) {}
    fn on_inference_failed(&self, _model_name: &str, _error: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_metrics_latency() {
        let metrics = InferenceMetrics {
            model_name: "test".to_string(),
            inference_time: Duration::from_millis(100),
            input_size: 1024,
            output_size: 1024,
            memory_peak_mb: 512,
            cache_hit: false,
            success: true,
            timestamp: Instant::now(),
        };

        assert_eq!(metrics.latency_ms(), 100.0);
        assert_eq!(metrics.throughput(), 10.0);
    }

    #[test]
    fn test_histogram_bucket() {
        let mut histogram = HistogramBucket::default();
        for i in 0..100 {
            histogram.record(i as f64);
        }

        assert_eq!(histogram.count(), 100);
        assert!(histogram.min() >= Some(0.0));
        assert!(histogram.max() <= Some(99.0));
    }

    #[test]
    fn test_metrics_aggregator() {
        let aggregator = MetricsAggregator::new(10);

        let metric = InferenceMetrics {
            model_name: "test".to_string(),
            inference_time: Duration::from_millis(50),
            input_size: 1024,
            output_size: 1024,
            memory_peak_mb: 256,
            cache_hit: true,
            success: true,
            timestamp: Instant::now(),
        };

        aggregator.record(metric);
        let snapshot = aggregator.snapshot();

        assert_eq!(snapshot.successful_inferences, 1);
        assert_eq!(snapshot.failed_inferences, 0);
        assert_eq!(snapshot.cache_hit_rate, 1.0);
        assert_eq!(snapshot.success_rate, 1.0);
    }
}
