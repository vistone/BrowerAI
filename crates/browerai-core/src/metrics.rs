/// Unified metrics and performance tracking for BrowerAI
///
/// This module provides centralized types for metrics and performance tracking,
/// unifying previously duplicated definitions across different modules.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Type of metric being tracked
///
/// This enum unifies metric type definitions that were previously duplicated in:
/// - `browerai-learning/src/metrics.rs` (MetricType)
/// - `browerai-ai-core/src/advanced_metrics.rs` (inferred types)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Parsing accuracy percentage (0.0 - 1.0)
    ParsingAccuracy,
    /// Rendering performance in milliseconds
    RenderingTime,
    /// Cache hit rate percentage (0.0 - 1.0)
    CacheHitRate,
    /// Model inference time in milliseconds
    InferenceTime,
    /// Memory usage in megabytes
    MemoryUsage,
    /// Throughput in requests per second
    Throughput,
    /// Error rate percentage (0.0 - 1.0)
    ErrorRate,
    /// Model load time in milliseconds
    ModelLoadTime,
    /// AI operation success rate (0.0 - 1.0)
    AiSuccessRate,
    /// Code generation confidence (0.0 - 1.0)
    GenerationConfidence,
    /// Custom metric with a name
    Custom(String),
}

impl Default for MetricType {
    fn default() -> Self {
        MetricType::Custom("unknown".to_string())
    }
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricType::ParsingAccuracy => write!(f, "parsing_accuracy"),
            MetricType::RenderingTime => write!(f, "rendering_time_ms"),
            MetricType::CacheHitRate => write!(f, "cache_hit_rate"),
            MetricType::InferenceTime => write!(f, "inference_time_ms"),
            MetricType::MemoryUsage => write!(f, "memory_mb"),
            MetricType::Throughput => write!(f, "throughput_per_sec"),
            MetricType::ErrorRate => write!(f, "error_rate"),
            MetricType::ModelLoadTime => write!(f, "model_load_time_ms"),
            MetricType::AiSuccessRate => write!(f, "ai_success_rate"),
            MetricType::GenerationConfidence => write!(f, "generation_confidence"),
            MetricType::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Individual metric value with timestamp
///
/// This struct unifies metric definitions that were previously duplicated in:
/// - `browerai-learning/src/metrics.rs` (Metric)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    /// Type of metric
    pub metric_type: MetricType,
    /// Metric value
    pub value: f64,
    /// Timestamp when metric was recorded
    pub timestamp: u64,
    /// Optional labels for grouping and filtering
    #[serde(default)]
    pub labels: HashMap<String, String>,
    /// Optional unit description
    #[serde(default)]
    pub unit: String,
}

impl Metric {
    /// Create a new metric with the current timestamp
    pub fn new(metric_type: MetricType, value: f64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();

        Self {
            metric_type,
            value,
            timestamp,
            labels: HashMap::new(),
            unit: String::new(),
        }
    }

    /// Add a label to the metric
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Add multiple labels to the metric
    pub fn with_labels(mut self, labels: impl Into<HashMap<String, String>>) -> Self {
        self.labels.extend(labels.into());
        self
    }

    /// Set the unit for this metric
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = unit.into();
        self
    }

    /// Create a timing metric (value in milliseconds)
    pub fn timing(value: f64) -> Self {
        Self::new(MetricType::InferenceTime, value).with_unit("ms")
    }

    /// Create a rate metric (0.0 - 1.0)
    pub fn rate(value: f64) -> Self {
        Self::new(
            MetricType::Custom("rate".to_string()),
            value.clamp(0.0, 1.0),
        )
    }
}

/// Statistics for a metric over time
///
/// This struct unifies statistics definitions that were previously duplicated in:
/// - `browerai-learning/src/metrics.rs` (MetricStats)
/// - `browerai-ai-core/src/advanced_metrics.rs` (HistogramBucket, MetricStats)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricStats {
    /// Number of samples
    pub count: usize,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Arithmetic mean
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Sum of all values
    pub sum: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

impl MetricStats {
    /// Calculate statistics from a list of values
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                sum: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let count = values.len();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum = values.iter().sum::<f64>();
        let mean = sum / count as f64;

        // Calculate median
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = if count.is_multiple_of(2) {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        // Calculate standard deviation
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        // Calculate percentiles
        let p95 = percentile(&sorted, 0.95);
        let p99 = percentile(&sorted, 0.99);

        Self {
            count,
            min,
            max,
            mean,
            median,
            std_dev,
            sum,
            p95,
            p99,
        }
    }

    /// Create empty statistics
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if this is empty (no data)
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the range (max - min)
    pub fn range(&self) -> f64 {
        self.max - self.min
    }
}

/// Calculate the p-th percentile of a sorted slice
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Histogram bucket for distributing values
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistogramBucket {
    /// Lower bound (inclusive)
    pub lower_bound: f64,
    /// Upper bound (exclusive)
    pub upper_bound: f64,
    /// Count of values in this bucket
    pub count: u64,
    /// Cumulative count from start
    pub cumulative_count: u64,
}

impl HistogramBucket {
    /// Create a new bucket
    pub fn new(lower_bound: f64, upper_bound: f64) -> Self {
        Self {
            lower_bound,
            upper_bound,
            count: 0,
            cumulative_count: 0,
        }
    }

    /// Check if a value falls in this bucket
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower_bound && value < self.upper_bound
    }
}

/// Histogram for distributing metric values
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Histogram {
    /// Buckets defining the ranges
    pub buckets: Vec<HistogramBucket>,
    /// Total count of values
    pub total_count: u64,
    /// Sum of all values
    pub total_sum: f64,
    /// Minimum value observed
    pub min: f64,
    /// Maximum value observed
    pub max: f64,
}

impl Histogram {
    /// Create a histogram with custom bucket boundaries
    pub fn new(boundaries: &[f64]) -> Self {
        let mut buckets = Vec::with_capacity(boundaries.len() + 1);
        for i in 0..boundaries.len().saturating_sub(1) {
            buckets.push(HistogramBucket::new(boundaries[i], boundaries[i + 1]));
        }
        // Add overflow bucket
        if let Some(&last) = boundaries.last() {
            buckets.push(HistogramBucket::new(last, f64::INFINITY));
        }
        Self {
            buckets,
            total_count: 0,
            total_sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Create a histogram with default latency boundaries (in ms)
    pub fn latency() -> Self {
        Self::new(&[0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
    }

    /// Record a value in the histogram
    pub fn record(&mut self, value: f64) {
        self.total_count += 1;
        self.total_sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        for bucket in &mut self.buckets {
            if bucket.contains(value) {
                bucket.count += 1;
            }
        }

        // Update cumulative counts
        let mut cumulative = 0;
        for bucket in &mut self.buckets {
            cumulative += bucket.count;
            bucket.cumulative_count = cumulative;
        }
    }

    /// Get statistics from the histogram
    pub fn stats(&self) -> MetricStats {
        let values: Vec<f64> = self
            .buckets
            .iter()
            .flat_map(|b| {
                std::iter::repeat_n(
                    b.lower_bound + (b.upper_bound - b.lower_bound) / 2.0,
                    b.count as usize,
                )
            })
            .collect();
        MetricStats::from_values(&values)
    }
}

/// Dashboard for monitoring and collecting metrics
#[derive(Debug, Clone, Default)]
pub struct MetricsDashboard {
    /// All recorded metrics
    metrics: Vec<Metric>,
    /// Maximum number of metrics to keep
    max_metrics: usize,
}

impl MetricsDashboard {
    /// Create a new metrics dashboard with default capacity
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            max_metrics: 100_000,
        }
    }

    /// Create a dashboard with custom capacity
    pub fn with_capacity(max_metrics: usize) -> Self {
        Self {
            metrics: Vec::with_capacity(max_metrics.min(10_000)),
            max_metrics,
        }
    }

    /// Record a new metric
    pub fn record(&mut self, metric: Metric) {
        self.metrics.push(metric);

        // Remove oldest metrics if we exceed max_metrics
        if self.metrics.len() > self.max_metrics {
            self.metrics.drain(0..self.metrics.len() - self.max_metrics);
        }
    }

    /// Record a simple metric
    pub fn record_value(&mut self, metric_type: MetricType, value: f64) {
        self.record(Metric::new(metric_type, value));
    }

    /// Get all metrics
    pub fn get_all(&self) -> &[Metric] {
        &self.metrics
    }

    /// Get metrics of a specific type
    pub fn get_by_type(&self, metric_type: &MetricType) -> Vec<&Metric> {
        self.metrics
            .iter()
            .filter(|m| &m.metric_type == metric_type)
            .collect()
    }

    /// Get metrics matching labels
    pub fn get_by_labels(&self, labels: &HashMap<String, String>) -> Vec<&Metric> {
        self.metrics
            .iter()
            .filter(|m| labels.iter().all(|(k, v)| m.labels.get(k) == Some(v)))
            .collect()
    }

    /// Get statistics for a specific metric type
    pub fn get_stats(&self, metric_type: &MetricType) -> MetricStats {
        let values: Vec<f64> = self
            .metrics
            .iter()
            .filter(|m| &m.metric_type == metric_type)
            .map(|m| m.value)
            .collect();
        MetricStats::from_values(&values)
    }

    /// Get a histogram for a specific metric type
    pub fn get_histogram(&self, metric_type: &MetricType) -> Histogram {
        let mut histogram = Histogram::latency();
        for metric in self
            .metrics
            .iter()
            .filter(|m| &m.metric_type == metric_type)
        {
            histogram.record(metric.value);
        }
        histogram
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        self.metrics.clear();
    }

    /// Get the total number of metrics recorded
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    /// Check if the dashboard is empty
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }
}

use std::fmt;

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}={}{} @ {}",
            self.metric_type,
            self.value,
            if self.unit.is_empty() {
                String::new()
            } else {
                format!(" {}", self.unit)
            },
            self.timestamp
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_creation() {
        let metric = Metric::new(MetricType::InferenceTime, 42.5);
        assert_eq!(metric.metric_type, MetricType::InferenceTime);
        assert_eq!(metric.value, 42.5);
        assert!(metric.timestamp > 0);
    }

    #[test]
    fn test_metric_with_labels() {
        let metric = Metric::new(MetricType::Throughput, 100.0)
            .with_label("model", "test")
            .with_labels(std::collections::HashMap::from([(
                "env".to_string(),
                "test".to_string(),
            )]));

        assert_eq!(metric.labels.len(), 2);
        assert_eq!(metric.labels.get("model"), Some(&"test".to_string()));
    }

    #[test]
    fn test_metric_stats_from_values() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = MetricStats::from_values(&values);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.mean - 3.0).abs() < 0.001);
        assert!((stats.median - 3.0).abs() < 0.001);
        assert!((stats.sum - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_metric_stats_empty() {
        let stats = MetricStats::from_values(&[]);
        assert!(stats.is_empty());
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_histogram() {
        let mut histogram = Histogram::new(&[0.0, 10.0, 20.0, 50.0]);
        histogram.record(5.0);
        histogram.record(15.0);
        histogram.record(25.0);
        histogram.record(75.0);

        assert_eq!(histogram.total_count, 4);
        assert_eq!(histogram.buckets[0].count, 1); // 0-10
        assert_eq!(histogram.buckets[1].count, 1); // 10-20
        assert_eq!(histogram.buckets[2].count, 1); // 20-50
        assert_eq!(histogram.buckets[3].count, 1); // 50+
    }

    #[test]
    fn test_metrics_dashboard() {
        let mut dashboard = MetricsDashboard::new();
        dashboard.record(Metric::new(MetricType::InferenceTime, 10.0));
        dashboard.record(Metric::new(MetricType::InferenceTime, 20.0));
        dashboard.record(Metric::new(MetricType::MemoryUsage, 100.0));

        assert_eq!(dashboard.len(), 3);
        assert_eq!(dashboard.get_by_type(&MetricType::InferenceTime).len(), 2);

        let stats = dashboard.get_stats(&MetricType::InferenceTime);
        assert_eq!(stats.count, 2);
        assert!((stats.mean - 15.0).abs() < 0.001);
    }
}
