/// Metrics dashboard for monitoring system performance
///
/// Tracks and visualizes various performance and quality metrics
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Type of metric being tracked
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Parsing accuracy percentage
    ParsingAccuracy,
    /// Rendering performance (ms)
    RenderingTime,
    /// Cache hit rate percentage
    CacheHitRate,
    /// Model inference time (ms)
    InferenceTime,
    /// Memory usage (MB)
    MemoryUsage,
    /// Throughput (requests/sec)
    Throughput,
    /// Error rate percentage
    ErrorRate,
    /// Custom metric
    Custom(String),
}

/// Individual metric value with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    /// Type of metric
    pub metric_type: MetricType,
    /// Metric value
    pub value: f64,
    /// Timestamp when metric was recorded
    pub timestamp: u64,
    /// Optional labels for grouping
    pub labels: HashMap<String, String>,
}

impl Metric {
    /// Create a new metric
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
        }
    }

    /// Add a label to the metric
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

/// Statistics for a metric over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
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
            };
        }

        let count = values.len();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = values.iter().sum::<f64>() / count as f64;

        // Calculate median
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b)); // Use total_cmp to handle NaN safely
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        // Calculate standard deviation
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        Self {
            count,
            min,
            max,
            mean,
            median,
            std_dev,
        }
    }
}

/// Dashboard for monitoring metrics
pub struct MetricsDashboard {
    /// All recorded metrics
    metrics: Vec<Metric>,
    /// Maximum number of metrics to keep
    max_metrics: usize,
}

impl MetricsDashboard {
    /// Create a new metrics dashboard
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            max_metrics: 100000,
        }
    }

    /// Create a dashboard with custom capacity
    pub fn with_capacity(max_metrics: usize) -> Self {
        Self {
            metrics: Vec::with_capacity(max_metrics.min(10000)),
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

    /// Record a simple metric value
    pub fn record_value(&mut self, metric_type: MetricType, value: f64) {
        self.record(Metric::new(metric_type, value));
    }

    /// Get all metrics
    pub fn get_all_metrics(&self) -> &[Metric] {
        &self.metrics
    }

    /// Get metrics by type
    pub fn get_metrics_by_type(&self, metric_type: &MetricType) -> Vec<&Metric> {
        self.metrics
            .iter()
            .filter(|m| &m.metric_type == metric_type)
            .collect()
    }

    /// Get metrics within a time range
    pub fn get_metrics_in_range(&self, start_time: u64, end_time: u64) -> Vec<&Metric> {
        self.metrics
            .iter()
            .filter(|m| m.timestamp >= start_time && m.timestamp <= end_time)
            .collect()
    }

    /// Get recent metrics (last n entries)
    pub fn get_recent_metrics(&self, count: usize) -> &[Metric] {
        let start = self.metrics.len().saturating_sub(count);
        &self.metrics[start..]
    }

    /// Calculate statistics for a metric type
    pub fn get_stats(&self, metric_type: &MetricType) -> MetricStats {
        let values: Vec<f64> = self
            .get_metrics_by_type(metric_type)
            .iter()
            .map(|m| m.value)
            .collect();

        MetricStats::from_values(&values)
    }

    /// Get the latest value for a metric type
    pub fn get_latest_value(&self, metric_type: &MetricType) -> Option<f64> {
        self.metrics
            .iter()
            .rev()
            .find(|m| &m.metric_type == metric_type)
            .map(|m| m.value)
    }

    /// Calculate average over recent period
    pub fn get_recent_average(&self, metric_type: &MetricType, count: usize) -> Option<f64> {
        let values: Vec<f64> = self
            .get_metrics_by_type(metric_type)
            .iter()
            .rev()
            .take(count)
            .map(|m| m.value)
            .collect();

        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    }

    /// Check if a metric is trending up or down
    pub fn get_trend(&self, metric_type: &MetricType, window_size: usize) -> Option<f64> {
        let values: Vec<f64> = self
            .get_metrics_by_type(metric_type)
            .iter()
            .rev()
            .take(window_size)
            .map(|m| m.value)
            .collect();

        if values.len() < 2 {
            return None;
        }

        // Simple linear trend: (last - first) / count
        let first = values.last()?;
        let last = values.first()?;
        Some((last - first) / values.len() as f64)
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        self.metrics.clear();
    }

    /// Generate a text report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("=== Metrics Dashboard Report ===\n\n");

        // Group metrics by type
        let mut metrics_by_type: HashMap<String, Vec<&Metric>> = HashMap::new();
        for metric in &self.metrics {
            let type_key = format!("{:?}", metric.metric_type);
            metrics_by_type
                .entry(type_key)
                .or_insert_with(Vec::new)
                .push(metric);
        }

        for (metric_type, metrics) in metrics_by_type {
            report.push_str(&format!("## {}\n", metric_type));

            let values: Vec<f64> = metrics.iter().map(|m| m.value).collect();
            let stats = MetricStats::from_values(&values);

            report.push_str(&format!("  Count: {}\n", stats.count));
            report.push_str(&format!("  Min: {:.2}\n", stats.min));
            report.push_str(&format!("  Max: {:.2}\n", stats.max));
            report.push_str(&format!("  Mean: {:.2}\n", stats.mean));
            report.push_str(&format!("  Median: {:.2}\n", stats.median));
            report.push_str(&format!("  Std Dev: {:.2}\n", stats.std_dev));
            report.push('\n');
        }

        report
    }
}

impl Default for MetricsDashboard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_creation() {
        let metric = Metric::new(MetricType::ParsingAccuracy, 95.5);
        assert_eq!(metric.metric_type, MetricType::ParsingAccuracy);
        assert_eq!(metric.value, 95.5);
    }

    #[test]
    fn test_metric_with_labels() {
        let metric = Metric::new(MetricType::RenderingTime, 100.0)
            .with_label("page", "home")
            .with_label("device", "mobile");

        assert_eq!(metric.labels.get("page"), Some(&"home".to_string()));
        assert_eq!(metric.labels.get("device"), Some(&"mobile".to_string()));
    }

    #[test]
    fn test_metric_stats_calculation() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = MetricStats::from_values(&values);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 50.0);
        assert_eq!(stats.mean, 30.0);
        assert_eq!(stats.median, 30.0);
    }

    #[test]
    fn test_metric_stats_empty() {
        let values: Vec<f64> = vec![];
        let stats = MetricStats::from_values(&values);

        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_dashboard_record() {
        let mut dashboard = MetricsDashboard::new();
        dashboard.record_value(MetricType::ParsingAccuracy, 95.0);
        dashboard.record_value(MetricType::RenderingTime, 100.0);

        assert_eq!(dashboard.get_all_metrics().len(), 2);
    }

    #[test]
    fn test_dashboard_get_by_type() {
        let mut dashboard = MetricsDashboard::new();
        dashboard.record_value(MetricType::ParsingAccuracy, 95.0);
        dashboard.record_value(MetricType::ParsingAccuracy, 96.0);
        dashboard.record_value(MetricType::RenderingTime, 100.0);

        let accuracy_metrics = dashboard.get_metrics_by_type(&MetricType::ParsingAccuracy);
        assert_eq!(accuracy_metrics.len(), 2);
    }

    #[test]
    fn test_dashboard_stats() {
        let mut dashboard = MetricsDashboard::new();
        dashboard.record_value(MetricType::ParsingAccuracy, 90.0);
        dashboard.record_value(MetricType::ParsingAccuracy, 95.0);
        dashboard.record_value(MetricType::ParsingAccuracy, 100.0);

        let stats = dashboard.get_stats(&MetricType::ParsingAccuracy);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, 90.0);
        assert_eq!(stats.max, 100.0);
        assert!((stats.mean - 95.0).abs() < 0.01);
    }

    #[test]
    fn test_dashboard_latest_value() {
        let mut dashboard = MetricsDashboard::new();
        dashboard.record_value(MetricType::ParsingAccuracy, 90.0);
        dashboard.record_value(MetricType::ParsingAccuracy, 95.0);

        let latest = dashboard.get_latest_value(&MetricType::ParsingAccuracy);
        assert_eq!(latest, Some(95.0));
    }

    #[test]
    fn test_dashboard_recent_average() {
        let mut dashboard = MetricsDashboard::new();
        dashboard.record_value(MetricType::CacheHitRate, 80.0);
        dashboard.record_value(MetricType::CacheHitRate, 85.0);
        dashboard.record_value(MetricType::CacheHitRate, 90.0);

        let avg = dashboard.get_recent_average(&MetricType::CacheHitRate, 2);
        assert_eq!(avg, Some(87.5)); // (85 + 90) / 2
    }

    #[test]
    fn test_dashboard_trend() {
        let mut dashboard = MetricsDashboard::new();
        dashboard.record_value(MetricType::Throughput, 100.0);
        dashboard.record_value(MetricType::Throughput, 110.0);
        dashboard.record_value(MetricType::Throughput, 120.0);

        let trend = dashboard.get_trend(&MetricType::Throughput, 3);
        assert!(trend.is_some());
        assert!(trend.unwrap() > 0.0); // Positive trend
    }

    #[test]
    fn test_dashboard_recent_metrics() {
        let mut dashboard = MetricsDashboard::new();
        for i in 0..10 {
            dashboard.record_value(MetricType::InferenceTime, i as f64);
        }

        let recent = dashboard.get_recent_metrics(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].value, 7.0);
        assert_eq!(recent[2].value, 9.0);
    }

    #[test]
    fn test_dashboard_max_capacity() {
        let mut dashboard = MetricsDashboard::with_capacity(5);
        for i in 0..10 {
            dashboard.record_value(MetricType::MemoryUsage, i as f64);
        }

        assert_eq!(dashboard.get_all_metrics().len(), 5);
    }

    #[test]
    fn test_dashboard_generate_report() {
        let mut dashboard = MetricsDashboard::new();
        dashboard.record_value(MetricType::ParsingAccuracy, 95.0);
        dashboard.record_value(MetricType::RenderingTime, 100.0);

        let report = dashboard.generate_report();
        assert!(report.contains("Metrics Dashboard Report"));
        assert!(report.contains("ParsingAccuracy"));
    }
}
