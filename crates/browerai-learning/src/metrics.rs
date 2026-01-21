/// Metrics dashboard for monitoring system performance
///
/// Tracks and visualizes various performance and quality metrics
///
/// This module re-exports and extends the core metrics types from browerai-core
/// to maintain backward compatibility while using unified types.
pub use browerai_core::{Histogram, Metric, MetricStats, MetricType, MetricsDashboard};

use std::collections::HashMap;

/// Extended metrics dashboard with additional functionality
pub trait MetricsDashboardExt {
    fn get_all_metrics(&self) -> &[Metric];
    fn get_metrics_by_type(&self, metric_type: &MetricType) -> Vec<&Metric>;
    fn get_metrics_in_range(&self, start_time: u64, end_time: u64) -> Vec<&Metric>;
    fn get_recent_metrics(&self, count: usize) -> &[Metric];
    fn get_latest_value(&self, metric_type: &MetricType) -> Option<f64>;
    fn get_recent_average(&self, metric_type: &MetricType, count: usize) -> Option<f64>;
    fn get_trend(&self, metric_type: &MetricType, window_size: usize) -> Option<f64>;
    fn generate_report(&self) -> String;
}

impl MetricsDashboardExt for MetricsDashboard {
    fn get_all_metrics(&self) -> &[Metric] {
        self.get_all()
    }

    fn get_metrics_by_type(&self, metric_type: &MetricType) -> Vec<&Metric> {
        self.get_by_type(metric_type)
    }

    fn get_metrics_in_range(&self, start_time: u64, end_time: u64) -> Vec<&Metric> {
        self.get_all()
            .iter()
            .filter(|m| m.timestamp >= start_time && m.timestamp <= end_time)
            .collect()
    }

    fn get_recent_metrics(&self, count: usize) -> &[Metric] {
        let metrics = self.get_all();
        let start = metrics.len().saturating_sub(count);
        &metrics[start..]
    }

    fn get_latest_value(&self, metric_type: &MetricType) -> Option<f64> {
        self.get_all()
            .iter()
            .rev()
            .find(|m| &m.metric_type == metric_type)
            .map(|m| m.value)
    }

    fn get_recent_average(&self, metric_type: &MetricType, count: usize) -> Option<f64> {
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

    fn get_trend(&self, metric_type: &MetricType, window_size: usize) -> Option<f64> {
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

        let first = values.last()?;
        let last = values.first()?;
        Some((last - first) / values.len() as f64)
    }

    fn generate_report(&self) -> String {
        let mut report = String::from("=== Metrics Dashboard Report ===\n\n");

        let mut metrics_by_type: HashMap<String, Vec<&Metric>> = HashMap::new();
        for metric in self.get_all() {
            let type_key = format!("{:?}", metric.metric_type);
            metrics_by_type.entry(type_key).or_default().push(metric);
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
        assert!((stats.mean - 30.0).abs() < 0.001);
        assert!((stats.median - 30.0).abs() < 0.001);
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
    fn test_dashboard_trend() {
        let mut dashboard = MetricsDashboard::new();
        dashboard.record_value(MetricType::Throughput, 100.0);
        dashboard.record_value(MetricType::Throughput, 110.0);
        dashboard.record_value(MetricType::Throughput, 120.0);

        let trend = dashboard.get_trend(&MetricType::Throughput, 3);
        assert!(trend.is_some());
        assert!(trend.unwrap() > 0.0);
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
    fn test_dashboard_generate_report() {
        let mut dashboard = MetricsDashboard::new();
        dashboard.record_value(MetricType::ParsingAccuracy, 95.0);
        dashboard.record_value(MetricType::RenderingTime, 100.0);

        let report = dashboard.generate_report();
        assert!(report.contains("Metrics Dashboard Report"));
        assert!(report.contains("ParsingAccuracy"));
    }
}
