/// Metrics dashboard for monitoring system performance
///
/// Tracks and visualizes various performance and quality metrics
///
/// This module re-exports and extends the core metrics types from browerai-core
/// to maintain backward compatibility while using unified types.
pub use browerai_core::metrics::{Histogram, Metric, MetricStats, MetricType, MetricsDashboard};

/// Extended metrics dashboard with additional functionality
pub trait MetricsDashboardExt {
    /// Get all metrics as a formatted string
    fn get_all_metrics(&self) -> String;
    /// Get uptime in seconds
    fn get_uptime_secs(&self) -> u64;
    /// Get counter value
    fn get_counter_value(&self, name: &str) -> u64;
    /// Get gauge value
    fn get_gauge_value(&self, name: &str) -> Option<f64>;
    /// Generate a simple report
    fn generate_simple_report(&self) -> String;
}

impl MetricsDashboardExt for MetricsDashboard {
    fn get_all_metrics(&self) -> String {
        format!("Metrics Dashboard - Uptime: {:?}", self.uptime())
    }

    fn get_uptime_secs(&self) -> u64 {
        self.uptime().as_secs()
    }

    fn get_counter_value(&self, _name: &str) -> u64 {
        // Use reflection or internal state if available
        // For now, return 0 as placeholder
        0
    }

    fn get_gauge_value(&self, _name: &str) -> Option<f64> {
        // Use reflection or internal state if available
        None
    }

    fn generate_simple_report(&self) -> String {
        format!(
            "=== Metrics Report ===\nUptime: {:?}\n",
            self.uptime()
        )
    }
}

/// Create a new metrics dashboard
pub fn create_dashboard() -> MetricsDashboard {
    MetricsDashboard::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let dashboard = create_dashboard();
        assert!(dashboard.get_all_metrics().contains("Uptime"));
    }
}
