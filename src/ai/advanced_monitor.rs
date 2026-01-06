/// Advanced AI performance monitoring and profiling
///
/// Provides detailed performance analysis for AI operations
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Performance profile for AI operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// Operation name
    pub operation: String,
    /// Total execution time
    pub total_time_ms: f64,
    /// Number of executions
    pub execution_count: usize,
    /// Average execution time
    pub avg_time_ms: f64,
    /// Minimum execution time
    pub min_time_ms: f64,
    /// Maximum execution time
    pub max_time_ms: f64,
    /// Memory usage estimate (MB)
    pub memory_mb: f64,
    /// CPU utilization percentage
    pub cpu_usage: f64,
}

impl PerformanceProfile {
    pub fn new(operation: String) -> Self {
        Self {
            operation,
            total_time_ms: 0.0,
            execution_count: 0,
            avg_time_ms: 0.0,
            min_time_ms: f64::MAX,
            max_time_ms: 0.0,
            memory_mb: 0.0,
            cpu_usage: 0.0,
        }
    }

    pub fn record_execution(&mut self, duration_ms: f64, memory_mb: f64, cpu_usage: f64) {
        self.total_time_ms += duration_ms;
        self.execution_count += 1;
        self.avg_time_ms = self.total_time_ms / self.execution_count as f64;
        self.min_time_ms = self.min_time_ms.min(duration_ms);
        self.max_time_ms = self.max_time_ms.max(duration_ms);

        // Update moving averages for memory and CPU
        let alpha = 0.3; // Smoothing factor
        self.memory_mb = alpha * memory_mb + (1.0 - alpha) * self.memory_mb;
        self.cpu_usage = alpha * cpu_usage + (1.0 - alpha) * self.cpu_usage;
    }

    pub fn get_summary(&self) -> String {
        format!(
            "{}: {} executions, avg {:.2}ms, min {:.2}ms, max {:.2}ms, mem {:.2}MB, cpu {:.1}%",
            self.operation,
            self.execution_count,
            self.avg_time_ms,
            self.min_time_ms,
            self.max_time_ms,
            self.memory_mb,
            self.cpu_usage
        )
    }
}

/// Advanced performance monitor with profiling capabilities
pub struct AdvancedPerformanceMonitor {
    /// Performance profiles by operation
    profiles: HashMap<String, PerformanceProfile>,
    /// Global performance metrics
    global_metrics: GlobalMetrics,
    /// Enable detailed profiling
    profiling_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct GlobalMetrics {
    pub total_operations: usize,
    pub total_time_ms: f64,
    pub peak_memory_mb: f64,
    pub avg_cpu_usage: f64,
}

impl Default for GlobalMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            total_time_ms: 0.0,
            peak_memory_mb: 0.0,
            avg_cpu_usage: 0.0,
        }
    }
}

impl AdvancedPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            global_metrics: GlobalMetrics::default(),
            profiling_enabled: true,
        }
    }

    pub fn enable_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
    }

    pub fn start_operation(&self, operation: &str) -> OperationTimer {
        OperationTimer {
            operation: operation.to_string(),
            start: Instant::now(),
            profiling_enabled: self.profiling_enabled,
        }
    }

    pub fn record_operation(
        &mut self,
        operation: &str,
        duration_ms: f64,
        memory_mb: f64,
        cpu_usage: f64,
    ) {
        if !self.profiling_enabled {
            return;
        }

        let profile = self
            .profiles
            .entry(operation.to_string())
            .or_insert_with(|| PerformanceProfile::new(operation.to_string()));

        profile.record_execution(duration_ms, memory_mb, cpu_usage);

        // Update global metrics
        self.global_metrics.total_operations += 1;
        self.global_metrics.total_time_ms += duration_ms;
        self.global_metrics.peak_memory_mb = self.global_metrics.peak_memory_mb.max(memory_mb);

        let n = self.global_metrics.total_operations as f64;
        self.global_metrics.avg_cpu_usage =
            (self.global_metrics.avg_cpu_usage * (n - 1.0) + cpu_usage) / n;
    }

    pub fn get_profile(&self, operation: &str) -> Option<&PerformanceProfile> {
        self.profiles.get(operation)
    }

    pub fn get_all_profiles(&self) -> Vec<&PerformanceProfile> {
        self.profiles.values().collect()
    }

    pub fn get_global_metrics(&self) -> &GlobalMetrics {
        &self.global_metrics
    }

    pub fn get_bottlenecks(&self, threshold_ms: f64) -> Vec<&PerformanceProfile> {
        self.profiles
            .values()
            .filter(|p| p.avg_time_ms > threshold_ms)
            .collect()
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::from("=== Advanced Performance Report ===\n\n");

        report.push_str(&format!(
            "Global Metrics:\n\
             - Total Operations: {}\n\
             - Total Time: {:.2}ms\n\
             - Peak Memory: {:.2}MB\n\
             - Avg CPU Usage: {:.1}%\n\n",
            self.global_metrics.total_operations,
            self.global_metrics.total_time_ms,
            self.global_metrics.peak_memory_mb,
            self.global_metrics.avg_cpu_usage
        ));

        report.push_str("Operation Profiles:\n");
        let mut profiles: Vec<_> = self.profiles.values().collect();
        profiles.sort_by(|a, b| b.avg_time_ms.partial_cmp(&a.avg_time_ms).unwrap());

        for profile in profiles {
            report.push_str(&format!("  - {}\n", profile.get_summary()));
        }

        report.push_str("\nBottlenecks (>100ms):\n");
        let bottlenecks = self.get_bottlenecks(100.0);
        if bottlenecks.is_empty() {
            report.push_str("  None detected\n");
        } else {
            for profile in bottlenecks {
                report.push_str(&format!("  - {}\n", profile.get_summary()));
            }
        }

        report
    }

    pub fn reset(&mut self) {
        self.profiles.clear();
        self.global_metrics = GlobalMetrics::default();
    }
}

impl Default for AdvancedPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer for measuring operation duration
pub struct OperationTimer {
    operation: String,
    start: Instant,
    profiling_enabled: bool,
}

impl OperationTimer {
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn operation(&self) -> &str {
        &self.operation
    }

    pub fn is_enabled(&self) -> bool {
        self.profiling_enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_profile_creation() {
        let profile = PerformanceProfile::new("test_op".to_string());
        assert_eq!(profile.operation, "test_op");
        assert_eq!(profile.execution_count, 0);
    }

    #[test]
    fn test_performance_profile_record() {
        let mut profile = PerformanceProfile::new("test_op".to_string());
        profile.record_execution(100.0, 50.0, 75.0);

        assert_eq!(profile.execution_count, 1);
        assert_eq!(profile.avg_time_ms, 100.0);
        assert_eq!(profile.min_time_ms, 100.0);
        assert_eq!(profile.max_time_ms, 100.0);
    }

    #[test]
    fn test_performance_profile_multiple_records() {
        let mut profile = PerformanceProfile::new("test_op".to_string());
        profile.record_execution(100.0, 50.0, 75.0);
        profile.record_execution(200.0, 60.0, 80.0);

        assert_eq!(profile.execution_count, 2);
        assert_eq!(profile.avg_time_ms, 150.0);
        assert_eq!(profile.min_time_ms, 100.0);
        assert_eq!(profile.max_time_ms, 200.0);
    }

    #[test]
    fn test_advanced_monitor_creation() {
        let monitor = AdvancedPerformanceMonitor::new();
        assert!(monitor.profiling_enabled);
        assert_eq!(monitor.global_metrics.total_operations, 0);
    }

    #[test]
    fn test_advanced_monitor_record_operation() {
        let mut monitor = AdvancedPerformanceMonitor::new();
        monitor.record_operation("parse_html", 150.0, 45.0, 70.0);

        let profile = monitor.get_profile("parse_html");
        assert!(profile.is_some());
        assert_eq!(profile.unwrap().execution_count, 1);
    }

    #[test]
    fn test_advanced_monitor_global_metrics() {
        let mut monitor = AdvancedPerformanceMonitor::new();
        monitor.record_operation("op1", 100.0, 50.0, 60.0);
        monitor.record_operation("op2", 200.0, 70.0, 80.0);

        let metrics = monitor.get_global_metrics();
        assert_eq!(metrics.total_operations, 2);
        assert_eq!(metrics.total_time_ms, 300.0);
        assert_eq!(metrics.peak_memory_mb, 70.0);
    }

    #[test]
    fn test_advanced_monitor_bottlenecks() {
        let mut monitor = AdvancedPerformanceMonitor::new();
        monitor.record_operation("fast_op", 50.0, 30.0, 40.0);
        monitor.record_operation("slow_op", 150.0, 60.0, 80.0);

        let bottlenecks = monitor.get_bottlenecks(100.0);
        assert_eq!(bottlenecks.len(), 1);
        assert_eq!(bottlenecks[0].operation, "slow_op");
    }

    #[test]
    fn test_advanced_monitor_generate_report() {
        let mut monitor = AdvancedPerformanceMonitor::new();
        monitor.record_operation("test_op", 100.0, 50.0, 70.0);

        let report = monitor.generate_report();
        assert!(report.contains("Advanced Performance Report"));
        assert!(report.contains("test_op"));
    }

    #[test]
    fn test_advanced_monitor_reset() {
        let mut monitor = AdvancedPerformanceMonitor::new();
        monitor.record_operation("test_op", 100.0, 50.0, 70.0);

        monitor.reset();
        assert_eq!(monitor.global_metrics.total_operations, 0);
        assert!(monitor.get_profile("test_op").is_none());
    }

    #[test]
    fn test_operation_timer() {
        let monitor = AdvancedPerformanceMonitor::new();
        let timer = monitor.start_operation("test");

        assert_eq!(timer.operation(), "test");
        assert!(timer.is_enabled());
        assert!(timer.elapsed_ms() >= 0.0);
    }

    #[test]
    fn test_enable_disable_profiling() {
        let mut monitor = AdvancedPerformanceMonitor::new();

        monitor.enable_profiling(false);
        monitor.record_operation("test_op", 100.0, 50.0, 70.0);
        assert!(monitor.get_profile("test_op").is_none());

        monitor.enable_profiling(true);
        monitor.record_operation("test_op2", 100.0, 50.0, 70.0);
        assert!(monitor.get_profile("test_op2").is_some());
    }
}
