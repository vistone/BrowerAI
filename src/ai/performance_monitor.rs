// AI Performance Monitoring System
// Tracks inference times, model accuracy, and resource usage

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Performance metrics for a single AI inference
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    pub model_name: String,
    pub inference_time: Duration,
    pub input_size: usize,
    pub output_size: usize,
    pub success: bool,
    pub timestamp: Instant,
}

/// Aggregated performance statistics for a model
#[derive(Debug, Clone)]
pub struct ModelStats {
    pub model_name: String,
    pub total_inferences: u64,
    pub successful_inferences: u64,
    pub failed_inferences: u64,
    pub total_inference_time: Duration,
    pub min_inference_time: Duration,
    pub max_inference_time: Duration,
    pub avg_inference_time: Duration,
    pub total_input_bytes: usize,
    pub total_output_bytes: usize,
}

impl ModelStats {
    fn new(model_name: String) -> Self {
        Self {
            model_name,
            total_inferences: 0,
            successful_inferences: 0,
            failed_inferences: 0,
            total_inference_time: Duration::ZERO,
            min_inference_time: Duration::MAX,
            max_inference_time: Duration::ZERO,
            avg_inference_time: Duration::ZERO,
            total_input_bytes: 0,
            total_output_bytes: 0,
        }
    }

    fn update(&mut self, metrics: &InferenceMetrics) {
        self.total_inferences += 1;
        if metrics.success {
            self.successful_inferences += 1;
        } else {
            self.failed_inferences += 1;
        }

        self.total_inference_time += metrics.inference_time;
        self.min_inference_time = self.min_inference_time.min(metrics.inference_time);
        self.max_inference_time = self.max_inference_time.max(metrics.inference_time);
        self.avg_inference_time = self.total_inference_time / (self.total_inferences as u32);

        self.total_input_bytes += metrics.input_size;
        self.total_output_bytes += metrics.output_size;
    }

    /// Calculate success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_inferences == 0 {
            0.0
        } else {
            (self.successful_inferences as f64 / self.total_inferences as f64) * 100.0
        }
    }

    /// Calculate throughput in inferences per second
    pub fn throughput(&self) -> f64 {
        if self.total_inference_time.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.total_inferences as f64 / self.total_inference_time.as_secs_f64()
        }
    }
}

/// Performance monitor for tracking AI model performance
#[derive(Clone)]
pub struct PerformanceMonitor {
    stats: Arc<RwLock<HashMap<String, ModelStats>>>,
    enabled: bool,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(enabled: bool) -> Self {
        Self {
            stats: Arc::new(RwLock::new(HashMap::new())),
            enabled,
        }
    }

    /// Record inference metrics for a model
    pub fn record_inference(&self, metrics: InferenceMetrics) {
        if !self.enabled {
            return;
        }

        let mut stats = self.stats.write().unwrap();
        let model_stats = stats
            .entry(metrics.model_name.clone())
            .or_insert_with(|| ModelStats::new(metrics.model_name.clone()));

        model_stats.update(&metrics);
    }

    /// Start timing an inference operation
    pub fn start_inference(&self, model_name: &str) -> InferenceTimer {
        InferenceTimer {
            model_name: model_name.to_string(),
            start_time: Instant::now(),
            monitor: if self.enabled {
                Some(Arc::clone(&self.stats))
            } else {
                None
            },
        }
    }

    /// Get statistics for a specific model
    pub fn get_stats(&self, model_name: &str) -> Option<ModelStats> {
        let stats = self.stats.read().unwrap();
        stats.get(model_name).cloned()
    }

    /// Get statistics for all models
    pub fn get_all_stats(&self) -> Vec<ModelStats> {
        let stats = self.stats.read().unwrap();
        stats.values().cloned().collect()
    }

    /// Reset statistics for a specific model
    pub fn reset_stats(&self, model_name: &str) {
        let mut stats = self.stats.write().unwrap();
        stats.remove(model_name);
    }

    /// Reset all statistics
    pub fn reset_all_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.clear();
    }

    /// Generate a performance report
    pub fn generate_report(&self) -> String {
        let stats = self.stats.read().unwrap();
        let mut report = String::from("=== AI Performance Report ===\n\n");

        if stats.is_empty() {
            report.push_str("No inference data recorded.\n");
            return report;
        }

        for (model_name, model_stats) in stats.iter() {
            report.push_str(&format!("Model: {}\n", model_name));
            report.push_str(&format!(
                "  Total Inferences: {}\n",
                model_stats.total_inferences
            ));
            report.push_str(&format!(
                "  Success Rate: {:.2}%\n",
                model_stats.success_rate()
            ));
            report.push_str(&format!(
                "  Avg Inference Time: {:.2}ms\n",
                model_stats.avg_inference_time.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Min Inference Time: {:.2}ms\n",
                model_stats.min_inference_time.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Max Inference Time: {:.2}ms\n",
                model_stats.max_inference_time.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Throughput: {:.2} inf/sec\n",
                model_stats.throughput()
            ));
            report.push_str(&format!(
                "  Total Input: {} bytes\n",
                model_stats.total_input_bytes
            ));
            report.push_str(&format!(
                "  Total Output: {} bytes\n\n",
                model_stats.total_output_bytes
            ));
        }

        report
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Timer for tracking inference duration
pub struct InferenceTimer {
    model_name: String,
    start_time: Instant,
    monitor: Option<Arc<RwLock<HashMap<String, ModelStats>>>>,
}

impl InferenceTimer {
    /// Complete the inference and record metrics
    pub fn complete(self, input_size: usize, output_size: usize, success: bool) {
        if let Some(monitor) = self.monitor {
            let inference_time = self.start_time.elapsed();
            let metrics = InferenceMetrics {
                model_name: self.model_name.clone(),
                inference_time,
                input_size,
                output_size,
                success,
                timestamp: self.start_time,
            };

            let mut stats = monitor.write().unwrap();
            let model_stats = stats
                .entry(self.model_name.clone())
                .or_insert_with(|| ModelStats::new(self.model_name.clone()));

            model_stats.update(&metrics);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new(true);
        assert_eq!(monitor.get_all_stats().len(), 0);
    }

    #[test]
    fn test_record_inference() {
        let monitor = PerformanceMonitor::new(true);

        let metrics = InferenceMetrics {
            model_name: "test_model".to_string(),
            inference_time: Duration::from_millis(10),
            input_size: 100,
            output_size: 50,
            success: true,
            timestamp: Instant::now(),
        };

        monitor.record_inference(metrics);

        let stats = monitor.get_stats("test_model").unwrap();
        assert_eq!(stats.total_inferences, 1);
        assert_eq!(stats.successful_inferences, 1);
        assert_eq!(stats.success_rate(), 100.0);
    }

    #[test]
    fn test_inference_timer() {
        let monitor = PerformanceMonitor::new(true);

        let timer = monitor.start_inference("timer_model");
        thread::sleep(Duration::from_millis(10));
        timer.complete(100, 50, true);

        let stats = monitor.get_stats("timer_model").unwrap();
        assert_eq!(stats.total_inferences, 1);
        assert!(stats.avg_inference_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_multiple_inferences() {
        let monitor = PerformanceMonitor::new(true);

        for i in 0..5 {
            let metrics = InferenceMetrics {
                model_name: "multi_model".to_string(),
                inference_time: Duration::from_millis(10 + i),
                input_size: 100,
                output_size: 50,
                success: i % 2 == 0,
                timestamp: Instant::now(),
            };
            monitor.record_inference(metrics);
        }

        let stats = monitor.get_stats("multi_model").unwrap();
        assert_eq!(stats.total_inferences, 5);
        assert_eq!(stats.successful_inferences, 3);
        assert_eq!(stats.failed_inferences, 2);
        assert_eq!(stats.success_rate(), 60.0);
    }

    #[test]
    fn test_model_stats_calculation() {
        let monitor = PerformanceMonitor::new(true);

        monitor.record_inference(InferenceMetrics {
            model_name: "calc_model".to_string(),
            inference_time: Duration::from_millis(5),
            input_size: 100,
            output_size: 50,
            success: true,
            timestamp: Instant::now(),
        });

        monitor.record_inference(InferenceMetrics {
            model_name: "calc_model".to_string(),
            inference_time: Duration::from_millis(15),
            input_size: 200,
            output_size: 100,
            success: true,
            timestamp: Instant::now(),
        });

        let stats = monitor.get_stats("calc_model").unwrap();
        assert_eq!(stats.min_inference_time, Duration::from_millis(5));
        assert_eq!(stats.max_inference_time, Duration::from_millis(15));
        assert_eq!(stats.avg_inference_time, Duration::from_millis(10));
        assert_eq!(stats.total_input_bytes, 300);
        assert_eq!(stats.total_output_bytes, 150);
    }

    #[test]
    fn test_reset_stats() {
        let monitor = PerformanceMonitor::new(true);

        monitor.record_inference(InferenceMetrics {
            model_name: "reset_model".to_string(),
            inference_time: Duration::from_millis(10),
            input_size: 100,
            output_size: 50,
            success: true,
            timestamp: Instant::now(),
        });

        assert!(monitor.get_stats("reset_model").is_some());
        monitor.reset_stats("reset_model");
        assert!(monitor.get_stats("reset_model").is_none());
    }

    #[test]
    fn test_disabled_monitor() {
        let monitor = PerformanceMonitor::new(false);

        monitor.record_inference(InferenceMetrics {
            model_name: "disabled_model".to_string(),
            inference_time: Duration::from_millis(10),
            input_size: 100,
            output_size: 50,
            success: true,
            timestamp: Instant::now(),
        });

        assert!(monitor.get_stats("disabled_model").is_none());
    }

    #[test]
    fn test_generate_report() {
        let monitor = PerformanceMonitor::new(true);

        monitor.record_inference(InferenceMetrics {
            model_name: "report_model".to_string(),
            inference_time: Duration::from_millis(10),
            input_size: 100,
            output_size: 50,
            success: true,
            timestamp: Instant::now(),
        });

        let report = monitor.generate_report();
        assert!(report.contains("AI Performance Report"));
        assert!(report.contains("report_model"));
        assert!(report.contains("Success Rate"));
    }
}
