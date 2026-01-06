use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct InferenceMetrics {
    pub model_name: String,
    pub inference_time: Duration,
    pub input_size: usize,
    pub output_size: usize,
    pub success: bool,
    pub timestamp: Instant,
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            inference_time: Duration::default(),
            input_size: 0,
            output_size: 0,
            success: false,
            timestamp: Instant::now(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PerformanceMonitor {
    enabled: bool,
}

impl PerformanceMonitor {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn record_inference(&self, _metrics: InferenceMetrics) {
        if self.enabled {
            // Placeholder: hook into real telemetry later
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }
}
