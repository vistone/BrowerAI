use crate::performance_monitor::{InferenceMetrics, PerformanceMonitor};

#[derive(Clone, Debug, Default)]
pub struct AdvancedPerformanceMonitor {
    inner: PerformanceMonitor,
}

impl AdvancedPerformanceMonitor {
    pub fn new(enabled: bool) -> Self {
        Self {
            inner: PerformanceMonitor::new(enabled),
        }
    }

    pub fn record_inference(&self, metrics: InferenceMetrics) {
        self.inner.record_inference(metrics);
    }

    pub fn enabled(&self) -> bool {
        self.inner.enabled()
    }
}
