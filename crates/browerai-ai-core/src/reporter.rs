use crate::performance_monitor::PerformanceMonitor;
use crate::runtime::AiRuntime;

#[derive(Clone)]
pub struct AiReporter {
    runtime: AiRuntime,
    monitor: PerformanceMonitor,
}

impl AiReporter {
    pub fn new(runtime: AiRuntime, monitor: PerformanceMonitor) -> Self {
        Self { runtime, monitor }
    }

    pub fn generate_full_report(&self) -> String {
        format!(
            "AI Reporter (placeholder)\nRuntime enabled: {}\nMonitor enabled: {}",
            self.runtime.is_ai_enabled(),
            self.monitor.enabled()
        )
    }
}
