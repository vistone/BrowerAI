use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use prometheus::{register_histogram_vec, register_int_counter_vec, HistogramVec, IntCounterVec};

/// 缓存性能指标（Prometheus 导出）。
#[derive(Clone)]
pub struct CacheMetrics {
    /// 命中次数（按层统计）
    hits: IntCounterVec,
    /// 未命中次数（按层统计）
    misses: IntCounterVec,
    /// 降级次数（Redis 故障等）
    degradations: IntCounterVec,
    /// 操作延迟直方图
    latency: HistogramVec,
    /// 批量操作大小直方图
    batch_size: HistogramVec,
}

impl CacheMetrics {
    pub fn new() -> Self {
        Self {
            hits: register_int_counter_vec!(
                "cache_hits_total",
                "Total cache hits by layer",
                &["layer"]
            )
            .unwrap(),
            misses: register_int_counter_vec!(
                "cache_misses_total",
                "Total cache misses by layer",
                &["layer"]
            )
            .unwrap(),
            degradations: register_int_counter_vec!(
                "cache_degradations_total",
                "Total degradation events by layer",
                &["layer", "reason"]
            )
            .unwrap(),
            latency: register_histogram_vec!(
                "cache_operation_latency_seconds",
                "Cache operation latency",
                &["operation", "layer"],
                vec![0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
            )
            .unwrap(),
            batch_size: register_histogram_vec!(
                "cache_batch_size",
                "Batch operation size",
                &["operation"],
                vec![1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
            )
            .unwrap(),
        }
    }

    pub fn record_hit(&self, layer: &str) {
        self.hits.with_label_values(&[layer]).inc();
    }

    pub fn record_miss(&self, layer: &str) {
        self.misses.with_label_values(&[layer]).inc();
    }

    pub fn record_degradation(&self, layer: &str, reason: &str) {
        self.degradations.with_label_values(&[layer, reason]).inc();
    }

    pub fn record_latency(&self, operation: &str, layer: &str, duration_secs: f64) {
        self.latency
            .with_label_values(&[operation, layer])
            .observe(duration_secs);
    }

    pub fn record_batch_size(&self, operation: &str, size: usize) {
        self.batch_size
            .with_label_values(&[operation])
            .observe(size as f64);
    }

    /// 获取 Prometheus 格式的指标文本。
    pub fn export(&self) -> String {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// 简化的内存指标（无 Prometheus 依赖）。
#[derive(Clone, Default)]
pub struct SimpleMetrics {
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    degradations: Arc<AtomicU64>,
}

impl SimpleMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_degradation(&self) {
        self.degradations.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn get_misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn get_degradations(&self) -> u64 {
        self.degradations.load(Ordering::Relaxed)
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = self.get_hits();
        let total = hits + self.get_misses();
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}
