//! 指标收集系统
//!
//! 提供 BrowerAI 的性能指标收集和导出

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 指标类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// 计数器（单调递增）
    Counter,
    /// 仪表盘（可上下变化）
    Gauge,
    /// 直方图（分布统计）
    Histogram,
    /// 摘要（分位数统计）
    Summary,
}

/// 指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    /// 指标名称
    pub name: String,
    /// 指标类型
    pub metric_type: MetricType,
    /// 指标值
    pub value: f64,
    /// 标签
    pub labels: HashMap<String, String>,
    /// 时间戳
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// 描述
    pub description: Option<String>,
}

impl Metric {
    /// 创建新的指标
    pub fn new(name: impl Into<String>, metric_type: MetricType, value: f64) -> Self {
        Self {
            name: name.into(),
            metric_type,
            value,
            labels: HashMap::new(),
            timestamp: chrono::Utc::now(),
            description: None,
        }
    }

    /// 添加标签
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// 添加描述
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// 创建计数器指标
    pub fn counter(name: impl Into<String>, value: f64) -> Self {
        Self::new(name, MetricType::Counter, value)
    }

    /// 创建仪表盘指标
    pub fn gauge(name: impl Into<String>, value: f64) -> Self {
        Self::new(name, MetricType::Gauge, value)
    }

    /// 创建直方图指标
    pub fn histogram(name: impl Into<String>, value: f64) -> Self {
        Self::new(name, MetricType::Histogram, value)
    }
}

/// 指标统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricStats {
    /// 样本数
    pub count: u64,
    /// 总和
    pub sum: f64,
    /// 最小值
    pub min: f64,
    /// 最大值
    pub max: f64,
    /// 平均值
    pub avg: f64,
    /// 分位数（P50, P90, P95, P99）
    pub percentiles: HashMap<String, f64>,
}

impl MetricStats {
    /// 从样本计算统计
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let count = samples.len() as u64;
        let sum: f64 = samples.iter().sum();
        let min = samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let avg = sum / count as f64;

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut percentiles = HashMap::new();
        percentiles.insert("p50".to_string(), Self::percentile(&sorted, 0.5));
        percentiles.insert("p90".to_string(), Self::percentile(&sorted, 0.9));
        percentiles.insert("p95".to_string(), Self::percentile(&sorted, 0.95));
        percentiles.insert("p99".to_string(), Self::percentile(&sorted, 0.99));

        Self {
            count,
            sum,
            min,
            max,
            avg,
            percentiles,
        }
    }

    /// 计算分位数
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let index = (p * (sorted.len() - 1) as f64) as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

/// 直方图
#[derive(Debug, Clone)]
pub struct Histogram {
    /// 桶边界
    pub buckets: Vec<f64>,
    /// 桶计数
    pub counts: Vec<u64>,
    /// 总和
    pub sum: f64,
    /// 总数
    pub total_count: u64,
}

impl Histogram {
    /// 创建新的直方图
    pub fn new(buckets: Vec<f64>) -> Self {
        let len = buckets.len();
        Self {
            buckets,
            counts: vec![0; len + 1], // +1 for +Inf bucket
            sum: 0.0,
            total_count: 0,
        }
    }

    /// 创建默认的延迟直方图（毫秒）
    pub fn default_latency() -> Self {
        Self::new(vec![
            1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
        ])
    }

    /// 观察值
    pub fn observe(&mut self, value: f64) {
        self.sum += value;
        self.total_count += 1;

        for (i, &bucket) in self.buckets.iter().enumerate() {
            if value <= bucket {
                self.counts[i] += 1;
                return;
            }
        }
        // +Inf bucket
        self.counts[self.buckets.len()] += 1;
    }

    /// 获取分位数
    pub fn percentile(&self, p: f64) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let target = (p * self.total_count as f64) as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return if i < self.buckets.len() {
                    self.buckets[i]
                } else {
                    f64::INFINITY
                };
            }
        }

        f64::INFINITY
    }
}

/// 指标仪表板
#[derive(Debug, Clone)]
pub struct MetricsDashboard {
    /// 计数器
    counters: HashMap<String, Arc<AtomicU64>>,
    /// 仪表盘
    gauges: HashMap<String, f64>,
    /// 直方图
    histograms: HashMap<String, Histogram>,
    /// 启动时间
    start_time: Instant,
}

impl Default for MetricsDashboard {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsDashboard {
    /// 创建新的仪表板
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            start_time: Instant::now(),
        }
    }

    /// 增加计数器
    pub fn increment_counter(&mut self, name: impl Into<String>) {
        let name = name.into();
        let counter = self
            .counters
            .entry(name)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)));
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// 增加计数器（指定值）
    pub fn add_counter(&mut self, name: impl Into<String>, value: u64) {
        let name = name.into();
        let counter = self
            .counters
            .entry(name)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)));
        counter.fetch_add(value, Ordering::Relaxed);
    }

    /// 获取计数器值
    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// 设置仪表盘
    pub fn set_gauge(&mut self, name: impl Into<String>, value: f64) {
        self.gauges.insert(name.into(), value);
    }

    /// 获取仪表盘
    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.get(name).copied()
    }

    /// 观察直方图
    pub fn observe_histogram(&mut self, name: impl Into<String>, value: f64) {
        let name = name.into();
        let histogram = self
            .histograms
            .entry(name)
            .or_insert_with(Histogram::default_latency);
        histogram.observe(value);
    }

    /// 获取直方图统计
    pub fn get_histogram_stats(&self, name: &str) -> Option<MetricStats> {
        self.histograms.get(name).map(|h| {
            let mut samples = Vec::new();
            for (i, &count) in h.counts.iter().enumerate() {
                let value = if i < h.buckets.len() {
                    h.buckets[i]
                } else {
                    h.buckets.last().copied().unwrap_or(0.0) * 2.0
                };
                for _ in 0..count {
                    samples.push(value);
                }
            }
            MetricStats::from_samples(&samples)
        })
    }

    /// 获取运行时间
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// 导出所有指标为 Prometheus 格式
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();

        // 计数器
        for (name, counter) in &self.counters {
            let value = counter.load(Ordering::Relaxed);
            output.push_str(&format!("# TYPE {} counter\n", name));
            output.push_str(&format!("{} {}\n\n", name, value));
        }

        // 仪表盘
        for (name, value) in &self.gauges {
            output.push_str(&format!("# TYPE {} gauge\n", name));
            output.push_str(&format!("{} {}\n\n", name, value));
        }

        // 直方图
        for (name, histogram) in &self.histograms {
            output.push_str(&format!("# TYPE {} histogram\n", name));

            let mut cumulative = 0u64;
            for (i, &bucket) in histogram.buckets.iter().enumerate() {
                cumulative += histogram.counts[i];
                output.push_str(&format!(
                    "{}_bucket{{le=\"{}\"}} {}\n",
                    name, bucket, cumulative
                ));
            }
            // +Inf bucket
            cumulative += histogram.counts[histogram.buckets.len()];
            output.push_str(&format!("{}_bucket{{le=\"+Inf\"}} {}\n", name, cumulative));

            output.push_str(&format!("{}_sum {}\n", name, histogram.sum));
            output.push_str(&format!("{}_count {}\n\n", name, histogram.total_count));
        }

        output
    }

    /// 导出为 JSON
    pub fn export_json(&self) -> String {
        let metrics: Vec<Metric> = self.to_metrics();
        serde_json::to_string_pretty(&metrics).unwrap_or_default()
    }

    /// 转换为指标列表
    pub fn to_metrics(&self) -> Vec<Metric> {
        let mut metrics = Vec::new();

        for (name, counter) in &self.counters {
            let value = counter.load(Ordering::Relaxed) as f64;
            metrics.push(Metric::counter(name.clone(), value));
        }

        for (name, value) in &self.gauges {
            metrics.push(Metric::gauge(name.clone(), *value));
        }

        for (name, histogram) in &self.histograms {
            metrics.push(
                Metric::histogram(name.clone(), histogram.sum)
                    .with_label("count", histogram.total_count.to_string())
                    .with_label("p50", histogram.percentile(0.5).to_string())
                    .with_label("p95", histogram.percentile(0.95).to_string()),
            );
        }

        metrics
    }
}

/// 计时器（自动记录持续时间）
pub struct Timer {
    start: Instant,
    name: String,
    dashboard: Option<Arc<std::sync::Mutex<MetricsDashboard>>>,
}

impl Timer {
    /// 创建新的计时器
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            name: name.into(),
            dashboard: None,
        }
    }

    /// 关联仪表板
    pub fn with_dashboard(mut self, dashboard: Arc<std::sync::Mutex<MetricsDashboard>>) -> Self {
        self.dashboard = Some(dashboard);
        self
    }

    /// 获取已过去的时间
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// 获取已过去的毫秒数
    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed().as_millis() as u64
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let duration_ms = self.elapsed().as_millis() as f64;

        if let Some(ref dashboard) = self.dashboard {
            if let Ok(mut d) = dashboard.lock() {
                d.observe_histogram(format!("{}_duration_ms", self.name), duration_ms);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_creation() {
        let metric = Metric::counter("test_counter", 42.0)
            .with_label("method", "GET")
            .with_description("Test counter");

        assert_eq!(metric.name, "test_counter");
        assert_eq!(metric.value, 42.0);
        assert_eq!(metric.labels.get("method"), Some(&"GET".to_string()));
    }

    #[test]
    fn test_histogram() {
        let mut hist = Histogram::default_latency();

        hist.observe(10.0);
        hist.observe(50.0);
        hist.observe(100.0);

        assert_eq!(hist.total_count, 3);
        assert_eq!(hist.sum, 160.0);
        assert!(hist.percentile(0.5) >= 10.0);
    }

    #[test]
    fn test_metric_stats() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = MetricStats::from_samples(&samples);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.sum, 15.0);
        assert_eq!(stats.avg, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_dashboard() {
        let mut dashboard = MetricsDashboard::new();

        dashboard.increment_counter("requests");
        dashboard.increment_counter("requests");
        dashboard.set_gauge("active_connections", 10.0);
        dashboard.observe_histogram("response_time", 100.0);

        assert_eq!(dashboard.get_counter("requests"), 2);
        assert_eq!(dashboard.get_gauge("active_connections"), Some(10.0));
    }
}
