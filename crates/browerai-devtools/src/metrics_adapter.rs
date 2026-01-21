//! 性能指标适配器 - 连接 browerai-metrics 到 DevTools
//!
//! 提供将实际性能指标数据转换为 WebView 面板格式的功能

use crate::webview::{MetricsProvider, PerformanceMetrics};
use anyhow::Result;

/// 基于 Prometheus 指标的性能提供者
pub struct PrometheusMetricsProvider {
    /// 参考性能基线数据（用于演示）
    baseline_metrics: PerformanceMetrics,
}

impl PrometheusMetricsProvider {
    /// 创建新的 Prometheus 指标提供者
    pub fn new() -> Self {
        Self {
            baseline_metrics: PerformanceMetrics {
                lcp_ms: 2000.0,
                inp_ms: 100.0,
                cls: 0.05,
                ttfb_ms: 500.0,
                total_load_time_ms: 3000.0,
                render_time_ms: 150.0,
            },
        }
    }

    /// 从真实指标源更新基线
    ///
    /// 在实际集成中，这会从 browerai-metrics::MetricsRegistry 读取数据
    pub fn with_baseline(mut self, baseline: PerformanceMetrics) -> Self {
        self.baseline_metrics = baseline;
        self
    }
}

impl Default for PrometheusMetricsProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsProvider for PrometheusMetricsProvider {
    fn collect_metrics(&self) -> Result<PerformanceMetrics> {
        // 在实际实现中，这里会访问 browerai-metrics::MetricsRegistry
        // 获取真实的 Prometheus 指标数据
        //
        // 示例集成：
        // let registry = MetricsRegistry::get_global();
        // let metrics = PerformanceMetrics {
        //     lcp_ms: registry.render_duration.get_sample_count() as f64,
        //     ...
        // };
        //
        // 现在返回基线数据用于演示
        Ok(self.baseline_metrics.clone())
    }
}

/// 模拟指标提供者（用于测试）
pub struct MockMetricsProvider {
    metrics: PerformanceMetrics,
}

impl MockMetricsProvider {
    /// 创建新的模拟指标提供者
    pub fn new(metrics: PerformanceMetrics) -> Self {
        Self { metrics }
    }
}

impl MetricsProvider for MockMetricsProvider {
    fn collect_metrics(&self) -> Result<PerformanceMetrics> {
        Ok(self.metrics.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_metrics_provider() -> Result<()> {
        let provider = PrometheusMetricsProvider::new();
        let metrics = provider.collect_metrics()?;

        assert_eq!(metrics.lcp_ms, 2000.0);
        assert_eq!(metrics.inp_ms, 100.0);
        assert_eq!(metrics.cls, 0.05);

        Ok(())
    }

    #[test]
    fn test_mock_metrics_provider() -> Result<()> {
        let test_metrics = PerformanceMetrics {
            lcp_ms: 1500.0,
            inp_ms: 80.0,
            cls: 0.02,
            ttfb_ms: 400.0,
            total_load_time_ms: 2500.0,
            render_time_ms: 100.0,
        };

        let provider = MockMetricsProvider::new(test_metrics);
        let collected = provider.collect_metrics()?;

        assert_eq!(collected.lcp_ms, 1500.0);

        Ok(())
    }
}
