//! 性能测试器

use crate::*;
use anyhow::Result;
use playwright::Playwright;

pub struct PerformanceTester;

impl PerformanceTester {
    pub fn new() -> Self {
        Self
    }

    pub async fn compare_performance(
        &self,
        original_url: &str,
        generated_url: &str,
    ) -> Result<TestResult> {
        let start_time = std::time::Instant::now();

        // 测试原始网站
        let original_metrics = self.measure_performance(original_url).await?;

        // 测试生成网站
        let generated_metrics = self.measure_performance(generated_url).await?;

        // 比较指标
        let comparisons = self.compare_metrics(&original_metrics, &generated_metrics);

        let duration = start_time.elapsed().as_millis() as u64;

        // 计算总体分数
        let avg_ratio: f64 = comparisons.iter().map(|c| c.ratio).sum::<f64>() / comparisons.len() as f64;
        let score = if avg_ratio <= 1.0 {
            // 生成版本更快或相同
            1.0
        } else if avg_ratio <= 1.5 {
            // 生成版本慢50%以内
            1.0 - (avg_ratio - 1.0)
        } else {
            // 生成版本慢超过50%
            0.0
        };

        let passed = score >= 0.8;

        let logs: Vec<String> = comparisons
            .iter()
            .map(|c| {
                format!(
                    "{}: original={:.0}ms, generated={:.0}ms, ratio={:.2}x",
                    c.metric, c.original_value, c.generated_value, c.ratio
                )
            })
            .collect();

        let errors: Vec<TestError> = comparisons
            .iter()
            .filter(|c| !c.passed)
            .map(|c| TestError {
                step: 0,
                message: format!("Performance regression in {}", c.metric),
                expected: format!("<= {:.0}ms", c.original_value * 1.2),
                actual: format!("{:.0}ms", c.generated_value),
                severity: ErrorSeverity::Warning,
            })
            .collect();

        Ok(TestResult {
            test_name: "Performance Comparison".to_string(),
            test_type: TestType::Performance,
            passed,
            score,
            duration_ms: duration,
            details: TestDetails {
                steps_executed: comparisons.len(),
                assertions_passed: comparisons.iter().filter(|c| c.passed).count(),
                assertions_failed: comparisons.iter().filter(|c| !c.passed).count(),
                screenshots: vec![],
                logs,
            },
            errors,
        })
    }

    async fn measure_performance(&self, url: &str) -> Result<PerformanceMetrics> {
        let playwright = Playwright::initialize().await?;
        let browser = playwright.chromium().launcher().headless(true).launch().await?;
        let context = browser.context_builder().build().await?;
        let page = context.new_page().await?;

        // 启用性能监控
        page.evaluate::<(), ()>(r#"
            window.performanceMetrics = {};
            const observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    window.performanceMetrics[entry.name] = entry.startTime;
                }
            });
            observer.observe({ entryTypes: ['navigation', 'paint', 'measure'] });
        "#, ()).await?;

        let start = std::time::Instant::now();
        page.goto_builder(url).goto().await?;
        // Use a simple delay instead of wait_for_load_state which doesn't exist
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        let load_time = start.elapsed().as_millis() as f64;

        // 获取性能指标
        let metrics: serde_json::Value = page.evaluate::<(), serde_json::Value>(r#"
            () => ({
                loadTime: performance.now(),
                domContentLoaded: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
                firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime,
                firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime,
                memory: performance.memory?.usedJSHeapSize
            })
        "#, ()).await?;

        browser.close().await?;

        Ok(PerformanceMetrics {
            load_time,
            dom_content_loaded: metrics.get("domContentLoaded").and_then(|v: &serde_json::Value| v.as_f64()).unwrap_or(0.0),
            first_paint: metrics.get("firstPaint").and_then(|v: &serde_json::Value| v.as_f64()).unwrap_or(0.0),
            first_contentful_paint: metrics.get("firstContentfulPaint").and_then(|v: &serde_json::Value| v.as_f64()).unwrap_or(0.0),
            memory_usage: metrics.get("memory").and_then(|v: &serde_json::Value| v.as_f64()).unwrap_or(0.0),
        })
    }

    fn compare_metrics(
        &self,
        original: &PerformanceMetrics,
        generated: &PerformanceMetrics,
    ) -> Vec<PerformanceResult> {
        vec![
            self.compare_metric("load_time", original.load_time, generated.load_time),
            self.compare_metric("dom_content_loaded", original.dom_content_loaded, generated.dom_content_loaded),
            self.compare_metric("first_paint", original.first_paint, generated.first_paint),
            self.compare_metric("first_contentful_paint", original.first_contentful_paint, generated.first_contentful_paint),
            self.compare_metric("memory_usage", original.memory_usage, generated.memory_usage),
        ]
    }

    fn compare_metric(&self, name: &str, original: f64, generated: f64) -> PerformanceResult {
        let ratio = if original > 0.0 {
            generated / original
        } else {
            1.0
        };

        // 允许20%的性能差异
        let passed = ratio <= 1.2;

        PerformanceResult {
            metric: name.to_string(),
            original_value: original,
            generated_value: generated,
            ratio,
            passed,
        }
    }
}

impl Default for PerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct PerformanceMetrics {
    load_time: f64,
    dom_content_loaded: f64,
    first_paint: f64,
    first_contentful_paint: f64,
    memory_usage: f64,
}
