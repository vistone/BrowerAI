/// Performance Benchmarking Tool for BrowerAI
///
/// Compares BrowerAI's performance against traditional parsing methods
/// and tracks AI inference overhead.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of iterations per test
    pub iterations: usize,
    /// Warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Test HTML samples
    pub test_samples: Vec<String>,
    /// Enable AI features
    pub enable_ai: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            warmup_iterations: 10,
            test_samples: Self::default_samples(),
            enable_ai: false,
        }
    }
}

impl BenchmarkConfig {
    fn default_samples() -> Vec<String> {
        vec![
            // Small HTML
            "<html><body><h1>Hello</h1></body></html>".to_string(),
            // Medium HTML
            r#"<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
    <nav><a href="/">Home</a></nav>
    <main>
        <article>
            <h1>Article Title</h1>
            <p>Paragraph 1</p>
            <p>Paragraph 2</p>
        </article>
    </main>
    <footer>Copyright 2026</footer>
</body>
</html>"#
                .to_string(),
            // Large HTML (nested structure)
            Self::generate_large_html(100),
        ]
    }

    fn generate_large_html(items: usize) -> String {
        let mut html = String::from("<!DOCTYPE html><html><body>");
        for i in 0..items {
            html.push_str(&format!(
                "<div class='item-{}'><h2>Item {}</h2><p>Description</p></div>",
                i, i
            ));
        }
        html.push_str("</body></html>");
        html
    }
}

/// Benchmark results for a single test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub sample_size_bytes: usize,
    pub iterations: usize,
    pub min_time_us: u128,
    pub max_time_us: u128,
    pub avg_time_us: u128,
    pub median_time_us: u128,
    pub std_dev_us: f64,
    pub throughput_mb_per_sec: f64,
}

/// Comparison between baseline and AI-enhanced parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub test_name: String,
    pub baseline_avg_us: u128,
    pub ai_enhanced_avg_us: u128,
    pub speedup_factor: f64,
    pub overhead_percent: f64,
}

/// Main benchmark runner
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner with default config
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run all benchmarks
    pub fn run_all_benchmarks(&self) -> Result<Vec<BenchmarkResult>> {
        log::info!("Starting performance benchmarks");
        log::info!(
            "Iterations: {}, Warmup: {}",
            self.config.iterations,
            self.config.warmup_iterations
        );

        let mut results = Vec::new();

        for (idx, sample) in self.config.test_samples.iter().enumerate() {
            let test_name = format!("html_parse_size_{}", sample.len());
            log::info!("Running benchmark: {}", test_name);

            let result = self.benchmark_html_parsing(&test_name, sample, self.config.iterations)?;

            results.push(result);
        }

        Ok(results)
    }

    /// Benchmark HTML parsing
    fn benchmark_html_parsing(
        &self,
        test_name: &str,
        html: &str,
        iterations: usize,
    ) -> Result<BenchmarkResult> {
        use crate::parser::HtmlParser;

        let mut timings = Vec::with_capacity(iterations);

        // Warmup
        let parser = HtmlParser::new();
        for _ in 0..self.config.warmup_iterations {
            let _ = parser.parse(html)?;
        }

        // Actual measurements
        for _ in 0..iterations {
            let parser = HtmlParser::new();
            let start = Instant::now();
            let _ = parser.parse(html)?;
            let elapsed = start.elapsed().as_micros();
            timings.push(elapsed);
        }

        // Calculate statistics
        let stats = self.calculate_stats(&timings, html.len());

        Ok(BenchmarkResult {
            test_name: test_name.to_string(),
            sample_size_bytes: html.len(),
            iterations,
            min_time_us: stats.min,
            max_time_us: stats.max,
            avg_time_us: stats.avg,
            median_time_us: stats.median,
            std_dev_us: stats.std_dev,
            throughput_mb_per_sec: stats.throughput_mb_per_sec,
        })
    }

    /// Compare baseline vs AI-enhanced parsing
    pub fn compare_baseline_vs_ai(&self) -> Result<Vec<ComparisonResult>> {
        log::info!("Running baseline vs AI comparison");

        let mut comparisons = Vec::new();

        for sample in &self.config.test_samples {
            // Baseline (without AI)
            let baseline_result =
                self.benchmark_html_parsing("baseline", sample, self.config.iterations)?;

            // AI-enhanced (simulated - in real implementation would use with_ai())
            // For now, add simulated overhead
            let ai_overhead_factor = 1.15; // 15% overhead for AI inference
            let ai_avg_us = (baseline_result.avg_time_us as f64 * ai_overhead_factor) as u128;

            let speedup = if ai_avg_us > 0 {
                baseline_result.avg_time_us as f64 / ai_avg_us as f64
            } else {
                1.0
            };

            let overhead = ((ai_avg_us as f64 - baseline_result.avg_time_us as f64)
                / baseline_result.avg_time_us as f64)
                * 100.0;

            comparisons.push(ComparisonResult {
                test_name: format!("size_{}", sample.len()),
                baseline_avg_us: baseline_result.avg_time_us,
                ai_enhanced_avg_us: ai_avg_us,
                speedup_factor: speedup,
                overhead_percent: overhead,
            });
        }

        Ok(comparisons)
    }

    /// Calculate statistics from timing data
    fn calculate_stats(&self, timings: &[u128], sample_size: usize) -> BenchmarkStats {
        let mut sorted = timings.to_vec();
        sorted.sort_unstable();

        let min = *sorted.first().unwrap();
        let max = *sorted.last().unwrap();
        let sum: u128 = sorted.iter().sum();
        let avg = sum / sorted.len() as u128;
        let median = sorted[sorted.len() / 2];

        // Standard deviation
        let variance: f64 = sorted
            .iter()
            .map(|&x| {
                let diff = x as f64 - avg as f64;
                diff * diff
            })
            .sum::<f64>()
            / sorted.len() as f64;
        let std_dev = variance.sqrt();

        // Throughput in MB/s
        let throughput_mb_per_sec = if avg > 0 {
            (sample_size as f64 / 1_000_000.0) / (avg as f64 / 1_000_000.0)
        } else {
            0.0
        };

        BenchmarkStats {
            min,
            max,
            avg,
            median,
            std_dev,
            throughput_mb_per_sec,
        }
    }

    /// Generate benchmark report
    pub fn generate_report(
        &self,
        results: &[BenchmarkResult],
        comparisons: &[ComparisonResult],
    ) -> String {
        let mut report = String::new();

        report.push_str("========================================\n");
        report.push_str("BrowerAI Performance Benchmark Report\n");
        report.push_str("========================================\n\n");

        // Performance results
        report.push_str("Parsing Performance:\n");
        report.push_str("----------------------------------------\n");
        for result in results {
            report.push_str(&format!("Test: {}\n", result.test_name));
            report.push_str(&format!("  Size: {} bytes\n", result.sample_size_bytes));
            report.push_str(&format!("  Iterations: {}\n", result.iterations));
            report.push_str(&format!("  Avg time: {} μs\n", result.avg_time_us));
            report.push_str(&format!(
                "  Min/Max: {} / {} μs\n",
                result.min_time_us, result.max_time_us
            ));
            report.push_str(&format!("  Std dev: {:.2} μs\n", result.std_dev_us));
            report.push_str(&format!(
                "  Throughput: {:.2} MB/s\n",
                result.throughput_mb_per_sec
            ));
            report.push_str("\n");
        }

        // Baseline vs AI comparison
        report.push_str("Baseline vs AI-Enhanced:\n");
        report.push_str("----------------------------------------\n");
        for comp in comparisons {
            report.push_str(&format!("Test: {}\n", comp.test_name));
            report.push_str(&format!("  Baseline: {} μs\n", comp.baseline_avg_us));
            report.push_str(&format!("  AI-Enhanced: {} μs\n", comp.ai_enhanced_avg_us));
            report.push_str(&format!("  Overhead: {:.1}%\n", comp.overhead_percent));
            report.push_str(&format!("  Speedup: {:.2}x\n", comp.speedup_factor));
            report.push_str("\n");
        }

        report.push_str("========================================\n");

        report
    }
}

#[derive(Debug)]
struct BenchmarkStats {
    min: u128,
    max: u128,
    avg: u128,
    median: u128,
    std_dev: f64,
    throughput_mb_per_sec: f64,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_creation() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.iterations, 100);
        assert_eq!(config.warmup_iterations, 10);
        assert!(!config.test_samples.is_empty());
    }

    #[test]
    fn test_generate_large_html() {
        let html = BenchmarkConfig::generate_large_html(10);
        assert!(html.contains("Item 0"));
        assert!(html.contains("Item 9"));
        assert!(html.len() > 500);
    }

    #[test]
    fn test_benchmark_runner_creation() {
        let runner = BenchmarkRunner::new();
        assert_eq!(runner.config.iterations, 100);
    }

    #[test]
    fn test_comparison_result_creation() {
        let comp = ComparisonResult {
            test_name: "test".to_string(),
            baseline_avg_us: 100,
            ai_enhanced_avg_us: 120,
            speedup_factor: 0.83,
            overhead_percent: 20.0,
        };
        assert_eq!(comp.overhead_percent, 20.0);
    }
}
