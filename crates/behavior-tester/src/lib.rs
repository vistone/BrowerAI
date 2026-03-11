//! 行为测试器
//! 端到端验证生成的代码与原始网站行为等价

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod equivalence_tester;
pub mod visual_regression;
pub mod performance_tester;
pub mod test_scenarios;

pub use equivalence_tester::EquivalenceTester;
pub use visual_regression::VisualRegressionTester;
pub use performance_tester::PerformanceTester;
pub use test_scenarios::{TestScenario, TestStep, TestAction};

/// 测试报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub test_run_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub original_url: String,
    pub generated_url: String,
    pub overall_score: f64,
    pub passed: bool,
    pub results: Vec<TestResult>,
    pub summary: TestSummary,
}

/// 测试结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_name: String,
    pub test_type: TestType,
    pub passed: bool,
    pub score: f64,
    pub duration_ms: u64,
    pub details: TestDetails,
    pub errors: Vec<TestError>,
}

/// 测试类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestType {
    Functional,
    Visual,
    Performance,
    Accessibility,
    CrossBrowser,
}

/// 测试详情
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDetails {
    pub steps_executed: usize,
    pub assertions_passed: usize,
    pub assertions_failed: usize,
    pub screenshots: Vec<String>,
    pub logs: Vec<String>,
}

/// 测试错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestError {
    pub step: usize,
    pub message: String,
    pub expected: String,
    pub actual: String,
    pub severity: ErrorSeverity,
}

/// 错误严重程度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Critical,
}

/// 测试摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub skipped_tests: usize,
    pub total_duration_ms: u64,
    pub average_score: f64,
}

/// 行为等价性结果
#[derive(Debug, Clone)]
pub struct EquivalenceResult {
    pub equivalent: bool,
    pub similarity_score: f64,
    pub differences: Vec<BehaviorDifference>,
    pub original_events: Vec<EventRecord>,
    pub generated_events: Vec<EventRecord>,
}

/// 行为差异
#[derive(Debug, Clone)]
pub struct BehaviorDifference {
    pub event_type: String,
    pub property: String,
    pub original_value: String,
    pub generated_value: String,
    pub tolerance: f64,
}

/// 事件记录
#[derive(Debug, Clone)]
pub struct EventRecord {
    pub timestamp: u64,
    pub event_type: String,
    pub target: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// 视觉回归结果
#[derive(Debug, Clone)]
pub struct VisualRegressionResult {
    pub similarity: f64,
    pub diff_image_path: Option<String>,
    pub pixel_diff_count: usize,
    pub pixel_diff_percentage: f64,
    pub passed: bool,
}

/// 性能测试结果
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    pub metric: String,
    pub original_value: f64,
    pub generated_value: f64,
    pub ratio: f64,
    pub passed: bool,
}

/// 行为测试引擎
pub struct BehaviorTestEngine {
    equivalence_tester: EquivalenceTester,
    visual_tester: VisualRegressionTester,
    performance_tester: PerformanceTester,
}

impl BehaviorTestEngine {
    pub fn new() -> Self {
        Self {
            equivalence_tester: EquivalenceTester::new(),
            visual_tester: VisualRegressionTester::new(),
            performance_tester: PerformanceTester::new(),
        }
    }

    /// 运行完整测试套件
    pub async fn run_full_test_suite(
        &self,
        original_url: &str,
        generated_url: &str,
        scenarios: &[TestScenario],
    ) -> Result<TestReport> {
        log::info!("Starting full test suite");
        
        let mut results = Vec::new();
        let start_time = std::time::Instant::now();

        // 1. 功能等价性测试
        for scenario in scenarios {
            let result = self.equivalence_tester.test_scenario(
                original_url,
                generated_url,
                scenario,
            ).await?;
            results.push(result);
        }

        // 2. 视觉回归测试
        let visual_result = self.visual_tester.compare_pages(
            original_url,
            generated_url,
        ).await?;
        results.push(visual_result);

        // 3. 性能测试
        let perf_result = self.performance_tester.compare_performance(
            original_url,
            generated_url,
        ).await?;
        results.push(perf_result);

        let total_duration = start_time.elapsed().as_millis() as u64;
        
        // 计算总分
        let average_score = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64
        };

        let passed_tests = results.iter().filter(|r| r.passed).count();
        let failed_tests = results.len() - passed_tests;

        // Clone results for summary before moving
        let total_tests = results.len();
        let summary = TestSummary {
            total_tests,
            passed_tests,
            failed_tests,
            skipped_tests: 0,
            total_duration_ms: total_duration,
            average_score,
        };

        Ok(TestReport {
            test_run_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            original_url: original_url.to_string(),
            generated_url: generated_url.to_string(),
            overall_score: average_score,
            passed: average_score >= 0.8,
            results,
            summary,
        })
    }

    /// 生成测试报告
    pub fn generate_report(&self, results: &TestReport) -> String {
        let mut report = String::new();

        report.push_str("# Test Report\n\n");
        report.push_str(&format!("**Test Run ID:** {}\n", results.test_run_id));
        report.push_str(&format!("**Timestamp:** {}\n", results.timestamp));
        report.push_str(&format!("**Overall Score:** {:.1}%\n", results.overall_score * 100.0));
        report.push_str(&format!("**Status:** {}\n\n", if results.passed { "✅ PASSED" } else { "❌ FAILED" }));

        report.push_str("## Summary\n\n");
        report.push_str(&format!("- Total Tests: {}\n", results.summary.total_tests));
        report.push_str(&format!("- Passed: {}\n", results.summary.passed_tests));
        report.push_str(&format!("- Failed: {}\n", results.summary.failed_tests));
        report.push_str(&format!("- Duration: {}ms\n\n", results.summary.total_duration_ms));

        report.push_str("## Detailed Results\n\n");
        for result in &results.results {
            report.push_str(&format!("### {}\n", result.test_name));
            report.push_str(&format!("- Type: {:?}\n", result.test_type));
            report.push_str(&format!("- Score: {:.1}%\n", result.score * 100.0));
            report.push_str(&format!("- Status: {}\n", if result.passed { "✅" } else { "❌" }));
            report.push_str(&format!("- Duration: {}ms\n", result.duration_ms));
            
            if !result.errors.is_empty() {
                report.push_str("\n**Errors:**\n");
                for error in &result.errors {
                    report.push_str(&format!("- Step {}: {}\n", error.step, error.message));
                }
            }
            report.push('\n');
        }

        report
    }
}

impl Default for BehaviorTestEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavior_test_engine_new() {
        let engine = BehaviorTestEngine::new();
        // Engine should be created successfully
    }

    #[test]
    fn test_behavior_test_engine_default() {
        let engine: BehaviorTestEngine = Default::default();
        // Should use default implementation
    }

    #[test]
    fn test_test_summary_creation() {
        let summary = TestSummary {
            total_tests: 10,
            passed_tests: 8,
            failed_tests: 2,
            skipped_tests: 0,
            total_duration_ms: 1000,
            average_score: 0.85,
        };
        assert_eq!(summary.total_tests, 10);
        assert_eq!(summary.passed_tests, 8);
        assert_eq!(summary.average_score, 0.85);
    }

    #[test]
    fn test_test_result_creation() {
        let result = TestResult {
            test_name: "functional_test".to_string(),
            test_type: TestType::Functional,
            passed: true,
            score: 0.95,
            duration_ms: 100,
            details: TestDetails {
                steps_executed: 5,
                assertions_passed: 10,
                assertions_failed: 0,
                screenshots: vec![],
                logs: vec![],
            },
            errors: vec![],
        };
        assert!(result.passed);
        assert_eq!(result.score, 0.95);
    }

    #[test]
    fn test_test_type_variants() {
        let types = vec![
            TestType::Functional,
            TestType::Visual,
            TestType::Performance,
            TestType::Accessibility,
            TestType::CrossBrowser,
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_error_severity_variants() {
        let severities = vec![
            ErrorSeverity::Info,
            ErrorSeverity::Warning,
            ErrorSeverity::Critical,
        ];
        assert_eq!(severities.len(), 3);
    }

    #[test]
    fn test_test_error_creation() {
        let error = TestError {
            step: 1,
            message: "Element not found".to_string(),
            expected: "button".to_string(),
            actual: "div".to_string(),
            severity: ErrorSeverity::Critical,
        };
        assert_eq!(error.step, 1);
        assert!(matches!(error.severity, ErrorSeverity::Critical));
    }

    #[test]
    fn test_equivalence_result_creation() {
        let result = EquivalenceResult {
            equivalent: true,
            similarity_score: 0.98,
            differences: vec![],
            original_events: vec![],
            generated_events: vec![],
        };
        assert!(result.equivalent);
        assert_eq!(result.similarity_score, 0.98);
    }

    #[test]
    fn test_visual_regression_result_creation() {
        let result = VisualRegressionResult {
            similarity: 0.99,
            diff_image_path: None,
            pixel_diff_count: 0,
            pixel_diff_percentage: 0.0,
            passed: true,
        };
        assert!(result.passed);
        assert_eq!(result.similarity, 0.99);
    }

    #[test]
    fn test_performance_result_creation() {
        let result = PerformanceResult {
            metric: "load_time".to_string(),
            original_value: 1000.0,
            generated_value: 900.0,
            ratio: 0.9,
            passed: true,
        };
        assert!(result.passed);
        assert_eq!(result.metric, "load_time");
    }
}
