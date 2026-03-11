//! 等价性测试器

use crate::*;
use anyhow::Result;
use playwright::Playwright;
use std::path::PathBuf;

pub struct EquivalenceTester;

impl EquivalenceTester {
    pub fn new() -> Self {
        Self
    }

    pub async fn test_scenario(
        &self,
        original_url: &str,
        generated_url: &str,
        scenario: &TestScenario,
    ) -> Result<TestResult> {
        let start_time = std::time::Instant::now();

        // 在原始网站上执行场景
        let original_result = self.execute_scenario(original_url, scenario).await?;

        // 在生成网站上执行场景
        let generated_result = self.execute_scenario(generated_url, scenario).await?;

        // 比较结果
        let comparison = self.compare_results(&original_result, &generated_result);

        let duration = start_time.elapsed().as_millis() as u64;

        Ok(TestResult {
            test_name: scenario.name.clone(),
            test_type: TestType::Functional,
            passed: comparison.passed,
            score: comparison.score,
            duration_ms: duration,
            details: TestDetails {
                steps_executed: scenario.steps.len(),
                assertions_passed: comparison.assertions_passed,
                assertions_failed: comparison.assertions_failed,
                screenshots: comparison.screenshots,
                logs: comparison.logs,
            },
            errors: comparison.errors,
        })
    }

    async fn execute_scenario(
        &self,
        url: &str,
        scenario: &TestScenario,
    ) -> Result<ScenarioExecutionResult> {
        let playwright = Playwright::initialize().await?;
        let browser = playwright
            .chromium()
            .launcher()
            .headless(true)
            .launch()
            .await?;
        let context = browser.context_builder().build().await?;
        let page = context.new_page().await?;

        page.goto_builder(url).goto().await?;
        // Use a simple delay instead of wait_for_load_state which doesn't exist
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

        let mut events = Vec::new();
        let mut screenshots = Vec::new();
        let mut logs = Vec::new();

        for (step_idx, step) in scenario.steps.iter().enumerate() {
            match &step.action {
                TestAction::Click(selector) => {
                    page.click_builder(selector).click().await?;
                    events.push(self.record_event("click", selector));
                }
                TestAction::Type { selector, value } => {
                    page.fill_builder(selector, value).fill().await?;
                    events.push(self.record_event("input", selector));
                }
                TestAction::Scroll { direction, amount } => {
                    let script = match direction {
                        test_scenarios::ScrollDirection::Down => {
                            format!("window.scrollBy(0, {})", amount)
                        }
                        test_scenarios::ScrollDirection::Up => {
                            format!("window.scrollBy(0, -{})", amount)
                        }
                        _ => String::new(),
                    };
                    page.evaluate::<(), ()>(&script, ()).await?;
                    events.push(self.record_event("scroll", "window"));
                }
                TestAction::Wait(ms) => {
                    tokio::time::sleep(tokio::time::Duration::from_millis(*ms)).await;
                }
                TestAction::Screenshot(name) => {
                    let path = format!("screenshot_{}_{}.png", step_idx, name);
                    page.screenshot_builder()
                        .path(PathBuf::from(&path))
                        .screenshot()
                        .await?;
                    screenshots.push(path);
                }
                TestAction::AssertVisible(selector) => {
                    let is_visible = page.is_visible(selector, None).await?;
                    if !is_visible {
                        logs.push(format!("Assertion failed: {} should be visible", selector));
                    }
                }
                TestAction::AssertText { selector, expected } => {
                    let text = page.text_content(selector, None).await?.unwrap_or_default();
                    if text != *expected {
                        logs.push(format!(
                            "Assertion failed: expected '{}', got '{}'",
                            expected, text
                        ));
                    }
                }
                TestAction::AssertUrl(expected) => {
                    let url = page.url()?;
                    if url != *expected {
                        logs.push(format!(
                            "Assertion failed: expected '{}', got '{}'",
                            expected, url
                        ));
                    }
                }
                TestAction::KeyPress(key) => {
                    page.keyboard.press(key, None).await?;
                    events.push(self.record_event("keydown", "document"));
                }
            }

            // 等待页面稳定
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        browser.close().await?;

        Ok(ScenarioExecutionResult {
            events,
            screenshots,
            logs,
        })
    }

    fn record_event(&self, event_type: &str, target: &str) -> EventRecord {
        EventRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            event_type: event_type.to_string(),
            target: target.to_string(),
            properties: HashMap::new(),
        }
    }

    fn compare_results(
        &self,
        original: &ScenarioExecutionResult,
        generated: &ScenarioExecutionResult,
    ) -> ComparisonResult {
        let mut passed = true;
        let mut assertions_passed = 0;
        let mut assertions_failed = 0;
        let mut errors = Vec::new();

        // 比较事件数量
        if original.events.len() != generated.events.len() {
            passed = false;
            assertions_failed += 1;
            errors.push(TestError {
                step: 0,
                message: format!(
                    "Event count mismatch: original={}, generated={}",
                    original.events.len(),
                    generated.events.len()
                ),
                expected: original.events.len().to_string(),
                actual: generated.events.len().to_string(),
                severity: ErrorSeverity::Critical,
            });
        } else {
            assertions_passed += 1;
        }

        // 比较事件类型
        for (i, (orig, gen)) in original
            .events
            .iter()
            .zip(generated.events.iter())
            .enumerate()
        {
            if orig.event_type != gen.event_type {
                passed = false;
                assertions_failed += 1;
                errors.push(TestError {
                    step: i,
                    message: format!(
                        "Event type mismatch at step {}: expected '{}', got '{}'",
                        i, orig.event_type, gen.event_type
                    ),
                    expected: orig.event_type.clone(),
                    actual: gen.event_type.clone(),
                    severity: ErrorSeverity::Critical,
                });
            } else {
                assertions_passed += 1;
            }
        }

        // 计算分数
        let total_assertions = assertions_passed + assertions_failed;
        let score = if total_assertions > 0 {
            assertions_passed as f64 / total_assertions as f64
        } else {
            0.0
        };

        ComparisonResult {
            passed: passed && score >= 0.8,
            score,
            assertions_passed,
            assertions_failed,
            screenshots: original.screenshots.clone(),
            logs: original.logs.clone(),
            errors,
        }
    }
}

impl Default for EquivalenceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct ScenarioExecutionResult {
    events: Vec<EventRecord>,
    screenshots: Vec<String>,
    logs: Vec<String>,
}

#[derive(Debug)]
struct ComparisonResult {
    passed: bool,
    score: f64,
    assertions_passed: usize,
    assertions_failed: usize,
    screenshots: Vec<String>,
    logs: Vec<String>,
    errors: Vec<TestError>,
}
