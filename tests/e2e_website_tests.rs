use anyhow::Result;
use browerai::{CssParser, HtmlParser, JsParser, RenderEngine, WebsiteTester};
use std::time::{Duration, Instant};

/// End-to-end test suite for real websites
/// Tests the complete pipeline: fetch -> parse -> render
#[derive(Debug)]
pub struct E2ETestSuite {
    test_urls: Vec<String>,
    timeout: Duration,
    enable_ai: bool,
}

/// Result of an end-to-end test
#[derive(Debug, Clone)]
pub struct E2ETestResult {
    pub url: String,
    pub success: bool,
    pub total_time_ms: u128,
    pub fetch_time_ms: u128,
    pub parse_time_ms: u128,
    pub render_time_ms: u128,
    pub html_size_bytes: usize,
    pub css_rules_count: usize,
    pub js_scripts_count: usize,
    pub error_message: Option<String>,
}

impl Default for E2ETestSuite {
    fn default() -> Self {
        Self {
            test_urls: Self::default_test_urls(),
            timeout: Duration::from_secs(30),
            enable_ai: false,
        }
    }
}

impl E2ETestSuite {
    /// Create a new E2E test suite with default test URLs
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a test suite with custom URLs
    pub fn with_urls(urls: Vec<String>) -> Self {
        Self {
            test_urls: urls,
            timeout: Duration::from_secs(30),
            enable_ai: false,
        }
    }

    /// Enable AI features for testing
    pub fn with_ai(mut self) -> Self {
        self.enable_ai = true;
        self
    }

    /// Set custom timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Default test URLs covering various website types
    fn default_test_urls() -> Vec<String> {
        vec![
            // Simple static sites
            "http://example.com".to_string(),
            "http://info.cern.ch".to_string(),
            // Medium complexity
            "https://www.wikipedia.org".to_string(),
            // Documentation sites
            "https://docs.rust-lang.org".to_string(),
            // News sites
            "https://news.ycombinator.com".to_string(),
            // Search engines (simple versions)
            "https://duckduckgo.com".to_string(),
            // Social media (basic pages)
            "https://github.com".to_string(),
            // E-commerce
            "https://www.amazon.com".to_string(),
        ]
    }

    /// Run all end-to-end tests
    pub async fn run_all_tests(&self) -> Vec<E2ETestResult> {
        log::info!("Starting E2E test suite with {} URLs", self.test_urls.len());

        let mut results = Vec::new();

        for url in &self.test_urls {
            log::info!("Testing: {}", url);
            let result = self.test_single_url(url).await;
            results.push(result);
        }

        results
    }

    /// Test a single URL through the complete pipeline
    async fn test_single_url(&self, url: &str) -> E2ETestResult {
        let start = Instant::now();

        let mut result = E2ETestResult {
            url: url.to_string(),
            success: false,
            total_time_ms: 0,
            fetch_time_ms: 0,
            parse_time_ms: 0,
            render_time_ms: 0,
            html_size_bytes: 0,
            css_rules_count: 0,
            js_scripts_count: 0,
            error_message: None,
        };

        // Step 1: Fetch HTML
        let fetch_start = Instant::now();
        let html_content = match self.fetch_html(url).await {
            Ok(content) => {
                result.fetch_time_ms = fetch_start.elapsed().as_millis();
                result.html_size_bytes = content.len();
                content
            }
            Err(e) => {
                result.error_message = Some(format!("Fetch failed: {}", e));
                result.total_time_ms = start.elapsed().as_millis();
                return result;
            }
        };

        // Step 2: Parse HTML
        let parse_start = Instant::now();
        let parser = if self.enable_ai {
            // In real implementation, would use with_ai()
            HtmlParser::new()
        } else {
            HtmlParser::new()
        };

        let dom = match parser.parse(&html_content) {
            Ok(d) => {
                result.parse_time_ms = parse_start.elapsed().as_millis();
                d
            }
            Err(e) => {
                result.error_message = Some(format!("HTML parse failed: {}", e));
                result.total_time_ms = start.elapsed().as_millis();
                return result;
            }
        };

        // Step 3: Parse CSS (extract from HTML or fetch external)
        let css_rules = self.extract_css_rules(&html_content);
        result.css_rules_count = css_rules.len();

        // Step 4: Count JS scripts
        result.js_scripts_count = self.count_js_scripts(&html_content);

        // Step 5: Render
        let render_start = Instant::now();
        let mut renderer = RenderEngine::new();
        match renderer.render(&dom, &css_rules) {
            Ok(_render_tree) => {
                result.render_time_ms = render_start.elapsed().as_millis();
                result.success = true;
            }
            Err(e) => {
                result.error_message = Some(format!("Render failed: {}", e));
            }
        }

        result.total_time_ms = start.elapsed().as_millis();
        result
    }

    /// Fetch HTML content from URL
    async fn fetch_html(&self, url: &str) -> Result<String> {
        let client = reqwest::Client::builder()
            .timeout(self.timeout)
            .user_agent("BrowerAI/0.1 (E2E Test Suite)")
            .build()?;

        let response = client.get(url).send().await?;
        let text = response.text().await?;
        Ok(text)
    }

    /// Extract CSS rules from HTML content (simplified)
    fn extract_css_rules(&self, html: &str) -> Vec<browerai::parser::css::CssRule> {
        // Simple extraction: count <style> tags
        // In real implementation, would parse CSS properly
        let style_count = html.matches("<style").count();

        // Return empty rules for now (proper CSS parsing would go here)
        Vec::with_capacity(style_count)
    }

    /// Count JavaScript scripts in HTML
    fn count_js_scripts(&self, html: &str) -> usize {
        html.matches("<script").count()
    }

    /// Generate test report
    pub fn generate_report(&self, results: &[E2ETestResult]) -> String {
        let mut report = String::new();

        report.push_str("========================================\n");
        report.push_str("E2E Test Suite Report\n");
        report.push_str("========================================\n\n");

        let total = results.len();
        let successful = results.iter().filter(|r| r.success).count();
        let failed = total - successful;

        report.push_str(&format!("Total Tests: {}\n", total));
        report.push_str(&format!(
            "Passed: {} ({:.1}%)\n",
            successful,
            (successful as f64 / total as f64) * 100.0
        ));
        report.push_str(&format!(
            "Failed: {} ({:.1}%)\n\n",
            failed,
            (failed as f64 / total as f64) * 100.0
        ));

        // Performance summary
        let avg_total_time: u128 =
            results.iter().map(|r| r.total_time_ms).sum::<u128>() / total as u128;
        let avg_fetch_time: u128 =
            results.iter().map(|r| r.fetch_time_ms).sum::<u128>() / total as u128;
        let avg_parse_time: u128 =
            results.iter().map(|r| r.parse_time_ms).sum::<u128>() / total as u128;
        let avg_render_time: u128 =
            results.iter().map(|r| r.render_time_ms).sum::<u128>() / total as u128;

        report.push_str("Average Timings:\n");
        report.push_str(&format!("  Total:  {}ms\n", avg_total_time));
        report.push_str(&format!("  Fetch:  {}ms\n", avg_fetch_time));
        report.push_str(&format!("  Parse:  {}ms\n", avg_parse_time));
        report.push_str(&format!("  Render: {}ms\n\n", avg_render_time));

        // Individual results
        report.push_str("Individual Results:\n");
        report.push_str("----------------------------------------\n");
        for result in results {
            let status = if result.success {
                "✓ PASS"
            } else {
                "✗ FAIL"
            };
            report.push_str(&format!("{} {}\n", status, result.url));
            report.push_str(&format!(
                "     Time: {}ms (fetch: {}ms, parse: {}ms, render: {}ms)\n",
                result.total_time_ms,
                result.fetch_time_ms,
                result.parse_time_ms,
                result.render_time_ms
            ));
            report.push_str(&format!(
                "     HTML size: {} bytes, CSS rules: {}, JS scripts: {}\n",
                result.html_size_bytes, result.css_rules_count, result.js_scripts_count
            ));
            if let Some(error) = &result.error_message {
                report.push_str(&format!("     Error: {}\n", error));
            }
            report.push_str("\n");
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e2e_suite_creation() {
        let suite = E2ETestSuite::new();
        assert!(!suite.test_urls.is_empty());
        assert_eq!(suite.timeout, Duration::from_secs(30));
        assert!(!suite.enable_ai);
    }

    #[test]
    fn test_custom_urls() {
        let urls = vec!["http://example.com".to_string()];
        let suite = E2ETestSuite::with_urls(urls.clone());
        assert_eq!(suite.test_urls, urls);
    }

    #[test]
    fn test_with_ai() {
        let suite = E2ETestSuite::new().with_ai();
        assert!(suite.enable_ai);
    }

    #[test]
    fn test_with_timeout() {
        let suite = E2ETestSuite::new().with_timeout(Duration::from_secs(60));
        assert_eq!(suite.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_count_js_scripts() {
        let suite = E2ETestSuite::new();
        let html = r#"
            <html>
                <head><script src="a.js"></script></head>
                <body><script>console.log('test');</script></body>
            </html>
        "#;
        assert_eq!(suite.count_js_scripts(html), 2);
    }

    #[test]
    fn test_result_creation() {
        let result = E2ETestResult {
            url: "http://test.com".to_string(),
            success: true,
            total_time_ms: 100,
            fetch_time_ms: 30,
            parse_time_ms: 40,
            render_time_ms: 30,
            html_size_bytes: 1024,
            css_rules_count: 5,
            js_scripts_count: 2,
            error_message: None,
        };
        assert!(result.success);
        assert_eq!(result.total_time_ms, 100);
    }
}
