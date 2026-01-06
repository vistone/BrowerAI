/// Website testing framework for real-world validation
///
/// This module provides utilities for testing BrowerAI against real-world websites,
/// measuring accuracy, performance, and compatibility.
pub mod benchmark;

use browerai_css_parser::CssParser;
use browerai_html_parser::HtmlParser;
use browerai_js_parser::JsParser;
use std::time::{Duration, Instant};

pub use benchmark::{BenchmarkConfig, BenchmarkResult, BenchmarkRunner, ComparisonResult};

/// Test result for a single website test
#[derive(Debug, Clone)]
pub struct WebsiteTestResult {
    /// URL that was tested
    pub url: String,
    /// Whether HTML parsing succeeded
    pub html_parse_success: bool,
    /// Whether CSS parsing succeeded
    pub css_parse_success: bool,
    /// Whether JS parsing succeeded  
    pub js_parse_success: bool,
    /// Time taken to parse HTML
    pub html_parse_time: Duration,
    /// Time taken to parse CSS
    pub css_parse_time: Duration,
    /// Time taken to parse JS
    pub js_parse_time: Duration,
    /// Total elements parsed
    pub elements_parsed: usize,
    /// Total CSS rules parsed
    pub css_rules_parsed: usize,
    /// Total JS statements parsed
    pub js_statements_parsed: usize,
    /// Any errors encountered
    pub errors: Vec<String>,
}

impl WebsiteTestResult {
    /// Create a new test result
    pub fn new(url: String) -> Self {
        Self {
            url,
            html_parse_success: false,
            css_parse_success: false,
            js_parse_success: false,
            html_parse_time: Duration::ZERO,
            css_parse_time: Duration::ZERO,
            js_parse_time: Duration::ZERO,
            elements_parsed: 0,
            css_rules_parsed: 0,
            js_statements_parsed: 0,
            errors: Vec::new(),
        }
    }

    /// Check if all parsing succeeded
    pub fn all_success(&self) -> bool {
        self.html_parse_success && self.css_parse_success && self.js_parse_success
    }

    /// Get total parse time
    pub fn total_time(&self) -> Duration {
        self.html_parse_time + self.css_parse_time + self.js_parse_time
    }

    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        let mut count = 0;
        let mut success = 0;

        count += 1;
        if self.html_parse_success {
            success += 1;
        }
        count += 1;
        if self.css_parse_success {
            success += 1;
        }
        count += 1;
        if self.js_parse_success {
            success += 1;
        }

        success as f64 / count as f64
    }
}

/// Website tester for real-world validation
pub struct WebsiteTester {
    html_parser: HtmlParser,
    css_parser: CssParser,
    js_parser: JsParser,
}

impl WebsiteTester {
    /// Create a new website tester
    pub fn new() -> Self {
        Self {
            html_parser: HtmlParser::new(),
            css_parser: CssParser::new(),
            js_parser: JsParser::new(),
        }
    }

    /// Test HTML content
    pub fn test_html(&self, html: &str) -> (bool, Duration, usize) {
        let start = Instant::now();
        match self.html_parser.parse(html) {
            Ok(dom) => {
                let duration = start.elapsed();
                // Count elements in DOM
                let elements = Self::count_elements(&dom);
                (true, duration, elements)
            }
            Err(_) => (false, start.elapsed(), 0),
        }
    }

    /// Test CSS content
    pub fn test_css(&self, css: &str) -> (bool, Duration, usize) {
        let start = Instant::now();
        match self.css_parser.parse(css) {
            Ok(rules) => {
                let duration = start.elapsed();
                (true, duration, rules.len())
            }
            Err(_) => (false, start.elapsed(), 0),
        }
    }

    /// Test JavaScript content
    pub fn test_js(&self, js: &str) -> (bool, Duration, usize) {
        let start = Instant::now();
        match self.js_parser.parse(js) {
            Ok(ast) => {
                let duration = start.elapsed();
                (true, duration, ast.statement_count)
            }
            Err(_) => (false, start.elapsed(), 0),
        }
    }

    /// Test a complete website with HTML, CSS, and JS
    pub fn test_website(&self, url: &str, html: &str, css: &str, js: &str) -> WebsiteTestResult {
        let mut result = WebsiteTestResult::new(url.to_string());

        // Test HTML
        let (success, time, count) = self.test_html(html);
        result.html_parse_success = success;
        result.html_parse_time = time;
        result.elements_parsed = count;
        if !success {
            result.errors.push("HTML parsing failed".to_string());
        }

        // Test CSS
        let (success, time, count) = self.test_css(css);
        result.css_parse_success = success;
        result.css_parse_time = time;
        result.css_rules_parsed = count;
        if !success {
            result.errors.push("CSS parsing failed".to_string());
        }

        // Test JS
        let (success, time, count) = self.test_js(js);
        result.js_parse_success = success;
        result.js_parse_time = time;
        result.js_statements_parsed = count;
        if !success {
            result.errors.push("JS parsing failed".to_string());
        }

        result
    }

    /// Count elements in DOM
    fn count_elements(dom: &markup5ever_rcdom::RcDom) -> usize {
        use markup5ever_rcdom::NodeData;
        let mut count = 0;

        fn walk_tree(node: &markup5ever_rcdom::Handle, count: &mut usize) {
            if let NodeData::Element { .. } = node.data {
                *count += 1;
            }
            for child in node.children.borrow().iter() {
                walk_tree(child, count);
            }
        }

        walk_tree(&dom.document, &mut count);
        count
    }
}

impl Default for WebsiteTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Test suite aggregator for multiple website tests
pub struct WebsiteTestSuite {
    results: Vec<WebsiteTestResult>,
}

impl WebsiteTestSuite {
    /// Create a new test suite
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add a test result
    pub fn add_result(&mut self, result: WebsiteTestResult) {
        self.results.push(result);
    }

    /// Get all results
    pub fn results(&self) -> &[WebsiteTestResult] {
        &self.results
    }

    /// Get overall success rate
    pub fn overall_success_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let total_rate: f64 = self.results.iter().map(|r| r.success_rate()).sum();
        total_rate / self.results.len() as f64
    }

    /// Get average parse time
    pub fn average_parse_time(&self) -> Duration {
        if self.results.is_empty() {
            return Duration::ZERO;
        }

        let total: Duration = self.results.iter().map(|r| r.total_time()).sum();
        total / self.results.len() as u32
    }

    /// Get total elements parsed
    pub fn total_elements_parsed(&self) -> usize {
        self.results.iter().map(|r| r.elements_parsed).sum()
    }

    /// Get total CSS rules parsed
    pub fn total_css_rules_parsed(&self) -> usize {
        self.results.iter().map(|r| r.css_rules_parsed).sum()
    }

    /// Get total JS statements parsed
    pub fn total_js_statements_parsed(&self) -> usize {
        self.results.iter().map(|r| r.js_statements_parsed).sum()
    }

    /// Generate a summary report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Website Test Suite Report ===\n\n");

        report.push_str(&format!("Total websites tested: {}\n", self.results.len()));
        report.push_str(&format!(
            "Overall success rate: {:.2}%\n",
            self.overall_success_rate() * 100.0
        ));
        report.push_str(&format!(
            "Average parse time: {:.2}ms\n",
            self.average_parse_time().as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "Total elements parsed: {}\n",
            self.total_elements_parsed()
        ));
        report.push_str(&format!(
            "Total CSS rules parsed: {}\n",
            self.total_css_rules_parsed()
        ));
        report.push_str(&format!(
            "Total JS statements parsed: {}\n\n",
            self.total_js_statements_parsed()
        ));

        report.push_str("Individual Results:\n");
        for result in &self.results {
            report.push_str(&format!("\nURL: {}\n", result.url));
            report.push_str(&format!(
                "  Success Rate: {:.2}%\n",
                result.success_rate() * 100.0
            ));
            report.push_str(&format!(
                "  Total Time: {:.2}ms\n",
                result.total_time().as_secs_f64() * 1000.0
            ));
            report.push_str(&format!("  Elements: {}\n", result.elements_parsed));
            report.push_str(&format!("  CSS Rules: {}\n", result.css_rules_parsed));
            report.push_str(&format!(
                "  JS Statements: {}\n",
                result.js_statements_parsed
            ));
            if !result.errors.is_empty() {
                report.push_str(&format!("  Errors: {:?}\n", result.errors));
            }
        }

        report
    }
}

impl Default for WebsiteTestSuite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_website_tester_creation() {
        let tester = WebsiteTester::new();
        assert!(std::ptr::addr_of!(tester.html_parser) as usize > 0);
    }

    #[test]
    fn test_html_parsing() {
        let tester = WebsiteTester::new();
        let html = "<html><body><h1>Test</h1><p>Content</p></body></html>";
        let (success, time, count) = tester.test_html(html);

        assert!(success);
        assert!(time.as_nanos() > 0);
        assert!(count > 0);
    }

    #[test]
    fn test_css_parsing() {
        let tester = WebsiteTester::new();
        let css = "body { color: red; } h1 { font-size: 20px; }";
        let (success, time, count) = tester.test_css(css);

        assert!(success);
        assert!(time.as_nanos() > 0);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_js_parsing() {
        let tester = WebsiteTester::new();
        let js = "function test() { return 42; }";
        let (success, time, count) = tester.test_js(js);

        assert!(success);
        assert!(time.as_nanos() > 0);
        assert!(count > 0);
    }

    #[test]
    fn test_complete_website() {
        let tester = WebsiteTester::new();
        let html = "<html><body><h1>Test</h1></body></html>";
        let css = "body { color: red; }";
        let js = "console.log('test');";

        let result = tester.test_website("http://example.com", html, css, js);

        assert!(result.all_success());
        assert_eq!(result.success_rate(), 1.0);
        assert!(result.total_time().as_nanos() > 0);
        assert!(result.elements_parsed > 0);
    }

    #[test]
    fn test_test_suite() {
        let mut suite = WebsiteTestSuite::new();

        let mut result1 = WebsiteTestResult::new("http://example.com".to_string());
        result1.html_parse_success = true;
        result1.css_parse_success = true;
        result1.js_parse_success = true;
        result1.elements_parsed = 10;

        let mut result2 = WebsiteTestResult::new("http://test.com".to_string());
        result2.html_parse_success = true;
        result2.css_parse_success = false;
        result2.js_parse_success = true;
        result2.elements_parsed = 5;

        suite.add_result(result1);
        suite.add_result(result2);

        assert_eq!(suite.results().len(), 2);
        assert!(suite.overall_success_rate() > 0.5);
        assert_eq!(suite.total_elements_parsed(), 15);
    }

    #[test]
    fn test_generate_report() {
        let mut suite = WebsiteTestSuite::new();

        let mut result = WebsiteTestResult::new("http://example.com".to_string());
        result.html_parse_success = true;
        result.css_parse_success = true;
        result.js_parse_success = true;
        result.elements_parsed = 10;
        result.css_rules_parsed = 5;
        result.js_statements_parsed = 20;

        suite.add_result(result);

        let report = suite.generate_report();
        assert!(report.contains("Website Test Suite Report"));
        assert!(report.contains("example.com"));
        assert!(report.contains("Success Rate:"));
    }

    #[test]
    fn test_result_all_success() {
        let mut result = WebsiteTestResult::new("http://test.com".to_string());
        assert!(!result.all_success());

        result.html_parse_success = true;
        result.css_parse_success = true;
        result.js_parse_success = true;
        assert!(result.all_success());
    }

    #[test]
    fn test_empty_suite() {
        let suite = WebsiteTestSuite::new();
        assert_eq!(suite.overall_success_rate(), 0.0);
        assert_eq!(suite.average_parse_time(), Duration::ZERO);
        assert_eq!(suite.total_elements_parsed(), 0);
    }
}
