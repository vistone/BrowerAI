/// End-to-end website testing demo
///
/// This example demonstrates how to use the testing infrastructure to validate
/// BrowerAI's parsing and rendering capabilities.
use browerai::testing::{WebsiteTestSuite, WebsiteTester};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("========================================");
    println!("BrowerAI E2E Testing Demo");
    println!("========================================\n");

    // Create tester and suite
    let tester = WebsiteTester::new();
    let mut suite = WebsiteTestSuite::new();

    // Test 1: Simple HTML
    println!("Test 1: Simple HTML");
    let html1 = r#"<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body><h1>Hello World</h1></body>
</html>"#;

    let result1 = tester.test_website("http://example.com", html1, "", "");
    println!(
        "  HTML parse: {}",
        if result1.html_parse_success {
            "✓"
        } else {
            "✗"
        }
    );
    println!("  Elements: {}", result1.elements_parsed);
    println!("  Parse time: {}ms\n", result1.html_parse_time.as_millis());
    suite.add_result(result1);

    // Test 2: HTML with CSS
    println!("Test 2: HTML with CSS");
    let html2 = r#"<!DOCTYPE html>
<html>
<head><title>Styled</title></head>
<body><div class="test">Content</div></body>
</html>"#;
    let css2 = ".test { color: blue; font-size: 16px; }";

    let result2 = tester.test_website("http://styled.example", html2, css2, "");
    println!(
        "  HTML parse: {}",
        if result2.html_parse_success {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  CSS parse: {}",
        if result2.css_parse_success {
            "✓"
        } else {
            "✗"
        }
    );
    println!("  CSS rules: {}", result2.css_rules_parsed);
    println!("  Parse time: {}ms\n", result2.css_parse_time.as_millis());
    suite.add_result(result2);

    // Test 3: Complete website
    println!("Test 3: Complete website with JS");
    let html3 = r#"<!DOCTYPE html>
<html>
<body><button id="btn">Click</button></body>
</html>"#;
    let css3 = "button { padding: 10px; }";
    let js3 = "document.getElementById('btn').addEventListener('click', () => alert('Hi'));";

    let result3 = tester.test_website("http://complete.example", html3, css3, js3);
    println!(
        "  HTML parse: {}",
        if result3.html_parse_success {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  CSS parse: {}",
        if result3.css_parse_success {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "  JS parse: {}",
        if result3.js_parse_success {
            "✓"
        } else {
            "✗"
        }
    );
    println!("  JS statements: {}\n", result3.js_statements_parsed);
    suite.add_result(result3);

    // Summary
    println!("========================================");
    println!("Summary");
    println!("========================================");
    println!("Total tests: {}", suite.results().len());
    println!("Success rate: {:.1}%", suite.overall_success_rate() * 100.0);
    println!("\nE2E Testing Complete!");

    Ok(())
}
