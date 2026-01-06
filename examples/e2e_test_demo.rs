/// End-to-end website testing demo
///
/// This example demonstrates how to use the E2E test suite to test
/// BrowerAI against real websites.
use browerai::testing::WebsiteTestSuite;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("========================================");
    println!("BrowerAI E2E Website Test Suite");
    println!("========================================\n");

    // Simple test with a few URLs
    let test_urls = vec![
        "http://example.com".to_string(),
        "http://info.cern.ch".to_string(),
    ];

    println!("Testing {} websites...\n", test_urls.len());

    // Create test suite
    let suite = WebsiteTestSuite::new();

    // Test each URL
    for url in &test_urls {
        println!("Testing: {}", url);

        match suite.test_website(url).await {
            Ok(result) => {
                println!("  ✓ Test passed");
                println!("    HTML parsed: {} bytes", result.html_parsed);
                println!("    CSS rules: {}", result.css_rules);
                println!("    Render time: {}ms", result.render_time_ms);
            }
            Err(e) => {
                println!("  ✗ Test failed: {}", e);
            }
        }
        println!();
    }

    println!("========================================");
    println!("E2E Testing Complete");
    println!("========================================");

    Ok(())
}
