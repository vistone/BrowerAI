//! Demonstration of deep crawling capabilities
//!
//! Shows how BrowerAI can crawl multiple levels of a website,
//! building a comprehensive understanding of site structure.

use browerai::network::deep_crawler::{analyze_site_structure, CrawlConfig, DeepCrawler};
use std::env;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("========================================");
    println!("BrowerAI Deep Crawling Demonstration");
    println!("========================================\n");

    // Get URL from command line or use default
    let args: Vec<String> = env::args().collect();
    let seed_url = if args.len() > 1 {
        &args[1]
    } else {
        "https://example.com"
    };

    println!("🌐 Seed URL: {}", seed_url);
    println!();

    // Configure crawler
    let config = CrawlConfig {
        max_depth: 2,  // Crawl 2 levels deep
        max_pages: 20, // Limit to 20 pages
        request_timeout: Duration::from_secs(15),
        request_delay: Duration::from_millis(1000), // 1 second between requests
        follow_external: false,                     // Stay on same domain
        respect_robots: true,
        max_concurrent: 3,
    };

    println!("⚙️  Crawler Configuration:");
    println!("   Max Depth: {}", config.max_depth);
    println!("   Max Pages: {}", config.max_pages);
    println!("   Request Delay: {:?}", config.request_delay);
    println!("   Follow External: {}", config.follow_external);
    println!();

    // Create crawler
    let mut crawler = DeepCrawler::new(config);

    println!("🚀 Starting deep crawl...\n");

    // Perform crawl
    match crawler.crawl(seed_url) {
        Ok(result) => {
            println!("========================================");
            println!("Crawl Results");
            println!("========================================\n");

            println!("✅ Crawl completed successfully!");
            println!("   Total time: {:?}", result.total_time);
            println!("   Pages crawled: {}", result.pages_crawled);
            println!("   Pages skipped: {}", result.pages_skipped);
            println!("   Errors: {}", result.errors.len());
            println!();

            // Show pages by depth
            println!("📊 Pages by Depth:");
            let mut depth_groups: std::collections::HashMap<usize, Vec<&str>> =
                std::collections::HashMap::new();

            for page in &result.pages {
                depth_groups
                    .entry(page.depth)
                    .or_insert_with(Vec::new)
                    .push(&page.url);
            }

            for depth in 0..=2 {
                if let Some(urls) = depth_groups.get(&depth) {
                    println!("   Depth {}: {} pages", depth, urls.len());
                    for (i, url) in urls.iter().take(3).enumerate() {
                        println!("      {}. {}", i + 1, url);
                    }
                    if urls.len() > 3 {
                        println!("      ... and {} more", urls.len() - 3);
                    }
                }
            }
            println!();

            // Show site map
            println!("🗺️  Site Map (parent → children):");
            let mut count = 0;
            for (parent, children) in result.site_map.iter().take(5) {
                count += 1;
                println!("   {}", parent);
                for child in children.iter().take(3) {
                    println!("      ↳ {}", child);
                }
                if children.len() > 3 {
                    println!("      ↳ ... {} more", children.len() - 3);
                }
            }
            if result.site_map.len() > 5 {
                println!("   ... and {} more parent pages", result.site_map.len() - 5);
            }
            println!();

            // Analyze structure
            println!("🔍 Site Structure Analysis:");
            let analysis = analyze_site_structure(&result);
            println!("   Total pages discovered: {}", analysis.total_pages);
            println!("   Maximum depth reached: {}", analysis.max_depth_reached);
            println!(
                "   Average links per page: {:.1}",
                analysis.avg_links_per_page
            );
            println!();

            if !analysis.hub_pages.is_empty() {
                println!("   Top hub pages (most outgoing links):");
                for (url, count) in analysis.hub_pages.iter().take(3) {
                    println!("      {} ({} links)", url, count);
                }
                println!();
            }

            // Show errors if any
            if !result.errors.is_empty() {
                println!("⚠️  Errors encountered:");
                for (i, error) in result.errors.iter().take(5).enumerate() {
                    println!("   {}. {}", i + 1, error);
                }
                if result.errors.len() > 5 {
                    println!("   ... and {} more errors", result.errors.len() - 5);
                }
                println!();
            }

            println!("========================================");
            println!("Next Steps with Crawled Data");
            println!("========================================\n");

            println!("✨ BrowerAI can now:");
            println!(
                "   1. Learn from ALL {} pages collectively",
                result.pages.len()
            );
            println!("   2. Understand complete site navigation structure");
            println!("   3. Identify common patterns across multiple pages");
            println!("   4. Generate consistent experiences for entire site");
            println!("   5. Map functionality across the site hierarchy");
            println!();

            println!("🎯 This demonstrates:");
            println!("   ✓ Multi-level depth crawling");
            println!("   ✓ Link discovery and following");
            println!("   ✓ Site structure mapping");
            println!("   ✓ Hierarchical understanding");
            println!("   ✓ Comprehensive data collection");
            println!();
        }
        Err(e) => {
            eprintln!("❌ Crawl failed: {}", e);
            return Err(e.into());
        }
    }

    println!("✅ Deep crawl demonstration complete!");

    Ok(())
}
