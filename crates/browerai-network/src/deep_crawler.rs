//! Deep web crawler for multi-level site analysis
//!
//! Handles hierarchical navigation, link following, and comprehensive site learning.

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use url::Url;

/// Configuration for deep crawling
#[derive(Debug, Clone)]
pub struct CrawlConfig {
    /// Maximum depth to crawl (0 = single page, 1 = page + direct links, etc.)
    pub max_depth: usize,
    /// Maximum number of pages to crawl
    pub max_pages: usize,
    /// Timeout for each request
    pub request_timeout: Duration,
    /// Delay between requests (politeness)
    pub request_delay: Duration,
    /// Follow external links
    pub follow_external: bool,
    /// Respect robots.txt
    pub respect_robots: bool,
    /// Maximum concurrent requests
    pub max_concurrent: usize,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_pages: 50,
            request_timeout: Duration::from_secs(30),
            request_delay: Duration::from_millis(500),
            follow_external: false,
            respect_robots: true,
            max_concurrent: 5,
        }
    }
}

/// A crawled page with metadata
#[derive(Debug, Clone)]
pub struct CrawledPage {
    pub url: String,
    pub depth: usize,
    pub html_content: String,
    pub links: Vec<String>,
    pub parent_url: Option<String>,
    pub crawl_time: Duration,
    pub status_code: u16,
}

/// Result of a deep crawl
#[derive(Debug)]
pub struct CrawlResult {
    pub pages: Vec<CrawledPage>,
    pub site_map: HashMap<String, Vec<String>>, // URL -> child URLs
    pub total_time: Duration,
    pub pages_crawled: usize,
    pub pages_skipped: usize,
    pub errors: Vec<String>,
}

/// Deep web crawler
pub struct DeepCrawler {
    config: CrawlConfig,
    visited: HashSet<String>,
    queue: VecDeque<(String, usize, Option<String>)>, // (URL, depth, parent)
    pages: Vec<CrawledPage>,
    site_map: HashMap<String, Vec<String>>,
    errors: Vec<String>,
    start_time: Instant,
}

impl DeepCrawler {
    /// Create a new deep crawler with configuration
    pub fn new(config: CrawlConfig) -> Self {
        Self {
            config,
            visited: HashSet::new(),
            queue: VecDeque::new(),
            pages: Vec::new(),
            site_map: HashMap::new(),
            errors: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Create a crawler with default configuration
    pub fn with_defaults() -> Self {
        Self::new(CrawlConfig::default())
    }

    /// Start crawling from a seed URL
    pub fn crawl(&mut self, seed_url: &str) -> Result<CrawlResult> {
        log::info!("Starting deep crawl from: {}", seed_url);
        self.start_time = Instant::now();

        // Validate and normalize seed URL
        let base_url = Url::parse(seed_url).context("Invalid seed URL")?;

        // Add seed to queue
        self.queue.push_back((seed_url.to_string(), 0, None));

        // Main crawl loop
        while let Some((url, depth, parent)) = self.queue.pop_front() {
            // Check limits
            if self.pages.len() >= self.config.max_pages {
                log::info!("Reached max pages limit: {}", self.config.max_pages);
                break;
            }

            if depth > self.config.max_depth {
                continue;
            }

            // Skip if already visited
            if self.visited.contains(&url) {
                continue;
            }

            // Mark as visited
            self.visited.insert(url.clone());

            // Crawl the page
            match self.crawl_page(&url, depth, parent.clone(), &base_url) {
                Ok(page) => {
                    // Extract and queue links
                    let links = page.links.clone();
                    self.pages.push(page);

                    // Add children to site map
                    if let Some(parent_url) = &parent {
                        self.site_map
                            .entry(parent_url.clone())
                            .or_default()
                            .push(url.clone());
                    }

                    // Queue discovered links
                    for link in links {
                        if !self.visited.contains(&link) {
                            self.queue.push_back((link, depth + 1, Some(url.clone())));
                        }
                    }
                }
                Err(e) => {
                    let error_msg = format!("Failed to crawl {}: {}", url, e);
                    log::warn!("{}", error_msg);
                    self.errors.push(error_msg);
                }
            }

            // Politeness delay
            if !self.queue.is_empty() {
                std::thread::sleep(self.config.request_delay);
            }
        }

        let total_time = self.start_time.elapsed();
        log::info!(
            "Crawl complete: {} pages crawled, {} errors, {:?}",
            self.pages.len(),
            self.errors.len(),
            total_time
        );

        Ok(CrawlResult {
            pages: self.pages.clone(),
            site_map: self.site_map.clone(),
            total_time,
            pages_crawled: self.pages.len(),
            pages_skipped: self.visited.len() - self.pages.len(),
            errors: self.errors.clone(),
        })
    }

    /// Crawl a single page
    fn crawl_page(
        &self,
        url: &str,
        depth: usize,
        parent: Option<String>,
        base_url: &Url,
    ) -> Result<CrawledPage> {
        log::debug!("Crawling [depth={}]: {}", depth, url);
        let page_start = Instant::now();

        // Make HTTP request
        let client = reqwest::blocking::Client::builder()
            .timeout(self.config.request_timeout)
            .user_agent("BrowerAI/0.1.0 DeepCrawler")
            .build()?;

        let response = client.get(url).send().context("HTTP request failed")?;

        let status_code = response.status().as_u16();
        let html_content = response.text().context("Failed to read response body")?;

        // Extract links from HTML
        let links = self.extract_links(&html_content, url, base_url)?;

        let crawl_time = page_start.elapsed();

        Ok(CrawledPage {
            url: url.to_string(),
            depth,
            html_content,
            links,
            parent_url: parent,
            crawl_time,
            status_code,
        })
    }

    /// Extract and normalize links from HTML
    fn extract_links(&self, html: &str, current_url: &str, base_url: &Url) -> Result<Vec<String>> {
        let mut links = Vec::new();
        let current_parsed = Url::parse(current_url)?;

        // Simple regex-based link extraction
        // In production, use a proper HTML parser like scraper or html5ever
        let link_regex = regex::Regex::new(r#"(?i)href=["']([^"']+)["']"#)?;

        for cap in link_regex.captures_iter(html) {
            if let Some(href) = cap.get(1) {
                let href_str = href.as_str();

                // Skip fragments and javascript links
                if href_str.starts_with('#') || href_str.starts_with("javascript:") {
                    continue;
                }

                // Resolve relative URLs
                if let Ok(absolute_url) = current_parsed.join(href_str) {
                    let url_str = absolute_url.to_string();

                    // Check if we should follow this link
                    if self.should_follow(&absolute_url, base_url) {
                        links.push(url_str);
                    }
                }
            }
        }

        // Deduplicate
        links.sort();
        links.dedup();

        Ok(links)
    }

    /// Determine if a URL should be followed
    fn should_follow(&self, url: &Url, base_url: &Url) -> bool {
        // Skip non-HTTP(S) URLs
        if url.scheme() != "http" && url.scheme() != "https" {
            return false;
        }

        // Check external links
        if !self.config.follow_external && url.host() != base_url.host() {
            return false;
        }

        // Skip common non-content URLs
        let path = url.path().to_lowercase();
        let skip_extensions = [".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe"];
        for ext in &skip_extensions {
            if path.ends_with(ext) {
                return false;
            }
        }

        true
    }
}

/// Analyze crawl results to understand site structure
pub fn analyze_site_structure(result: &CrawlResult) -> SiteStructureAnalysis {
    let mut analysis = SiteStructureAnalysis::default();

    // Calculate depth distribution
    for page in &result.pages {
        *analysis.depth_distribution.entry(page.depth).or_insert(0) += 1;
    }

    // Find entry points (pages at depth 0 or 1)
    analysis.entry_points = result
        .pages
        .iter()
        .filter(|p| p.depth <= 1)
        .map(|p| p.url.clone())
        .collect();

    // Calculate average links per page
    let total_links: usize = result.pages.iter().map(|p| p.links.len()).sum();
    analysis.avg_links_per_page = if !result.pages.is_empty() {
        total_links as f64 / result.pages.len() as f64
    } else {
        0.0
    };

    // Identify hub pages (pages with many outgoing links)
    let mut hub_candidates: Vec<_> = result
        .pages
        .iter()
        .filter(|p| p.links.len() > 10)
        .map(|p| (p.url.clone(), p.links.len()))
        .collect();
    hub_candidates.sort_by(|a, b| b.1.cmp(&a.1));
    analysis.hub_pages = hub_candidates.into_iter().take(5).collect();

    // Site map summary
    analysis.total_pages = result.pages.len();
    analysis.max_depth_reached = result.pages.iter().map(|p| p.depth).max().unwrap_or(0);

    analysis
}

/// Analysis of site structure from crawl
#[derive(Debug, Default)]
pub struct SiteStructureAnalysis {
    pub total_pages: usize,
    pub max_depth_reached: usize,
    pub depth_distribution: HashMap<usize, usize>,
    pub entry_points: Vec<String>,
    pub hub_pages: Vec<(String, usize)>, // (URL, link count)
    pub avg_links_per_page: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crawler_creation() {
        let crawler = DeepCrawler::with_defaults();
        assert_eq!(crawler.config.max_depth, 3);
        assert_eq!(crawler.config.max_pages, 50);
    }

    #[test]
    fn test_should_follow_same_domain() {
        let config = CrawlConfig {
            follow_external: false,
            ..Default::default()
        };
        let crawler = DeepCrawler::new(config);

        let base = Url::parse("https://example.com").unwrap();
        let same_domain = Url::parse("https://example.com/page").unwrap();
        let external = Url::parse("https://other.com/page").unwrap();

        assert!(crawler.should_follow(&same_domain, &base));
        assert!(!crawler.should_follow(&external, &base));
    }

    #[test]
    fn test_extract_links() {
        let crawler = DeepCrawler::with_defaults();
        let html = r##"
            <a href="/page1">Page 1</a>
            <a href="https://example.com/page2">Page 2</a>
            <a href="#fragment">Fragment</a>
            <a href="javascript:void(0)">JS Link</a>
        "##;

        let base = Url::parse("https://example.com").unwrap();
        let links = crawler
            .extract_links(html, "https://example.com", &base)
            .unwrap();

        // Should extract only valid links
        assert!(links.len() >= 1);
        assert!(links
            .iter()
            .any(|l| l.contains("/page1") || l.contains("/page2")));
    }
}
