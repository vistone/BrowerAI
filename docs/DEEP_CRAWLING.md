# Deep Crawling Architecture

## Overview

BrowerAI's deep crawling system enables comprehensive multi-level website analysis, going beyond single-page processing to understand entire site hierarchies.

## Problem Statement

Modern websites are hierarchical with multiple levels:
- **Depth 0**: Entry page (homepage, landing page)
- **Depth 1**: Primary navigation (main sections, categories)
- **Depth 2**: Content pages (articles, products, details)
- **Depth 3+**: Deep content (sub-categories, related items)

Traditional single-page processing cannot:
- Understand navigation structure
- Discover all functionality across the site
- Learn patterns that span multiple pages
- Generate consistent experiences for entire site

## Architecture

### Core Components

```
┌─────────────────────────────────────────┐
│         Deep Crawler                     │
├─────────────────────────────────────────┤
│                                          │
│  ┌──────────┐      ┌──────────────┐    │
│  │  Queue   │ ───▶ │ Page Fetcher │    │
│  │ Manager  │      │              │    │
│  └──────────┘      └──────────────┘    │
│       │                    │            │
│       │                    ▼            │
│       │            ┌──────────────┐    │
│       │            │ Link         │    │
│       │            │ Extractor    │    │
│       │            └──────────────┘    │
│       │                    │            │
│       └────────────────────┘            │
│                                          │
│  ┌─────────────────────────────┐       │
│  │   Visited Set  (HashSet)    │       │
│  └─────────────────────────────┘       │
│                                          │
│  ┌─────────────────────────────┐       │
│  │   Site Map  (Tree)          │       │
│  └─────────────────────────────┘       │
│                                          │
└─────────────────────────────────────────┘
```

### Crawl Configuration

```rust
pub struct CrawlConfig {
    pub max_depth: usize,           // How many levels deep
    pub max_pages: usize,           // Total page limit
    pub request_timeout: Duration,  // Per-request timeout
    pub request_delay: Duration,    // Politeness delay
    pub follow_external: bool,      // Cross-domain links
    pub respect_robots: bool,       // robots.txt compliance
    pub max_concurrent: usize,      // Parallel requests
}
```

## Crawling Algorithm

### Breadth-First Traversal

```
1. Initialize:
   - Queue: [(seed_url, depth=0, parent=None)]
   - Visited: {}
   - Pages: []

2. While queue not empty AND limits not reached:
   a. Dequeue (url, depth, parent)
   b. Skip if visited or depth > max_depth
   c. Mark as visited
   d. Fetch page:
      - Make HTTP request
      - Extract HTML content
      - Parse and extract links
   e. Store page with metadata
   f. Update site map: parent → url
   g. Enqueue discovered links:
      For each link:
        If not visited:
          Queue.push((link, depth+1, url))
   h. Sleep (politeness delay)

3. Return CrawlResult with all pages
```

### Link Extraction & Filtering

**Extraction:**
- Parse `<a href="...">` tags
- Resolve relative URLs to absolute
- Handle base tags
- Support various URL formats

**Filtering:**
- ✅ Same domain (if !follow_external)
- ✅ HTTP/HTTPS only
- ❌ Skip fragments (#section)
- ❌ Skip javascript: links
- ❌ Skip binary files (.pdf, .zip, etc.)
- ❌ Skip already visited

### Politeness & Respect

1. **Request Delay**: Wait between requests (default: 500ms)
2. **User Agent**: Identify as "BrowerAI/0.1.0 DeepCrawler"
3. **robots.txt**: Respect crawl directives (optional)
4. **Timeouts**: Limit per-request time (default: 30s)
5. **Rate Limiting**: Max concurrent requests (default: 5)

## Data Structures

### CrawledPage

```rust
pub struct CrawledPage {
    pub url: String,              // Page URL
    pub depth: usize,             // Distance from seed
    pub html_content: String,     // Full HTML
    pub links: Vec<String>,       // Discovered links
    pub parent_url: Option<String>, // Parent page
    pub crawl_time: Duration,     // Time to fetch
    pub status_code: u16,         // HTTP status
}
```

### Site Map

Hierarchical structure mapping parent → children:

```
example.com
├── /products
│   ├── /products/category1
│   │   ├── /products/category1/item1
│   │   └── /products/category1/item2
│   └── /products/category2
├── /about
└── /contact
```

Stored as: `HashMap<String, Vec<String>>`

## Site Structure Analysis

After crawling, analyze to extract insights:

### Metrics

1. **Depth Distribution**: Pages at each level
2. **Hub Pages**: Pages with many outgoing links
3. **Entry Points**: Pages at depth 0-1
4. **Average Links**: Links per page
5. **Max Depth Reached**: Deepest level crawled

### Example Analysis

```
Total Pages: 47
Max Depth: 3
Depth Distribution:
  - Depth 0: 1 page   (homepage)
  - Depth 1: 8 pages  (main navigation)
  - Depth 2: 25 pages (content)
  - Depth 3: 13 pages (deep content)

Hub Pages:
  - /products: 45 links
  - /categories: 32 links
  - /sitemap: 28 links

Average Links per Page: 12.3
```

## Integration with Learning System

### Multi-Page Learning

After deep crawl, learn from ALL pages:

```rust
let crawl_result = crawler.crawl(seed_url)?;

// Learn from entire site
let mut collective_learning = SiteUnderstanding::new();

for page in crawl_result.pages {
    // Extract patterns from each page
    let patterns = analyze_page(&page.html_content)?;
    collective_learning.add_page_patterns(patterns);
}

// Identify site-wide patterns
let site_structure = collective_learning.identify_common_patterns();
let navigation_patterns = collective_learning.extract_navigation();
let functionality_map = collective_learning.map_functionality();
```

### Hierarchical Understanding

Use depth information for context:

- **Depth 0-1**: Entry pages, global navigation, branding
- **Depth 2**: Main content, primary functionality
- **Depth 3+**: Detail pages, specific content

### Consistent Generation

Generate experiences that maintain:
- Same navigation across all pages
- Consistent branding and style
- Preserved functionality hierarchy
- Coherent site-wide patterns

## Use Cases

### 1. E-commerce Site

```
Seed: https://shop.example.com

Crawl Result:
- Homepage
- Category pages (Electronics, Clothing, etc.)
- Product listing pages
- Individual product pages
- Cart, Checkout (if authenticated)

Learning Outcome:
- Product card patterns across categories
- Navigation structure
- Search functionality
- Add-to-cart interactions

Generated Experience:
- Consistent product display
- Unified navigation
- Optimized checkout flow
```

### 2. Documentation Site

```
Seed: https://docs.example.com

Crawl Result:
- Doc homepage
- Guide sections
- API reference pages
- Tutorial pages
- Search results

Learning Outcome:
- Table of contents structure
- Code block formatting
- Navigation sidebar patterns
- Search integration

Generated Experience:
- Unified documentation layout
- Improved navigation
- Better code highlighting
```

### 3. News Site

```
Seed: https://news.example.com

Crawl Result:
- Homepage with latest articles
- Category pages
- Individual article pages
- Author pages
- Tag pages

Learning Outcome:
- Article card patterns
- Category organization
- Related articles links
- Comment sections

Generated Experience:
- Card-based layouts
- Better article discovery
- Enhanced readability
```

## Performance Considerations

### Time Complexity

- **Per Page**: O(1) for fetch + O(n) for link extraction (n = links in page)
- **Total**: O(P × L) where P = pages, L = avg links per page
- **With Depth Limit**: Bounded by geometric series: P ≤ Σ(L^d) for d=0 to max_depth

### Space Complexity

- **Visited Set**: O(P) where P = unique URLs
- **Queue**: O(P) worst case
- **Pages Storage**: O(P × S) where S = avg page size
- **Site Map**: O(P) for structure

### Optimization Strategies

1. **Incremental Storage**: Stream pages to disk for large crawls
2. **Concurrent Fetching**: Parallel requests (respecting limits)
3. **Early Termination**: Stop when specific content found
4. **Smart Queuing**: Priority queue for important pages
5. **Content Deduplication**: Hash-based similarity detection

## Configuration Examples

### Conservative (Fast, Limited)

```rust
CrawlConfig {
    max_depth: 1,           // Only direct links
    max_pages: 10,          // Small sample
    request_delay: Duration::from_millis(200),
    follow_external: false,
    ..Default::default()
}
```

### Moderate (Balanced)

```rust
CrawlConfig {
    max_depth: 3,           // Three levels deep
    max_pages: 50,          // Reasonable coverage
    request_delay: Duration::from_millis(500),
    follow_external: false,
    ..Default::default()
}
```

### Aggressive (Comprehensive)

```rust
CrawlConfig {
    max_depth: 5,           // Deep crawl
    max_pages: 200,         // Extensive coverage
    request_delay: Duration::from_millis(1000),
    follow_external: true,  // Follow external links
    ..Default::default()
}
```

## Error Handling

### Common Errors

1. **Network Errors**: Timeout, connection refused
2. **HTTP Errors**: 404, 500, etc.
3. **Parse Errors**: Invalid HTML, malformed URLs
4. **Rate Limiting**: 429 Too Many Requests
5. **Authentication**: 401, 403 (pages requiring login)

### Recovery Strategies

- **Retry**: Exponential backoff for transient errors
- **Skip**: Continue with other pages
- **Log**: Record errors for debugging
- **Partial Success**: Return successfully crawled pages

## Testing

### Unit Tests

- Link extraction accuracy
- URL normalization
- Filter logic
- Data structure integrity

### Integration Tests

- Full crawl workflow
- Depth limiting
- Site map construction
- Error handling

### Mock Server Tests

Use local HTTP server to test:
- Multi-level navigation
- Link extraction
- Error scenarios
- Performance limits

## Future Enhancements

1. **JavaScript Execution**: Render SPAs with headless browser
2. **Authentication**: Handle login-protected pages
3. **Form Submission**: POST requests, search queries
4. **Sitemap.xml**: Use sitemaps for efficient discovery
5. **Content Change Detection**: Incremental crawls
6. **Distributed Crawling**: Scale across multiple machines
7. **Machine Learning**: Intelligent page prioritization

## Running the Demo

```bash
# Basic usage
cargo run --example deep_crawl_demo https://example.com

# Different sites
cargo run --example deep_crawl_demo https://news.ycombinator.com
cargo run --example deep_crawl_demo https://docs.rs

# With logging
RUST_LOG=info cargo run --example deep_crawl_demo https://example.com
```

## Conclusion

Deep crawling transforms BrowerAI from a single-page processor into a comprehensive website understanding system. By traversing multiple levels, following links, and building hierarchical site maps, BrowerAI can:

✅ Understand complete site structure
✅ Learn patterns across multiple pages
✅ Generate consistent experiences site-wide
✅ Preserve functionality hierarchies
✅ Handle real-world website complexity

This directly addresses the need for "deep request" handling and multi-level site analysis.
