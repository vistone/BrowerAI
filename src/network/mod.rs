pub mod cache;
pub mod http;
pub mod deep_crawler;

pub use cache::{CacheStats, CacheStrategy, CachedResource, ResourceCache};
pub use http::{HttpClient, HttpMethod, HttpRequest, HttpResponse};
pub use deep_crawler::{DeepCrawler, CrawlConfig, CrawledPage, CrawlResult, analyze_site_structure};
