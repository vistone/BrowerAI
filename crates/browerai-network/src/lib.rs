pub mod cache;
pub mod deep_crawler;
// pub mod deobfuscation;  // 暂时禁用：需要browerai_deobfuscation crate
pub mod http;

pub use cache::{CacheStats, CacheStrategy, CachedResource, ResourceCache};
pub use deep_crawler::{
    analyze_site_structure, CrawlConfig, CrawlResult, CrawledPage, DeepCrawler,
};
// pub use deobfuscation::{JsDeobfuscationProcessor, ProcessedJs};
pub use http::{HttpClient, HttpMethod, HttpRequest, HttpResponse};
