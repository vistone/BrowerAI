use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Represents a cached resource
#[derive(Debug, Clone)]
pub struct CachedResource {
    pub url: String,
    pub data: Vec<u8>,
    pub content_type: String,
    pub cached_at: Instant,
    pub ttl: Duration,
}

impl CachedResource {
    /// Check if the cached resource is still valid
    pub fn is_valid(&self) -> bool {
        self.cached_at.elapsed() < self.ttl
    }

    /// Get the age of the cached resource
    pub fn age(&self) -> Duration {
        self.cached_at.elapsed()
    }
}

/// Cache strategy for resources
#[derive(Debug, Clone, PartialEq)]
pub enum CacheStrategy {
    /// Cache everything
    CacheAll,
    /// Cache only static resources (images, CSS, JS)
    CacheStatic,
    /// Don't cache anything
    NoCache,
}

/// Resource cache with TTL support
pub struct ResourceCache {
    cache: Arc<RwLock<HashMap<String, CachedResource>>>,
    strategy: CacheStrategy,
    default_ttl: Duration,
    max_size: usize,
}

impl ResourceCache {
    /// Create a new resource cache
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            strategy: CacheStrategy::CacheStatic,
            default_ttl: Duration::from_secs(3600), // 1 hour
            max_size: 100 * 1024 * 1024,            // 100 MB
        }
    }

    /// Set cache strategy
    pub fn with_strategy(mut self, strategy: CacheStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set default TTL
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = ttl;
        self
    }

    /// Check if a URL should be cached
    fn should_cache(&self, url: &str) -> bool {
        match self.strategy {
            CacheStrategy::CacheAll => true,
            CacheStrategy::CacheStatic => {
                url.ends_with(".css")
                    || url.ends_with(".js")
                    || url.ends_with(".png")
                    || url.ends_with(".jpg")
                    || url.ends_with(".jpeg")
                    || url.ends_with(".gif")
                    || url.ends_with(".svg")
                    || url.ends_with(".woff")
                    || url.ends_with(".woff2")
            }
            CacheStrategy::NoCache => false,
        }
    }

    /// Get a resource from cache
    pub fn get(&self, url: &str) -> Option<CachedResource> {
        let cache = self.cache.read().ok()?;
        let resource = cache.get(url)?;

        if resource.is_valid() {
            log::debug!("Cache hit for {}", url);
            Some(resource.clone())
        } else {
            log::debug!("Cache expired for {}", url);
            None
        }
    }

    /// Put a resource in cache
    pub fn put(&self, url: String, data: Vec<u8>, content_type: String) -> Result<()> {
        if !self.should_cache(&url) {
            log::debug!("Not caching {} (strategy: {:?})", url, self.strategy);
            return Ok(());
        }

        let resource = CachedResource {
            url: url.clone(),
            data,
            content_type,
            cached_at: Instant::now(),
            ttl: self.default_ttl,
        };

        let mut cache = self
            .cache
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock on cache: {}", e))?;

        // Check cache size and evict if necessary
        let current_size: usize = cache.values().map(|r| r.data.len()).sum();
        if current_size + resource.data.len() > self.max_size {
            self.evict_oldest(&mut cache);
        }

        cache.insert(url, resource);
        log::debug!("Cached resource, total entries: {}", cache.len());

        Ok(())
    }

    /// Evict the oldest entry from cache
    fn evict_oldest(&self, cache: &mut HashMap<String, CachedResource>) {
        if let Some((oldest_url, _)) = cache.iter().min_by_key(|(_, r)| r.cached_at) {
            let oldest_url = oldest_url.clone();
            cache.remove(&oldest_url);
            log::debug!("Evicted oldest entry: {}", oldest_url);
        }
    }

    /// Clear all cached resources
    pub fn clear(&self) -> Result<()> {
        let mut cache = self
            .cache
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock on cache: {}", e))?;
        cache.clear();
        log::info!("Cache cleared");
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().unwrap();
        let total_entries = cache.len();
        let total_size: usize = cache.values().map(|r| r.data.len()).sum();
        let valid_entries = cache.values().filter(|r| r.is_valid()).count();

        CacheStats {
            total_entries,
            valid_entries,
            total_size,
        }
    }
}

impl Default for ResourceCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub valid_entries: usize,
    pub total_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = ResourceCache::new();
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 0);
    }

    #[test]
    fn test_cache_put_and_get() {
        let cache = ResourceCache::new().with_strategy(CacheStrategy::CacheAll);

        cache
            .put(
                "https://example.com/style.css".to_string(),
                b"body { color: red; }".to_vec(),
                "text/css".to_string(),
            )
            .unwrap();

        let resource = cache.get("https://example.com/style.css");
        assert!(resource.is_some());
        assert_eq!(resource.unwrap().content_type, "text/css");
    }

    #[test]
    fn test_cache_strategy_static() {
        let cache = ResourceCache::new().with_strategy(CacheStrategy::CacheStatic);

        // Should cache CSS
        cache
            .put(
                "https://example.com/style.css".to_string(),
                b"test".to_vec(),
                "text/css".to_string(),
            )
            .unwrap();
        assert!(cache.get("https://example.com/style.css").is_some());

        // Should not cache HTML
        cache
            .put(
                "https://example.com/page.html".to_string(),
                b"test".to_vec(),
                "text/html".to_string(),
            )
            .unwrap();
        assert!(cache.get("https://example.com/page.html").is_none());
    }

    #[test]
    fn test_cache_clear() {
        let cache = ResourceCache::new().with_strategy(CacheStrategy::CacheAll);

        cache
            .put(
                "https://example.com/test".to_string(),
                b"data".to_vec(),
                "text/plain".to_string(),
            )
            .unwrap();

        assert_eq!(cache.stats().total_entries, 1);

        cache.clear().unwrap();
        assert_eq!(cache.stats().total_entries, 0);
    }

    #[test]
    fn test_cache_stats() {
        let cache = ResourceCache::new().with_strategy(CacheStrategy::CacheAll);

        cache
            .put("url1".to_string(), vec![0; 100], "text/plain".to_string())
            .unwrap();
        cache
            .put("url2".to_string(), vec![0; 200], "text/plain".to_string())
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.total_size, 300);
    }

    #[test]
    fn test_cached_resource_validity() {
        let resource = CachedResource {
            url: "test".to_string(),
            data: vec![],
            content_type: "text/plain".to_string(),
            cached_at: Instant::now(),
            ttl: Duration::from_secs(3600),
        };

        assert!(resource.is_valid());
    }
}
