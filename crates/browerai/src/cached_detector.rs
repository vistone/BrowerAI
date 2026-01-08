/// Caching wrapper for FrameworkKnowledgeBase
///
/// Provides caching for analysis results to improve performance when analyzing
/// the same or similar code multiple times.
use anyhow::Result;
use browerai_learning::{DetectionResult, FrameworkKnowledgeBase};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Cached framework detection result
#[derive(Clone)]
struct CachedResult {
    detections: Vec<DetectionResult>,
    timestamp: Instant,
}

/// Configuration for the cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache
    pub max_entries: usize,
    /// Time-to-live for cache entries
    pub ttl: Duration,
    /// Enable cache statistics
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Duration::from_secs(300), // 5 minutes
            enable_stats: true,
        }
    }
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub current_size: usize,
}

impl CacheStats {
    /// Calculate hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

/// Cached FrameworkKnowledgeBase for improved performance
pub struct CachedFrameworkDetector {
    kb: FrameworkKnowledgeBase,
    cache: Arc<Mutex<HashMap<u64, CachedResult>>>,
    config: CacheConfig,
    stats: Arc<Mutex<CacheStats>>,
}

impl CachedFrameworkDetector {
    /// Create a new cached detector with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new cached detector with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            kb: FrameworkKnowledgeBase::new(),
            cache: Arc::new(Mutex::new(HashMap::new())),
            config,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Analyze code with caching
    pub fn analyze_code(&self, code: &str) -> Result<Vec<DetectionResult>> {
        // Calculate cache key (simple hash)
        let key = self.hash_code(code);

        // Check cache
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&key) {
                // Check if entry is still valid
                if cached.timestamp.elapsed() < self.config.ttl {
                    // Cache hit
                    if self.config.enable_stats {
                        let mut stats = self.stats.lock().unwrap();
                        stats.hits += 1;
                    }
                    return Ok(cached.detections.clone());
                }
            }
        }

        // Cache miss - perform analysis
        if self.config.enable_stats {
            let mut stats = self.stats.lock().unwrap();
            stats.misses += 1;
        }

        let detections = self.kb.analyze_code(code)?;

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();

            // Evict old entries if cache is full
            if cache.len() >= self.config.max_entries {
                // Simple eviction: remove oldest entry
                if let Some(oldest_key) = self.find_oldest_entry(&cache) {
                    cache.remove(&oldest_key);
                    if self.config.enable_stats {
                        let mut stats = self.stats.lock().unwrap();
                        stats.evictions += 1;
                    }
                }
            }

            cache.insert(
                key,
                CachedResult {
                    detections: detections.clone(),
                    timestamp: Instant::now(),
                },
            );

            if self.config.enable_stats {
                let mut stats = self.stats.lock().unwrap();
                stats.current_size = cache.len();
            }
        }

        Ok(detections)
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();

        if self.config.enable_stats {
            let mut stats = self.stats.lock().unwrap();
            stats.current_size = 0;
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = CacheStats::default();
    }

    /// Simple hash function for code
    fn hash_code(&self, code: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        code.hash(&mut hasher);
        hasher.finish()
    }

    /// Find the oldest entry in the cache
    fn find_oldest_entry(&self, cache: &HashMap<u64, CachedResult>) -> Option<u64> {
        cache
            .iter()
            .min_by_key(|(_, v)| v.timestamp)
            .map(|(k, _)| *k)
    }

    /// Get the underlying knowledge base (for advanced usage)
    pub fn knowledge_base(&self) -> &FrameworkKnowledgeBase {
        &self.kb
    }
}

impl Default for CachedFrameworkDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() -> Result<()> {
        let detector = CachedFrameworkDetector::new();

        let code = r#"
            const { createApp, ref } = Vue;
            const app = createApp({});
        "#;

        // First call - cache miss
        let _ = detector.analyze_code(code)?;
        let stats1 = detector.stats();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.hits, 0);

        // Second call - cache hit
        let _ = detector.analyze_code(code)?;
        let stats2 = detector.stats();
        assert_eq!(stats2.misses, 1);
        assert_eq!(stats2.hits, 1);

        println!("✅ Cache hit rate: {:.1}%", stats2.hit_rate());

        Ok(())
    }

    #[test]
    fn test_cache_eviction() -> Result<()> {
        let config = CacheConfig {
            max_entries: 2,
            ttl: Duration::from_secs(300),
            enable_stats: true,
        };

        let detector = CachedFrameworkDetector::with_config(config);

        // Add 3 entries (should evict 1)
        detector.analyze_code("code1")?;
        detector.analyze_code("code2")?;
        detector.analyze_code("code3")?;

        let stats = detector.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.current_size, 2);

        println!("✅ Cache eviction working: {} evictions", stats.evictions);

        Ok(())
    }

    #[test]
    fn test_cache_ttl() -> Result<()> {
        let config = CacheConfig {
            max_entries: 100,
            ttl: Duration::from_millis(100), // Very short TTL
            enable_stats: true,
        };

        let detector = CachedFrameworkDetector::with_config(config);

        let code = "test code";

        // First call
        detector.analyze_code(code)?;

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(150));

        // Second call - should be cache miss due to expired TTL
        detector.analyze_code(code)?;

        let stats = detector.stats();
        assert_eq!(stats.misses, 2); // Both should be misses
        assert_eq!(stats.hits, 0);

        println!("✅ TTL expiration working");

        Ok(())
    }

    #[test]
    fn test_clear_cache() -> Result<()> {
        let detector = CachedFrameworkDetector::new();

        // Add some entries
        detector.analyze_code("code1")?;
        detector.analyze_code("code2")?;

        let stats1 = detector.stats();
        assert_eq!(stats1.current_size, 2);

        // Clear cache
        detector.clear_cache();

        let stats2 = detector.stats();
        assert_eq!(stats2.current_size, 0);

        // Next call should be cache miss
        detector.analyze_code("code1")?;
        let stats3 = detector.stats();
        assert_eq!(stats3.misses, 3); // Original 2 + 1 new

        println!("✅ Cache clear working");

        Ok(())
    }
}
