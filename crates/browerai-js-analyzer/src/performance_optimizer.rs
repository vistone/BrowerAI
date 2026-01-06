//! Performance Optimization Module
//!
//! Provides caching, incremental analysis, and memory optimization.

use std::collections::HashMap;
use std::sync::Arc;

const DEFAULT_CACHE_SIZE: usize = 100;

/// Analysis result cache using LRU eviction
#[derive(Debug, Clone)]
pub struct AnalysisCache {
    cache: HashMap<String, CacheEntry>,
    access_order: Vec<String>,
    max_size: usize,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Arc<str>,
    input_hash: u64,
    _created_at: std::time::SystemTime,
}

impl AnalysisCache {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_SIZE)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            max_size: capacity,
        }
    }

    pub fn get(&mut self, key: &str, input_hash: u64) -> Option<Arc<str>> {
        if let Some(entry) = self.cache.get(key) {
            if entry.input_hash == input_hash {
                self.access_order.retain(|k| k != key);
                self.access_order.push(key.to_string());
                return Some(entry.data.clone());
            }
        }
        None
    }

    pub fn put(&mut self, key: String, data: Arc<str>, input_hash: u64) {
        if self.cache.contains_key(&key) {
            self.access_order.retain(|k| k != &key);
        }

        if self.cache.len() >= self.max_size && !self.cache.contains_key(&key) {
            if let Some(lru_key) = self.access_order.first().cloned() {
                self.cache.remove(&lru_key);
                self.access_order.remove(0);
            }
        }

        self.cache.insert(
            key.clone(),
            CacheEntry {
                data,
                input_hash,
                _created_at: std::time::SystemTime::now(),
            },
        );
        self.access_order.push(key);
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            capacity: self.max_size,
            hit_rate: if self.access_order.is_empty() {
                0.0
            } else {
                (self.cache.len() as f64 / self.max_size as f64) * 100.0
            },
        }
    }
}

impl Default for AnalysisCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hit_rate: f64,
}

/// Incremental analysis tracker
#[derive(Debug, Clone)]
pub struct IncrementalAnalyzer {
    analyzed_functions: HashMap<String, u64>,
    dependencies: HashMap<String, Vec<String>>,
    dirty_set: Vec<String>,
}

impl IncrementalAnalyzer {
    pub fn new() -> Self {
        Self {
            analyzed_functions: HashMap::new(),
            dependencies: HashMap::new(),
            dirty_set: Vec::new(),
        }
    }

    pub fn mark_analyzed(&mut self, func_name: String, hash: u64) {
        self.analyzed_functions.insert(func_name, hash);
    }

    pub fn add_dependency(&mut self, func_a: String, func_b: String) {
        self.dependencies
            .entry(func_a)
            .or_default()
            .push(func_b);
    }

    pub fn needs_analysis(&self, func_name: &str, current_hash: u64) -> bool {
        match self.analyzed_functions.get(func_name) {
            None => true,
            Some(&stored_hash) => stored_hash != current_hash,
        }
    }

    pub fn get_affected_functions(&self, changed_func: &str) -> Vec<String> {
        let mut affected = vec![changed_func.to_string()];

        for (func, deps) in &self.dependencies {
            if deps.contains(&changed_func.to_string()) {
                affected.push(func.clone());
            }
        }

        affected
    }

    pub fn invalidate_transitive(&mut self, func_name: &str) {
        let affected = self.get_affected_functions(func_name);
        self.dirty_set.extend(affected);
    }

    pub fn get_dirty_functions(&self) -> Vec<String> {
        self.dirty_set.clone()
    }

    pub fn clear_dirty(&mut self) {
        self.dirty_set.clear();
    }

    pub fn stats(&self) -> IncrementalStats {
        IncrementalStats {
            analyzed_count: self.analyzed_functions.len(),
            dependency_count: self.dependencies.len(),
            dirty_count: self.dirty_set.len(),
        }
    }
}

impl Default for IncrementalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct IncrementalStats {
    pub analyzed_count: usize,
    pub dependency_count: usize,
    pub dirty_count: usize,
}

/// Performance metrics tracker
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_time_ms: f64,
    pub analysis_count: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub peak_memory_bytes: usize,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_time_ms: 0.0,
            analysis_count: 0,
            cache_hits: 0,
            cache_misses: 0,
            peak_memory_bytes: 0,
        }
    }

    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    pub fn record_analysis(&mut self, time_ms: f64) {
        self.total_time_ms += time_ms;
        self.analysis_count += 1;
    }

    pub fn avg_time_ms(&self) -> f64 {
        if self.analysis_count == 0 {
            0.0
        } else {
            self.total_time_ms / self.analysis_count as f64
        }
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            (self.cache_hits as f64 / total as f64) * 100.0
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimized analyzer with caching and incremental support
pub struct OptimizedAnalyzer {
    cache: std::sync::Arc<std::sync::Mutex<AnalysisCache>>,
    incremental: std::sync::Arc<std::sync::Mutex<IncrementalAnalyzer>>,
    metrics: std::sync::Arc<std::sync::Mutex<PerformanceMetrics>>,
}

impl OptimizedAnalyzer {
    pub fn new() -> Self {
        Self {
            cache: std::sync::Arc::new(std::sync::Mutex::new(AnalysisCache::new())),
            incremental: std::sync::Arc::new(std::sync::Mutex::new(IncrementalAnalyzer::new())),
            metrics: std::sync::Arc::new(std::sync::Mutex::new(PerformanceMetrics::new())),
        }
    }

    pub fn cache(&self, key: &str, input_hash: u64) -> Option<Arc<str>> {
        self.cache.lock().unwrap().get(key, input_hash)
    }

    pub fn cache_put(&mut self, key: String, data: Arc<str>, input_hash: u64) {
        self.cache.lock().unwrap().put(key, data, input_hash);
    }

    pub fn cache_clear(&mut self) {
        self.cache.lock().unwrap().clear();
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.lock().unwrap().stats()
    }

    pub fn incremental(&mut self) -> std::sync::Arc<std::sync::Mutex<IncrementalAnalyzer>> {
        self.incremental.clone()
    }

    pub fn metrics(&self) -> PerformanceMetrics {
        self.metrics.lock().unwrap().clone()
    }

    pub fn metrics_mut(&mut self) -> std::sync::Arc<std::sync::Mutex<PerformanceMetrics>> {
        self.metrics.clone()
    }

    pub fn record_cache_hit(&mut self) {
        self.metrics.lock().unwrap().record_cache_hit();
    }

    pub fn record_cache_miss(&mut self) {
        self.metrics.lock().unwrap().record_cache_miss();
    }

    pub fn record_analysis(&mut self, time_ms: f64) {
        self.metrics.lock().unwrap().record_analysis(time_ms);
    }

    pub fn reset(&mut self) {
        self.cache.lock().unwrap().clear();
        *self.incremental.lock().unwrap() = IncrementalAnalyzer::new();
        *self.metrics.lock().unwrap() = PerformanceMetrics::new();
    }
}

impl Default for OptimizedAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

pub fn hash_string(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = AnalysisCache::new();
        assert_eq!(cache.cache.len(), 0);
    }

    #[test]
    fn test_cache_put_and_get() {
        let mut cache = AnalysisCache::new();
        let key = "test_key".to_string();
        let data: Arc<str> = Arc::from("test_data");

        cache.put(key.clone(), data.clone(), 123);
        assert_eq!(cache.cache.len(), 1);

        let retrieved = cache.get(&key, 123);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_cache_hash_validation() {
        let mut cache = AnalysisCache::new();
        cache.put("key".to_string(), Arc::<str>::from("data"), 111);
        
        let retrieved = cache.get("key", 222);
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = AnalysisCache::with_capacity(2);

        cache.put("key1".to_string(), Arc::<str>::from("data1"), 1);
        cache.put("key2".to_string(), Arc::<str>::from("data2"), 2);
        cache.put("key3".to_string(), Arc::<str>::from("data3"), 3);

        assert_eq!(cache.cache.len(), 2);
        assert!(cache.cache.get("key1").is_none());
    }

    #[test]
    fn test_incremental_needs_analysis() {
        let mut analyzer = IncrementalAnalyzer::new();
        analyzer.mark_analyzed("func1".to_string(), 123);
        
        assert!(!analyzer.needs_analysis("func1", 123));
        assert!(analyzer.needs_analysis("func1", 456));
    }

    #[test]
    fn test_dependencies() {
        let mut analyzer = IncrementalAnalyzer::new();
        analyzer.add_dependency("a".to_string(), "b".to_string());
        analyzer.add_dependency("c".to_string(), "b".to_string());

        let affected = analyzer.get_affected_functions("b");
        assert!(affected.contains(&"a".to_string()));
        assert!(affected.contains(&"c".to_string()));
    }

    #[test]
    fn test_metrics() {
        let mut metrics = PerformanceMetrics::new();
        metrics.record_analysis(10.0);
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        assert_eq!(metrics.analysis_count, 1);
        assert_eq!(metrics.cache_hits, 1);
        assert!(metrics.cache_hit_rate() < 100.0 && metrics.cache_hit_rate() > 0.0);
    }

    #[test]
    fn test_hash_string() {
        let h1 = hash_string("test");
        let h2 = hash_string("test");
        let h3 = hash_string("other");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
