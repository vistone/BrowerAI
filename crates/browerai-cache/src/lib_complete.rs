// Week 3 缓存集成实现计划
// 文件位置: crates/browerai-cache/src/lib.rs
// 这是完整的生产级缓存实现框架

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use dashmap::DashMap;
use parking_lot::RwLock;

/// 缓存统计指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// 总命中次数
    pub total_hits: u64,
    /// 总未命中次数
    pub total_misses: u64,
    /// 当前缓存项数量
    pub current_entries: u64,
    /// 平均响应时间（毫秒）
    pub avg_response_time_ms: f64,
    /// p95 响应时间（毫秒）
    pub p95_response_time_ms: f64,
    /// 缓存命中率
    pub hit_rate: f64,
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            total_hits: 0,
            total_misses: 0,
            current_entries: 0,
            avg_response_time_ms: 0.0,
            p95_response_time_ms: 0.0,
            hit_rate: 0.0,
        }
    }
}

/// 缓存项结构
#[derive(Clone, Serialize, Deserialize)]
struct CacheEntry<T> {
    /// 缓存值
    value: T,
    /// 过期时间戳（Unix 秒数）
    expires_at: u64,
    /// 访问计数
    access_count: u64,
    /// 最后访问时间戳
    last_accessed_at: u64,
    /// 创建时间戳
    created_at: u64,
}

impl<T> CacheEntry<T> {
    /// 检查是否已过期
    fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now > self.expires_at
    }
}

/// 缓存统计数据
#[derive(Debug, Clone)]
struct CacheStats {
    /// 命中次数
    hits: u64,
    /// 未命中次数
    misses: u64,
    /// 响应时间样本
    response_times: Vec<f64>,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            response_times: Vec::with_capacity(10000),
        }
    }
}

/// 分布式缓存存储（使用 DashMap 支持并发）
pub struct CacheStore<T: Clone + Serialize + for<'de> Deserialize<'de>> {
    /// 缓存数据（键值对）
    data: Arc<DashMap<String, CacheEntry<T>>>,
    /// 统计信息
    stats: Arc<RwLock<CacheStats>>,
    /// 指标
    metrics: Arc<RwLock<CacheMetrics>>,
}

impl<T: Clone + Serialize + for<'de> Deserialize<'de>> CacheStore<T> {
    /// 创建新的缓存存储
    pub fn new() -> Self {
        Self {
            data: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
        }
    }

    /// 从缓存获取值（异步）
    pub async fn get(&self, key: &str) -> Result<Option<T>> {
        let start = std::time::Instant::now();

        // 尝试获取值
        let value = if let Some(entry) = self.data.get(key) {
            if entry.is_expired() {
                // 已过期，删除并返回 None
                drop(entry);
                self.data.remove(key);
                None
            } else {
                // 更新访问信息
                let mut entry = entry.clone();
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                entry.access_count += 1;
                entry.last_accessed_at = now;

                // 更新存储中的条目
                drop(entry.clone());
                if let Some(mut stored) = self.data.get_mut(key) {
                    stored.access_count += 1;
                    stored.last_accessed_at = now;
                }

                Some(entry.value)
            }
        } else {
            None
        };

        // 更新统计信息
        let elapsed = start.elapsed().as_secs_f64() * 1000.0; // 转换为毫秒
        let mut stats = self.stats.write();
        if value.is_some() {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }
        stats.response_times.push(elapsed);

        // 更新指标
        let total = stats.hits + stats.misses;
        let mut metrics = self.metrics.write();
        metrics.total_hits = stats.hits;
        metrics.total_misses = stats.misses;
        metrics.current_entries = self.data.len() as u64;
        metrics.hit_rate = if total > 0 {
            stats.hits as f64 / total as f64
        } else {
            0.0
        };

        // 计算 p95 响应时间
        if !stats.response_times.is_empty() {
            let mut times = stats.response_times.clone();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = (times.len() as f64 * 0.95) as usize;
            metrics.p95_response_time_ms = times[idx];
            metrics.avg_response_time_ms =
                times.iter().sum::<f64>() / times.len() as f64;
        }

        Ok(value)
    }

    /// 设置缓存值（异步）
    pub async fn set(&self, key: &str, value: T, ttl: Duration) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Failed to get current time")?
            .as_secs();

        let expires_at = now + ttl.as_secs();

        let entry = CacheEntry {
            value,
            expires_at,
            access_count: 0,
            last_accessed_at: now,
            created_at: now,
        };

        self.data.insert(key.to_string(), entry);

        // 更新指标
        let mut metrics = self.metrics.write();
        metrics.current_entries = self.data.len() as u64;

        Ok(())
    }

    /// 删除缓存项
    pub async fn delete(&self, key: &str) -> Result<()> {
        self.data.remove(key);

        // 更新指标
        let mut metrics = self.metrics.write();
        metrics.current_entries = self.data.len() as u64;

        Ok(())
    }

    /// 清空所有缓存
    pub async fn clear(&self) -> Result<()> {
        self.data.clear();

        // 重置指标
        let mut metrics = self.metrics.write();
        metrics.current_entries = 0;

        Ok(())
    }

    /// 获取缓存统计信息
    pub fn get_metrics(&self) -> CacheMetrics {
        self.metrics.read().clone()
    }

    /// 获取缓存大小
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 检查缓存是否为空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 清理过期的缓存项（可定期调用）
    pub async fn cleanup_expired(&self) -> Result<u64> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut removed = 0;
        self.data.retain(|_, entry| {
            if entry.is_expired() {
                removed += 1;
                false
            } else {
                true
            }
        });

        // 更新指标
        let mut metrics = self.metrics.write();
        metrics.current_entries = self.data.len() as u64;

        Ok(removed)
    }
}

impl<T: Clone + Serialize + for<'de> Deserialize<'de>> Default for CacheStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Serialize + for<'de> Deserialize<'de>> Clone for CacheStore<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            stats: Arc::clone(&self.stats),
            metrics: Arc::clone(&self.metrics),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_set_and_get() {
        let cache: CacheStore<String> = CacheStore::new();
        let key = "test_key";
        let value = "test_value".to_string();

        cache
            .set(key, value.clone(), Duration::from_secs(60))
            .await
            .unwrap();

        let retrieved = cache.get(key).await.unwrap();
        assert_eq!(retrieved, Some(value));
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache: CacheStore<String> = CacheStore::new();
        let retrieved = cache.get("nonexistent").await.unwrap();
        assert_eq!(retrieved, None);
    }

    #[tokio::test]
    async fn test_cache_delete() {
        let cache: CacheStore<String> = CacheStore::new();
        let key = "delete_test";

        cache
            .set(key, "value".to_string(), Duration::from_secs(60))
            .await
            .unwrap();

        cache.delete(key).await.unwrap();

        let retrieved = cache.get(key).await.unwrap();
        assert_eq!(retrieved, None);
    }

    #[tokio::test]
    async fn test_cache_metrics() {
        let cache: CacheStore<String> = CacheStore::new();

        cache
            .set("key1", "value1".to_string(), Duration::from_secs(60))
            .await
            .unwrap();

        // 多次命中
        let _ = cache.get("key1").await;
        let _ = cache.get("key1").await;

        // 多次未命中
        let _ = cache.get("nonexistent").await;
        let _ = cache.get("nonexistent").await;

        let metrics = cache.get_metrics();
        assert_eq!(metrics.total_hits, 2);
        assert_eq!(metrics.total_misses, 2);
        assert_eq!(metrics.current_entries, 1);
        assert_eq!(metrics.hit_rate, 0.5);
    }

    #[tokio::test]
    async fn test_cache_ttl_expiration() {
        let cache: CacheStore<String> = CacheStore::new();
        let key = "ttl_test";

        // 设置 1 毫秒 TTL
        cache
            .set(key, "value".to_string(), Duration::from_millis(1))
            .await
            .unwrap();

        // 立即获取应该成功
        let retrieved = cache.get(key).await.unwrap();
        assert_eq!(retrieved, Some("value".to_string()));

        // 等待过期
        tokio::time::sleep(Duration::from_millis(10)).await;

        // 再次获取应该返回 None
        let retrieved = cache.get(key).await.unwrap();
        assert_eq!(retrieved, None);
    }

    #[tokio::test]
    async fn test_cache_concurrent_access() {
        let cache: CacheStore<i32> = CacheStore::new();
        let cache = Arc::new(cache);
        let mut handles = vec![];

        for i in 0..10 {
            let cache = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                cache
                    .set(&format!("key_{}", i), i, Duration::from_secs(60))
                    .await
                    .unwrap();

                for _ in 0..100 {
                    let _ = cache.get(&format!("key_{}", i)).await;
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(cache.len(), 10);
        let metrics = cache.get_metrics();
        assert!(metrics.hit_rate > 0.9); // 应该有很高的命中率
    }

    #[tokio::test]
    async fn test_cache_cleanup_expired() {
        let cache: CacheStore<String> = CacheStore::new();

        // 添加多个过期和有效的项
        cache
            .set("key1", "value1".to_string(), Duration::from_millis(1))
            .await
            .unwrap();
        cache
            .set("key2", "value2".to_string(), Duration::from_secs(60))
            .await
            .unwrap();

        // 等待 key1 过期
        tokio::time::sleep(Duration::from_millis(10)).await;

        let removed = cache.cleanup_expired().await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(cache.len(), 1);
    }
}
