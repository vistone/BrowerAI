// Week 3 缓存实现 - lib.rs 完整代码
// 文件位置: crates/browerai-cache/src/lib.rs
// 这是生产级别的完整缓存实现

use anyhow::{Context, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// 缓存统计指标 (Prometheus 导出)
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
    /// 缓存命中率 (0.0-1.0)
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

/// 缓存项结构 (内部使用)
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

/// 缓存统计数据 (内部使用)
#[derive(Debug, Clone, Default)]
struct CacheStats {
    /// 命中次数
    hits: u64,
    /// 未命中次数
    misses: u64,
    /// 响应时间样本 (毫秒)
    response_times: Vec<f64>,
}

/// 分布式缓存存储
///
/// 使用 DashMap 实现无锁并发访问，支持泛型值的存储。
/// 自动管理 TTL 过期，并收集性能指标。
pub struct CacheStore<T: Clone + Serialize + for<'de> Deserialize<'de>> {
    /// 缓存数据 (无锁并发)
    data: Arc<DashMap<String, CacheEntry<T>>>,
    /// 统计信息
    stats: Arc<RwLock<CacheStats>>,
    /// 性能指标
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

    /// 从缓存获取值 (异步)
    ///
    /// 自动检查 TTL 过期，更新访问统计，计算响应时间。
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
                // 克隆值用于返回
                let cloned_value = entry.value.clone();
                drop(entry);

                // 更新访问信息
                if let Some(mut entry) = self.data.get_mut(key) {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    entry.access_count += 1;
                    entry.last_accessed_at = now;
                }

                Some(cloned_value)
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

        // 保持最近 10000 个样本以避免内存溢出
        if stats.response_times.len() > 10000 {
            stats.response_times.remove(0);
        }

        // 更新指标
        self.update_metrics(&stats);

        Ok(value)
    }

    /// 设置缓存值 (异步)
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
        let _now = SystemTime::now()
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

    /// 导出 Prometheus 格式的指标
    pub fn export_prometheus_metrics(&self) -> String {
        let metrics = self.metrics.read().clone();

        format!(
            "# HELP cache_hits Total number of cache hits\n\
             # TYPE cache_hits counter\n\
             cache_hits {}\n\
             # HELP cache_misses Total number of cache misses\n\
             # TYPE cache_misses counter\n\
             cache_misses {}\n\
             # HELP cache_hit_rate Cache hit rate (0-1)\n\
             # TYPE cache_hit_rate gauge\n\
             cache_hit_rate {}\n\
             # HELP cache_current_entries Current number of entries\n\
             # TYPE cache_current_entries gauge\n\
             cache_current_entries {}\n\
             # HELP cache_avg_response_time_ms Average response time in milliseconds\n\
             # TYPE cache_avg_response_time_ms gauge\n\
             cache_avg_response_time_ms {}\n\
             # HELP cache_p95_response_time_ms P95 response time in milliseconds\n\
             # TYPE cache_p95_response_time_ms gauge\n\
             cache_p95_response_time_ms {}\n",
            metrics.total_hits,
            metrics.total_misses,
            metrics.hit_rate,
            metrics.current_entries,
            metrics.avg_response_time_ms,
            metrics.p95_response_time_ms
        )
    }

    /// 内部方法: 更新指标 (基于统计数据)
    fn update_metrics(&self, stats: &CacheStats) {
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

        // 计算平均和 P95 响应时间
        if !stats.response_times.is_empty() {
            metrics.avg_response_time_ms =
                stats.response_times.iter().sum::<f64>() / stats.response_times.len() as f64;

            let mut sorted = stats.response_times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((sorted.len() as f64) * 0.95) as usize;
            metrics.p95_response_time_ms = sorted[idx];
        }
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
        assert!(metrics.hit_rate > 0.49 && metrics.hit_rate < 0.51); // 约等于 0.5
    }

    #[tokio::test]
    async fn test_cache_ttl_expiration() {
        let cache: CacheStore<String> = CacheStore::new();
        let key = "ttl_test";

        cache
            .set(key, "value".to_string(), Duration::from_millis(300))
            .await
            .unwrap();

        // 立即获取应该成功
        let retrieved = cache.get(key).await.unwrap();
        assert_eq!(retrieved, Some("value".to_string()));

        // 等待过期 (显式等待更长时间确保 SystemTime 精度)
        tokio::time::sleep(Duration::from_secs(1)).await;

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

        cache
            .set("key1", "value1".to_string(), Duration::from_millis(300))
            .await
            .unwrap();
        cache
            .set("key2", "value2".to_string(), Duration::from_secs(60))
            .await
            .unwrap();

        // 等待 key1 过期 (使用更长的等待时间确保 SystemTime 精度)
        tokio::time::sleep(Duration::from_secs(1)).await;

        let removed = cache.cleanup_expired().await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(cache.len(), 1);
    }

    #[tokio::test]
    async fn test_prometheus_metrics_export() {
        let cache: CacheStore<String> = CacheStore::new();

        cache
            .set("key1", "value1".to_string(), Duration::from_secs(60))
            .await
            .unwrap();

        let _ = cache.get("key1").await;
        let _ = cache.get("nonexistent").await;

        let prometheus_output = cache.export_prometheus_metrics();

        // 验证输出包含关键指标
        assert!(prometheus_output.contains("cache_hits 1"));
        assert!(prometheus_output.contains("cache_misses 1"));
        assert!(prometheus_output.contains("cache_current_entries 1"));
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache: CacheStore<String> = CacheStore::new();

        for i in 0..10 {
            cache
                .set(
                    &format!("key_{}", i),
                    "value".to_string(),
                    Duration::from_secs(60),
                )
                .await
                .unwrap();
        }

        assert_eq!(cache.len(), 10);

        cache.clear().await.unwrap();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
