//! 多层缓存系统 - Week 6
//!
//! L1 (内存) → L2 (本地) → L3 (分布式/Redis)

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::trace;

pub mod l1_memory;
pub mod l2_local;
pub mod l3_distributed;
pub mod metrics;
pub mod strategy;

pub use l1_memory::MemoryLayer;
pub use l2_local::LocalLayer;
pub use l3_distributed::NoopDistributedLayer;
pub use metrics::{CacheMetrics, SimpleMetrics};
pub use strategy::{CacheLayer, PromotePolicy, WriteStrategy};

/// 多层缓存调度器。
pub struct MultiLayerCache<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    layers: Vec<Arc<dyn CacheLayer<V>>>,
    write_strategy: WriteStrategy,
    promote_policy: PromotePolicy,
    default_ttl: Duration,
    metrics: SimpleMetrics,
}

impl<V> MultiLayerCache<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    /// 使用默认写穿策略与回填策略创建实例。
    pub fn new(layers: Vec<Arc<dyn CacheLayer<V>>>, default_ttl: Duration) -> Result<Self> {
        Self::with_options(
            layers,
            default_ttl,
            WriteStrategy::WriteThrough,
            PromotePolicy::PromoteOnHit,
        )
    }

    /// 指定写入与回填策略创建实例。
    pub fn with_options(
        layers: Vec<Arc<dyn CacheLayer<V>>>,
        default_ttl: Duration,
        write_strategy: WriteStrategy,
        promote_policy: PromotePolicy,
    ) -> Result<Self> {
        if layers.is_empty() {
            return Err(anyhow!("MultiLayerCache 需要至少一个缓存层"));
        }

        Ok(Self {
            layers,
            write_strategy,
            promote_policy,
            default_ttl,
            metrics: SimpleMetrics::new(),
        })
    }

    /// 仅使用 L1 内存层的便捷构造函数。
    pub fn memory_only(default_ttl: Duration) -> Self {
        let l1: Arc<dyn CacheLayer<V>> = Arc::new(MemoryLayer::new("l1-memory"));
        Self::new(vec![l1], default_ttl).expect("memory_only 至少包含一层")
    }

    /// 获取缓存值，若命中低层可选择回填高层。
    pub async fn get(&self, key: &str) -> Result<Option<V>> {
        let mut hit_value: Option<V> = None;
        let mut hit_layer_index: Option<usize> = None;

        for (idx, layer) in self.layers.iter().enumerate() {
            match layer.get(key).await? {
                Some(value) => {
                    self.metrics.record_hit();
                    hit_value = Some(value);
                    hit_layer_index = Some(idx);
                    break;
                }
                None => {
                    trace!(layer = layer.name(), key, "cache miss");
                }
            }
        }

        if hit_value.is_none() {
            self.metrics.record_miss();
        }

        if let (Some(value), Some(hit_idx)) = (hit_value.clone(), hit_layer_index) {
            if self.promote_policy == PromotePolicy::PromoteOnHit && hit_idx > 0 {
                for upper in 0..hit_idx {
                    if let Err(err) = self.layers[upper]
                        .set(key, value.clone(), self.default_ttl)
                        .await
                    {
                        trace!(layer = self.layers[upper].name(), %err, "promote to upper layer failed");
                    }
                }
            }
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    /// 批量获取，命中低层时回填高层。
    pub async fn get_batch(&self, keys: &[String]) -> Result<HashMap<String, V>> {
        let mut remaining: HashSet<String> = keys.iter().cloned().collect();
        let mut hits: HashMap<String, V> = HashMap::new();

        for (idx, layer) in self.layers.iter().enumerate() {
            if remaining.is_empty() {
                break;
            }

            let query_keys: Vec<String> = remaining.iter().cloned().collect();
            let batch = layer.get_batch(&query_keys).await?;

            if batch.is_empty() {
                continue;
            }

            // 填充命中结果并准备回填
            for (k, v) in batch.iter() {
                remaining.remove(k);
                hits.insert(k.clone(), v.clone());
            }

            // 命中低层，批量向上回填（优化：使用 set_batch）
            if self.promote_policy == PromotePolicy::PromoteOnHit && idx > 0 {
                for upper in 0..idx {
                    if let Err(err) = self.layers[upper]
                        .set_batch(batch.clone(), self.default_ttl)
                        .await
                    {
                        trace!(layer = self.layers[upper].name(), count = batch.len(), %err, "promote batch failed");
                    }
                }
            }
        }

        Ok(hits)
    }

    /// 写入单个键，使用默认 TTL。
    pub async fn set(&self, key: &str, value: V) -> Result<()> {
        self.set_with_ttl(key, value, self.default_ttl).await
    }

    /// 写入单个键，指定 TTL。
    pub async fn set_with_ttl(&self, key: &str, value: V, ttl: Duration) -> Result<()> {
        match self.write_strategy {
            WriteStrategy::WriteThrough => {
                for layer in &self.layers {
                    layer.set(key, value.clone(), ttl).await?;
                }
            }
            WriteStrategy::WriteAround => {
                if let Some(layer) = self.layers.first() {
                    layer.set(key, value, ttl).await?;
                }
            }
        }

        Ok(())
    }

    /// 批量写入，使用默认 TTL。
    pub async fn set_batch(&self, items: HashMap<String, V>) -> Result<()> {
        self.set_batch_with_ttl(items, self.default_ttl).await
    }

    /// 批量写入，指定 TTL。
    pub async fn set_batch_with_ttl(&self, items: HashMap<String, V>, ttl: Duration) -> Result<()> {
        match self.write_strategy {
            WriteStrategy::WriteThrough => {
                for layer in &self.layers {
                    layer.set_batch(items.clone(), ttl).await?;
                }
            }
            WriteStrategy::WriteAround => {
                if let Some(layer) = self.layers.first() {
                    layer.set_batch(items, ttl).await?;
                }
            }
        }

        Ok(())
    }

    /// 删除键：写穿删除所有层。
    pub async fn delete(&self, key: &str) -> Result<()> {
        for layer in &self.layers {
            layer.delete(key).await?;
        }
        Ok(())
    }

    /// 获取缓存统计指标。
    pub fn metrics(&self) -> &SimpleMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn build_two_layer_cache() -> MultiLayerCache<String> {
        let l1: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l1"));
        let l2: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l2"));

        MultiLayerCache::with_options(
            vec![l1, l2],
            Duration::from_secs(60),
            WriteStrategy::WriteThrough,
            PromotePolicy::PromoteOnHit,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn memory_only_set_get() {
        let cache: MultiLayerCache<String> = MultiLayerCache::memory_only(Duration::from_secs(30));

        cache.set("hello", "world".to_string()).await.unwrap();
        let got = cache.get("hello").await.unwrap();

        assert_eq!(got.as_deref(), Some("world"));
    }

    #[tokio::test]
    async fn promote_on_lower_hit() {
        let cache = build_two_layer_cache();

        // 直接写入第二层，模拟 L1 miss + L2 hit
        let l2 = cache.layers.get(1).unwrap().clone();
        l2.set("k", "v".to_string(), Duration::from_secs(120))
            .await
            .unwrap();

        // 第一次读取应命中 L2，随后回填 L1
        let got = cache.get("k").await.unwrap();
        assert_eq!(got.as_deref(), Some("v"));

        // 再次读取应直接命中 L1
        let hit = cache.get("k").await.unwrap();
        assert_eq!(hit.as_deref(), Some("v"));
    }

    #[tokio::test]
    async fn batch_set_and_get() {
        let cache = build_two_layer_cache();
        let mut batch = HashMap::new();
        batch.insert("a".to_string(), "1".to_string());
        batch.insert("b".to_string(), "2".to_string());

        cache
            .set_batch_with_ttl(batch.clone(), Duration::from_secs(10))
            .await
            .unwrap();

        let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = cache.get_batch(&keys).await.unwrap();

        assert_eq!(result.get("a").map(String::as_str), Some("1"));
        assert_eq!(result.get("b").map(String::as_str), Some("2"));
        assert!(!result.contains_key("c"));
    }

    #[tokio::test]
    async fn ttl_expiration() {
        let cache: MultiLayerCache<String> =
            MultiLayerCache::memory_only(Duration::from_millis(300));

        cache.set("short", "live".to_string()).await.unwrap();
        let immediate = cache.get("short").await.unwrap();
        assert_eq!(immediate.as_deref(), Some("live"));

        tokio::time::sleep(Duration::from_secs(1)).await;
        let expired = cache.get("short").await.unwrap();
        assert_eq!(expired, None);
    }
}
