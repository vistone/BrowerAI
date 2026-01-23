use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::{debug, warn};

use crate::connection::{RedisConfig, RedisPool};

/// Redis 缓存层，实现 CacheLayer 接口（可选降级）。
pub struct RedisLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    pool: Arc<RedisPool>,
    name: String,
    graceful_degradation: bool,
    _phantom: std::marker::PhantomData<V>,
}

impl<V> RedisLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    /// 创建 Redis 层（默认启用降级）。
    pub fn new(name: impl Into<String>, config: RedisConfig) -> Result<Self> {
        Self::with_degradation(name, config, true)
    }

    /// 创建 Redis 层，可指定是否降级。
    pub fn with_degradation(
        name: impl Into<String>,
        config: RedisConfig,
        graceful_degradation: bool,
    ) -> Result<Self> {
        let pool = RedisPool::new(config)?;

        Ok(Self {
            pool: Arc::new(pool),
            name: name.into(),
            graceful_degradation,
            _phantom: std::marker::PhantomData,
        })
    }

    /// 内部方法：get 单键（暴露为 pub 用于测试）。
    pub async fn get_internal(&self, key: &str) -> Result<Option<V>> {
        match self.pool.get(key).await {
            Ok(v) => {
                debug!(layer = %self.name, key, "Redis get success");
                Ok(v)
            }
            Err(err) if self.graceful_degradation => {
                warn!(layer = %self.name, key, %err, "Redis get failed, degrading");
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }

    /// 内部方法：get 批量（暴露为 pub 用于测试）。
    pub async fn get_batch_internal(&self, keys: &[String]) -> Result<HashMap<String, V>> {
        match self.pool.get_batch(keys).await {
            Ok(values) => {
                let mut map = HashMap::new();
                for (idx, key) in keys.iter().enumerate() {
                    if let Some(value) = values.get(idx).and_then(|opt| opt.clone()) {
                        map.insert(key.clone(), value);
                    }
                }
                debug!(layer = %self.name, count = map.len(), "Redis get_batch success");
                Ok(map)
            }
            Err(err) if self.graceful_degradation => {
                warn!(layer = %self.name, %err, "Redis get_batch failed, degrading");
                Ok(HashMap::new())
            }
            Err(err) => Err(err),
        }
    }

    /// 内部方法：set 单键（暴露为 pub 用于测试）。
    pub async fn set_internal(&self, key: &str, value: V, ttl: Duration) -> Result<()> {
        match self.pool.set(key, &value, ttl).await {
            Ok(_) => {
                debug!(layer = %self.name, key, "Redis set success");
                Ok(())
            }
            Err(err) if self.graceful_degradation => {
                warn!(layer = %self.name, key, %err, "Redis set failed, degrading");
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    /// 内部方法：set 批量（暴露为 pub 用于测试）。
    pub async fn set_batch_internal(&self, items: HashMap<String, V>, ttl: Duration) -> Result<()> {
        let pairs: Vec<(String, V)> = items.into_iter().collect();

        match self.pool.set_batch(&pairs, ttl).await {
            Ok(_) => {
                debug!(layer = %self.name, count = pairs.len(), "Redis set_batch success");
                Ok(())
            }
            Err(err) if self.graceful_degradation => {
                warn!(layer = %self.name, %err, "Redis set_batch failed, degrading");
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    /// 内部方法：删除键（暴露为 pub 用于测试）。
    pub async fn delete_internal(&self, key: &str) -> Result<()> {
        match self.pool.delete(key).await {
            Ok(_) => {
                debug!(layer = %self.name, key, "Redis delete success");
                Ok(())
            }
            Err(err) if self.graceful_degradation => {
                warn!(layer = %self.name, key, %err, "Redis delete failed, degrading");
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    /// 检查 Redis 连接健康状态。
    pub async fn is_healthy(&self) -> bool {
        self.pool.is_healthy().await
    }
}

// 实现动态调度接口（仅在集成到 MultiLayerCache 时使用）
#[cfg(feature = "cache-integration")]
use browerai_multilayer_cache::strategy::{CacheLayer, LayerFuture};

#[cfg(feature = "cache-integration")]
impl<V> CacheLayer<V> for RedisLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn get<'a>(&'a self, key: &'a str) -> LayerFuture<'a, Option<V>> {
        Box::pin(self.get_internal(key))
    }

    fn get_batch<'a>(&'a self, keys: &'a [String]) -> LayerFuture<'a, HashMap<String, V>> {
        Box::pin(self.get_batch_internal(keys))
    }

    fn set<'a>(&'a self, key: &'a str, value: V, ttl: Duration) -> LayerFuture<'a, ()> {
        Box::pin(self.set_internal(key, value, ttl))
    }

    fn set_batch<'a>(&'a self, items: HashMap<String, V>, ttl: Duration) -> LayerFuture<'a, ()> {
        Box::pin(self.set_batch_internal(items, ttl))
    }

    fn delete<'a>(&'a self, key: &'a str) -> LayerFuture<'a, ()> {
        Box::pin(self.delete_internal(key))
    }
}
