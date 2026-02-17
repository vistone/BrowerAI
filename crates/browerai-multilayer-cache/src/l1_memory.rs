use std::collections::HashMap;
use std::time::Duration;

use browerai_cache::CacheStore;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::strategy::{CacheLayer, LayerFuture};

/// L1 内存缓存层，基于 `browerai-cache` 的并发安全实现。
pub struct MemoryLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    store: CacheStore<V>,
    name: String,
}

impl<V> MemoryLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            store: CacheStore::new(),
            name: name.into(),
        }
    }
}

impl<V> CacheLayer<V> for MemoryLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn get<'a>(&'a self, key: &'a str) -> LayerFuture<'a, Option<V>> {
        Box::pin(async move { self.store.get(key).await })
    }

    fn get_batch<'a>(&'a self, keys: &'a [String]) -> LayerFuture<'a, HashMap<String, V>> {
        Box::pin(async move {
            let mut results = HashMap::new();
            // 内存缓存足够快，直接串行查询避免 spawn 开销
            for key in keys {
                if let Some(value) = self.store.get(key).await? {
                    results.insert(key.clone(), value);
                }
            }
            Ok(results)
        })
    }

    fn set<'a>(&'a self, key: &'a str, value: V, ttl: Duration) -> LayerFuture<'a, ()> {
        Box::pin(async move { self.store.set(key, value, ttl).await })
    }

    fn set_batch<'a>(&'a self, items: HashMap<String, V>, ttl: Duration) -> LayerFuture<'a, ()> {
        Box::pin(async move {
            for (key, value) in items.into_iter() {
                self.store.set(&key, value, ttl).await?;
            }
            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> LayerFuture<'a, ()> {
        Box::pin(async move { self.store.delete(key).await })
    }
}
