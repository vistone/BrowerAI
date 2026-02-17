use std::collections::HashMap;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::l1_memory::MemoryLayer;
use crate::strategy::{CacheLayer, LayerFuture};

/// L2 本地缓存层（当前复用内存实现，占位以便后续替换为文件/持久化方案）。
pub struct LocalLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    inner: MemoryLayer<V>,
}

impl<V> LocalLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            inner: MemoryLayer::new(name),
        }
    }
}

impl<V> CacheLayer<V> for LocalLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn get<'a>(&'a self, key: &'a str) -> LayerFuture<'a, Option<V>> {
        self.inner.get(key)
    }

    fn get_batch<'a>(&'a self, keys: &'a [String]) -> LayerFuture<'a, HashMap<String, V>> {
        self.inner.get_batch(keys)
    }

    fn set<'a>(&'a self, key: &'a str, value: V, ttl: Duration) -> LayerFuture<'a, ()> {
        self.inner.set(key, value, ttl)
    }

    fn set_batch<'a>(&'a self, items: HashMap<String, V>, ttl: Duration) -> LayerFuture<'a, ()> {
        self.inner.set_batch(items, ttl)
    }

    fn delete<'a>(&'a self, key: &'a str) -> LayerFuture<'a, ()> {
        self.inner.delete(key)
    }
}

impl<V> From<MemoryLayer<V>> for LocalLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn from(inner: MemoryLayer<V>) -> Self {
        Self { inner }
    }
}
