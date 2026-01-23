use std::collections::HashMap;
use std::time::Duration;

use crate::strategy::{CacheLayer, LayerFuture};

/// 占位的分布式缓存层（将在 Redis 集成阶段替换）。
#[derive(Default)]
pub struct NoopDistributedLayer;

impl NoopDistributedLayer {
    pub fn new() -> Self {
        Self
    }
}

impl<V> CacheLayer<V> for NoopDistributedLayer
where
    V: Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        "noop-distributed"
    }

    fn get<'a>(&'a self, _key: &'a str) -> LayerFuture<'a, Option<V>> {
        Box::pin(async { Ok(None) })
    }

    fn get_batch<'a>(&'a self, _keys: &'a [String]) -> LayerFuture<'a, HashMap<String, V>> {
        Box::pin(async { Ok(HashMap::new()) })
    }

    fn set<'a>(&'a self, _key: &'a str, _value: V, _ttl: Duration) -> LayerFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn set_batch<'a>(&'a self, _items: HashMap<String, V>, _ttl: Duration) -> LayerFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn delete<'a>(&'a self, _key: &'a str) -> LayerFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}
