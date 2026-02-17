use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use anyhow::Result;

/// 统一的缓存层接口，支持异步操作。
pub trait CacheLayer<V>: Send + Sync
where
    V: Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str;

    fn get<'a>(&'a self, key: &'a str) -> LayerFuture<'a, Option<V>>;

    fn get_batch<'a>(&'a self, keys: &'a [String]) -> LayerFuture<'a, HashMap<String, V>>;

    fn set<'a>(&'a self, key: &'a str, value: V, ttl: Duration) -> LayerFuture<'a, ()>;

    fn set_batch<'a>(&'a self, items: HashMap<String, V>, ttl: Duration) -> LayerFuture<'a, ()>;

    fn delete<'a>(&'a self, key: &'a str) -> LayerFuture<'a, ()>;
}

pub type LayerFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteStrategy {
    /// 写穿：同时写入所有层，保证一致性。
    WriteThrough,
    /// 写绕过：只写入顶层，底层由后续淘汰/回填。
    WriteAround,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotePolicy {
    /// 命中低层时，向上回填。
    PromoteOnHit,
    /// 不回填，保持只读链路。
    NoPromote,
}
