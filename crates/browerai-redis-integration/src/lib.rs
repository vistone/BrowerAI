//! Redis 分布式缓存集成 - Week 6
//!
//! 支持连接池、降级、分布式锁和故障转移
//! 注意：集群模式暂时禁用，等待redis_cluster_async更新

// pub mod cluster_connection;  // Temporarily disabled - redis_cluster_async uses old redis
// pub mod cluster_hash_tag;
// pub mod cluster_layer;
// pub mod cluster_sentinel;
pub mod connection;
pub mod distributed_lock;
pub mod layer;
pub mod sentinel;

// pub use cluster_connection::{RedisClusterConfig, RedisClusterPool};  // Temporarily disabled
// pub use cluster_hash_tag::ClusterHashTag;
// pub use cluster_layer::RedisClusterLayer;
// pub use cluster_sentinel::{RedisClusterPoolWithSentinel, SentinelConfig};
pub use connection::{RedisConfig, RedisPool};
pub use distributed_lock::{DistributedLock, LockGuard};
pub use layer::RedisLayer;
pub use sentinel::resolve_master_addr;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn redis_connection_basic() {
        // 仅在本地 Redis 可用时执行
        let config = RedisConfig::default();
        if let Ok(pool) = RedisPool::new(config) {
            let healthy = pool.is_healthy().await;
            println!("Redis health check: {}", healthy);
        } else {
            println!("Redis unavailable, skipping test");
        }
    }
}
