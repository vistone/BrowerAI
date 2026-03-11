use std::time::Duration;

use anyhow::{anyhow, Result};
use deadpool_redis::redis;
use tracing::{debug, warn};

use crate::connection::RedisPool;

/// 分布式锁实现（基于 Redis SET NX EX）。
pub struct DistributedLock {
    pool: std::sync::Arc<RedisPool>,
    key: String,
    value: String,
    ttl: Duration,
}

impl DistributedLock {
    /// 创建锁实例（未获取）。
    pub fn new(pool: std::sync::Arc<RedisPool>, key: impl Into<String>, ttl: Duration) -> Self {
        let value = uuid::Uuid::new_v4().to_string();
        Self {
            pool,
            key: format!("lock:{}", key.into()),
            value,
            ttl,
        }
    }

    /// 尝试获取锁（非阻塞）。
    pub async fn try_acquire(&self) -> Result<bool> {
        let timeout = Duration::from_secs(2);
        let key = self.key.clone();
        let value = self.value.clone();
        let ttl_secs = self.ttl.as_secs();

        let result = tokio::time::timeout(timeout, async {
            let mut conn = self.pool.get_connection().await?;

            // SET key value NX EX ttl
            let acquired: bool = redis::cmd("SET")
                .arg(&key)
                .arg(&value)
                .arg("NX")
                .arg("EX")
                .arg(ttl_secs)
                .query_async(&mut *conn)
                .await
                .unwrap_or(false);

            Ok::<bool, anyhow::Error>(acquired)
        })
        .await;

        match result {
            Ok(Ok(acquired)) => {
                if acquired {
                    debug!(key = %self.key, "distributed lock acquired");
                }
                Ok(acquired)
            }
            Ok(Err(err)) => {
                warn!(key = %self.key, %err, "failed to acquire lock");
                Err(err)
            }
            Err(_) => {
                warn!(key = %self.key, "lock acquire timeout");
                Err(anyhow!("Lock acquire timeout"))
            }
        }
    }

    /// 阻塞获取锁（带重试）。
    pub async fn acquire(&self, max_retries: usize, retry_delay: Duration) -> Result<()> {
        for attempt in 0..max_retries {
            if self.try_acquire().await? {
                return Ok(());
            }
            debug!(key = %self.key, attempt, "lock acquisition retry");
            tokio::time::sleep(retry_delay).await;
        }
        Err(anyhow!(
            "Failed to acquire lock after {} retries",
            max_retries
        ))
    }

    /// 释放锁（简化实现：直接删除 key，生产环境建议使用 Lua 脚本确保原子性）。
    pub async fn release(&self) -> Result<()> {
        let timeout = Duration::from_secs(2);
        let key = self.key.clone();

        let result = tokio::time::timeout(timeout, async {
            self.pool.delete(&key).await?;
            debug!(key = %self.key, "distributed lock released");
            Ok::<_, anyhow::Error>(())
        })
        .await;

        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(err)) => Err(err),
            Err(_) => Err(anyhow!("Lock release timeout")),
        }
    }
}

/// RAII 风格的锁守卫。
pub struct LockGuard {
    lock: DistributedLock,
}

impl LockGuard {
    /// 获取锁并返回守卫。
    pub async fn acquire(
        lock: DistributedLock,
        max_retries: usize,
        retry_delay: Duration,
    ) -> Result<Self> {
        lock.acquire(max_retries, retry_delay).await?;
        Ok(Self { lock })
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        // 异步释放（在同步 Drop 中启动 blocking task）
        let lock_clone = DistributedLock {
            pool: std::sync::Arc::clone(&self.lock.pool),
            key: self.lock.key.clone(),
            value: self.lock.value.clone(),
            ttl: self.lock.ttl,
        };

        tokio::task::spawn(async move {
            if let Err(err) = lock_clone.release().await {
                warn!(%err, "failed to release lock in drop");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::RedisConfig;

    #[tokio::test]
    async fn distributed_lock_basic() {
        let config = RedisConfig::default();
        let pool = match RedisPool::new(config) {
            Ok(p) => std::sync::Arc::new(p),
            Err(_) => {
                println!("Redis unavailable, skipping test");
                return;
            }
        };

        if !pool.is_healthy().await {
            println!("Redis not healthy, skipping test");
            return;
        }

        let lock = DistributedLock::new(pool, "test_lock", Duration::from_secs(10));

        // 第一次获取应该成功
        assert!(lock.try_acquire().await.unwrap());

        // 第二次获取应该失败（已被占用）
        let lock2 = DistributedLock::new(
            std::sync::Arc::clone(&lock.pool),
            "test_lock",
            Duration::from_secs(10),
        );
        assert!(!lock2.try_acquire().await.unwrap());

        // 释放后再次获取应该成功
        lock.release().await.unwrap();
        assert!(lock2.try_acquire().await.unwrap());

        lock2.release().await.unwrap();
    }
}
