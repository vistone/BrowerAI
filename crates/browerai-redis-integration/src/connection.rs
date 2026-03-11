use std::time::Duration;

use anyhow::{Context, Result};
use deadpool_redis::redis::AsyncCommands;
use deadpool_redis::{Config, Pool, Runtime};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::{debug, warn};

/// Redis 连接池配置。
#[derive(Debug, Clone)]
pub struct RedisConfig {
    pub url: String,
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub operation_timeout: Duration,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            max_connections: 10,
            connection_timeout: Duration::from_secs(5),
            operation_timeout: Duration::from_secs(2),
        }
    }
}

/// Redis 连接池封装，支持序列化存储。
pub struct RedisPool {
    pool: Pool,
    operation_timeout: Duration,
}

impl RedisPool {
    /// 创建连接池。
    pub fn new(config: RedisConfig) -> Result<Self> {
        let cfg = Config::from_url(config.url);
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1))
            .context("Failed to create Redis connection pool")?;

        debug!(
            max_connections = config.max_connections,
            "Redis connection pool created"
        );

        Ok(Self {
            pool,
            operation_timeout: config.operation_timeout,
        })
    }

    /// 获取原始连接（用于高级操作如分布式锁）。
    pub async fn get_connection(&self) -> Result<deadpool_redis::Connection> {
        self.pool
            .get()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get connection: {}", e))
    }

    /// 获取值（JSON 反序列化）。
    pub async fn get<V>(&self, key: &str) -> Result<Option<V>>
    where
        V: DeserializeOwned,
    {
        let timeout = self.operation_timeout;
        let key = key.to_string();

        let result = tokio::time::timeout(timeout, async {
            let mut conn = self
                .pool
                .get()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let raw: Option<String> = conn.get(&key).await?;
            match raw {
                Some(json) => {
                    let value: V =
                        serde_json::from_str(&json).context("Failed to deserialize Redis value")?;
                    Ok(Some(value))
                }
                None => Ok(None),
            }
        })
        .await;

        match result {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(err)) => {
                warn!(key = %key, %err, "Redis get failed");
                Err(err)
            }
            Err(_) => {
                warn!(key = %key, "Redis get timeout");
                Err(anyhow::anyhow!("Redis operation timeout"))
            }
        }
    }

    /// 设置值（JSON 序列化，支持 TTL）。
    pub async fn set<V>(&self, key: &str, value: &V, ttl: Duration) -> Result<()>
    where
        V: Serialize,
    {
        let timeout = self.operation_timeout;
        let key = key.to_string();
        let json = serde_json::to_string(value).context("Failed to serialize value")?;
        let ttl_secs = ttl.as_secs();

        let result = tokio::time::timeout(timeout, async {
            let mut conn = self
                .pool
                .get()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            conn.set_ex::<_, _, ()>(&key, json, ttl_secs).await?;
            Ok::<_, anyhow::Error>(())
        })
        .await;

        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(err)) => {
                warn!(key = %key, %err, "Redis set failed");
                Err(err)
            }
            Err(_) => {
                warn!(key = %key, "Redis set timeout");
                Err(anyhow::anyhow!("Redis operation timeout"))
            }
        }
    }

    /// 删除键。
    pub async fn delete(&self, key: &str) -> Result<()> {
        let timeout = self.operation_timeout;
        let key = key.to_string();

        let result = tokio::time::timeout(timeout, async {
            let mut conn = self
                .pool
                .get()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            conn.del::<_, ()>(&key).await?;
            Ok::<_, anyhow::Error>(())
        })
        .await;

        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(err)) => {
                warn!(key = %key, %err, "Redis delete failed");
                Err(err)
            }
            Err(_) => {
                warn!(key = %key, "Redis delete timeout");
                Err(anyhow::anyhow!("Redis operation timeout"))
            }
        }
    }

    /// 批量获取（MGET + 反序列化）。
    pub async fn get_batch<V>(&self, keys: &[String]) -> Result<Vec<Option<V>>>
    where
        V: DeserializeOwned,
    {
        if keys.is_empty() {
            return Ok(vec![]);
        }

        let timeout = self.operation_timeout;

        let result = tokio::time::timeout(timeout, async {
            let mut conn = self
                .pool
                .get()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            // Use redis::cmd directly for MGET to avoid trait-bound issues across redis versions
            let raw: Vec<Option<String>> = {
                let mut cmd = redis::cmd("MGET");
                for k in keys {
                    cmd.arg(k.as_str());
                }
                cmd.query_async(&mut *conn).await?
            };

            let mut values = vec![];
            for opt_json in raw {
                match opt_json {
                    Some(json) => {
                        let value: V = serde_json::from_str(&json)
                            .context("Failed to deserialize batch value")?;
                        values.push(Some(value));
                    }
                    None => {
                        values.push(None);
                    }
                }
            }

            Ok::<_, anyhow::Error>(values)
        })
        .await;

        match result {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(err)) => {
                warn!(%err, "Redis get_batch failed");
                Err(err)
            }
            Err(_) => {
                warn!("Redis get_batch timeout");
                Err(anyhow::anyhow!("Redis operation timeout"))
            }
        }
    }

    /// 批量设置（使用 MSET + 后续 EXPIRE）。
    pub async fn set_batch<V>(&self, items: &[(String, V)], ttl: Duration) -> Result<()>
    where
        V: Serialize,
    {
        if items.is_empty() {
            return Ok(());
        }

        let timeout = self.operation_timeout;
        let ttl_secs = ttl.as_secs();

        let mut pairs = vec![];
        for (key, value) in items {
            let json = serde_json::to_string(value).context("Failed to serialize batch value")?;
            pairs.push((key.clone(), json));
        }

        let result = tokio::time::timeout(timeout, async {
            let mut conn = self
                .pool
                .get()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            // MSET 设置所有键值对
            let kv_refs: Vec<(&str, &str)> = pairs
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            // 使用单个命令替代pipeline（简化实现）
            for (key, value) in &kv_refs {
                conn.set::<_, _, ()>(key, value).await?;
            }

            // 批量设置过期时间
            for (key, _) in &pairs {
                conn.expire::<_, ()>(key, ttl_secs as i64).await?;
            }

            Ok::<_, anyhow::Error>(())
        })
        .await;

        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(err)) => {
                warn!(%err, "Redis set_batch failed");
                Err(err)
            }
            Err(_) => {
                warn!("Redis set_batch timeout");
                Err(anyhow::anyhow!("Redis operation timeout"))
            }
        }
    }

    /// 检查 Redis 连接是否可用。
    pub async fn is_healthy(&self) -> bool {
        matches!(
            tokio::time::timeout(self.operation_timeout, async {
                let mut conn = self
                    .pool
                    .get()
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                let result: Result<String, redis::RedisError> =
                    redis::cmd("PING").query_async(&mut *conn).await;
                result.map_err(|e| anyhow::anyhow!("{}", e))
            })
            .await,
            Ok(Ok(_))
        )
    }
}
