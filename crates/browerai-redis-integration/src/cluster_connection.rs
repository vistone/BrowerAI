use std::time::Duration;

use anyhow::{Context, Result};
use redis_cluster_async::redis::{self, AsyncCommands};
use redis_cluster_async::{Client as ClusterClient, Connection as ClusterConnection};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::{debug, warn};

/// Redis Cluster 配置。
#[derive(Debug, Clone)]
pub struct RedisClusterConfig {
    pub nodes: Vec<String>,
    pub operation_timeout: Duration,
}

impl Default for RedisClusterConfig {
    fn default() -> Self {
        Self {
            nodes: vec!["redis://127.0.0.1:6379".to_string()],
            operation_timeout: Duration::from_secs(2),
        }
    }
}

/// Redis Cluster 客户端封装。
pub struct RedisClusterPool {
    client: ClusterClient,
    operation_timeout: Duration,
}

impl RedisClusterPool {
    /// 创建 Cluster 客户端。
    pub fn new(config: RedisClusterConfig) -> Result<Self> {
        let client = ClusterClient::open(config.nodes.clone())
            .context("Failed to create Redis Cluster client")?;
        debug!(nodes = ?config.nodes, "Redis Cluster client created");
        Ok(Self {
            client,
            operation_timeout: config.operation_timeout,
        })
    }

    /// 获取值（JSON 反序列化）。
    pub async fn get<V>(&self, key: &str) -> Result<Option<V>>
    where
        V: DeserializeOwned,
    {
        let timeout = self.operation_timeout;
        let key_owned = key.to_string();

        let result = tokio::time::timeout(timeout, async {
            let mut conn: ClusterConnection = self
                .client
                .get_connection()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let raw: Option<String> = conn.get(&key_owned).await?;
            match raw {
                Some(json) => {
                    let value: V = serde_json::from_str(&json)
                        .context("Failed to deserialize Redis Cluster value")?;
                    Ok(Some(value))
                }
                None => Ok(None),
            }
        })
        .await;

        match result {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(err)) => {
                warn!(key = %key_owned, %err, "Redis Cluster get failed");
                Err(err)
            }
            Err(_) => {
                warn!(key = %key_owned, "Redis Cluster get timeout");
                Err(anyhow::anyhow!("Redis Cluster operation timeout"))
            }
        }
    }

    /// 设置值（JSON 序列化，支持 TTL）。
    pub async fn set<V>(&self, key: &str, value: &V, ttl: Duration) -> Result<()>
    where
        V: Serialize,
    {
        let timeout = self.operation_timeout;
        let key_owned = key.to_string();
        let json = serde_json::to_string(value).context("Failed to serialize value")?;
        let ttl_secs = ttl.as_secs();

        let result = tokio::time::timeout(timeout, async {
            let mut conn: ClusterConnection = self
                .client
                .get_connection()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            conn.set_ex::<_, _, ()>(&key_owned, json, ttl_secs as usize)
                .await?;
            Ok::<_, anyhow::Error>(())
        })
        .await;

        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(err)) => {
                warn!(key = %key_owned, %err, "Redis Cluster set failed");
                Err(err)
            }
            Err(_) => {
                warn!(key = %key_owned, "Redis Cluster set timeout");
                Err(anyhow::anyhow!("Redis Cluster operation timeout"))
            }
        }
    }

    /// 删除键。
    pub async fn delete(&self, key: &str) -> Result<()> {
        let timeout = self.operation_timeout;
        let key_owned = key.to_string();

        let result = tokio::time::timeout(timeout, async {
            let mut conn: ClusterConnection = self
                .client
                .get_connection()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            conn.del::<_, ()>(&key_owned).await?;
            Ok::<_, anyhow::Error>(())
        })
        .await;

        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(err)) => {
                warn!(key = %key_owned, %err, "Redis Cluster delete failed");
                Err(err)
            }
            Err(_) => {
                warn!(key = %key_owned, "Redis Cluster delete timeout");
                Err(anyhow::anyhow!("Redis Cluster operation timeout"))
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
        let _key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();

        let result = tokio::time::timeout(timeout, async {
            let mut conn: ClusterConnection = self
                .client
                .get_connection()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            // 使用逐个 GET 以避免 trait 版本冲突
            let mut raw: Vec<Option<String>> = Vec::with_capacity(keys.len());
            for k in keys {
                let val: Option<String> = redis::cmd("GET")
                    .arg(k.as_str())
                    .query_async(&mut conn)
                    .await?;
                raw.push(val);
            }
            let mut values = Vec::with_capacity(raw.len());
            for opt_json in raw {
                match opt_json {
                    Some(json) => {
                        let value: V = serde_json::from_str(&json)
                            .context("Failed to deserialize batch value")?;
                        values.push(Some(value));
                    }
                    None => values.push(None),
                }
            }
            Ok::<_, anyhow::Error>(values)
        })
        .await;

        match result {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(err)) => {
                warn!(%err, "Redis Cluster get_batch failed");
                Err(err)
            }
            Err(_) => {
                warn!("Redis Cluster get_batch timeout");
                Err(anyhow::anyhow!("Redis Cluster operation timeout"))
            }
        }
    }

    /// 批量设置（使用管道 + EXPIRE）。
    pub async fn set_batch<V>(&self, items: &[(String, V)], ttl: Duration) -> Result<()>
    where
        V: Serialize,
    {
        if items.is_empty() {
            return Ok(());
        }
        let timeout = self.operation_timeout;
        let ttl_secs = ttl.as_secs();

        let mut pairs = Vec::with_capacity(items.len());
        for (key, value) in items {
            let json = serde_json::to_string(value).context("Failed to serialize batch value")?;
            pairs.push((key.clone(), json));
        }

        let result = tokio::time::timeout(timeout, async {
            let mut conn: ClusterConnection = self
                .client
                .get_connection()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            // Cluster 模式下不保证跨 slot 的多键原子性，改为逐键 SETEX
            for (key, value) in &pairs {
                redis::cmd("SETEX")
                    .arg(key.as_str())
                    .arg(ttl_secs)
                    .arg(value.as_str())
                    .query_async::<_, ()>(&mut conn)
                    .await?;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await;

        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(err)) => {
                warn!(%err, "Redis Cluster set_batch failed");
                Err(err)
            }
            Err(_) => {
                warn!("Redis Cluster set_batch timeout");
                Err(anyhow::anyhow!("Redis Cluster operation timeout"))
            }
        }
    }

    /// 检查 Cluster 健康。
    pub async fn is_healthy(&self) -> bool {
        matches!(
            tokio::time::timeout(self.operation_timeout, async {
                let mut conn: ClusterConnection = self
                    .client
                    .get_connection()
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                let pong: Result<String, redis::RedisError> =
                    redis::cmd("PING").query_async(&mut conn).await;
                pong.map_err(|e| anyhow::anyhow!("{}", e))
            })
            .await,
            Ok(Ok(_))
        )
    }
}
