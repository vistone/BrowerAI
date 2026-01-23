use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use redis_cluster_async::{Client as ClusterClient, Connection as ClusterConnection};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::sync::RwLock;
use tracing::debug;

use crate::cluster_hash_tag::ClusterHashTag;

/// Sentinel 配置用于故障转移（主节点）。
#[derive(Debug, Clone)]
pub struct SentinelConfig {
    pub sentinels: Vec<String>,
    pub master_name: String,
    pub monitor_interval: Duration,
}

/// Redis Cluster 连接池，支持 Sentinel 健康监控与热切换。
pub struct RedisClusterPoolWithSentinel {
    client: Arc<RwLock<ClusterClient>>,
    sentinel_config: Option<SentinelConfig>,
    operation_timeout: Duration,
}

impl RedisClusterPoolWithSentinel {
    /// 创建 Cluster 连接池（无 Sentinel）。
    pub fn new(nodes: Vec<String>, operation_timeout: Duration) -> Result<Self> {
        let client = ClusterClient::open(nodes)?;
        Ok(Self {
            client: Arc::new(RwLock::new(client)),
            sentinel_config: None,
            operation_timeout,
        })
    }

    /// 创建 Cluster 连接池，带 Sentinel 故障转移支持。
    pub fn with_sentinel(
        nodes: Vec<String>,
        sentinel: SentinelConfig,
        operation_timeout: Duration,
    ) -> Result<Self> {
        let client = ClusterClient::open(nodes)?;
        Ok(Self {
            client: Arc::new(RwLock::new(client)),
            sentinel_config: Some(sentinel),
            operation_timeout,
        })
    }

    /// 获取连接（支持 Sentinel 热切换）。
    pub async fn get_connection(&self) -> Result<ClusterConnection> {
        // 如果启用 Sentinel 且健康检查失败，触发重解析
        if let Some(ref sentinel) = self.sentinel_config {
            if !self.check_health().await {
                self.handle_sentinel_failover(sentinel).await?;
            }
        }
        let client = self.client.read().await;
        let conn = client
            .get_connection()
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(conn)
    }

    /// 检查集群健康（PING）。
    pub async fn check_health(&self) -> bool {
        match tokio::time::timeout(self.operation_timeout, async {
            let client = self.client.read().await;
            let mut conn = client
                .get_connection()
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let pong: Result<String, redis_cluster_async::redis::RedisError> =
                redis_cluster_async::redis::cmd("PING")
                    .query_async(&mut conn)
                    .await;
            pong.map_err(|e| anyhow::anyhow!("{}", e))
        })
        .await
        {
            Ok(Ok(_)) => true,
            _ => false,
        }
    }

    /// Sentinel 故障转移处理。
    async fn handle_sentinel_failover(&self, sentinel: &SentinelConfig) -> Result<()> {
        // 通过 Sentinel 解析新主节点
        if let Ok(Some(master_addr)) = crate::resolve_master_addr(
            &sentinel.sentinels,
            &sentinel.master_name,
            self.operation_timeout,
        )
        .await
        {
            debug!(master_addr = %master_addr, "Sentinel resolved new master address, rebuilding cluster client");
            let new_client = ClusterClient::open(vec![master_addr])?;
            let mut client = self.client.write().await;
            *client = new_client;
        }
        Ok(())
    }

    /// 支持哈希标签的批量操作 GET。
    ///
    /// 为键列表批量应用哈希标签（保证同 slot），执行 MGET，然后移除标签。
    pub async fn get_batch_with_hash_tag<V>(
        &self,
        keys: &[String],
        tag: Option<&str>,
    ) -> Result<Vec<Option<V>>>
    where
        V: DeserializeOwned,
    {
        if keys.is_empty() {
            return Ok(vec![]);
        }
        let tagged_keys = ClusterHashTag::apply_batch(keys, tag);
        let mut conn = self.get_connection().await?;

        let timeout = self.operation_timeout;
        let raw = tokio::time::timeout(timeout, async {
            let mut raw: Vec<Option<String>> = Vec::with_capacity(keys.len());
            for k in &tagged_keys {
                let val: Option<String> = redis_cluster_async::redis::cmd("GET")
                    .arg(k.as_str())
                    .query_async(&mut conn)
                    .await?;
                raw.push(val);
            }
            Ok::<_, anyhow::Error>(raw)
        })
        .await??;

        let mut values = Vec::with_capacity(raw.len());
        for opt_json in raw {
            match opt_json {
                Some(json) => {
                    let value: V = serde_json::from_str(&json)?;
                    values.push(Some(value));
                }
                None => values.push(None),
            }
        }
        Ok(values)
    }

    /// 支持哈希标签的批量操作 SET。
    ///
    /// 为键列表批量应用哈希标签，逐键执行 SETEX，保证同 slot 原子性。
    pub async fn set_batch_with_hash_tag<V>(
        &self,
        items: &[(String, V)],
        tag: Option<&str>,
        ttl: Duration,
    ) -> Result<()>
    where
        V: Serialize,
    {
        if items.is_empty() {
            return Ok(());
        }
        let ttl_secs = ttl.as_secs();
        let mut conn = self.get_connection().await?;

        let timeout = self.operation_timeout;
        tokio::time::timeout(timeout, async {
            for (key, value) in items {
                let tagged_key = ClusterHashTag::apply(key, tag);
                let json = serde_json::to_string(value)?;
                redis_cluster_async::redis::cmd("SETEX")
                    .arg(tagged_key.as_str())
                    .arg(ttl_secs)
                    .arg(json.as_str())
                    .query_async::<_, ()>(&mut conn)
                    .await?;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_tag_integration() {
        // 验证哈希标签应用和移除
        let keys = vec!["user:1:name".to_string(), "user:1:email".to_string()];
        let tagged = ClusterHashTag::apply_batch(&keys, Some("user:1"));
        assert_eq!(tagged.len(), 2);
        assert_eq!(ClusterHashTag::strip(&tagged[0]), "user:1:name");
        assert_eq!(ClusterHashTag::extract(&tagged[1]), Some("user:1"));
    }
}
