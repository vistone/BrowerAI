use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::Serialize;
use sled::Db;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tracing::{debug, warn};

/// 持久化层配置
#[derive(Clone, Debug)]
pub struct PersistentConfig {
    pub path: PathBuf,
    pub cleanup_interval: Duration,
}

impl Default for PersistentConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("/tmp/browerai_rocksdb"),
            cleanup_interval: Duration::from_secs(10),
        }
    }
}

/// 基于 RocksDB 的持久化缓存层，支持 TTL 索引与批量写入
pub struct PersistentLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    db: Arc<Db>,
    ttl_db: Arc<Db>,
    name: String,
    cleaner: Mutex<Option<JoinHandle<()>>>,
    _phantom: std::marker::PhantomData<V>,
}

impl<V> PersistentLayer<V>
where
    V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    pub fn new(name: impl Into<String>, cfg: PersistentConfig) -> Result<Self> {
        let db_path = cfg.path.join("data");
        let ttl_path = cfg.path.join("ttl");
        let db = sled::open(db_path).context("Failed to open sled data")?;
        let ttl_db = sled::open(ttl_path).context("Failed to open sled ttl")?;
        let layer = Self {
            db: Arc::new(db),
            ttl_db: Arc::new(ttl_db),
            name: name.into(),
            cleaner: Mutex::new(None),
            _phantom: std::marker::PhantomData,
        };
        layer.start_cleaner(cfg.cleanup_interval);
        Ok(layer)
    }

    fn start_cleaner(&self, interval: Duration) {
        let db = self.db.clone();
        let ttl_db = self.ttl_db.clone();
        let name = self.name.clone();
        let handle = tokio::spawn(async move {
            loop {
                sleep(interval).await;
                // 扫描 TTL 索引，删除过期键
                let now_ms = current_millis();
                let mut to_delete: Vec<Vec<u8>> = Vec::new();
                let iter = ttl_db.iter();
                for item in iter {
                    if let Ok((key, val)) = item {
                        if let Ok(ts_str) = std::str::from_utf8(&val) {
                            if let Ok(ts) = ts_str.parse::<u64>() {
                                if ts <= now_ms {
                                    to_delete.push(key.to_vec());
                                }
                            }
                        }
                    }
                }
                let deleted_count = to_delete.len();
                for k in &to_delete {
                    // 删除 ttl 索引与数据
                    let _ = ttl_db.remove(&k);
                    let _ = db.remove(&k);
                }
                debug!(layer = %name, deleted = deleted_count, "TTL cleanup run");
            }
        });
        *self.cleaner.lock() = Some(handle);
    }

    pub async fn get_internal(&self, key: &str) -> Result<Option<V>> {
        match self.db.get(key.as_bytes()) {
            Ok(Some(bytes)) => {
                let json = String::from_utf8(bytes.as_ref().to_vec())
                    .context("Invalid UTF-8 in sled value")?;
                let v: V = serde_json::from_str(&json).context("Failed to deserialize value")?;
                Ok(Some(v))
            }
            Ok(None) => Ok(None),
            Err(e) => {
                warn!(layer = %self.name, %e, "Sled get failed");
                Ok(None)
            }
        }
    }

    pub async fn get_batch_internal(&self, keys: &[String]) -> Result<HashMap<String, V>> {
        let mut map = HashMap::new();
        for k in keys {
            if let Some(v) = self.get_internal(k).await? {
                map.insert(k.clone(), v);
            }
        }
        Ok(map)
    }

    pub async fn set_internal(&self, key: &str, value: V, ttl: Duration) -> Result<()> {
        let json = serde_json::to_string(&value).context("Failed to serialize value")?;
        self.db
            .insert(key.as_bytes(), json.as_bytes())
            .context("Sled insert failed")?;
        let expire_at = current_millis() + ttl.as_millis() as u64;
        self.ttl_db
            .insert(key.as_bytes(), expire_at.to_string().as_bytes())
            .ok();
        Ok(())
    }

    pub async fn set_batch_internal(&self, items: HashMap<String, V>, ttl: Duration) -> Result<()> {
        let expire_at = current_millis() + ttl.as_millis() as u64;
        // sled 批量写入：使用事务确保一致性
        self.db
            .transaction(|tree| {
                for (k, v) in &items {
                    let json = serde_json::to_string(&v)
                        .map_err(|_| sled::transaction::ConflictableTransactionError::Abort(()))?;
                    tree.insert(k.as_bytes(), json.as_bytes())
                        .map_err(|_| sled::transaction::ConflictableTransactionError::Abort(()))?;
                }
                Ok(())
            })
            .map_err(|_| anyhow::anyhow!("Sled transaction data failed"))?;
        self.ttl_db
            .transaction(|tree| {
                for (k, _) in &items {
                    tree.insert(k.as_bytes(), expire_at.to_string().as_bytes())
                        .map_err(|_| sled::transaction::ConflictableTransactionError::Abort(()))?;
                }
                Ok(())
            })
            .map_err(|_| anyhow::anyhow!("Sled transaction ttl failed"))?;
        Ok(())
    }

    pub async fn delete_internal(&self, key: &str) -> Result<()> {
        let _ = self.db.remove(key.as_bytes());
        let _ = self.ttl_db.remove(key.as_bytes());
        Ok(())
    }
}

fn current_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis() as u64
}

#[cfg(feature = "cache-integration")]
use browerai_multilayer_cache::strategy::{CacheLayer, LayerFuture};

#[cfg(feature = "cache-integration")]
impl<V> CacheLayer<V> for PersistentLayer<V>
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn persistent_basic_ops() {
        let mut cfg = PersistentConfig::default();
        cfg.cleanup_interval = Duration::from_millis(200);
        let layer: PersistentLayer<String> = PersistentLayer::new("rocksdb-l2", cfg).unwrap();
        layer
            .set_internal("a", "b".to_string(), Duration::from_millis(500))
            .await
            .unwrap();
        let v = layer.get_internal("a").await.unwrap();
        assert_eq!(v, Some("b".to_string()));
        sleep(Duration::from_secs(1)).await;
        let v2 = layer.get_internal("a").await.unwrap();
        assert!(v2.is_none(), "value should expire by ttl cleanup");
    }
}
