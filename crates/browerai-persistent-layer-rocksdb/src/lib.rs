//! RocksDB 持久化层 - 生产性能优化版
//!
//! 此模块在生产环境安装完整构建工具链（CMake、LLVM）后使用，
//! 相比纯 Rust 的 sled 提供更高的写入吞吐与空间效率。
//!
//! 构建需求：
//! - Ubuntu/Debian: apt-get install cmake clang llvm
//! - macOS: brew install cmake llvm
//! - RHEL/CentOS: yum install cmake clang llvm-devel

#[cfg(feature = "rocksdb-support")]
pub mod rocksdb_layer {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use anyhow::{Context, Result};
    use parking_lot::Mutex;
    use rocksdb::{Options, DB, WriteBatch};
    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use tokio::task::JoinHandle;
    use tokio::time::sleep;
    use tracing::{debug, warn};

    /// RocksDB 持久化层配置。
    #[derive(Clone, Debug)]
    pub struct RocksDBConfig {
        pub data_path: PathBuf,
        pub ttl_path: PathBuf,
        pub cleanup_interval: Duration,
        pub cache_size_mb: usize,
    }

    impl Default for RocksDBConfig {
        fn default() -> Self {
            Self {
                data_path: PathBuf::from("/tmp/browerai_rocksdb/data"),
                ttl_path: PathBuf::from("/tmp/browerai_rocksdb/ttl"),
                cleanup_interval: Duration::from_secs(10),
                cache_size_mb: 512,
            }
        }
    }

    /// 基于 RocksDB 的持久化缓存层，支持 TTL 索引与批量写入。
    pub struct RocksDBLayer<V>
    where
        V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
    {
        db: Arc<DB>,
        ttl_db: Arc<DB>,
        name: String,
        cleaner: Mutex<Option<JoinHandle<()>>>,
        _phantom: std::marker::PhantomData<V>,
    }

    impl<V> RocksDBLayer<V>
    where
        V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
    {
        pub fn new(name: impl Into<String>, cfg: RocksDBConfig) -> Result<Self> {
            let mut opts = Options::default();
            opts.create_if_missing(true);

            // RocksDB 优化配置
            // - 块缓存：提升随机读性能
            // - 写缓冲区：批量写入合并
            // - 压缩：减少磁盘占用
            opts.set_block_cache_size_mb(cfg.cache_size_mb as u64);
            opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB

            let db = DB::open(&opts, &cfg.data_path).context("Failed to open rocksdb data")?;
            let ttl_db = DB::open(&opts, &cfg.ttl_path).context("Failed to open rocksdb ttl")?;

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
                    let now_ms = current_millis();
                    let mut to_delete = Vec::new();

                    // 扫描 TTL 索引，收集过期键
                    let iter = ttl_db.iterator(rocksdb::IteratorMode::Start);
                    for (key, val) in iter {
                        if let Ok(ts_str) = std::str::from_utf8(&val) {
                            if let Ok(ts) = ts_str.parse::<u64>() {
                                if ts <= now_ms {
                                    to_delete.push(key.to_vec());
                                }
                            }
                        }
                    }

                    // 删除过期键
                    for k in &to_delete {
                        let _ = ttl_db.delete(k);
                        let _ = db.delete(k);
                    }

                    debug!(layer = %name, deleted = to_delete.len(), "RocksDB TTL cleanup run");
                }
            });

            *self.cleaner.lock() = Some(handle);
        }

        pub async fn get_internal(&self, key: &str) -> Result<Option<V>> {
            match self.db.get(key.as_bytes()) {
                Ok(Some(bytes)) => {
                    let json = String::from_utf8(bytes).context("Invalid UTF-8 in rocksdb value")?;
                    let v: V = serde_json::from_str(&json).context("Failed to deserialize value")?;
                    Ok(Some(v))
                }
                Ok(None) => Ok(None),
                Err(e) => {
                    warn!(layer = %self.name, %e, "RocksDB get failed");
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
            self.db.put(key.as_bytes(), json.as_bytes()).context("RocksDB put failed")?;
            let expire_at = current_millis() + ttl.as_millis() as u64;
            self.ttl_db.put(key.as_bytes(), expire_at.to_string().as_bytes()).ok();
            Ok(())
        }

        pub async fn set_batch_internal(&self, items: HashMap<String, V>, ttl: Duration) -> Result<()> {
            let expire_at = current_millis() + ttl.as_millis() as u64;
            let mut wb = WriteBatch::default();
            let mut wb_ttl = WriteBatch::default();

            for (k, v) in items {
                let json = serde_json::to_string(&v).context("Failed to serialize batch value")?;
                wb.put(k.as_bytes(), json.as_bytes());
                wb_ttl.put(k.as_bytes(), expire_at.to_string().as_bytes());
            }

            self.db.write(wb).context("RocksDB WriteBatch data failed")?;
            self.ttl_db.write(wb_ttl).context("RocksDB WriteBatch ttl failed")?;
            Ok(())
        }

        pub async fn delete_internal(&self, key: &str) -> Result<()> {
            let _ = self.db.delete(key.as_bytes());
            let _ = self.ttl_db.delete(key.as_bytes());
            Ok(())
        }
    }

    #[cfg(feature = "cache-integration")]
    use browerai_multilayer_cache::strategy::{CacheLayer, LayerFuture};

    #[cfg(feature = "cache-integration")]
    impl<V> CacheLayer<V> for RocksDBLayer<V>
    where
        V: Clone + Serialize + DeserializeOwned + Send + Sync + 'static,
    {
        fn name(&self) -> &str { &self.name }
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

    fn current_millis() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or(Duration::from_secs(0)).as_millis() as u64
    }
}

// 在未启用 rocksdb-support feature 时，提供编译友好的占位符模块
#[cfg(not(feature = "rocksdb-support"))]
pub mod rocksdb_layer {
    compile_error!("Enable 'rocksdb-support' feature and install build tools (CMake, LLVM) to use RocksDB layer");
}

pub use rocksdb_layer::*;
