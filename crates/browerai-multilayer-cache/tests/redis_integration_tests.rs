use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use browerai_multilayer_cache::{CacheLayer, MemoryLayer, MultiLayerCache};
use browerai_redis_integration::{RedisConfig, RedisLayer};

#[tokio::test]
async fn multi_layer_with_redis_l2() {
    let _ = tracing_subscriber::fmt::try_init();

    let l1: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l1-memory"));

    // Redis 作为 L2，降级模式
    let redis_config = RedisConfig {
        url: "redis://127.0.0.1:6379".to_string(),
        ..Default::default()
    };

    let redis_layer_result = RedisLayer::new("l2-redis", redis_config);
    let l2: Arc<dyn CacheLayer<String>> = match redis_layer_result {
        Ok(layer) => Arc::new(layer),
        Err(_) => {
            println!("Redis unavailable, using memory fallback for L2");
            Arc::new(MemoryLayer::new("l2-fallback"))
        }
    };

    let cache =
        MultiLayerCache::new(vec![l1.clone(), l2.clone()], Duration::from_secs(30)).unwrap();

    // 写入并读取
    cache.set("test", "value".to_string()).await.unwrap();
    let got = cache.get("test").await.unwrap();
    assert_eq!(got.as_deref(), Some("value"));

    // 批量操作
    let mut batch = HashMap::new();
    batch.insert("a".to_string(), "1".to_string());
    batch.insert("b".to_string(), "2".to_string());

    cache.set_batch(batch).await.unwrap();

    let keys = vec!["a".to_string(), "b".to_string()];
    let results = cache.get_batch(&keys).await.unwrap();

    assert_eq!(results.get("a").map(String::as_str), Some("1"));
    assert_eq!(results.get("b").map(String::as_str), Some("2"));

    println!("Multi-layer cache with Redis L2 test passed");
}

#[tokio::test]
async fn redis_degradation_fallback_to_l1() {
    let _ = tracing_subscriber::fmt::try_init();

    let l1: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l1-memory"));

    // 故意使用不可用的 Redis
    let bad_config = RedisConfig {
        url: "redis://127.0.0.1:9999".to_string(),
        connection_timeout: Duration::from_millis(100),
        operation_timeout: Duration::from_millis(100),
        ..Default::default()
    };

    let l2: Arc<dyn CacheLayer<String>> =
        Arc::new(RedisLayer::new("l2-broken", bad_config).unwrap());

    let cache =
        MultiLayerCache::new(vec![l1.clone(), l2.clone()], Duration::from_secs(30)).unwrap();

    // 写入（L1 成功，L2 降级静默失败）
    cache.set("key", "value".to_string()).await.unwrap();

    // 读取应从 L1 命中
    let got = cache.get("key").await.unwrap();
    assert_eq!(got.as_deref(), Some("value"));

    println!("Redis degradation with L1 fallback test passed");
}
