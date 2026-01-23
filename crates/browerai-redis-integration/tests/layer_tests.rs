use std::collections::HashMap;
use std::time::Duration;

use browerai_redis_integration::{RedisConfig, RedisLayer};

#[tokio::test]
async fn redis_layer_basic_operations() {
    let _ = tracing_subscriber::fmt::try_init();

    let config = RedisConfig {
        url: "redis://127.0.0.1:6379".to_string(),
        ..Default::default()
    };

    // 如果 Redis 不可用，跳过测试
    let layer: RedisLayer<String> = match RedisLayer::new("test-redis", config) {
        Ok(l) => l,
        Err(_) => {
            println!("Redis unavailable, skipping test");
            return;
        }
    };

    if !layer.is_healthy().await {
        println!("Redis not healthy, skipping test");
        return;
    }

    // 单键操作
    layer
        .set_internal("hello", "world".to_string(), Duration::from_secs(60))
        .await
        .unwrap();

    let got = layer.get_internal("hello").await.unwrap();
    assert_eq!(got.as_deref(), Some("world"));

    // 批量操作
    let mut batch = HashMap::new();
    batch.insert("k1".to_string(), "v1".to_string());
    batch.insert("k2".to_string(), "v2".to_string());

    layer
        .set_batch_internal(batch, Duration::from_secs(60))
        .await
        .unwrap();

    let keys = vec!["k1".to_string(), "k2".to_string(), "k3".to_string()];
    let result = layer.get_batch_internal(&keys).await.unwrap();

    assert_eq!(result.get("k1").map(String::as_str), Some("v1"));
    assert_eq!(result.get("k2").map(String::as_str), Some("v2"));
    assert!(!result.contains_key("k3"));

    // 删除
    layer.delete_internal("hello").await.unwrap();
    let deleted = layer.get_internal("hello").await.unwrap();
    assert_eq!(deleted, None);
}

#[tokio::test]
async fn redis_layer_graceful_degradation() {
    let _ = tracing_subscriber::fmt::try_init();

    // 故意使用错误的 URL 测试降级
    let config = RedisConfig {
        url: "redis://127.0.0.1:9999".to_string(),
        connection_timeout: Duration::from_millis(100),
        operation_timeout: Duration::from_millis(100),
        ..Default::default()
    };

    let layer: RedisLayer<String> = RedisLayer::new("degradation-test", config).unwrap();

    // 降级模式下，get 应该返回 None 而不是 panic
    let got = layer.get_internal("any").await.unwrap();
    assert_eq!(got, None);

    // 降级模式下，set 应该静默成功
    layer
        .set_internal("test", "value".to_string(), Duration::from_secs(10))
        .await
        .unwrap();
}
