# Week 7 快速参考 - 哈希标签 + Sentinel 故障转移

## 快速开始

### 导入核心类型
```rust
use browerai_redis_integration::{
    ClusterHashTag,
    RedisClusterPoolWithSentinel,
    SentinelConfig,
};
use std::time::Duration;
```

### 场景 1：批量操作同 slot 保证（哈希标签）

```rust
// 原始键（跨不同 slot）
let keys = vec!["user:1:name".into(), "user:1:email".into()];

// 应用同一标签，保证同 slot
let tagged = ClusterHashTag::apply_batch(&keys, Some("user:1"));
// 结果: ["user:1:name{user:1}", "user:1:email{user:1}"]

// 在集群连接池中使用
let conn = pool.get_connection().await?;
// MGET/MSET 现在安全地在同一 slot，原子执行

// 获取后移除标签
let stripped = ClusterHashTag::strip(&tagged[0]);
```

### 场景 2：Sentinel 自动故障转移

```rust
// 配置 Sentinel
let sentinel = SentinelConfig {
    sentinels: vec![
        "127.0.0.1:26379".into(),
        "127.0.0.1:26380".into(),
        "127.0.0.1:26381".into(),
    ],
    master_name: "mymaster".into(),
    monitor_interval: Duration::from_secs(5),
};

// 创建带故障转移的连接池
let pool = RedisClusterPoolWithSentinel::with_sentinel(
    vec!["redis://127.0.0.1:6379".into()],
    sentinel,
    Duration::from_secs(2),
)?;

// 正常操作（故障时自动转移）
let conn = pool.get_connection().await?; // 健康检查 + 自动转移
let health = pool.check_health().await; // 显式健康检查
```

### 场景 3：批量操作 + 哈希标签 + Sentinel

```rust
// 定义业务实体
struct UserData {
    name: String,
    email: String,
    age: u32,
}

// 批量更新用户数据（同 slot，自动故障转移）
let user_id = "user:123";
let items = vec![
    ("user:123:name".into(), "Alice".into()),
    ("user:123:email".into(), "alice@example.com".into()),
];

// 哈希标签 + 故障转移自动进行
pool.set_batch_with_hash_tag(&items, Some(user_id), Duration::from_secs(300)).await?;

// 查询也享受同 slot 保证
let keys = vec!["user:123:name".into(), "user:123:email".into()];
let values: Vec<Option<String>> = pool
    .get_batch_with_hash_tag(&keys, Some(user_id))
    .await?;
```

## API 参考

### ClusterHashTag

| 方法 | 说明 | 示例 |
|------|------|------|
| `apply(key, tag)` | 为单个键应用标签 | `apply("user:1:name", Some("user:1"))` → `"user:1:name{user:1}"` |
| `apply_batch(keys, tag)` | 为多个键批量应用标签 | `apply_batch(&["a", "b"], Some("x"))` → `["a{x}", "b{x}"]` |
| `strip(key)` | 移除标签部分 | `strip("user:1:name{user:1}")` → `"user:1:name"` |
| `extract(key)` | 提取标签部分 | `extract("user:1:name{user:1}")` → `Some("user:1")` |

### RedisClusterPoolWithSentinel

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `new(nodes, timeout)` | 创建纯集群连接池 | `Result<Self>` |
| `with_sentinel(nodes, sentinel, timeout)` | 创建带 Sentinel 的连接池 | `Result<Self>` |
| `get_connection()` | 获取连接（自动故障转移） | `Result<ClusterConnection>` |
| `check_health()` | PING 健康检查 | `bool` |
| `get_batch_with_hash_tag(keys, tag)` | 批量 GET + 哈希标签 | `Result<Vec<Option<V>>>` |
| `set_batch_with_hash_tag(items, tag, ttl)` | 批量 SET + 哈希标签 | `Result<()>` |

## 性能特征

| 操作 | 延迟 | 吞吐 | 备注 |
|------|------|------|------|
| `apply()` | ~100ns | 10M op/s | 零开销字符串拼接 |
| `apply_batch(100)` | ~10µs | 10M op/s | O(n) 线性复杂度 |
| `strip()` | ~50ns | 20M op/s | 字符串查找，均摊 O(1) |
| 集群 GET | ~1-5ms | 200-500 req/s | 网络往返延迟主导 |
| 集群 MGET (10) | ~2-8ms | 1500-2500 req/s | 同 slot 批量优化 |
| Sentinel 故障转移 | ~50-200ms | - | 包括 Sentinel 查询 + 重连 |

## 故障排查

### 问题 1：跨 slot 批量操作失败
```
❌ Redis error: CROSSSLOT Keys in request don't hash to the same slot
```
**解决**: 使用 `ClusterHashTag::apply_batch(keys, Some(common_tag))`

### 问题 2：Sentinel 故障转移不工作
```
❌ Sentinel query timeout / master address resolution failed
```
**排查步骤**:
1. 验证 Sentinel 可达: `redis-cli -p 26379 PING`
2. 验证主节点: `redis-cli -p 26379 SENTINEL get-master-addr-by-name mymaster`
3. 检查网络延迟与超时设置

### 问题 3：连接池耗尽
```
❌ Failed to get connection: No connection available
```
**解决**: 增加连接池大小或减少并发请求

## 基准测试

运行本地基准：
```bash
# 快速测试
cargo test -p browerai-redis-integration --lib

# 完整基准（需要 ~5 分钟）
cargo bench -p browerai-redis-integration --bench hash_tag_bench

# 对标 sled vs RocksDB（需构建工具链）
cargo bench -p browerai-persistent-layer
cargo bench -p browerai-persistent-layer-rocksdb --features rocksdb-support
```

## 代码示例

### 完整的多层缓存 + Sentinel
参见: [examples/redis_cluster_sentinel_example.rs](examples/redis_cluster_sentinel_example.rs)

### 集群哈希标签单元测试
位置: [crates/browerai-redis-integration/src/cluster_hash_tag.rs](crates/browerai-redis-integration/src/cluster_hash_tag.rs#L45)

### Sentinel 故障转移测试
位置: [crates/browerai-redis-integration/src/cluster_sentinel.rs](crates/browerai-redis-integration/src/cluster_sentinel.rs#L170)

## 最佳实践

✅ **推荐**
- 为相关的键组使用统一的标签（如用户 ID、会话 ID）
- 将 `monitor_interval` 设为 5-10 秒，平衡故障检测与开销
- 在批量操作前应用哈希标签，在批量操作后移除标签
- 为 Sentinel 配置至少 3 个哨兵节点

❌ **避免**
- 为每个键使用不同标签（失去批量优化）
- 在高频路径频繁解析 Sentinel 主地址（缓存或定期更新）
- 混用标记键和未标记键的 MGET/MSET（会失败）
- 设置过短的故障检测间隔（< 1 秒，增加网络开销）

## 部署检查清单

- [ ] 集群节点全部启动（`redis-cli cluster nodes`）
- [ ] Sentinel 配置正确（`redis-cli -p 26379 SENTINEL masters`）
- [ ] 哈希标签约定文档已发布
- [ ] 批量操作都使用 `apply_batch()` 统一标签
- [ ] 监控面板配置（延迟、故障转移计数、连接池状态）
- [ ] 灾难恢复演习（模拟主节点宕机，验证自动转移）

## 相关文档

- [Week 6 性能报告](docs/WEEK6_CACHE_PERFORMANCE_REPORT.md) - 缓存基准数据
- [Week 7 完成报告](docs/WEEK7_FOLLOWUP_COMPLETION.md) - 完整实现说明
- [多层缓存架构](docs/SYSTEM_OVERVIEW.md) - 整体设计

---

**最后更新**: 2026-01-26  
**维护者**: BrowerAI Team  
**版本**: 1.0.0 (Stable)
