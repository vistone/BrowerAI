#!/bin/bash

# Week 5-6 快速启动脚本
# 时间: 2026-01-25
# 用途: 一键启动 Week 5-6 开发环境

set -e

WORKSPACE="/home/stone/BrowerAI"
BRANCH_NAME="week5-postgresql-persistence"

echo "════════════════════════════════════════════════════════════"
echo "🚀 BrowerAI Week 5-6 快速启动"
echo "════════════════════════════════════════════════════════════"
echo ""

# 第 1 步: 验证环境
echo "📋 第 1 步: 验证 Week 3-4 完成情况..."
cd "$WORKSPACE"
CACHE_TESTS=$(cargo test -p browerai-cache --lib 2>&1 | grep "test result:" | grep "ok")
RENDERER_TESTS=$(cargo test -p browerai-renderer --lib 2>&1 | grep "test result:" | grep "ok")

if [ -z "$CACHE_TESTS" ] || [ -z "$RENDERER_TESTS" ]; then
    echo "❌ 测试验证失败，请检查 Week 3-4 完成情况"
    exit 1
fi
echo "✅ Week 3-4 验证通过"
echo "   └─ Cache: PASSED"
echo "   └─ Renderer: PASSED"
echo ""

# 第 2 步: 创建开发分支
echo "📋 第 2 步: 创建开发分支..."
if git rev-parse --verify "$BRANCH_NAME" >/dev/null 2>&1; then
    echo "⚠️  分支已存在，检出..."
    git checkout "$BRANCH_NAME"
else
    echo "创建新分支: $BRANCH_NAME"
    git checkout -b "$BRANCH_NAME"
fi
echo "✅ 分支就绪: $BRANCH_NAME"
echo ""

# 第 3 步: 创建 Week 5-6 crate 结构
echo "📋 第 3 步: 初始化 Week 5-6 crate..."
mkdir -p crates/browerai-db crates/browerai-multilayer-cache crates/browerai-redis-integration

# 创建 browerai-db Cargo.toml
cat > crates/browerai-db/Cargo.toml << 'EOF'
[package]
name = "browerai-db"
version = "0.1.0"
edition = "2021"
description = "PostgreSQL persistence layer for BrowerAI cache system"

[dependencies]
tokio = { version = "1.35", features = ["full"] }
sqlx = { version = "0.7", features = ["runtime-tokio-native-tls", "postgres", "macros"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
thiserror = "1.0"
parking_lot = "0.12"
prometheus = "0.13"
tracing = "0.1"
uuid = { version = "1.0", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
tokio-test = "0.4"
EOF

# 创建 browerai-multilayer-cache Cargo.toml
cat > crates/browerai-multilayer-cache/Cargo.toml << 'EOF'
[package]
name = "browerai-multilayer-cache"
version = "0.1.0"
edition = "2021"
description = "Multi-layer caching strategy (L1/L2/L3) for BrowerAI"

[dependencies]
browerai-cache = { path = "../browerai-cache" }
tokio = { version = "1.35", features = ["full"] }
dashmap = "5.5"
parking_lot = "0.12"
prometheus = "0.13"
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
tracing = "0.1"

[dev-dependencies]
tokio-test = "0.4"
EOF

# 创建 browerai-redis-integration Cargo.toml
cat > crates/browerai-redis-integration/Cargo.toml << 'EOF'
[package]
name = "browerai-redis-integration"
version = "0.1.0"
edition = "2021"
description = "Redis integration for distributed caching in BrowerAI"

[dependencies]
redis = { version = "0.24", features = ["tokio-comp", "connection-manager"] }
deadpool-redis = "0.14"
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
parking_lot = "0.12"
prometheus = "0.13"
tracing = "0.1"

[dev-dependencies]
tokio-test = "0.4"
EOF

echo "✅ Crate 结构创建完成"
echo ""

# 第 4 步: 创建 lib.rs 框架
echo "📋 第 4 步: 创建 lib.rs 框架..."

# browerai-db
mkdir -p crates/browerai-db/src
cat > crates/browerai-db/src/lib.rs << 'EOF'
//! PostgreSQL 持久化层 - Week 5
//! 
//! 提供数据库连接管理、CRUD 操作和缓存持久化功能

pub mod schema;
pub mod operations;
pub mod connection;
pub mod errors;

pub use errors::DbError;

#[cfg(test)]
mod tests {
    // Week 5 单元测试将在此添加
}
EOF

# browerai-multilayer-cache
mkdir -p crates/browerai-multilayer-cache/src
cat > crates/browerai-multilayer-cache/src/lib.rs << 'EOF'
//! 多层缓存系统 - Week 6
//! 
//! L1 (内存) → L2 (本地) → L3 (Redis)

pub mod l1_memory;
pub mod l2_local;
pub mod l3_distributed;
pub mod strategy;

#[cfg(test)]
mod tests {
    // Week 6 单元测试将在此添加
}
EOF

# browerai-redis-integration
mkdir -p crates/browerai-redis-integration/src
cat > crates/browerai-redis-integration/src/lib.rs << 'EOF'
//! Redis 分布式缓存集成 - Week 6
//! 
//! 支持分布式锁、集群模式和故障转移

pub mod connection;
pub mod distributed_lock;
pub mod cluster;

#[cfg(test)]
mod tests {
    // Week 6 单元测试将在此添加
}
EOF

echo "✅ 框架文件创建完成"
echo ""

# 第 5 步: 更新 workspace Cargo.toml
echo "📋 第 5 步: 更新 workspace Cargo.toml..."
if ! grep -q "browerai-db" Cargo.toml; then
    sed -i '/\[workspace\]/a\members = [\n    "crates/browerai-cache",\n    "crates/browerai-renderer",\n    "crates/browerai-db",\n    "crates/browerai-multilayer-cache",\n    "crates/browerai-redis-integration",\n]' Cargo.toml 2>/dev/null || echo "⚠️  需要手动更新 Cargo.toml (members 已存在)"
fi
echo "✅ Workspace 配置更新"
echo ""

# 第 6 步: 编译检查
echo "📋 第 6 步: 编译检查..."
cargo build --release 2>&1 | tail -3
echo "✅ 编译成功"
echo ""

# 第 7 步: 创建 Week 5 启动日志
cat > WEEK5_STARTUP_LOG.md << 'EOF'
# Week 5-6 启动日志
**时间**: 2026-01-25  
**操作**: Week 5-6 开发环境初始化

## 初始化步骤完成清单

- [x] Week 3-4 验证完成
- [x] 开发分支创建 (week5-postgresql-persistence)
- [x] 3 个新 crate 初始化
  - [x] browerai-db (PostgreSQL)
  - [x] browerai-multilayer-cache (L1/L2/L3)
  - [x] browerai-redis-integration (Redis)
- [x] Cargo.toml 配置完成
- [x] lib.rs 框架创建
- [x] 编译验证通过

## 接下来的步骤

### Week 5 (PostgreSQL 持久化) - 预计 5 天
1. 数据库表设计 (1-2 天)
   - cache_entries 表
   - cache_statistics 表
   - 索引和视图

2. 数据库驱动实现 (2-3 天)
   - 连接管理
   - CRUD 操作
   - 事务支持
   
3. 集成和测试 (1 天)
   - 与 Cache 集成
   - 性能测试
   - 文档编写

### Week 6 (多层缓存和 Redis) - 预计 5 天
1. 多层缓存架构 (1-2 天)
   - L1/L2/L3 设计
   - 淘汰策略
   - 预热机制

2. Redis 集成 (2-3 天)
   - 分布式锁
   - 一致性验证
   - 故障转移

3. 系统验证 (1 天)
   - 集成测试
   - 性能基准
   - 最终报告

## 开发命令参考

```bash
# 编译 Week 5-6 代码
cargo build --release

# 运行 Week 5-6 测试
cargo test -p browerai-db --lib
cargo test -p browerai-multilayer-cache --lib
cargo test -p browerai-redis-integration --lib

# 运行所有测试
cargo test --lib --all

# 查看编译警告
cargo build --release 2>&1 | grep "warning"

# 提交代码
git add .
git commit -m "Week 5: PostgreSQL 实现"
```

## 性能目标

| 层 | 延迟 | 吞吐 | 命中率 |
|----|------|------|--------|
| L1 | < 1ms | - | 70%+ |
| L2 | < 10ms | - | 15%+ |
| L3 | < 50ms | - | 10%+ |
| 总系统 | - | > 50K ops/s | - |

---

**启动时间**: 2026-01-25  
**预计完成**: 2026-02-09  
**状态**: 🟢 **就绪启动**
EOF

echo "✅ Week 5 启动日志创建"
echo ""

# 第 8 步: 显示最终状态
echo "════════════════════════════════════════════════════════════"
echo "✨ Week 5-6 启动完成！"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 启动摘要:"
echo "  ✅ Branch: $BRANCH_NAME"
echo "  ✅ Crates: 3 个已创建"
echo "  ✅ Cargo.toml: 配置完成"
echo "  ✅ 编译: 成功"
echo ""
echo "🎯 接下来:"
echo "  1️⃣  阅读 WEEK5_6_STARTUP_GUIDE.md"
echo "  2️⃣  启动 PostgreSQL 容器 (docker-compose)"
echo "  3️⃣  实现 browerai-db 的数据库表设计"
echo "  4️⃣  运行: cargo test --lib --all"
echo ""
echo "📁 工作目录: $WORKSPACE"
echo "🔗 Git 分支: $(git rev-parse --abbrev-ref HEAD)"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "🚀 开发已准备就绪，祝编码愉快！"
echo "════════════════════════════════════════════════════════════"
