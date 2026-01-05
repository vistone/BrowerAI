# BrowerAI - AI驱动浏览器快速参考

## 一分钟快速开始

```rust
use browerai::{ai::*, SeamlessBrowser};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 初始化
    let model_dir = std::path::PathBuf::from("./models/local");
    let model_manager = ModelManager::new(model_dir)?;
    let perf_monitor = performance_monitor::PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor)?;
    let ai_runtime = Arc::new(AiRuntime::with_models(inference_engine, model_manager));
    
    // 创建浏览器并启动学习
    let mut browser = SeamlessBrowser::new(ai_runtime);
    browser.start_learning()?;
    
    // 访问网站（自动AI增强）
    let result = browser.navigate("https://example.com").await?;
    
    println!("AI Enhanced: {}", result.ai_enhanced);
    println!("Functionality Verified: {}", result.functionality_verified);
    
    Ok(())
}
```

## 命令行快速运行

```bash
# 运行完整演示
cargo run --example autonomous_browser_demo

# 运行测试
cargo test --lib

# 构建项目
cargo build --release
```

## 核心组件

### SeamlessBrowser（无感浏览器）

```rust
// 创建
let mut browser = SeamlessBrowser::new(ai_runtime);

// 导航
let result = browser.navigate("https://example.com").await?;

// 浏览器操作
browser.go_back();
browser.go_forward();
browser.refresh().await?;

// 学习控制
browser.start_learning()?;
browser.stop_learning()?;

// 统计信息
let stats = browser.get_session_stats();
```

### AutonomousCoordinator（自主协调器）

```rust
// 配置
let config = AutonomousConfig {
    learning_mode: LearningMode::Transparent,
    preservation_strategy: PreservationStrategy::Strict,
    optimization_threshold: 0.7,
    ..Default::default()
};

// 创建
let coordinator = AutonomousCoordinator::new(config, ai_runtime);

// 处理网站
let result = coordinator.process_website(url, html).await?;
```

## 配置选项

### 学习模式

```rust
LearningMode::Transparent    // 透明学习，用户无感
LearningMode::Background     // 后台学习
LearningMode::Explicit       // 显式显示学习过程
```

### 保持策略

```rust
PreservationStrategy::Strict              // 100%保持原始功能
PreservationStrategy::Intelligent         // AI判断关键功能
PreservationStrategy::OptimizationFirst   // 优化优先
```

### 用户偏好

```rust
let preferences = UserPreferences {
    enable_ai_features: true,
    performance_priority: true,
    accessibility_priority: false,
    custom_styles: HashMap::new(),
};
browser.set_user_preferences(preferences);
```

## 处理阶段

网站访问经历5个阶段：

1. **Learning** - 学习网站结构
2. **Reasoning** - 推理优化方案
3. **Generation** - 生成增强版本
4. **Validation** - 验证功能完整性
5. **Rendering** - 渲染最终结果

```rust
// 检查完成的阶段
for phase in result.phases_completed {
    println!("Completed: {:?}", phase);
}
```

## 统计信息

```rust
let stats = browser.get_session_stats();

// 会话统计
stats.pages_visited
stats.ai_enhancements_applied

// AI协调器统计
stats.coordinator_stats.total_sites_processed
stats.coordinator_stats.functionality_validations_passed
stats.coordinator_stats.total_patterns_learned
stats.coordinator_stats.avg_performance_improvement
```

## 使用场景

### 场景1: 生产环境

```rust
let config = AutonomousConfig {
    learning_mode: LearningMode::Transparent,
    preservation_strategy: PreservationStrategy::Strict,
    optimization_threshold: 0.8,
    ..Default::default()
};
```

### 场景2: 开发环境

```rust
let config = AutonomousConfig {
    learning_mode: LearningMode::Explicit,
    preservation_strategy: PreservationStrategy::Intelligent,
    ..Default::default()
};

env_logger::Builder::from_default_env()
    .filter_level(log::LevelFilter::Debug)
    .init();
```

### 场景3: 性能优化

```rust
let config = AutonomousConfig {
    learning_mode: LearningMode::Background,
    preservation_strategy: PreservationStrategy::OptimizationFirst,
    optimization_threshold: 0.5,
    ..Default::default()
};

let preferences = UserPreferences {
    performance_priority: true,
    ..Default::default()
};
```

## 错误处理

```rust
match browser.navigate(url).await {
    Ok(result) => {
        if result.functionality_verified {
            // 成功
        } else {
            // 降级版本
            log::warn!("Using fallback");
        }
    }
    Err(e) => {
        log::error!("Error: {}", e);
    }
}
```

## 性能调优

```rust
// 限制并发学习
let config = AutonomousConfig {
    max_concurrent_learning: 2,
    ..Default::default()
};

// 调整优化阈值
let config = AutonomousConfig {
    optimization_threshold: 0.8,  // 只有显著改进才应用
    ..Default::default()
};
```

## 常用命令

```bash
# 构建
cargo build

# 测试
cargo test --lib

# 运行演示
cargo run --example autonomous_browser_demo

# 查看AI报告
cargo run -- --ai-report

# 学习模式
cargo run -- --learn https://example.com

# 发布构建
cargo build --release
```

## 核心API

### PageRenderResult

```rust
result.html                    // 渲染的HTML
result.ai_enhanced             // 是否AI增强
result.render_time_ms          // 渲染时间
result.functionality_verified  // 功能验证结果
```

### AutonomousResult

```rust
result.original_html           // 原始HTML
result.enhanced_html           // 增强版本（Option）
result.ai_enhanced             // 是否增强
result.phases_completed        // 完成的阶段
result.functionality_preserved // 功能是否保持
result.learned_patterns        // 学习的模式
```

## 日志级别

```rust
// 详细日志
env_logger::Builder::from_default_env()
    .filter_level(log::LevelFilter::Debug)
    .init();

// 正常日志
env_logger::Builder::from_default_env()
    .filter_level(log::LevelFilter::Info)
    .init();
```

## 文档链接

- [架构文档](AI_DRIVEN_BROWSER_ARCHITECTURE.md)
- [使用指南](AI_DRIVEN_BROWSER_USAGE.md)
- [实现总结](IMPLEMENTATION_SUMMARY.md)
- [完整文档](zh-CN/README.md)

## 测试

```bash
# 所有测试
cargo test --lib

# 特定测试
cargo test seamless_browser

# 带输出
cargo test -- --nocapture

# 单个测试
cargo test test_navigate_basic
```

## 帮助

遇到问题？

1. 查看[使用指南](AI_DRIVEN_BROWSER_USAGE.md)的故障排除部分
2. 运行演示查看完整示例：`cargo run --example autonomous_browser_demo`
3. 启用DEBUG日志查看详细信息
4. 检查统计信息定位问题

## 核心特性一览

✅ **自主学习** - 从网站自动学习  
✅ **智能推理** - 理解和分析  
✅ **代码生成** - 智能优化  
✅ **无感体验** - 完全透明  
✅ **功能保持** - 确保兼容  

---

**更多信息**: 查看 [完整文档](zh-CN/README.md)
