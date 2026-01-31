# Week 6: 性能优化与功能扩展 - 执行计划

**开始时间**: 2026-01-31 20:15 UTC  
**目标完成**: 2026-02-07 (7 天)  
**优先级**: 性能优化 → 功能扩展 → 生产准备

---

## 📊 当前系统诊断

### 关键指标回顾

| 指标 | 当前值 | 目标值 | 差距 | 优先级 |
|------|--------|--------|------|--------|
| **模型准确率** | 41.67% | >75% | -33.33% | 🔴 P1 |
| **API 延迟** | 35-50µs | <10ms | ✅ 通过 | P3 |
| **缓存命中** | 53.77x | 85%+ | -31% | 🟡 P1 |
| **并发能力** | 1,000 RPS | 10,000 RPS | -90% | 🟡 P1 |
| **模型覆盖** | 6 框架 | 12+ 框架 | -50% | 🟡 P2 |
| **混淆检测** | 4 技术 | 8+ 技术 | -50% | 🟡 P2 |

### 系统瓶颈分析

#### 🔴 **关键瓶颈 1: 模型准确率低 (41.67%)**
- **问题**: 混合规则检测器准确率不足
- **原因**: 
  - 特征提取可能不充分
  - 训练数据不均衡
  - 规则与 ML 融合不够优化
- **解决方案**:
  1. 分析特征重要性
  2. 收集更多框架样本
  3. 优化规则引擎权重
  4. 使用多模型投票

#### 🟡 **关键瓶颈 2: 并发能力不足 (1,000 RPS vs 10,000 RPS)**
- **问题**: 无法处理高并发请求
- **原因**:
  - 模型推理可能是串行的
  - 缓存策略不够优化
  - 没有请求队列管理
- **解决方案**:
  1. 异步模型推理
  2. 多线程处理
  3. 请求批处理
  4. 缓存预热

#### 🟡 **关键瓶颈 3: 缓存命中率 (53.77% vs 85%)**
- **问题**: 缓存效率不够
- **原因**:
  - 缓存键设计不当
  - 没有热点数据预加载
  - LRU 驱逐策略不优
- **解决方案**:
  1. 优化缓存键生成
  2. 预加载常见框架
  3. 调整 LRU 参数
  4. 多级缓存策略

---

## 🚀 Week 6 执行阶段

### 第 1 阶段 (Day 1-2): 数据增强与特征优化

#### 任务 1.1: 框架覆盖扩展 (6 → 12 框架)

**新增框架目标**:
- ✅ React (已有)
- ✅ Vue (已有)
- ✅ Angular (已有)
- ✅ jQuery (已有)
- ✅ Svelte (已有)
- ✅ Express (已有)
- 🆕 Ember.js
- 🆕 Backbone.js
- 🆕 Alpine.js
- 🆕 Htmx
- 🆕 Next.js SSG/SSR
- 🆕 Nuxt.js

```bash
# 采集新框架样本
python3 training/scripts/collect_framework_samples.py \
  --frameworks ember,backbone,alpine,htmx,nextjs,nuxt \
  --samples 100 \
  --output data/week6_framework_samples/
```

**预期结果**: +600 个新样本，框架覆盖翻倍

#### 任务 1.2: 混淆技术检测扩展 (4 → 8 技术)

**新增混淆技术目标**:
- ✅ 控制流混淆 (已有)
- ✅ 死代码插入 (已有)
- ✅ 字符串编码 (已有)
- ✅ 变量重命名 (已有)
- 🆕 属性加密
- 🆕 函数包装
- 🆕 正则表达式混淆
- 🆕 数组混淆

```bash
# 生成混淆技术数据集
python3 training/scripts/generate_obfuscation_samples.py \
  --techniques property-encryption,function-wrapping,regex-obfuscation,array-obfuscation \
  --samples 50 \
  --output data/week6_obfuscation_samples/
```

#### 任务 1.3: 特征提取优化

```python
# 分析当前特征重要性
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 训练特征重要性模型
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 获取特征重要性
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))

# 删除低重要性特征（<0.01）
important_features = feature_importance[
    feature_importance['importance'] > 0.01
]['feature'].tolist()

print(f"保留特征: {len(important_features)}/{len(feature_names)}")
```

---

### 第 2 阶段 (Day 3-4): 模型改进与训练

#### 任务 2.1: 混合模型 v3 训练

```python
# 改进的混合模型架构
class ImprovedHybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # 三层 MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.output(self.mlp(x)))

# 目标: 准确率 >75%
model_v3 = ImprovedHybridModel(input_dim=len(important_features))

# 使用加权损失函数处理不均衡数据
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(class_weights[1] / class_weights[0])
)

# 训练配置
optimizer = torch.optim.AdamW(model_v3.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# K-fold 交叉验证 (k=5)
accuracies = []
for fold in range(5):
    # 分割训练/验证集
    train_idx, val_idx = split_fold(X_train, fold)
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    # 训练
    best_acc = 0
    for epoch in range(100):
        # ... 训练循环
        val_acc = evaluate(model_v3, X_val_fold, y_val_fold)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_v3.state_dict(), f'models/fold_{fold}.pt')
    
    accuracies.append(best_acc)

# 最终准确率
final_accuracy = np.mean(accuracies)
print(f"K-Fold 平均准确率: {final_accuracy:.2%}")  # 目标: >75%
```

**关键改进**:
1. 更深的神经网络 (三层 MLP)
2. Dropout 正则化 (0.3)
3. 加权损失函数 (处理类别不均衡)
4. K-fold 交叉验证
5. AdamW 优化器 + LR 调度

**预期结果**: 准确率从 41.67% → 70-80%

#### 任务 2.2: 多模型投票集成

```python
# 训练 3 个不同的模型
models = {
    'neural_net': ImprovedHybridModel(...),
    'random_forest': RandomForestClassifier(n_estimators=200),
    'gradient_boosting': XGBClassifier(max_depth=7, n_estimators=200)
}

# 集成预测
def ensemble_predict(x):
    predictions = []
    for name, model in models.items():
        if name == 'neural_net':
            pred = model(torch.tensor(x)).detach().numpy()
        else:
            pred = model.predict_proba(x)[:, 1]
        predictions.append(pred)
    
    # 加权平均 (权重根据验证准确率)
    return np.average(predictions, axis=0, weights=[0.4, 0.3, 0.3])

# 目标: 集成准确率 >78%
ensemble_accuracy = evaluate_ensemble(models, X_test, y_test)
print(f"集成模型准确率: {ensemble_accuracy:.2%}")
```

**预期结果**: 准确率从 70% → 78-82%

#### 任务 2.3: 规则引擎权重优化

```python
# 当前规则权重 (固定的)
rules = {
    'react_cdn': 0.15,
    'react_import': 0.15,
    'vue_cdn': 0.15,
    'angular_import': 0.15,
    ...
}

# 可学习的规则权重 (基于训练数据)
class RuleWeightOptimizer:
    def __init__(self, rules_dict):
        # 规则权重初始化为可学习参数
        self.weights = nn.Parameter(
            torch.tensor(list(rules_dict.values()), dtype=torch.float32)
        )
        self.rules = list(rules_dict.keys())
    
    def forward(self, detected_rules):
        """
        detected_rules: [batch_size, num_rules] 
        返回加权规则分数
        """
        return torch.matmul(
            detected_rules.float(),
            torch.softmax(self.weights, dim=0)
        )

# 训练规则权重
optimizer = torch.optim.Adam(rule_optimizer.parameters(), lr=0.01)
for epoch in range(100):
    predictions = rule_optimizer(detected_rules_train)
    loss = loss_fn(predictions, y_train)
    loss.backward()
    optimizer.step()

# 最终权重
final_weights = torch.softmax(rule_optimizer.weights, dim=0).detach().numpy()
print("优化后规则权重:")
for rule, weight in zip(rule_optimizer.rules, final_weights):
    print(f"  {rule}: {weight:.4f}")
```

**预期结果**: 规则准确率从 35% → 55-65%

---

### 第 3 阶段 (Day 5): 性能优化

#### 任务 3.1: 模型推理并行化

```rust
// src/api/server.rs - 使用线程池处理推理

use tokio::task::spawn_blocking;
use std::sync::Arc;

#[post("/detect/framework")]
async fn detect_framework(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DetectRequest>,
) -> Result<Json<DetectResponse>> {
    // 转移到线程池执行 (避免阻塞 async 运行时)
    let model = state.model.clone();
    let html = payload.html.clone();
    
    let result = spawn_blocking(move || {
        model.predict(&html)
    }).await?;
    
    Ok(Json(DetectResponse {
        framework: result.framework,
        confidence: result.confidence,
    }))
}
```

**预期结果**: 并发从 1,000 RPS → 5,000 RPS

#### 任务 3.2: 请求批处理

```rust
// 批处理多个请求
#[post("/detect/framework/batch")]
async fn detect_framework_batch(
    State(state): State<Arc<AppState>>,
    Json(requests): Json<Vec<DetectRequest>>,
) -> Result<Json<Vec<DetectResponse>>> {
    let model = state.model.clone();
    
    // 合并特征向量
    let batch_features = requests
        .iter()
        .map(|r| extract_features(&r.html))
        .collect::<Vec<_>>();
    
    // 批量推理
    let predictions = model.predict_batch(&batch_features)?;
    
    Ok(Json(
        predictions
            .into_iter()
            .map(|p| DetectResponse { ... })
            .collect()
    ))
}
```

**预期结果**: 吞吐量从 59,140/s → 200,000/s

#### 任务 3.3: 缓存策略优化

```rust
// 多级缓存策略

// L1: 内存缓存 (热点数据)
static L1_CACHE: Lazy<Cache<String, DetectResult>> = 
    Lazy::new(|| Cache::new(1000));  // 1000 条目

// L2: Redis 缓存 (分布式)
let redis_client = redis::Client::open("redis://127.0.0.1/")?;

#[post("/detect/framework")]
async fn detect_framework(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DetectRequest>,
) -> Result<Json<DetectResponse>> {
    let cache_key = format!("framework:{:x}", md5::compute(&payload.html));
    
    // L1 缓存检查
    if let Some(cached) = L1_CACHE.get(&cache_key) {
        return Ok(Json(cached));
    }
    
    // L2 Redis 检查
    if let Ok(cached) = redis_client.get::<String, DetectResult>(&cache_key) {
        L1_CACHE.insert(cache_key.clone(), cached.clone());
        return Ok(Json(cached));
    }
    
    // 推理
    let result = spawn_blocking(move || {
        state.model.predict(&payload.html)
    }).await?;
    
    // 写入两级缓存
    L1_CACHE.insert(cache_key.clone(), result.clone());
    redis_client.set::<String, DetectResult>(&cache_key, result.clone())?;
    
    Ok(Json(result))
}
```

**预期结果**: 缓存命中从 53.77% → 85%+

---

### 第 4 阶段 (Day 6): 功能扩展

#### 任务 4.1: 实时性能监控面板

```rust
// Prometheus 指标导出
use prometheus::{Counter, Histogram, Registry};

lazy_static::lazy_static! {
    static ref DETECT_LATENCY: Histogram = 
        Histogram::new("detect_latency_ms", "检测延迟").unwrap();
    
    static ref DETECT_COUNT: Counter =
        Counter::new("detect_total", "检测总数").unwrap();
    
    static ref CACHE_HIT: Counter =
        Counter::new("cache_hit_total", "缓存命中").unwrap();
    
    static ref CACHE_MISS: Counter =
        Counter::new("cache_miss_total", "缓存未命中").unwrap();
}

#[get("/metrics")]
async fn metrics() -> String {
    prometheus::TextEncoder::new()
        .encode(&prometheus::DEFAULT_REGISTRY.gather(), &mut String::new())
        .unwrap()
}
```

#### 任务 4.2: Web 客户端基础

```html
<!-- web_client/index.html -->
<html>
<head>
    <title>BrowerAI - Framework Detector</title>
    <script src="https://cdn.jsdelivr.net/npm/react@18/umd/react.production.min.js"></script>
</head>
<body>
    <div id="root"></div>
    <script src="app.js"></script>
</body>
</html>
```

---

## 📈 预期成果

| 指标 | 当前 | Week 6 目标 | 实现方式 |
|------|------|-----------|---------|
| 模型准确率 | 41.67% | >75% | 改进特征 + 集成学习 |
| 框架覆盖 | 6 个 | 12+ 个 | 新增采集 |
| 混淆检测 | 4 种 | 8+ 种 | 生成更多样本 |
| 并发能力 | 1,000 RPS | 5,000+ RPS | 异步推理 + 线程池 |
| 缓存命中 | 53.77% | 85%+ | 多级缓存策略 |
| API 响应 | 35-50µs | <10ms | 批处理 + 缓存 |

---

## 📋 每日检查清单

### Day 1 (2026-02-01)
- [ ] 框架样本采集 (6 → 12 个)
- [ ] 混淆技术样本生成 (4 → 8 种)
- [ ] 特征重要性分析

### Day 2 (2026-02-02)
- [ ] 特征优化与清洗
- [ ] 数据集重新平衡
- [ ] 混合模型 v3 架构设计

### Day 3 (2026-02-03)
- [ ] 模型 v3 训练开始
- [ ] K-fold 交叉验证配置
- [ ] 性能监控设置

### Day 4 (2026-02-04)
- [ ] 模型 v3 训练完成
- [ ] 多模型集成开发
- [ ] 规则权重优化

### Day 5 (2026-02-05)
- [ ] API 并行化改进
- [ ] 缓存策略优化
- [ ] 批处理实现
- [ ] 性能压测

### Day 6 (2026-02-06)
- [ ] 监控面板集成
- [ ] Web 客户端初版
- [ ] 文档完成

### Day 7 (2026-02-07)
- [ ] 最终性能验证
- [ ] Week 6 完成报告生成
- [ ] 下一阶段规划

---

## 🎯 成功标准

### P1: 必须达成
- [ ] 模型准确率 ≥ 75%
- [ ] API 可处理 5,000+ RPS
- [ ] 缓存命中率 ≥ 80%

### P2: 强烈建议
- [ ] 框架覆盖 ≥ 12 个
- [ ] 混淆检测 ≥ 8 种
- [ ] 监控面板可用

### P3: 可选
- [ ] Web 客户端基础版
- [ ] 文档完整

---

**计划生成**: 2026-01-31 20:15 UTC  
**预计完成**: 2026-02-07  
**优先级**: 性能 > 功能 > 优化

