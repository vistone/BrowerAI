# Week 6 架构分析：BrowerAI 自主学习系统

## 📋 概述

BrowerAI 的核心目标是**自动访问真实网站 → 学习 → 改进解析/渲染能力**。

系统采用**两层架构**：
- **Rust 层**（主要）：负责网站访问、解析、反馈收集
- **Python 层**（机器学习）：模型训练、代码生成、质量验证

---

## 🏗️ 完整学习流程

```
[网站访问]  →  [解析分析]  →  [反馈收集]  →  [模型训练]  →  [代码生成]
    ↓            ↓            ↓             ↓             ↓
  Rust/Fetcher  Intent       Feedback      Python        Generators
                Analyzer     Collector     OnlineLearning
                
                ↑ ← ← ← ← ← ← ← ← ← 反馈循环 ← ← ← ← ← ← ↓
                
[改进模型部署] ← [质量验证] ← [测试新代码]
```

---

## 1️⃣ Rust 层：网站学习基础设施

### 1.1 核心组件

#### **A. LearningPipeline** (`crates/browerai-learning/src/pipeline/learning_pipeline.rs`)
主要的学习管道，协调整个流程。

**输入**：`LearningInput`
```rust
pub struct LearningInput {
    pub url: String,                    // 目标网站 URL
    pub preferences: Option<UserPreferences>,  // 用户偏好
    pub record_behavior: bool,          // 是否记录行为
    pub enable_validation: bool,        // 是否验证
}
```

**输出**：`LearningOutput`
```rust
pub struct LearningOutput {
    pub success: bool,
    pub output: OutputBundle,           // 生成的 HTML/CSS/JS
    pub intent: WebsiteIntent,          // 网站意图分析
    pub behavior_record: Option<BehaviorRecord>,  // 行为记录
    pub validation_report: Option<ValidationReport>,  // 验证报告
    pub metadata: LearningMetadata,     // 元数据
}
```

**执行流程**（`pub async fn run()`）：
1. **Stage 1: 获取网页** → `PageFetcher::fetch(url)`
   - 获取原始 HTML/CSS/JS 内容
   - 返回 `PageContent`

2. **Stage 2: 记录行为** → `BehaviorRecorder::record()`
   - 可选：注入检测脚本
   - 捕获用户交互、API 调用、状态变化

3. **Stage 3: 分析意图** → `IntentAnalyzer::analyze()`
   - 识别网站类型（电商、新闻、博客等）
   - 分析设计风格、页面结构、商业模式

4. **Stage 4: 生成代码** → 生成器
   - 基于意图生成 HTML/CSS/JS
   - 调用 Python 学习系统

5. **Stage 5: 验证质量** → `WebsiteValidator`
   - 可选：验证生成代码质量

6. **Stage 6: 格式化输出** → 准备最终输出

#### **B. IntentAnalyzer** (`crates/browerai-learning/src/learning_sandbox/intent_analyzer.rs`)
分析网站的目的和结构。

**输出**：`WebsiteIntent`
```rust
pub struct WebsiteIntent {
    pub website_type: String,           // "电商", "博客", "新闻", etc
    pub confidence: f32,                // 置信度 0.0-1.0
    pub core_features: Vec<String>,     // ["产品列表", "购物车", "登录"]
    pub target_audience: String,        // 目标受众
    pub design_style: DesignStyle,      // 设计分析
    pub structure: PageStructure,       // 页面结构
    pub business_model: String,         // 商业模式
    pub type_scores: HashMap<String, f32>, // 各类型置信度
}
```

**DesignStyle** - 审美特征：
- `formality`: 0.0(随意) ~ 1.0(正式)
- `colorfulness`: 0.0(单调) ~ 1.0(五彩)
- `minimalism`: 0.0(复杂) ~ 1.0(简约)
- `modernity`: 0.0(传统) ~ 1.0(现代)

**PageStructure** - 结构特征：
- `has_header`, `has_navigation`, `has_sidebar`, `has_footer`
- `layout_type`: 单列/两列/多列
- `complexity`: 简单/中等/复杂

#### **C. FeedbackCollector** (`crates/browerai-learning/src/feedback.rs`)
收集解析和渲染反馈。

**Feedback 结构**：
```rust
pub struct Feedback {
    pub id: String,
    pub feedback_type: FeedbackType,    // ParsingAccuracy, RenderingQuality, etc
    pub timestamp: u64,
    pub score: f32,                     // 0.0-1.0
    pub comment: Option<String>,        // 详细说明
    pub context: HashMap<String, String>, // 上下文元数据
    pub model_id: Option<String>,       // 哪个模型产生的
}
```

**FeedbackType**：
- `ParsingAccuracy` - 解析准确性
- `RenderingQuality` - 渲染质量
- `Performance` - 性能
- `LayoutCorrectness` - 布局正确性
- `Custom(String)` - 自定义反馈

#### **D. ContinuousLearningLoop** (`crates/browerai-learning/src/continuous_loop.rs`)
自主学习循环（learn-infer-generate 周期）。

**核心循环**（`pub fn run_iteration()`）：
```
1. 推理阶段 → 分析现有代码的混淆特征
2. 学习阶段 → 从反馈收集样本，提取特征
3. 更新阶段 → 触发模型更新（样本足够时）
4. 生成阶段 → 生成新的 HTML/CSS/JS 样本

↑ ← ← ← ← ← ← ← ← ← ← ← ← ↓
→ → → → → → 重复运行
```

**关键数据**：
```rust
pub struct ContinuousLearningConfig {
    pub learning_rate: f32,           // 学习率 (default: 0.001)
    pub batch_size: usize,            // 批大小 (default: 32)
    pub update_interval_secs: u64,    // 更新间隔 (default: 60s)
    pub auto_generate: bool,          // 自动生成样本 (default: true)
    pub max_iterations: Option<usize>,// 最大迭代次数
}
```

---

## 2️⃣ Python 层：模型训练系统

### 2.1 核心系统

**文件**：`training/pipelines/enhanced_learning_system.py` (720 行)

#### **A. 模型架构**

**1. WebsiteFeatureEncoder**
```python
# 输入：网站特征向量 (100 维)
# 处理：Embedding → 编码器 → 高维特征向量
# 输出：编码特征 (128 维)

输入 → Embedding(128维) 
    → Linear(128→256) + ReLU + Dropout
    → Linear(256→256) + ReLU + Dropout  
    → Linear(256→128)
    → 输出编码特征
```

**2. CodeGenerator**
```python
# 输入：编码特征 (128 维)
# 处理：并行生成 HTML/CSS/JS
# 输出：HTML logits + CSS logits + JS logits

特征 → Linear(128→256) + ReLU
    → 三个生成器并行
       ├→ HTML Generator → vocab_size logits
       ├→ CSS Generator  → vocab_size logits
       └→ JS Generator   → vocab_size logits
```

**3. QualityVerifier**
```python
# 输入：原始代码 + 生成代码
# 处理：编码比较 → 质量计算
# 输出：质量分数 (0.0-1.0)

原始代码 → 编码 ┐
生成代码 → 编码 ┴→ 连接 → 比较器 → Sigmoid → 质量分数
```

#### **B. OnlineLearningEngine**
在线学习，支持实时数据流的增量学习。

**核心方法**：`learn_from_sample()`
```python
def learn_from_sample(self, 
                     website_features,    # 网站特征张量
                     reference_code,      # 参考代码 {html, css, js}
                     feedback=None) → Dict:  # 反馈对象
    """
    从单个网站样本学习：
    1. 编码网站特征
    2. 生成对应代码
    3. 计算与参考代码的差异
    4. 根据反馈权重调整
    5. 反向传播更新模型
    """
```

**学习损失**（`_compute_learning_loss()`）：
```
total_loss = reconstruction_loss × feedback_weight

其中：
  reconstruction_loss = 生成代码与参考代码的差异
  feedback_weight = 1.0 + (quality_score/100) × 0.5
  
高质量反馈 → 权重更大 → 对模型的影响更大
```

#### **C. KnowledgeDistillation**
知识蒸馏：从大教师模型转移知识到小学生模型。

```python
class KnowledgeDistillation:
    """
    教师模型（大、准确） → 学生模型（小、快速）
    通过 soft targets 转移知识
    """
```

---

## 3️⃣ 数据流与集成点

### 3.1 从网站访问到模型更新

```
Step 1: 网站访问
  User Input: URL → LearningPipeline.run()
  ↓
Step 2: 内容获取与分析
  PageFetcher.fetch(url) → PageContent (HTML/CSS/JS)
    ↓
  IntentAnalyzer.analyze() → WebsiteIntent
    - website_type: "电商"
    - core_features: ["产品列表", "购物车"]
    - design_style: DesignStyle { formality: 0.7, ... }
    - structure: PageStructure { has_header: true, ... }
    ↓
Step 3: 行为记录（可选）
  BehaviorRecorder.record() → BehaviorRecord
    - 用户交互轨迹
    - API 调用
    - 状态变化
    ↓
Step 4: 特征提取
  ExtractWebsiteFeatures(PageContent) → 48-dim float vector
    - HTML 结构复杂度
    - CSS 颜色数量
    - JavaScript 函数数
    - 页面大小
    - ... 其他统计特征
    ↓
Step 5: 代码生成（Python）
  OnlineLearningEngine.learn_from_sample(
      website_features: Tensor[1, 48],      # 网站特征
      reference_code: {html, css, js},      # 原始网页代码
      feedback: LearningFeedback            # 反馈
  )
  ↓
  WebsiteFeatureEncoder(features) → encoded[1, 128]
  CodeGenerator(encoded) → (html_logits, css_logits, js_logits)
  
  计算损失：
    loss = reconstruction_loss × quality_weight
  
  反向传播：
    loss.backward()
    optimizer.step() → 更新模型参数
  ↓
Step 6: 反馈收集
  FeedbackCollector.collect(
      parsing_accuracy: 0.85,  # 解析准确率
      rendering_quality: 0.92, # 渲染质量
      comment: "布局正确，字体大小合适"
  )
  ↓
Step 7: 质量验证
  QualityVerifier(original_code, generated_code) → score
    ↓
Step 8: 模型部署
  SaveModel(encoder, generator, verifier)
    → /models/local/week6_model_v1.pt
    ↓
  → 下一次网站访问时使用改进的模型
```

### 3.2 反馈循环

```
+──────────────────────────────────────────────┐
│          周期性学习循环                       │
└──────────────────────────────────────────────┘

1. 访问网站集合 {site1, site2, site3, ...}
   
2. 对每个网站：
   ├─ 用当前模型生成代码
   ├─ 与原始网页对比，收集反馈
   │   - 布局正确性：是否布局相同
   │   - 样式准确性：颜色、字体、间距
   │   - 功能完整性：是否包含必要功能
   └─ 反馈分数：0.0(失败) ~ 1.0(完美)

3. 批量反馈 (feedback_batch)
   ├─ 筛选高质量反馈 (score > 0.7)
   ├─ 标记低质量反馈 (score < 0.4)
   └─ 计算平均改进率

4. 模型更新
   ├─ OnlineLearningEngine.learn_from_sample()
   ├─ 每个样本：loss.backward() → optimizer.step()
   └─ 累积学习：参数逐步改进

5. 验证改进
   ├─ QualityVerifier 评分
   ├─ 与上一版本对比
   └─ 如果改进 > 阈值：部署新模型

6. 继续下一轮
   └─ 用改进后的模型访问新网站
```

---

## 4️⃣ 现有基础设施对接点

### 4.1 什么已经存在

| 组件 | 文件 | 功能 | 状态 |
|------|------|------|------|
| **LearningPipeline** | `crates/browerai-learning/src/pipeline/` | 网站→解析→生成 | ✅ 完成 |
| **IntentAnalyzer** | `learning_sandbox/intent_analyzer.rs` | 网站意图分析 | ✅ 完成 |
| **FeedbackCollector** | `feedback.rs` | 收集反馈 | ✅ 完成 |
| **ContinuousLearningLoop** | `continuous_loop.rs` | 自主学习循环 | ✅ 完成 |
| **OnlineLearningEngine** | `enhanced_learning_system.py` | 模型训练 | ✅ 完成 |
| **CodeGenerator** | `enhanced_learning_system.py` | 生成 HTML/CSS/JS | ✅ 完成 |
| **QualityVerifier** | `enhanced_learning_system.py` | 质量验证 | ✅ 完成 |

### 4.2 Week 6 应该做的

实现**数据流打通**：

```
LearningPipeline (Rust)
    ↓
[获取网站] → PageFetcher
    ↓
[分析意图] → IntentAnalyzer → WebsiteIntent
    ↓
[提取特征] → FeatureExtractor → Vec<f32>
    ↓
[发送到 Python] ← ← ← ← ← ← IPC/HTTP
    ↓
OnlineLearningEngine (Python)
    ↓
[编码特征] → WebsiteFeatureEncoder
[生成代码] → CodeGenerator
[计算损失] → loss.backward()
[更新模型] → optimizer.step()
    ↓
[反馈评分] → QualityVerifier
    ↓
[保存模型] → /models/local/
    ↓
[下一次迭代] ← ← ← ← ← ← 循环
```

### 4.3 缺失的中间件

需要实现：

1. **WebsiteFeatureExtractor** (Rust)
   - 输入：`PageContent`
   - 输出：`Vec<f32>` (48 维特征向量)
   - 提取：HTML 结构、CSS 复杂度、JS 特征等

2. **Rust↔Python 通信**
   - 序列化网站特征为 JSON/Binary
   - 调用 Python 学习引擎
   - 接收模型更新

3. **反馈收集器**
   - 比较原始代码与生成代码
   - 计算相似度分数
   - 收集质量反馈

---

## 5️⃣ 数据格式规范

### 5.1 网站特征向量 (48 维)

```python
features = [
    # HTML 结构特征 (10)
    html_lines,              # HTML 行数
    html_size,               # HTML 大小 (KB)
    tag_count,               # 总标签数
    div_count,               # <div> 数量
    class_count,             # 类名数量
    
    # CSS 特征 (8)
    css_size,                # CSS 大小 (KB)
    css_rules,               # CSS 规则数
    color_count,             # 颜色数量
    font_families,           # 字体族数
    
    # JavaScript 特征 (10)
    js_size,                 # JS 大小 (KB)
    js_lines,                # JS 行数
    function_count,          # 函数数
    class_count_js,          # 类数
    variable_count,          # 变量数
    
    # 页面结构特征 (8)
    has_header,              # 有无头部
    has_footer,              # 有无底部
    has_nav,                 # 有无导航
    has_sidebar,             # 有无侧边栏
    section_count,           # 节区数
    form_count,              # 表单数
    link_count,              # 链接数
    image_count,             # 图片数
    
    # 复杂度指标 (5)
    nesting_depth,           # 最大嵌套深度
    css_specificity,         # CSS 选择器特异性
    js_complexity,           # JS 复杂度分数
    page_size_total,         # 页面总大小 (KB)
    media_queries,           # Media Query 数量
    
    # 可访问性和性能 (7)
    aria_labels,             # ARIA 标签数
    semantic_tags,           # 语义标签数
    responsive_meta,         # 响应式标签
    performance_score,       # 初步性能分数
    ...
]
```

### 5.2 网站意图结构 (JSON)

```json
{
    "website_type": "e-commerce",
    "confidence": 0.92,
    "core_features": [
        "product_listing",
        "shopping_cart",
        "user_login",
        "search"
    ],
    "target_audience": "online shoppers",
    "design_style": {
        "formality": 0.7,
        "colorfulness": 0.6,
        "minimalism": 0.3,
        "modernity": 0.8,
        "primary_colors": ["#FF6B35", "#004E89"],
        "layout_type": "grid"
    },
    "structure": {
        "has_header": true,
        "has_footer": true,
        "layout_type": "multi-column",
        "section_count": 5,
        "complexity": "medium"
    },
    "business_model": "direct_sales",
    "type_scores": {
        "e-commerce": 0.92,
        "blog": 0.05,
        "news": 0.02
    }
}
```

### 5.3 学习反馈结构

```json
{
    "id": "fb_1234567890",
    "feedback_type": "ParsingAccuracy",
    "timestamp": 1234567890,
    "score": 0.87,
    "comment": "布局正确，颜色准确，但字体大小偏小",
    "context": {
        "website_url": "https://example.com",
        "model_id": "week6_v1",
        "device_type": "desktop",
        "browser": "Chrome"
    }
}
```

---

## 🎯 总结：Week 6 的任务

### 现状
- ✅ Rust 学习基础设施完整（访问、分析、反馈）
- ✅ Python 模型系统完整（编码、生成、验证）
- ❌ 两层之间没有集成

### Week 6 需要做的
1. **特征提取器** - PageContent → 48-dim vector
2. **Rust↔Python 通信** - 序列化、IPC/HTTP、异步调用
3. **反馈收集** - 比较原始代码与生成代码，计算分数
4. **端到端测试** - 从 URL 到模型更新的完整流程

### 数据流
```
URL → Fetcher → Parser → IntentAnalyzer 
            ↓
        特征提取 (48-dim)
            ↓
        [IPC/HTTP]
            ↓
    OnlineLearningEngine
            ↓
        特征编码 → 代码生成 → 损失计算 → 模型更新
            ↓
        质量验证
            ↓
        保存模型 → 部署
```

---

## 📚 关键文件清单

**Rust 核心**：
- `crates/browerai-learning/src/pipeline/learning_pipeline.rs` - 主管道
- `crates/browerai-learning/src/learning_sandbox/intent_analyzer.rs` - 意图分析
- `crates/browerai-learning/src/feedback.rs` - 反馈收集
- `crates/browerai-learning/src/continuous_loop.rs` - 自主循环

**Python 核心**：
- `training/pipelines/enhanced_learning_system.py` - 学习引擎
  - WebsiteFeatureEncoder
  - CodeGenerator
  - QualityVerifier
  - OnlineLearningEngine

**待开发**：
- `WeekFeatureExtractor` - 特征提取
- `RustPythonBridge` - Rust-Python 通信
- `IntegrationTests` - 集成测试

