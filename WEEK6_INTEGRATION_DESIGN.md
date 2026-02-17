# Week 6 集成设计：从网站学习到模型改进

## 🔗 核心集成架构

### 数据流架构图

```
┌─────────────────── Rust 层 ────────────────────────────────────────┐
│                                                                      │
│  [1. 网站访问]                                                      │
│       ↓                                                              │
│  LearningPipeline::run(url)                                         │
│       ├─→ PageFetcher::fetch(url) → PageContent                    │
│       │   {html, css, js, ...}                                      │
│       │                                                             │
│       ├─→ [2. 分析网站]                                            │
│       │   IntentAnalyzer::analyze(PageContent)                      │
│       │   → WebsiteIntent {                                         │
│       │       website_type,                                         │
│       │       core_features,                                        │
│       │       design_style,                                         │
│       │       structure,                                            │
│       │       ...                                                   │
│       │     }                                                       │
│       │                                                             │
│       ├─→ [3. 提取特征] ← ← ← NEW!                                │
│       │   WebsiteFeatureExtractor::extract(PageContent, Intent)   │
│       │   → Vec<f32> [48-dim]                                      │
│       │   {                                                        │
│       │       html_structure,                                      │
│       │       css_complexity,                                      │
│       │       js_features,                                         │
│       │       page_layout,                                         │
│       │       ...                                                  │
│       │   }                                                        │
│       │                                                            │
│       ├─→ [4. 发送到 Python] ← ← ← NEW!                          │
│       │   RustPythonBridge::send_features(                         │
│       │       features: Vec<f32>,                                  │
│       │       reference_code: {html, css, js},                     │
│       │       intent: WebsiteIntent                                │
│       │   )                                                        │
│       │                                                            │
│       └─→ [行为记录 - 可选]                                       │
│           BehaviorRecorder::record(PageContent)                    │
│           → BehaviorRecord                                         │
│                                                                    │
└──────────────────────────────────────────────────────────────────────┘
                              ↓ (IPC/HTTP)
┌──────────────────── Python 层 ────────────────────────────────────┐
│                                                                     │
│  [5. 模型训练]                                                     │
│       ↓                                                             │
│  OnlineLearningEngine::learn_from_sample(                          │
│      features: Tensor[1, 48],     # 网站特征                      │
│      reference_code: Dict,        # 原始代码                      │
│      intent: Dict,                # 网站意图                      │
│      feedback: LearningFeedback   # 反馈                          │
│  )                                                                 │
│       │                                                            │
│       ├─→ WebsiteFeatureEncoder(features)                         │
│       │   → encoded[1, 128]                                        │
│       │                                                            │
│       ├─→ CodeGenerator(encoded)                                  │
│       │   → (html_logits, css_logits, js_logits)                 │
│       │                                                            │
│       ├─→ QualityVerifier(reference, generated)                   │
│       │   → quality_score [0.0-1.0]                               │
│       │                                                            │
│       ├─→ [6. 计算损失]                                           │
│       │   loss = reconstruction_loss × feedback_weight             │
│       │   feedback_weight = 1.0 + (quality_score/100) × 0.5       │
│       │                                                            │
│       ├─→ [7. 反向传播] ← ← ← GPU 加速                           │
│       │   loss.backward()                                         │
│       │   torch.nn.utils.clip_grad_norm(...)                      │
│       │   optimizer.step()                                        │
│       │                                                            │
│       └─→ [8. 记录学习历史]                                       │
│           learning_history.append({                               │
│               'loss': loss.item(),                                │
│               'timestamp': now(),                                 │
│               'quality_score': score                              │
│           })                                                      │
│                                                                    │
│  [9. 保存模型] ← ← ← NEW!                                        │
│       ↓                                                            │
│       SaveModel(encoder, generator, verifier)                     │
│       → /models/local/week6_model_epoch_N.pt                      │
│                                                                    │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────── Rust 层 (反馈) ────────────────────────────┐
│                                                                  │
│  [10. 收集反馈] ← ← ← NEW!                                    │
│       ↓                                                          │
│  FeedbackCollector::collect_feedback(                           │
│      original_code: PageContent,                               │
│      generated_code: GeneratedCode,                            │
│      parsing_metadata: Map                                     │
│  )                                                             │
│       → Feedback {                                             │
│           feedback_type: ParsingAccuracy,                      │
│           score: [0.0-1.0],                                    │
│           comment: "...",                                      │
│           context: {...}                                      │
│       }                                                        │
│                                                                │
│  [11. 循环继续]                                               │
│       访问下一个网站 → 重复流程                                │
│                                                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 数据接口规范

### 1. 特征向量格式 (48-dim)

**输出位置**：`WebsiteFeatureExtractor::extract()`

```rust
pub fn extract(page_content: &PageContent, intent: &WebsiteIntent) -> Vec<f32> {
    vec![
        // 0-9: HTML 结构特征
        html_line_count as f32,
        html_size_kb as f32,
        tag_count as f32,
        div_count as f32,
        class_count as f32,
        id_count as f32,
        semantic_tag_count as f32,
        form_count as f32,
        input_count as f32,
        button_count as f32,
        
        // 10-17: CSS 特征
        css_size_kb as f32,
        css_rule_count as f32,
        color_count as f32,
        font_family_count as f32,
        media_query_count as f32,
        animation_count as f32,
        gradient_count as f32,
        border_radius_count as f32,
        
        // 18-27: JavaScript 特征
        js_size_kb as f32,
        js_line_count as f32,
        function_count as f32,
        class_count as f32,
        variable_count as f32,
        event_listener_count as f32,
        api_call_count as f32,
        library_count as f32,
        async_function_count as f32,
        comment_ratio as f32,
        
        // 28-35: 页面结构特征
        has_header as u32 as f32,
        has_footer as u32 as f32,
        has_navigation as u32 as f32,
        has_sidebar as u32 as f32,
        max_nesting_depth as f32,
        section_count as f32,
        article_count as f32,
        aside_count as f32,
        
        // 36-42: 设计指标 (从 Intent 导出)
        intent.design_style.formality,
        intent.design_style.colorfulness,
        intent.design_style.minimalism,
        intent.design_style.modernity,
        (intent.structure.complexity as u32 as f32) / 10.0,  // normalize
        
        // 43-47: 复杂度和性能
        total_page_size_kb as f32,
        image_count as f32,
        video_count as f32,
        external_script_count as f32,
        cdn_resource_ratio as f32,
    ]
}
```

**转换为张量**（Python）：
```python
# 接收 48-dim 特征向量
features_np = np.array(features)  # shape: (48,)
features_tensor = torch.from_numpy(features_np).float()
features_tensor = features_tensor.unsqueeze(0)  # shape: (1, 48)

# 传入编码器
encoded = encoder(features_tensor)  # shape: (1, 128)
```

---

### 2. WebsiteIntent 结构 (传递给 Python)

**Rust 结构**：
```rust
pub struct WebsiteIntent {
    pub website_type: String,
    pub confidence: f32,
    pub core_features: Vec<String>,
    pub target_audience: String,
    pub design_style: DesignStyle,
    pub structure: PageStructure,
    pub business_model: String,
    pub type_scores: HashMap<String, f32>,
}

pub struct DesignStyle {
    pub formality: f32,
    pub colorfulness: f32,
    pub minimalism: f32,
    pub modernity: f32,
}

pub struct PageStructure {
    pub layout_type: LayoutType,
    pub section_count: usize,
    pub complexity: ComplexityLevel,
}
```

**序列化为 JSON** (Rust → Python)：
```json
{
    "website_type": "e-commerce",
    "confidence": 0.92,
    "core_features": ["product_list", "cart", "checkout"],
    "target_audience": "online shoppers",
    "design_style": {
        "formality": 0.7,
        "colorfulness": 0.6,
        "minimalism": 0.3,
        "modernity": 0.8
    },
    "structure": {
        "layout_type": "grid",
        "section_count": 5,
        "complexity": "medium"
    },
    "business_model": "direct_sales"
}
```

---

### 3. 反馈结构 (收集)

**Rust 结构**：
```rust
pub struct Feedback {
    pub id: String,
    pub feedback_type: FeedbackType,
    pub timestamp: u64,
    pub score: f32,  // 0.0-1.0
    pub comment: Option<String>,
    pub context: HashMap<String, String>,
    pub model_id: Option<String>,
}

pub enum FeedbackType {
    ParsingAccuracy,     // 解析准确性
    RenderingQuality,    // 渲染质量
    LayoutCorrectness,   // 布局正确性
    Performance,         // 性能
    Custom(String),
}
```

**计算反馈分数的方法**：
```rust
pub fn calculate_parsing_feedback(
    original_code: &PageContent,
    generated_code: &GeneratedCode,
) -> f32 {
    let mut score = 0.0;
    
    // HTML 布局相似度
    let html_similarity = compare_html_structure(&original_code.html, &generated_code.html);
    score += html_similarity * 0.3;
    
    // CSS 样式相似度
    let css_similarity = compare_css_styles(&original_code.css, &generated_code.css);
    score += css_similarity * 0.4;
    
    // JavaScript 功能完整性
    let js_completeness = check_js_completeness(&original_code.js, &generated_code.js);
    score += js_completeness * 0.3;
    
    score.clamp(0.0, 1.0)
}
```

---

## 🔧 实现路线

### Phase 1: 特征提取器 (Week 6 - Part A)

**文件**：`crates/browerai-learning/src/feature_extractor.rs`

```rust
pub struct WebsiteFeatureExtractor;

impl WebsiteFeatureExtractor {
    /// 从页面内容和意图提取 48-dim 特征向量
    pub fn extract(
        page_content: &PageContent,
        intent: &WebsiteIntent,
    ) -> Result<Vec<f32>> {
        // 1. 解析 HTML/CSS/JS
        // 2. 计算结构特征
        // 3. 分析复杂度
        // 4. 整合设计指标
        // 5. 返回 48-dim 向量
        Ok(vec![...])
    }
}
```

**测试**：
```rust
#[test]
fn test_feature_extraction() {
    let content = create_test_page_content();
    let intent = create_test_intent();
    let features = WebsiteFeatureExtractor::extract(&content, &intent).unwrap();
    
    assert_eq!(features.len(), 48);
    assert!(features.iter().all(|f| f.is_finite()));
}
```

---

### Phase 2: Rust-Python 通信桥 (Week 6 - Part B)

**选项 A：HTTP API**
```rust
// Rust 端
pub async fn send_to_learning_engine(
    features: Vec<f32>,
    reference_code: &PageContent,
    intent: &WebsiteIntent,
) -> Result<LearningResult> {
    let client = reqwest::Client::new();
    let response = client
        .post("http://localhost:5000/api/learn")
        .json(&LearningRequest {
            features,
            reference_code,
            intent,
        })
        .send()
        .await?;
    
    let result: LearningResult = response.json().await?;
    Ok(result)
}
```

**Python 端** (`training/api/learning_api.py`):
```python
from flask import Flask, request
from enhanced_learning_system import OnlineLearningEngine

app = Flask(__name__)
engine = OnlineLearningEngine(LearningConfig())

@app.route('/api/learn', methods=['POST'])
def learn():
    data = request.json
    features = torch.tensor(data['features'], dtype=torch.float32)
    reference_code = data['reference_code']
    
    result = engine.learn_from_sample(features, reference_code)
    return {'loss': result['loss'], 'model_version': 'v1'}
```

**选项 B：共享内存 (更快)**
```rust
// 使用 IPC channel 或 Unix socket
// 更适合高频调用
```

---

### Phase 3: 反馈收集器增强 (Week 6 - Part C)

**文件**：`crates/browerai-learning/src/feedback_collector.rs`

```rust
pub struct FeedbackCollector {
    comparator: CodeComparator,
}

impl FeedbackCollector {
    pub fn collect_feedback(
        &self,
        original: &PageContent,
        generated: &GeneratedCode,
    ) -> Result<Feedback> {
        // 1. 比较 HTML 结构
        let html_score = self.comparator.compare_html(&original.html, &generated.html)?;
        
        // 2. 比较 CSS 样式
        let css_score = self.comparator.compare_css(&original.css, &generated.css)?;
        
        // 3. 检查 JS 功能
        let js_score = self.comparator.check_js_functionality(&original.js, &generated.js)?;
        
        // 4. 综合评分
        let total_score = html_score * 0.3 + css_score * 0.4 + js_score * 0.3;
        
        Ok(Feedback::new(FeedbackType::ParsingAccuracy, total_score)
            .with_comment(format!("HTML: {:.2}, CSS: {:.2}, JS: {:.2}", html_score, css_score, js_score))
            .with_context("model_id", "week6_v1"))
    }
}
```

---

### Phase 4: 端到端测试 (Week 6 - Part D)

**文件**：`tests/e2e_website_learning_tests.rs`

```rust
#[tokio::test]
async fn test_complete_learning_loop() {
    // 1. 创建测试网站
    let test_url = start_test_server();
    
    // 2. 运行学习管道
    let pipeline = LearningPipeline::new()?;
    let output = pipeline.run(&LearningInput {
        url: test_url,
        record_behavior: true,
        enable_validation: true,
        ..Default::default()
    }).await?;
    
    // 3. 验证意图分析
    assert_eq!(output.intent.website_type, "blog");
    
    // 4. 提取特征
    let features = WebsiteFeatureExtractor::extract(
        &output.output,
        &output.intent
    )?;
    assert_eq!(features.len(), 48);
    
    // 5. 发送到学习引擎
    let learning_result = send_to_learning_engine(
        features,
        &output.output,
        &output.intent
    ).await?;
    
    // 6. 验证反馈
    let feedback = FeedbackCollector::collect_feedback(
        &original_content,
        &learning_result.generated_code
    )?;
    
    assert!(feedback.score > 0.6);
}
```

---

## 🎯 Integration Checklist

- [ ] **Phase 1A**: 实现 `WebsiteFeatureExtractor`
- [ ] **Phase 1B**: 单元测试 (48-dim 特征)
- [ ] **Phase 2A**: 实现 HTTP API (Flask)
- [ ] **Phase 2B**: 实现 Rust HTTP 客户端
- [ ] **Phase 2C**: 异步通信测试
- [ ] **Phase 3A**: 增强 `FeedbackCollector`
- [ ] **Phase 3B**: 代码对比逻辑
- [ ] **Phase 3C**: 反馈评分测试
- [ ] **Phase 4A**: 端到端测试脚本
- [ ] **Phase 4B**: 真实网站测试 (3+ 网站)
- [ ] **Phase 4C**: 性能基准测试
- [ ] **文档**: API 文档、使用指南

---

## 📝 关键 API 总结

### Rust 层

```rust
// 1. 特征提取
WebsiteFeatureExtractor::extract(
    page_content: &PageContent,
    intent: &WebsiteIntent
) → Vec<f32> (48-dim)

// 2. 发送到 Python
RustPythonBridge::learn(
    features: Vec<f32>,
    reference_code: PageContent,
    intent: WebsiteIntent
) → Result<LearningResult>

// 3. 收集反馈
FeedbackCollector::collect_feedback(
    original: &PageContent,
    generated: &GeneratedCode
) → Feedback
```

### Python 层

```python
# 1. 学习
OnlineLearningEngine.learn_from_sample(
    website_features: Tensor[1, 48],
    reference_code: Dict[str, str],
    feedback: LearningFeedback
) → Dict[loss, timestamp, quality_score]

# 2. 生成
CodeGenerator(encoded_features) 
    → (html_logits, css_logits, js_logits)

// 3. 验证
QualityVerifier(original, generated) → score [0.0-1.0]
```

---

## 💾 保存路径

**模型保存**：
```
/models/local/
├── week6_model_epoch_1.pt
├── week6_model_epoch_2.pt
├── ...
├── week6_model_latest.pt  # 最新
└── week6_model_best.pt    # 性能最好

/data/learning/
├── training_history.jsonl
├── feedback_log.jsonl
└── feature_statistics.json
```

