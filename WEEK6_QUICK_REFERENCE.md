# Week 6 快速参考：数据流和接口

## 🎯 核心目标

**从错误的方向纠正过来**：

| 之前（错误） | 现在（正确） |
|-----------|-----------|
| ❌ 收集项目代码 | ✅ 访问真实网站 |
| ❌ 应用混淆技术 | ✅ 分析网站结构 |
| ❌ 训练检测混淆 | ✅ 训练改进解析能力 |
| ❌ 独立的学习系统 | ✅ 集成到 BrowerAI 循环 |

---

## 📊 数据流简化版

```
[1. URL]
  ↓
[2. 获取网页] ← PageFetcher
  ↓ (PageContent: html, css, js)
[3. 分析意图] ← IntentAnalyzer
  ↓ (WebsiteIntent: type, features, style, structure)
[4. 提取特征] ← WebsiteFeatureExtractor ← NEW!
  ↓ (Vec<f32>, 48-dim)
[5. 编码特征] ← WebsiteFeatureEncoder (Python)
  ↓ (Tensor, 128-dim)
[6. 生成代码] ← CodeGenerator (Python)
  ↓ (html_logits, css_logits, js_logits)
[7. 计算损失] ← loss = reconstruction × feedback_weight
  ↓
[8. 更新模型] ← optimizer.step() (GPU)
  ↓
[9. 评分反馈] ← QualityVerifier (Python)
  ↓ (score: 0.0-1.0)
[10. 收集反馈] ← FeedbackCollector ← NEW!
  ↓ (Feedback struct)
[11. 保存模型] → /models/local/week6_model_vN.pt
  ↓
[12. 下一个网站] ← ← ← 循环
```

---

## 🔗 关键接口映射

### 入口点：LearningPipeline (Rust)

```rust
// ===== Rust 主流程 =====
let pipeline = LearningPipeline::new()?;

let output = pipeline.run(&LearningInput {
    url: "https://example.com",
    record_behavior: true,
    enable_validation: true,
    preferences: None,
}).await?;

// 输出结构：
// - output.output: GeneratedCode (HTML/CSS/JS)
// - output.intent: WebsiteIntent (网站分析)
// - output.behavior_record: BehaviorRecord (可选)
// - output.validation_report: ValidationReport (可选)
// - output.metadata: 处理元数据

// ===== 新增：特征提取 =====
let features = WebsiteFeatureExtractor::extract(
    &output.output,
    &output.intent
)?;  // Vec<f32>, 48 elements

// ===== 新增：发送到 Python =====
let learning_result = RustPythonBridge::send_learning_request(
    features.clone(),
    output.output.clone(),
    output.intent.clone()
).await?;

// 返回：
// {
//   "loss": 0.0234,
//   "model_version": "v1.2.3",
//   "quality_score": 0.87,
//   "timestamp": "2026-02-01T..."
// }

// ===== 新增：反馈收集 =====
let feedback = FeedbackCollector::new().collect_feedback(
    &original_page_content,
    &learning_result.generated_code
)?;

// 反馈包含：
// - feedback_type: ParsingAccuracy
// - score: 0.85 (0.0-1.0)
// - comment: "HTML 布局正确，但 CSS 需要微调"
// - context: {model_id, timestamp, ...}
```

---

### 48-维特征向量

```
┌─ 0-9: HTML 结构 (10)
│  [html_lines, html_size, tag_count, div_count, class_count,
│   id_count, semantic_tags, forms, inputs, buttons]
│
├─ 10-17: CSS 特征 (8)
│  [css_size, css_rules, colors, fonts, media_queries,
│   animations, gradients, border_radius]
│
├─ 18-27: JavaScript (10)
│  [js_size, js_lines, functions, classes, variables,
│   event_listeners, api_calls, libraries, async_funcs, comments]
│
├─ 28-35: 页面结构 (8)
│  [has_header, has_footer, has_nav, has_sidebar,
│   max_depth, sections, articles, asides]
│
├─ 36-42: 设计风格 (从 WebsiteIntent) (7)
│  [formality, colorfulness, minimalism, modernity,
│   complexity_score, total_size, image_count]
│
└─ 43-47: 复杂度指标 (5)
   [page_size, videos, external_scripts, cdn_ratio, ...]
   
   总计: 48 维
```

---

### Python 模型调用

```python
# ===== 数据准备 =====
import torch
from enhanced_learning_system import OnlineLearningEngine, LearningConfig

config = LearningConfig(
    batch_size=1,
    learning_rate=0.001,
    device='cuda'  # GPU 加速
)
engine = OnlineLearningEngine(config)

# ===== 接收来自 Rust 的特征 =====
features_np = np.array(incoming_features)  # 48-dim
features = torch.from_numpy(features_np).float().unsqueeze(0)  # [1, 48]

reference_code = {
    'html': original_html_content,
    'css': original_css_content,
    'js': original_js_content,
}

feedback = LearningFeedback(
    sample_id="sample_001",
    quality_score=87.0,  # 0-100
    correctness=True,
    confidence=0.92,
)

# ===== 学习 =====
result = engine.learn_from_sample(
    features,
    reference_code,
    feedback
)

# 返回：
# {
#   'loss': 0.0234,
#   'timestamp': '2026-02-01T...',
#   'sample_id': 'sample_001',
#   'quality_score': 87.0
# }

# ===== 检查学习进度 =====
summary = engine.get_learning_summary()
# {
#   'total_samples': 1234,
#   'avg_loss': 0.0145,
#   'avg_quality': 85.3,
#   'status': 'learning_active'
# }

# ===== 保存模型 =====
torch.save(engine.encoder.state_dict(), 'models/encoder_v1.pt')
torch.save(engine.generator.state_dict(), 'models/generator_v1.pt')
torch.save(engine.verifier.state_dict(), 'models/verifier_v1.pt')
```

---

## 🏗️ 实现步骤（优先顺序）

### Step 1: 特征提取器 (3-4 小时)

**文件**：`crates/browerai-learning/src/feature_extractor.rs`

```rust
pub struct WebsiteFeatureExtractor;

impl WebsiteFeatureExtractor {
    pub fn extract(
        page_content: &PageContent,
        intent: &WebsiteIntent,
    ) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(48);
        
        // 解析 HTML 结构
        let html_metrics = Self::extract_html_metrics(&page_content.html);
        features.extend(html_metrics);  // 0-9
        
        // 解析 CSS
        let css_metrics = Self::extract_css_metrics(&page_content.css);
        features.extend(css_metrics);   // 10-17
        
        // 解析 JavaScript
        let js_metrics = Self::extract_js_metrics(&page_content.js);
        features.extend(js_metrics);    // 18-27
        
        // 页面结构
        let structure_metrics = Self::extract_structure_metrics(&intent.structure);
        features.extend(structure_metrics);  // 28-35
        
        // 设计风格
        features.push(intent.design_style.formality);
        features.push(intent.design_style.colorfulness);
        // ... 更多设计指标
        
        Ok(features)
    }
    
    fn extract_html_metrics(html: &str) -> Vec<f32> {
        vec![
            html.lines().count() as f32,
            (html.len() / 1024) as f32,
            html.matches('<').count() as f32,
            // ...
        ]
    }
    
    // ... 其他方法
}
```

**测试**：
```rust
#[test]
fn test_feature_extraction_returns_48_dims() {
    let content = create_sample_page_content();
    let intent = create_sample_intent();
    let features = WebsiteFeatureExtractor::extract(&content, &intent).unwrap();
    
    assert_eq!(features.len(), 48);
    assert!(features.iter().all(|f| f.is_finite() && *f >= 0.0));
}
```

---

### Step 2: Rust-Python 通信 (4-5 小时)

**选项 A：HTTP API** (推荐)

**Python 端** - `training/api/learning_api.py`:
```python
from flask import Flask, request
import torch
from enhanced_learning_system import OnlineLearningEngine, LearningConfig

app = Flask(__name__)
engine = None

@app.before_first_request
def init():
    global engine
    config = LearningConfig(device='cuda')
    engine = OnlineLearningEngine(config)

@app.route('/api/health', methods=['GET'])
def health():
    return {'status': 'ok', 'device': str(engine.config.device)}

@app.route('/api/learn', methods=['POST'])
def learn():
    """接收特征向量，执行学习"""
    try:
        data = request.json
        
        # 转换数据格式
        features = torch.tensor(data['features'], dtype=torch.float32)
        features = features.unsqueeze(0)  # [1, 48]
        
        reference_code = data['reference_code']
        feedback_data = data.get('feedback')
        
        from enhanced_learning_system import LearningFeedback
        feedback = LearningFeedback(
            sample_id=feedback_data.get('sample_id'),
            quality_score=feedback_data.get('quality_score', 50),
        )
        
        # 执行学习
        result = engine.learn_from_sample(features, reference_code, feedback)
        
        return {
            'success': True,
            'loss': float(result['loss']),
            'timestamp': result['timestamp'],
            'model_version': 'week6_v1',
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}, 500

@app.route('/api/model/save', methods=['POST'])
def save_model():
    """保存当前模型"""
    import os
    path = request.json.get('path', '/models/local/week6_model_latest.pt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save(engine.encoder.state_dict(), path + '.encoder')
    torch.save(engine.generator.state_dict(), path + '.generator')
    torch.save(engine.verifier.state_dict(), path + '.verifier')
    
    return {'success': True, 'path': path}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**Rust 端** - `crates/browerai-learning/src/python_bridge.rs`:
```rust
pub struct RustPythonBridge {
    client: reqwest::Client,
    endpoint: String,
}

impl RustPythonBridge {
    pub fn new(endpoint: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.to_string(),
        }
    }
    
    pub async fn send_learning_request(
        &self,
        features: Vec<f32>,
        reference_code: PageContent,
        feedback: Feedback,
    ) -> Result<LearningResponse> {
        let payload = serde_json::json!({
            "features": features,
            "reference_code": {
                "html": reference_code.html,
                "css": reference_code.css,
                "js": reference_code.js,
            },
            "feedback": {
                "sample_id": feedback.id,
                "quality_score": (feedback.score * 100.0),
            },
        });
        
        let response = self.client
            .post(&format!("{}/api/learn", self.endpoint))
            .json(&payload)
            .send()
            .await?;
        
        let result: LearningResponse = response.json().await?;
        Ok(result)
    }
}

#[derive(Serialize, Deserialize)]
pub struct LearningResponse {
    pub success: bool,
    pub loss: f32,
    pub timestamp: String,
    pub model_version: String,
}
```

**启动脚本** - `training/scripts/run_learning_api.sh`:
```bash
#!/bin/bash

# 设置 Python 路径
export PYTHONPATH=/home/stone/BrowerAI/training:/home/stone/BrowerAI/training/pipelines

# 安装依赖
pip install flask torch numpy

# 运行 API 服务器
python -u /home/stone/BrowerAI/training/api/learning_api.py
```

---

### Step 3: 反馈收集器 (2-3 小时)

**增强** `crates/browerai-learning/src/feedback.rs`:

```rust
pub struct CodeComparator;

impl CodeComparator {
    /// 比较 HTML 结构相似度
    pub fn compare_html_structure(
        original_html: &str,
        generated_html: &str,
    ) -> Result<f32> {
        // 使用 html5ever 解析两者的 DOM 树
        // 计算树的结构相似度 (可以用编辑距离或树匹配)
        // 返回 0.0-1.0 的相似度分数
        
        let original_dom = parse_html(original_html)?;
        let generated_dom = parse_html(generated_html)?;
        
        let similarity = tree_similarity(&original_dom, &generated_dom);
        Ok(similarity)
    }
    
    /// 比较 CSS 样式
    pub fn compare_css_styles(
        original_css: &str,
        generated_css: &str,
    ) -> Result<f32> {
        // 解析 CSS 规则
        // 比较：选择器、属性、值
        // 计算覆盖率和准确度
        
        let original_rules = parse_css_rules(original_css)?;
        let generated_rules = parse_css_rules(generated_css)?;
        
        let similarity = rule_similarity(&original_rules, &generated_rules);
        Ok(similarity)
    }
    
    /// 检查 JavaScript 功能完整性
    pub fn check_js_completeness(
        original_js: &str,
        generated_js: &str,
    ) -> Result<f32> {
        // 提取关键函数、事件监听器、API 调用
        // 检查是否在生成的 JS 中都有对应的函数
        
        let original_funcs = extract_functions(original_js)?;
        let generated_funcs = extract_functions(generated_js)?;
        
        let completeness = (generated_funcs.len() as f32) / (original_funcs.len() as f32);
        Ok(completeness.clamp(0.0, 1.0))
    }
}

pub fn calculate_parsing_feedback(
    original_code: &PageContent,
    generated_code: &GeneratedCode,
) -> Result<Feedback> {
    let comparator = CodeComparator;
    
    // 计算各部分相似度
    let html_score = comparator.compare_html_structure(
        &original_code.html,
        &generated_code.html
    )?;  // 权重: 0.3
    
    let css_score = comparator.compare_css_styles(
        &original_code.css,
        &generated_code.css
    )?;  // 权重: 0.4
    
    let js_score = comparator.check_js_completeness(
        &original_code.js,
        &generated_code.js
    )?;  // 权重: 0.3
    
    // 加权平均
    let total_score = 
        html_score * 0.3 + 
        css_score * 0.4 + 
        js_score * 0.3;
    
    Ok(Feedback::new(FeedbackType::ParsingAccuracy, total_score)
        .with_comment(format!(
            "HTML: {:.2}, CSS: {:.2}, JS: {:.2}",
            html_score, css_score, js_score
        ))
        .with_model_id("week6_v1"))
}
```

---

### Step 4: 集成测试 (2-3 小时)

**文件**：`tests/e2e_website_learning_tests.rs`

```rust
#[tokio::test]
async fn test_complete_learning_cycle() {
    // 1. 启动测试服务器
    let test_server = TestWebsiteServer::new();
    let test_url = test_server.url().to_string();
    
    // 2. 创建学习管道
    let pipeline = LearningPipeline::new().unwrap();
    
    // 3. 运行学习
    let output = pipeline.run(&LearningInput {
        url: test_url,
        record_behavior: true,
        enable_validation: true,
        preferences: None,
    }).await.unwrap();
    
    // 4. 验证网站分析
    assert!(!output.intent.website_type.is_empty());
    assert!(output.intent.confidence > 0.5);
    
    // 5. 提取特征
    let features = WebsiteFeatureExtractor::extract(
        &output.output,
        &output.intent
    ).unwrap();
    
    assert_eq!(features.len(), 48);
    assert!(features.iter().all(|f| f.is_finite()));
    
    // 6. 模拟反馈收集
    let feedback = calculate_parsing_feedback(
        &test_server.original_content,
        &output.output
    ).unwrap();
    
    assert!(feedback.score >= 0.0 && feedback.score <= 1.0);
    
    // 7. 验证反馈结构
    assert_eq!(feedback.feedback_type, FeedbackType::ParsingAccuracy);
    assert!(feedback.comment.is_some());
}

#[test]
fn test_feature_extraction_consistency() {
    // 特征提取应该对同样的内容返回相同的结果
    let content = create_test_page_content();
    let intent = create_test_intent();
    
    let features1 = WebsiteFeatureExtractor::extract(&content, &intent).unwrap();
    let features2 = WebsiteFeatureExtractor::extract(&content, &intent).unwrap();
    
    assert_eq!(features1, features2);
}
```

---

## 📋 完成清单

### Phase 1: 架构分析 ✅
- [x] 分析 Rust 层 (LearningPipeline, IntentAnalyzer, FeedbackCollector)
- [x] 分析 Python 层 (OnlineLearningEngine, CodeGenerator, QualityVerifier)
- [x] 理解数据流和接口
- [x] 创建集成设计文档

### Phase 2: 实现 (待做)
- [ ] 特征提取器 (WebsiteFeatureExtractor)
- [ ] Rust-Python 通信 (HTTP API)
- [ ] 反馈收集器增强
- [ ] 端到端测试

### Phase 3: 验证 (待做)
- [ ] 单元测试 (48-dim 特征)
- [ ] 集成测试 (完整流程)
- [ ] 真实网站测试
- [ ] 性能基准测试

---

## 🔗 关键文件位置

| 组件 | 文件 | 行数 |
|------|------|------|
| LearningPipeline | `crates/browerai-learning/src/pipeline/learning_pipeline.rs` | 441 |
| IntentAnalyzer | `crates/browerai-learning/src/learning_sandbox/intent_analyzer.rs` | 807 |
| FeedbackCollector | `crates/browerai-learning/src/feedback.rs` | 358 |
| ContinuousLearningLoop | `crates/browerai-learning/src/continuous_loop.rs` | 449 |
| OnlineLearningEngine | `training/pipelines/enhanced_learning_system.py` | 720 |

---

## 🚀 下一步

现在的问题是：**是否要直接开始实现？**

选项：
1. **现在就开始** - 直接实现 Step 1 (特征提取器)
2. **继续分析** - 深入分析某个特定组件
3. **审查文档** - 确保理解无误后再开始

你的选择？

