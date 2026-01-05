# BrowerAI - 真正的AI学习系统路线图

## 核心问题
**当前系统** = 分类器 + 规则匹配  
**需要升级到** = 深度学习 + 生成模型 + 在线学习

---

## 第一阶段：深度学习基础 (替代简单分类器)

### 1.1 HTML理解模型 - Transformer架构
```
输入: HTML DOM树序列
模型: BERT-like Transformer (12层, 768维)
输出: 
  - 语义向量 (768维)
  - 结构类型 (单页/多页/SPA)
  - 框架预测 (React/Vue/Angular)
```

**训练数据**：
- 10万+ HTML样本（当前632个不够）
- 标注：框架类型、页面结构、语义标签

**模型大小**：~50MB（压缩后10-20MB）
**推理速度**：5-10ms（GPU）/ 50-100ms（CPU）

### 1.2 CSS理解模型 - 序列模型
```
输入: CSS规则序列（选择器 + 属性）
模型: LSTM/GRU (双向, 256维)
输出:
  - 设计风格向量 (256维)
  - 颜色主题类别
  - 布局模式 (grid/flex/float)
```

### 1.3 JS理解模型 - 代码语言模型
```
输入: JS AST序列
模型: CodeBERT / GraphCodeBERT
输出:
  - 代码意图向量 (768维)
  - 混淆程度分数 (0-1)
  - 功能分类 (UI/业务逻辑/工具)
```

---

## 第二阶段：生成能力 (不只是理解)

### 2.1 HTML生成模型 - Seq2Seq
```
输入: 语义描述 + 原始HTML
编码器: Transformer Encoder
解码器: Transformer Decoder
输出: 简化的HTML (去混淆、重构)
```

**关键能力**：
- 理解DOM语义 → 生成等价但更简洁的HTML
- 保留功能 → 移除冗余标签
- 优化结构 → 更符合语义化

### 2.2 CSS生成模型 - Style Transfer
```
输入: 原始CSS + 目标风格向量
模型: VAE (变分自编码器)
输出: 新的CSS (不同配色/布局，相同结构)
```

**5种主题 → 无限主题**：
- 学习CSS潜在空间
- 插值生成新风格
- 保持布局不变，只改视觉

### 2.3 JS去混淆模型 - Code-to-Code Translation
```
输入: 混淆JS代码
模型: Seq2Seq + AST注意力
输出: 去混淆JS (可读、保留功能)
```

**技术**：
- AST级别的变换（不是字符串替换）
- 变量重命名 (a,b,c → meaningful names)
- 死代码消除
- 控制流简化

---

## 第三阶段：在线学习 (持续改进)

### 3.1 增量学习架构
```python
class OnlineLearner:
    def __init__(self):
        self.base_model = load_pretrained()
        self.adapter_layers = AdapterModule()  # 轻量级适配器
        self.experience_buffer = ReplayBuffer(10000)
    
    def learn_from_visit(self, html, css, js, user_feedback):
        # 1. 存储经验
        self.experience_buffer.add(html, css, js, user_feedback)
        
        # 2. 小批量更新（不重训整个模型）
        if len(self.experience_buffer) >= 32:
            batch = self.experience_buffer.sample(32)
            loss = self.adapter_layers.train_step(batch)
            
        # 3. 定期合并到主模型（每1000次访问）
        if self.visit_count % 1000 == 0:
            self.base_model = merge_adapter(self.base_model, self.adapter_layers)
```

**关键技术**：
- **Adapter Layers** (只训练小部分参数, <1MB)
- **Experience Replay** (稳定学习)
- **Low-Rank Adaptation (LoRA)** (高效微调)

### 3.2 用户反馈闭环
```
用户访问网站 
  → 模型预测(站点类型/主题)
  → 生成简化版本
  → 用户选择 (原版/简化版/主题1/主题2...)
  → 记录选择作为标签
  → 更新模型权重
```

### 3.3 联邦学习 (可选，隐私保护)
```
多个用户的浏览器 → 本地训练 → 只上传梯度（不上传数据）
→ 中央服务器聚合 → 分发更新模型
```

---

## 第四阶段：语义理解 (深层智能)

### 4.1 多模态理解
```
输入: HTML + CSS + JS + 截图
模型: Vision-Language模型 (CLIP-like)
输出: 
  - 页面意图理解 ("这是一个购物车页面")
  - 功能识别 ("这个按钮是'加入购物车'")
  - 视觉-代码对应关系
```

### 4.2 知识图谱
```
构建: 网站类型 ← 使用框架 ← 混淆技术 ← 功能模块
查询: "购物网站通常使用什么框架？" → React/Vue占70%
推理: "这个网站用了webpack → 很可能有代码分割"
```

### 4.3 意图驱动生成
```
输入: "我想要一个极简风格的新闻页面"
推理:
  1. 提取关键词: 极简风格, 新闻页面
  2. 查询知识库: 新闻页面 → 常见布局模式
  3. 查询用户历史: 用户偏好的极简风格特征
  4. 生成: 定制化HTML/CSS
```

---

## 技术栈升级

### 当前 (简单分类)
```
- scikit-learn (质心分类器)
- 规则匹配
- 模板生成
```

### 升级后 (深度学习)
```python
# 模型训练
- PyTorch 2.9+ (深度学习框架)
- Transformers (Hugging Face)
- CodeBERT / GraphCodeBERT (代码理解)
- ONNX Runtime (推理优化)

# 在线学习
- Ray/RLlib (分布式学习)
- Weights & Biases (实验跟踪)

# 数据处理
- BeautifulSoup4 / lxml (HTML解析)
- cssutils (CSS解析)
- esprima-python (JS AST)
```

### Rust集成
```rust
// src/ai/deep_learning.rs
use ort::Session;

pub struct DeepLearningEngine {
    html_model: Session,    // 50MB Transformer
    css_model: Session,     // 20MB LSTM
    js_model: Session,      // 100MB CodeBERT
    online_adapter: Session, // 1MB Adapter
}

impl DeepLearningEngine {
    pub fn understand_html(&self, html: &str) -> SemanticVector {
        // 前向推理，返回768维语义向量
    }
    
    pub fn generate_theme(&self, style_vec: &[f32]) -> String {
        // 解码生成CSS
    }
    
    pub fn deobfuscate_js(&self, js: &str) -> String {
        // Seq2Seq转换
    }
    
    pub fn learn_online(&mut self, feedback: UserFeedback) {
        // 更新adapter层
    }
}
```

---

## 数据需求

### 当前数据规模 (不够!)
- 632个网站
- 3737个事件
- 1.5MB特征

### 需要的数据规模
```
1. HTML理解: 10万+ 网站 (100GB+)
   - Alexa Top 100K
   - Common Crawl数据集
   
2. CSS风格: 5万+ 样式表
   - GitHub CSS仓库
   - Bootstrap/Tailwind变体
   
3. JS去混淆: 1万+ 混淆/非混淆配对
   - 手工混淆 (uglify, terser, obfuscator)
   - 标注原始版本
   
4. 用户反馈: 持续积累
   - 每次访问的选择
   - A/B测试结果
```

---

## 实施计划

### Phase 1: 预训练模型 (2-3个月)
1. **数据收集**：爬取10万网站
2. **标注**：自动标注 + 人工验证1000个
3. **训练HTML模型**：BERT-base微调
4. **训练CSS模型**：LSTM从头训练
5. **训练JS模型**：CodeBERT微调
6. **导出ONNX**：优化推理速度

### Phase 2: 生成能力 (1-2个月)
1. **HTML生成器**：Seq2Seq训练
2. **CSS生成器**：VAE训练
3. **JS去混淆器**：AST-based Seq2Seq
4. **集成测试**：端到端生成流程

### Phase 3: 在线学习 (1个月)
1. **Adapter架构**：轻量级适配层
2. **反馈循环**：用户选择 → 标签 → 更新
3. **增量更新**：每1000次访问合并模型
4. **A/B测试**：验证学习效果

### Phase 4: 部署优化 (持续)
1. **模型压缩**：量化(INT8)、剪枝
2. **推理加速**：TensorRT、ONNX优化
3. **边缘计算**：浏览器内推理
4. **云端协同**：重模型在服务器，轻模型在本地

---

## 性能目标

### 当前性能 (简单分类)
- 特征提取: <1ms
- 分类: <0.1ms
- 主题生成: <5ms
- **总计**: <10ms

### 深度学习目标
- HTML理解: 50-100ms (CPU)
- CSS理解: 20-30ms
- JS去混淆: 100-200ms
- 主题生成: 50ms
- **总计**: 200-400ms (首次加载可接受)

### 优化后目标
- 缓存语义向量 → 二次访问<10ms
- 边缘推理(WebGPU) → 减半延迟
- 预测性加载 → 用户无感知

---

## 评估指标

### 理解准确度
- 站点分类准确率: >90%
- 框架识别准确率: >95%
- 混淆检测F1: >0.85

### 生成质量
- HTML简化后可用性: >98% (功能不破坏)
- CSS美学评分: 用户满意度>80%
- JS去混淆可读性: 代码复杂度降低>50%

### 学习效果
- 在线学习收敛: <10000次访问
- 个性化准确率: >85%
- 知识迁移成功率: >70%

---

## 对比：分类 vs 真正AI学习

| 维度 | 当前(分类器) | 真正AI学习 |
|------|-------------|-----------|
| **模型** | 质心分类(994B) | Transformer(50MB+) |
| **能力** | 标签预测 | 语义理解+生成 |
| **学习** | 离线训练一次 | 在线持续学习 |
| **数据** | 632网站 | 10万+网站 |
| **输出** | 5个预定义主题 | 无限定制主题 |
| **智能** | 规则匹配 | 语义推理 |

---

## 下一步行动建议

### 选项1: 快速原型 (1周)
```bash
# 使用现成的预训练模型
pip install transformers torch
python download_codebert.py  # 下载CodeBERT
python finetune_html.py      # 在632个样本上微调
python export_onnx.py        # 导出推理模型
```

### 选项2: 完整系统 (3-6个月)
1. 扩展数据收集到10万网站
2. 训练专用的HTML/CSS/JS模型
3. 实现在线学习架构
4. 部署到浏览器

### 选项3: 混合方案 (推荐, 1-2个月)
1. **第1周**: 集成预训练CodeBERT (理解能力提升)
2. **第2-3周**: 训练轻量级生成模型 (去混淆)
3. **第4周**: 实现简单的在线学习 (Adapter)
4. **后续**: 根据效果逐步扩展

---

## 需要的资源

### 计算资源
- **训练**: GPU (V100/A100) 或 Google Colab Pro
- **推理**: CPU即可 (ONNX优化后)

### 数据资源
- **爬虫**: 10万网站 (约1TB原始数据)
- **存储**: 100GB特征数据

### 开发时间
- **核心模型**: 2-3个月 (全职)
- **集成优化**: 1个月
- **测试部署**: 1个月

---

## 总结

**当前系统**: 入门级分类器（准确但不智能）  
**目标系统**: 深度学习驱动的理解+生成+学习系统

**核心差异**:
1. 从 **分类** → **理解+生成**
2. 从 **规则** → **神经网络**
3. 从 **静态** → **在线学习**
4. 从 **模板** → **无限可能**

**实现路径**: 预训练模型 → 生成能力 → 在线学习 → 持续优化
