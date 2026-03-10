# 🎓 BrowerAI 学习路径手册

**版本**: 1.0  
**日期**: 2026-02-17  
**适用对象**: 新开发者、贡献者、您自己重新理解项目

---

## 📋 使用说明

本手册为您提供**从零开始理解BrowerAI**的完整学习路径。无论您是项目维护者重新梳理设计思路，还是新加入的开发者，都可以按照这个路径逐步深入。

**预计完成时间**: 10-15 小时（分5个阶段）

---

## 🎯 核心口号（必读）

**保功能、换体验 | Preserve Functionality, Change Experience**

这是BrowerAI的灵魂 - 我们不是简单地压缩代码或复制网站，而是：
1. **深度理解** Web代码的功能本质（通过反混淆、深度分析）
2. **智能推理** 识别核心功能与可优化区域
3. **自由生成** 用全新体验重新呈现（企业风格、政府合规、极简设计等）
4. **功能保证** 所有交互功能100%保留（按钮、表单、事件处理）

**实际案例**:
- 输入：混淆的电商网站（JS混淆、CSS复杂）
- 输出：3个变体（现代风格、高对比度、极简）
- 保证：所有"加入购物车"、"提交订单"功能完全一致

---

## 阶段1: 概念入门（30分钟）

### 1.1 核心定位

BrowerAI是一个**AI驱动的自主学习浏览器引擎**，它：
- ✅ 不是传统浏览器（不替代Chrome/Firefox）
- ✅ 不是简单的解析器（不止于html5ever/cssparser）
- ✅ 是一个**理解和重构Web的AI系统**

```
传统浏览器流程:
网页 → 解析 → 渲染 → 显示

BrowerAI流程:
网页 → 解析 → 理解 → 推理 → 生成多种体验 → 验证功能完整性
     ↑________学习反馈_________|
```

### 1.2 快速理解项目

**第1步**: 阅读 [README.md](../README.md) 前100行
- 项目状态：Phase 3 Week 3完成，459+测试通过
- 核心特性：模块化架构、可选AI/ML、纯Rust解析器

**第2步**: 浏览 [DESIGN_ESSENCE.md](DESIGN_ESSENCE.md)（215行，15分钟）
- 七大设计原则：AI增强而非替代、模块化、类型安全等
- 五大创新亮点：混合引擎、完整流水线、真实数据学习等
- 核心能力矩阵：与传统浏览器的对比

**第3步**: 理解关键数字
- **27个crate**: 极致模块化的工作空间架构
- **17,542样本**: 100%真实NPM混淆代码训练数据
- **7阶段分析**: JS深度分析管道（scope → dataflow → controlflow → call graph → loop → performance → unified）
- **18种策略**: JavaScript反混淆策略（AST、控制流、数据流、符号执行等）
- **98.49%**: fast_enhanced.onnx模型的准确率

### 1.3 核心哲学（3个必记原则）

**原则1: AI增强而非替代**
```rust
// 所有Parser都有两个构造函数
pub fn new() -> Self               // 传统解析（无AI依赖）
pub fn with_ai(model: AIModel) -> Self  // AI增强（可选）
```

**原则2: 功能完整性第一**
```rust
// 验证功能保留度（from model_orchestrator.rs:428）
fn verify_functionality(&self, original_js: &str, generated_js: &str) -> Result<f32> {
    let preservation = generated_funcs / original_funcs;  // 必须 >= 0.8
    Ok(preservation)
}
```

**原则3: 真实数据驱动**
- 不用合成数据
- 只用真实混淆代码（17,542个NPM包样本）
- 训练50 epochs GPU加速

---

## 阶段2: 架构理解（2小时）

### 2.1 六层架构概览

```
┌─────────────────────────────────────────┐
│  应用层 Application Layer                │
│  API · 插件 · UI                         │ ← 用户交互
├─────────────────────────────────────────┤
│  学习层 Learning Layer                   │
│  反馈 · 模型更新 · 工作流提取            │ ← 持续改进
├─────────────────────────────────────────┤
│  AI增强层 AI Enhancement Layer           │
│  ONNX推理 · 模型管理 · 智能优化          │ ← 可选增强
├─────────────────────────────────────────┤
│  业务层 Business Logic Layer             │
│  反混淆 · 渲染 · 代码生成                │ ← 核心功能
├─────────────────────────────────────────┤
│  解析层 Parsing Layer                    │
│  HTML · CSS · JS 解析和分析              │ ← 基础解析
├─────────────────────────────────────────┤
│  核心层 Core Layer                       │
│  DOM · 类型 · 配置 · 缓存               │ ← 底层支撑
└─────────────────────────────────────────┘
```

### 2.2 27个Crate导览

**核心层（5个crate）**:
- [browerai-core](../crates/browerai-core/) - 中央类型系统、traits、metrics
- [browerai-dom](../crates/browerai-dom/) - DOM模型、Web APIs（Console、Timer、URL等）
- [browerai-cache](../crates/browerai-cache/) - 基础缓存
- [browerai-db](../crates/browerai-db/) - PostgreSQL持久化
- [browerai-metrics](../crates/browerai-metrics/) - Prometheus指标

**解析层（4个crate）** ⭐ 重点:
- [browerai-html-parser](../crates/browerai-html-parser/) - HTML5解析（html5ever）
- [browerai-css-parser](../crates/browerai-css-parser/) - CSS解析（cssparser + 现代特性）
- [browerai-js-parser](../crates/browerai-js-parser/) - JS解析（Boa + ES模块）
- [browerai-js-analyzer](../crates/browerai-js-analyzer/) ⭐⭐⭐ - **7阶段深度分析管道**（核心中的核心）

**AI层（3个crate）** - 可选特性:
- [browerai-ai-core](../crates/browerai-ai-core/) - ONNX Runtime集成、热重载
- [browerai-ai-integration](../crates/browerai-ai-integration/) - AI接口层
- [browerai-ml](../crates/browerai-ml/) - ML工具包（需LibTorch）

**渲染层（4个crate）**:
- [browerai-renderer-core](../crates/browerai-renderer-core/) - 核心渲染算法
- [browerai-renderer-predictive](../crates/browerai-renderer-predictive/) - 预测性渲染
- [browerai-renderer](../crates/browerai-renderer/) - 完整渲染器
- [browerai-intelligent-rendering](../crates/browerai-intelligent-rendering/) ⭐⭐ - **AI驱动的多样式生成**

**学习层（2个crate）** ⭐⭐:
- [browerai-learning](../crates/browerai-learning/) - **60+模块学习系统**（真实网站学习、工作流提取、质量评估）
- [browerai-deobfuscation](../crates/browerai-deobfuscation/) - **18种反混淆策略**

**支持层（9个crate）**:
- [browerai-network](../crates/browerai-network/) - HTTP客户端、爬虫
- [browerai-devtools](../crates/browerai-devtools/) - 开发工具、DOM检查器
- [browerai-testing](../crates/browerai-testing/) - 测试工具
- [browerai-plugins](../crates/browerai-plugins/) - 插件系统
- [browerai-multilayer-cache](../crates/browerai-multilayer-cache/) - 多层缓存（DashMap → Redis → RocksDB）
- [browerai-redis-integration](../crates/browerai-redis-integration/) - Redis后端
- [browerai-persistent-layer](../crates/browerai-persistent-layer/) - Sled/RocksDB持久化
- [browerai-api-server](../crates/browerai-api-server/) - REST API（5个端点）
- [browerai-js-v8](../crates/browerai-js-v8/) - 可选V8引擎

### 2.3 关键依赖关系

```
browerai (主程序)
├─ browerai-learning (学习系统)
│  ├─ browerai-deobfuscation (反混淆)
│  ├─ browerai-js-analyzer (深度分析) ⭐
│  └─ browerai-network (网络爬虫)
│
├─ browerai-intelligent-rendering (智能渲染) ⭐
│  ├─ browerai-renderer (渲染器)
│  ├─ browerai-js-analyzer (JS分析)
│  └─ browerai-ai-core (AI推理)
│
└─ browerai-ai-core (AI核心)
   └─ ort = "2.0.0-rc.10" (ONNX Runtime)
```

### 2.4 模块选择建议

**最小配置**（学习基础解析）:
```toml
browerai-core + browerai-html-parser + browerai-css-parser + browerai-js-parser
```

**标准配置**（包含AI增强）:
```toml
+ browerai-ai-core + browerai-intelligent-rendering
```

**完整配置**（生产部署）:
```toml
+ browerai-learning + browerai-deobfuscation + browerai-api-server + 缓存层
```

---

## 阶段3: 核心流程（3小时）

### 3.1 完整流水线：learn_and_generate

**代码位置**: [crates/browerai/src/main.rs:295](../crates/browerai/src/main.rs#L295)

```rust
async fn learn_and_generate(url: &str, output_dir: &PathBuf, variant_count: usize) -> Result<()>
```

**5个阶段**:

```
[1/5] 网站理解 (Site Understanding)
      ↓ 抓取HTML → 解析DOM → 提取功能 → 分类
      
[2/5] 运行时推理 (Runtime Inference)  
      ↓ V8追踪 → 事件监听 → 状态管理识别
      
[3/5] 智能推理 (Intelligent Reasoning)
      ↓ 识别核心功能 → 发现优化区域 → 生成布局建议
      
[4/5] 多样式生成 (Multi-Style Generation)
      ↓ HTML重构 + CSS重写 + JS功能桥接
      
[5/5] 验证与输出 (Verification & Output)
      ↓ 功能完整性验证 → 生成变体 → 保存报告
```

**实际输出结构**:
```
output/
├── variant_1/           # 现代风格
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── preserved_features.json
├── variant_2/           # 高对比度（政府合规）
│   └── ...
├── variant_3/           # 极简风格
│   └── ...
└── pipeline_report.json
```

### 3.2 核心流程详解

#### 3.2.1 阶段1: 网站理解

**代码**: [crates/browerai-intelligent-rendering/src/website_learning_engine.rs:33](../crates/browerai-intelligent-rendering/src/website_learning_engine.rs#L33)

```rust
pub struct SiteUnderstanding {
    pub core_structure: CoreStructure,      // DOM结构
    pub feature_categories: Vec<FeatureCategory>,  // 功能分类
    pub event_handlers: Vec<EventHandler>,  // 事件处理
    pub state_management: StateManagement,  // 状态管理
}
```

**关键步骤**:
1. HTML解析 → DOM树构建
2. CSS提取 → 样式规则分析
3. JavaScript分析 → 7阶段深度管道（见3.3）
4. 功能识别 → 标记核心功能（critical: true/false）

#### 3.2.2 阶段3: 智能推理

**代码**: [crates/browerai-intelligent-rendering/src/reasoning.rs:81](../crates/browerai-intelligent-rendering/src/reasoning.rs#L81)

```rust
pub fn reason(&self) -> Result<ReasoningResult> {
    // 步骤1: 识别核心功能（必须100%保留）
    let core_functions = self.identify_core_functions()?;
    
    // 步骤2: 发现可优化区域（styling, layout, performance）
    let optimizable = self.find_optimizable_regions()?;
    
    // 步骤3: 生成布局建议（Traditional, Modern, Minimal）
    let layouts = self.generate_layout_suggestions()?;
    
    // 步骤4: 创建体验变体（功能映射 + 视觉风格）
    let variants = self.create_experience_variants(&core_functions, &layouts)?;
    
    Ok(ReasoningResult { core_functions, optimizable_regions, layout_suggestions, experience_variants })
}
```

**优化区域类型**:
- `Styling`: 颜色、字体、间距（自由修改）
- `Layout`: 布局结构（可重构）
- `Performance`: 性能优化（可改进）
- ❌ `Functionality`: 功能逻辑（禁止修改！）

#### 3.2.3 阶段4: 多样式生成

**代码**: [crates/browerai-intelligent-rendering/src/generation.rs:41](../crates/browerai-intelligent-rendering/src/generation.rs#L41)

```rust
pub fn generate(&self) -> Result<Vec<GeneratedExperience>> {
    for variant in &self.reasoning.experience_variants {
        // 1. 生成新HTML结构（保持data-original-function属性）
        let html = self.generate_html_for_variant(variant)?;
        
        // 2. 生成新CSS样式（全新视觉体验）
        let css = self.generate_css_for_variant(variant)?;
        
        // 3. 生成功能桥接JS（确保原始功能可用）
        let bridge_js = self.generate_function_bridge(variant)?;
        
        // 4. 验证功能完整性
        let validation = self.validate_functions(&html, &bridge_js)?;
        
        if validation.all_functions_present {
            experiences.push(GeneratedExperience { ... });
        }
    }
    Ok(experiences)
}
```

**功能桥接示例**:
```javascript
// 原始网站: <button id="old-submit-btn" onclick="submitForm()">
// 新变体:   <button id="new-modern-submit" data-original-function="submitForm">

const BrowerAI = {
    functionBridge: {
        'new-modern-submit': function() {
            // 调用原始submitForm逻辑
            window.originalHandlers.submitForm();
        }
    }
};
```

### 3.3 ⭐ 7阶段JS深度分析管道

**代码**: [crates/browerai-js-analyzer/src/analysis_pipeline.rs](../crates/browerai-js-analyzer/src/analysis_pipeline.rs)

```
Stage 1: Scope Analysis (作用域分析)
         ↓ 词法作用域 → 变量声明 → 闭包检测
         
Stage 2: SWC AST Extraction (TypeScript/JSX支持)
         ↓ swc_core → 现代JS特性 → 类型信息
         
Stage 3: Data Flow Analysis (数据流分析)
         ↓ def-use链 → 常量识别 → 未使用变量
         
Stage 4: Control Flow Graph (控制流图)
         ↓ CFG构建 → 可达性分析 → 循环检测
         
Stage 5: Enhanced Call Graph (增强调用图)
         ↓ 调用关系 → 递归检测 → 深度计算
         
Stage 6: Loop Analysis (循环优化)
         ↓ 循环识别 → 不变量 → 性能建议
         
Stage 7: Unified Pipeline (统一编排)
         ↓ 整合所有分析结果 → 生成报告
```

**关键算法**:
- **DFS循环检测**: O(V+E)，检测递归调用链
- **BFS可达性**: O(V+E)，标记可达节点
- **调用图深度**: BFS计算函数调用层级

**输出数据结构**:
```rust
pub struct CompleteAnalysisResult {
    pub scope_tree: ScopeTree,           // Stage 1
    pub swc_ast: ExtractedAst,           // Stage 2
    pub dataflow: DataFlowGraph,         // Stage 3
    pub controlflow: ControlFlowGraph,   // Stage 4
    pub call_graph: EnhancedCallGraph,   // Stage 5
    pub loops: Vec<LoopAnalysis>,        // Stage 6
    pub summary: AnalysisSummary,        // Stage 7
}
```

### 3.4 真实网站学习流程

**代码**: [crates/browerai-learning/src/real_website_learner.rs:56](../crates/browerai-learning/src/real_website_learner.rs#L56)

```rust
pub async fn learn_website(&self, task: WebsiteLearningTask) -> Result<LearningSession>
```

**7步流程**:

```
步骤1: 获取页面 (fetch_page)
       ↓ HTTP GET → 保存原始HTML
       
步骤2: 注入V8追踪器 (inject_tracers_to_html)
       ↓ 插入追踪代码 → 监听function_calls、dom_operations、event_listeners
       
步骤3: 模拟用户交互 (simulate_interactions)
       ↓ 点击按钮 → 填写表单 → 滚动页面
       
步骤4: 提取追踪数据 (extract_traces_from_window)
       ↓ ExecutionTrace → function_calls数组、dom_operations数组
       
步骤5: 识别工作流 (extract_workflows)
       ↓ 事件序列 → 工作流模式 → 关键路径
       
步骤6: 评估学习质量 (evaluate)
       ↓ 质量评分（0-1）→ <0.7警告，>0.9优秀
       
步骤7: 生成学习代码 (generate_learning_code)
       ↓ 可复现的JS代码 → 用于训练数据
```

**质量评估标准**:
- 函数调用覆盖率 > 80%
- DOM操作捕获完整性 > 70%
- 事件处理识别准确率 > 85%

---

## 阶段4: 动手实践（5小时）

### 4.1 环境准备

```bash
# 1. 克隆项目
git clone https://github.com/vistone/BrowerAI.git
cd BrowerAI

# 2. 安装Rust（如未安装）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 3. 构建项目（无AI特性）
cargo build --release

# 4. 运行测试（验证环境）
cargo test --workspace --release
```

### 4.2 实验1: JS反混淆演示

```bash
# 运行反混淆demo
cargo run --example enhanced_js_deobfuscation_demo
```

**预期输出**:
```
=== Test 1: String Array Unpacking ===
Before: var _0xabc = ['Hello', 'World', 'JavaScript'];
After: console.log('Hello World JavaScript');
Transformations: 3
String arrays unpacked: 1

=== Test 2: Proxy Function Removal ===
...
```

**代码分析**:
打开 [crates/browerai/examples/enhanced_js_deobfuscation_demo.rs](../crates/browerai/examples/enhanced_js_deobfuscation_demo.rs)，学习：
- `EnhancedDeobfuscator::new()` 初始化
- `deobfuscate()` 方法的调用
- `DeobfuscationResult` 结构体

### 4.3 实验2: 完整学习与生成流水线

```bash
# 运行complete demo
cargo run --bin browerai -- learn https://example.com output/ 3
```

**参数说明**:
- `learn`: 命令模式
- `https://example.com`: 目标URL
- `output/`: 输出目录
- `3`: 生成3个变体

**输出检查**:
```bash
# 查看生成的变体
ls -lh output/variant_*/
cat output/variant_1/preserved_features.json
cat output/pipeline_report.json
```

### 4.4 实验3: 真实数据训练

```bash
# 使用Justfile快捷命令
just learn-real

# 或手动运行Python训练脚本
cd training
python scripts/train_mixed_model_v2.py \
    --data-dir ../data/real_codes \
    --epochs 50 \
    --batch-size 32 \
    --device cuda
```

**观察训练过程**:
```
Epoch 1/50: loss=0.532, accuracy=0.654
Epoch 10/50: loss=0.231, accuracy=0.842
Epoch 50/50: loss=0.089, accuracy=0.984
```

### 4.5 实验4: 阅读关键测试

**测试1: 调用图分析**
```bash
cargo test --test phase3_week3_enhanced_call_graph_tests -- --nocapture
```

**打开测试代码**: [tests/phase3/phase3_week3_enhanced_call_graph_tests.rs](../tests/phase3/phase3_week3_enhanced_call_graph_tests.rs)

**学习重点**:
- `test_recursive_chain_detection` - 递归检测算法
- `test_call_depth_calculation` - 深度计算算法
- `test_integration_with_dataflow` - 多分析器集成

**测试2: 智能渲染**
```bash
cargo test -p browerai-intelligent-rendering -- --nocapture
```

---

## 阶段5: 深入源码（持续学习）

### 5.1 源码阅读顺序

**第1周: 解析层**
1. [crates/browerai-html-parser/src/parser.rs](../crates/browerai-html-parser/src/parser.rs) - HTML解析入口
2. [crates/browerai-css-parser/src/parser.rs](../crates/browerai-css-parser/src/parser.rs) - CSS解析流程
3. [crates/browerai-js-parser/src/parser.rs](../crates/browerai-js-parser/src/parser.rs) - Boa集成

**第2周: JS分析管道** ⭐⭐⭐
1. [crates/browerai-js-analyzer/src/scope_analyzer.rs](../crates/browerai-js-analyzer/src/scope_analyzer.rs) - Stage 1
2. [crates/browerai-js-analyzer/src/dataflow_analyzer.rs](../crates/browerai-js-analyzer/src/dataflow_analyzer.rs) - Stage 3
3. [crates/browerai-js-analyzer/src/controlflow_analyzer.rs](../crates/browerai-js-analyzer/src/controlflow_analyzer.rs) - Stage 4
4. [crates/browerai-js-analyzer/src/enhanced_call_graph.rs](../crates/browerai-js-analyzer/src/enhanced_call_graph.rs) - Stage 5
5. [crates/browerai-js-analyzer/src/analysis_pipeline.rs](../crates/browerai-js-analyzer/src/analysis_pipeline.rs) - Stage 7

**第3周: 智能渲染系统**
1. [crates/browerai-intelligent-rendering/src/website_learning_engine.rs](../crates/browerai-intelligent-rendering/src/website_learning_engine.rs)
2. [crates/browerai-intelligent-rendering/src/reasoning.rs](../crates/browerai-intelligent-rendering/src/reasoning.rs)
3. [crates/browerai-intelligent-rendering/src/generation.rs](../crates/browerai-intelligent-rendering/src/generation.rs)
4. [crates/browerai-intelligent-rendering/src/model_orchestrator.rs](../crates/browerai-intelligent-rendering/src/model_orchestrator.rs)

**第4周: 学习系统**
1. [crates/browerai-learning/src/real_website_learner.rs](../crates/browerai-learning/src/real_website_learner.rs)
2. [crates/browerai-learning/src/workflow_extractor.rs](../crates/browerai-learning/src/workflow_extractor.rs)
3. [crates/browerai-learning/src/learning_quality.rs](../crates/browerai-learning/src/learning_quality.rs)
4. [crates/browerai-learning/src/website_generator.rs](../crates/browerai-learning/src/website_generator.rs)

**第5周: 反混淆系统**
1. [crates/browerai-deobfuscation/src/lib.rs](../crates/browerai-deobfuscation/src/lib.rs) - 总览
2. [crates/browerai-deobfuscation/src/strategies/](../crates/browerai-deobfuscation/src/strategies/) - 18种策略

### 5.2 关键数据结构

**ExtractedAst** (JS分析核心):
```rust
pub struct ExtractedAst {
    pub semantic: JsSemanticInfo,    // 函数、变量、作用域
    pub module_info: JsModuleInfo,   // ES模块导入导出
    pub locations: HashMap<String, LocationInfo>,  // 代码位置
}
```

**JsCallGraph** (调用图):
```rust
pub struct JsCallGraph {
    pub nodes: Vec<CallGraphNode>,   // 函数节点
    pub cycles: Vec<Vec<String>>,    // 递归循环
    pub entry_points: Vec<String>,   // 入口函数
}
```

**SiteUnderstanding** (网站理解):
```rust
pub struct SiteUnderstanding {
    pub core_structure: CoreStructure,        // DOM结构
    pub feature_categories: Vec<FeatureCategory>,  // 功能分类
    pub event_handlers: Vec<EventHandler>,    // 事件处理
    pub state_management: StateManagement,    // 状态管理
}
```

### 5.3 调试技巧

**技巧1: 启用详细日志**
```bash
RUST_LOG=debug cargo test test_name -- --nocapture
```

**技巧2: 查看分析结果**
```rust
let result = analyzer.analyze(code)?;
println!("{:#?}", result);  // 美化输出
```

**技巧3: 单步测试**
```bash
# 只运行特定测试
cargo test --test phase3_week3_enhanced_call_graph_tests::test_recursive_chain_detection
```

**技巧4: 检查生成的代码**
```bash
# 查看实际生成的HTML/CSS/JS
cat output/variant_1/index.html | head -50
```

---

## 📊 学习检查清单

完成以下检查项，确认您已掌握BrowerAI：

### 基础理解 ✓
- [ ] 能用一句话解释"保功能、换体验"
- [ ] 理解AI增强而非替代的设计原则
- [ ] 知道27个crate的分层结构
- [ ] 了解7阶段JS分析管道的作用

### 流程掌握 ✓
- [ ] 能描述`learn_and_generate`的5个阶段
- [ ] 理解智能推理的4个步骤（识别→发现→生成→创建）
- [ ] 知道如何验证功能完整性
- [ ] 了解真实网站学习的7步流程

### 实践能力 ✓
- [ ] 成功运行反混淆demo
- [ ] 能本地训练一个简单模型
- [ ] 可以阅读并理解关键测试代码
- [ ] 能调试一个简单的解析问题

### 深度理解 ✓
- [ ] 阅读过至少5个核心源码文件
- [ ] 理解DFS/BFS算法在调用图中的应用
- [ ] 知道18种反混淆策略的分类
- [ ] 了解多层缓存的设计原理

---

## 🚀 下一步学习资源

### 深入文档
- [CORE_DESIGN_PHILOSOPHY.md](CORE_DESIGN_PHILOSOPHY.md) - 完整设计哲学（1020行）
- [TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md) - 技术实现细节（800行）
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) - 设计决策日志（600行）
- [PROJECT_EVOLUTION_STORY.md](PROJECT_EVOLUTION_STORY.md) - 项目演进故事（400行）

### 架构文档
- [docs/architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md) - 系统架构
- [docs/architecture/WORKSPACE_ARCHITECTURE.md](architecture/WORKSPACE_ARCHITECTURE.md) - Workspace详解

### 专题指南
- [docs/deobfuscation/](deobfuscation/) - 反混淆技术指南
- [docs/learning/](learning/) - 学习系统详解
- [docs/testing/COMPREHENSIVE_TESTING.md](testing/COMPREHENSIVE_TESTING.md) - 测试策略

### 实战教程
- [DEVELOPMENT_GUIDE.md](../DEVELOPMENT_GUIDE.md) - 开发指南（509行）
- [CONTRIBUTING.md](../CONTRIBUTING.md) - 贡献指南
- [training/QUICKSTART.md](../training/QUICKSTART.md) - Python训练快速开始

---

## 💬 获取帮助

**遇到问题？**
1. 查看 [docs/guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)
2. 搜索现有测试代码（459+测试用例）
3. 阅读 [CHANGELOG.md](../CHANGELOG.md) 查看功能演进

**想贡献？**
1. 从小任务开始（标记为`good-first-issue`的issue）
2. 阅读 [PROJECT_STANDARDS.md](PROJECT_STANDARDS.md)
3. 提交PR前运行`cargo test --workspace`

---

**祝您学习愉快！记住核心口号：保功能、换体验！** 🎉
