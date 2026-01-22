# BrowerAI 学习模块 - 快速参考

## 主要组件使用示例

### 1. 代码验证器 (Code Verifier)
```rust
use browerai_learning::code_verifier::CodeVerifier;

// 验证 HTML 代码
let verifier = CodeVerifier::new();
let result = verifier.verify_html("<div>Hello</div>").unwrap();
println!("HTML 有效性: {}", result.is_valid);  // true
println!("评分: {}", result.score);  // 0.0-1.0

// 获取改进建议
for suggestion in result.suggestions {
    println!("建议: {}", suggestion);
}
```

### 2. 语义比较器 (Semantic Comparator)
```rust
use browerai_learning::semantic_comparator::SemanticComparator;

let comparator = SemanticComparator::new();

// 比较两个 HTML 结构
let original = "<div><p>Hello</p><p>World</p></div>";
let generated = "<div><p>Hello</p><p>World</p></div>";
let dom_similarity = comparator.compare_dom(original, generated)?;
println!("DOM 相似度: {}", dom_similarity.similarity); // 0.0-1.0

// 比较 JavaScript 函数
let original_js = "function add(a, b) { return a + b; }";
let generated_js = "function sum(x, y) { return x + y; }";
let func_similarity = comparator.compare_functions(original_js, generated_js)?;
println!("函数相似度: {}", func_similarity.similarity);

// 获取综合比较结果
let all_comparison = comparator.compare_all(original, generated, "", "")?;
println!("综合相似度: {}", all_comparison.overall_similarity); // 0.0-1.0
```

### 3. 学习质量评估
```rust
use browerai_learning::learning_quality::{LearningQuality, SemanticComparisonResult};

let mut quality = LearningQuality {
    original_code: "...".to_string(),
    generated_code: "...".to_string(),
    semantic_comparison: Some(SemanticComparisonResult {
        dom_similarity: 0.95,
        event_similarity: 0.88,
        css_similarity: 0.92,
        function_similarity: 0.85,
        overall_similarity: 0.90,
    }),
    code_equivalence_score: Some(0.87),
    // ... 其他字段
};

// 使用语义比较进行评估
let score = quality.evaluate_with_comparison();
println!("学习质量评分: {}", score); // 0.0-1.0
```

### 4. 双沙箱学习器
```rust
use browerai_learning::dual_sandbox_learner::DualSandboxLearner;

let mut learner = DualSandboxLearner::new();

// 从参考代码学习
let result = learner.learn_and_generate_with_reference(
    original_code,
    reference_code,
    "html"
)?;

println!("生成的代码: {}", result.generated_code);
println!("相似度: {}", result.similarity_score);
println!("学习质量: {}", result.learning_quality.overall_score);
```

## 关键功能说明

### CodeVerifier
- **目的**：验证 HTML/CSS/JS 代码的语法和结构
- **输入**：代码字符串
- **输出**：验证结果 + 改进建议
- **应用**：质量保证、代码审查

### SemanticComparator
- **目的**：比较原始代码和生成代码的相似度
- **方法**：
  - `compare_dom()` - 比较 HTML 结构
  - `compare_events()` - 比较事件处理
  - `compare_css()` - 比较 CSS 规则
  - `compare_functions()` - 比较函数实现
  - `compare_all()` - 综合比较
- **相似度度量**：Jaccard 指数 (0.0-1.0)
- **应用**：学习评估、生成质量验证

### LearningQuality
- **增强字段**：
  - `semantic_comparison: Option<SemanticComparisonResult>` - 语义比较结果
  - `code_equivalence_score: Option<f64>` - 代码等价性评分
- **方法**：
  - `evaluate()` - 基础评估
  - `evaluate_with_comparison()` - 使用语义比较的评估

### DualSandboxLearner
- **特性**：
  - 从原始代码和参考代码学习
  - 生成语义上相似的代码
  - 记录学习质量指标
- **应用**：智能代码生成、学习-生成循环

## 测试覆盖

| 模块 | 测试数 | 状态 |
|------|--------|------|
| code_verifier | 5 | ✅ |
| semantic_comparator | 4 | ✅ |
| learning_quality | 3 | ✅ |
| dual_sandbox_learner | 1 | ✅ |
| 其他模块 | 205+ | ✅ |
| **总计** | **223** | **✅** |

## 构建和测试

```bash
# 编译学习模块
cargo build -p browerai-learning

# 运行所有单元测试
cargo test -p browerai-learning --lib

# 运行特定模块的测试
cargo test -p browerai-learning --lib code_verifier::tests
cargo test -p browerai-learning --lib semantic_comparator::tests

# 运行包括集成测试的所有测试
cargo test -p browerai-learning

# 生成文档
cargo doc -p browerai-learning --open
```

## 依赖项

- **anyhow** - 错误处理
- **regex** - 正则表达式
- **serde** - 序列化
- **tokio** - 异步运行时
- **reqwest** - HTTP 客户端
- 以及其他工作区依赖

## 扩展指南

### 添加新的验证规则
在 `code_verifier.rs` 中的 `verify_*()` 方法中添加新的检查逻辑

### 添加新的相似度指标
在 `semantic_comparator.rs` 中实现新的 `compare_*()` 方法

### 自定义学习策略
扩展 `DualSandboxLearner` 或创建新的学习器实现

---

**文档版本**：1.0  
**最后更新**：2025-01-09  
**项目阶段**：Phase B (语义比较) 完成
