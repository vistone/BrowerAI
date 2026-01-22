# 双沙箱学习模块增强计划

## 1. 当前状态分析

### 现有功能架构
```
DualSandboxLearner (主入口)
  ├─ Step 1: V8 执行追踪 (v8_tracer.rs)
  │   └─ ExecutionTrace 记录：函数调用、DOM操作、用户事件
  │
  ├─ Step 2: 工作流提取 (workflow_extractor.rs)
  │   ├─ 识别用户交互 → 操作链 → 关键函数
  │   ├─ 计算工作流复杂度和重要性
  │   └─ WorkflowExtractionResult 输出
  │
  ├─ Step 3: 学习质量评估 (learning_quality.rs)
  │   ├─ 函数覆盖率 (学到的函数/总函数)
  │   ├─ 工作流完整性 (完整工作流/总工作流)
  │   └─ 功能保留度 (可重建功能/总功能)
  │
  ├─ Step 4: 代码生成 (dual_sandbox_learner.rs lines 100-200)
  │   ├─ 生成语义 HTML (基于工作流)
  │   ├─ 生成语义 CSS (工作流特定的样式)
  │   └─ 生成 JavaScript 框架 (函数骨架)
  │
  └─ Step 5: 学习总结 (learningSummary生成)
      └─ 关键函数、DOM模式、交互流、综合评分
```

### 现有的限制

#### 1️⃣ **缺乏对比分析机制** (Comparative Learning Gap)
- **当前**: 生成代码后无法与原始代码进行语义对比
- **问题**: 无法衡量"是否保留了原始逻辑"
- **影响**: 学习质量评估基于启发式规则，不够准确

```rust
// 现有代码质量评估只看数量，不看内容
pub fn calc_functionality_preserved(workflows) {
    // 只计算：能重建的函数数 / 总函数数
    // ❌ 无法验证生成的函数是否等价于原始函数
}
```

#### 2️⃣ **缺乏验证机制** (Verification Gap)
- **当前**: 生成代码后无法自动验证正确性
- **问题**: 无法发现"生成的代码是否能正常工作"
- **影响**: 生成的代码质量参差不齐

```rust
// 生成代码后直接输出，无验证步骤
fn learn_and_generate() {
    // Step 4: 生成代码
    let generated = self.generate_semantic_code()?;
    
    // ❌ 缺少这一步：验证生成的代码
    // - 是否能解析成有效的HTML/CSS/JS？
    // - 是否能执行而不报错？
    // - 是否与原始代码功能等价？
}
```

#### 3️⃣ **缺乏反馈优化循环** (Learning Loop Gap)
- **当前**: 单向学习 → 生成，无反馈优化
- **问题**: 无法基于验证结果改进模型
- **影响**: 多次学习相同代码也不会改进

#### 4️⃣ **缺乏语义对比工具** (Semantic Comparison Gap)
- **当前**: 无法比较原始和生成代码的语义等价性
- **问题**: 无法衡量"学到了多少真正的逻辑"
- **影响**: 工作流提取可能遗漏重要交互

---

## 2. 增强方案设计

### Phase A: 代码验证层 (Code Verification Layer)

#### 目标
在生成代码后立即进行自动验证，确保生成的代码：
- ✅ 语法正确（能解析）
- ✅ 能执行（无运行时错误）
- ✅ 符合原始意图（功能保留）

#### 实现

**A1. 新增: `CodeVerifier` 模块**

```rust
// crates/browerai-learning/src/code_verifier.rs

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeVerificationResult {
    /// HTML验证结果
    pub html_valid: bool,
    pub html_errors: Vec<String>,
    
    /// CSS验证结果
    pub css_valid: bool,
    pub css_errors: Vec<String>,
    
    /// JavaScript验证结果
    pub js_valid: bool,
    pub js_parse_errors: Vec<String>,
    pub js_runtime_errors: Vec<String>,
    
    /// 综合验证评分 (0-1)
    pub verification_score: f64,
    
    /// 建议的修复
    pub suggested_fixes: Vec<(String, String)>, // (错误, 修复建议)
}

pub struct CodeVerifier;

impl CodeVerifier {
    /// 验证生成的HTML
    pub fn verify_html(html: &str) -> Result<HtmlVerification> {
        // 使用 html5ever 解析
        // 记录解析错误、无效标签等
    }
    
    /// 验证生成的CSS
    pub fn verify_css(css: &str) -> Result<CssVerification> {
        // 使用 cssparser 验证
        // 检查选择器有效性、属性合法性等
    }
    
    /// 验证生成的JavaScript
    pub fn verify_js(js: &str) -> Result<JsVerification> {
        // 使用 boa_parser 解析检查语法
        // 尝试在沙箱中执行检查运行时错误
    }
    
    /// 综合验证
    pub fn verify_all(
        html: &str,
        css: &str,
        js: &str,
    ) -> Result<CodeVerificationResult> {
        let html_result = Self::verify_html(html)?;
        let css_result = Self::verify_css(css)?;
        let js_result = Self::verify_js(js)?;
        
        // 组合评分
        let score = (
            html_result.score() * 0.3 +
            css_result.score() * 0.2 +
            js_result.score() * 0.5
        );
        
        Ok(CodeVerificationResult { /* ... */ })
    }
}
```

**A2. 集成到 DualSandboxLearner**

```rust
// crates/browerai-learning/src/dual_sandbox_learner.rs

impl DualSandboxLearner {
    pub async fn learn_and_generate(&self, url: &str) -> Result<DualSandboxLearningResult> {
        // ... 现有 Step 1-4 ...
        
        // 🆕 Step 4.5: 验证生成的代码
        let verification = CodeVerifier::verify_all(
            &generated_html,
            &generated_css,
            &generated_js,
        )?;
        
        // 如果验证失败，记录问题但继续
        if verification.verification_score < 0.8 {
            log::warn!(
                "⚠️ 代码验证评分低: {:.1}%",
                verification.verification_score * 100.0
            );
            
            // 应用建议的修复
            let fixed_html = Self::apply_fixes(&generated_html, &verification.html_errors)?;
            let fixed_js = Self::apply_fixes(&generated_js, &verification.js_errors)?;
            // ...
        }
        
        // ... 现有 Step 5 ...
    }
}
```

---

### Phase B: 语义对比层 (Semantic Comparison Layer)

#### 目标
比较原始代码和生成代码的语义等价性，衡量学习质量

#### 实现

**B1. 新增: `SemanticComparator` 模块**

```rust
// crates/browerai-learning/src/semantic_comparator.rs

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticComparisonResult {
    /// 函数级别对比
    pub function_similarity: FunctionSimilarity,
    
    /// DOM结构相似度 (0-1)
    pub dom_structure_similarity: f64,
    
    /// 事件处理相似度 (0-1)
    pub event_handling_similarity: f64,
    
    /// 样式相似度 (0-1)
    pub style_similarity: f64,
    
    /// 综合相似度 (0-1)
    pub overall_similarity: f64,
    
    /// 缺失的部分
    pub missing_features: Vec<String>,
    
    /// 额外的部分
    pub extra_features: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionSimilarity {
    /// 每个关键函数的相似度评分
    pub function_scores: HashMap<String, f64>,
    
    /// 覆盖的函数 (生成代码中出现)
    pub covered_functions: Vec<String>,
    
    /// 遗漏的函数 (原始代码中有但生成代码无)
    pub missing_functions: Vec<String>,
}

pub struct SemanticComparator;

impl SemanticComparator {
    /// 对比HTML DOM结构
    pub fn compare_dom_structure(
        original_html: &str,
        generated_html: &str,
    ) -> Result<f64> {
        // 解析两个HTML为DOM树
        let original_tree = parse_html(original_html)?;
        let generated_tree = parse_html(generated_html)?;
        
        // 计算树的编辑距离
        // 返回相似度: 1 - (编辑距离 / 最大可能距离)
    }
    
    /// 对比JavaScript函数
    pub fn compare_js_functions(
        original_js: &str,
        generated_js: &str,
        key_functions: &[String],
    ) -> Result<FunctionSimilarity> {
        // 解析JS AST
        let original_ast = parse_js(original_js)?;
        let generated_ast = parse_js(generated_js)?;
        
        // 对每个关键函数，计算AST相似度
        // 使用结构化比较：参数数量、返回语句、调用链等
    }
    
    /// 对比CSS样式规则
    pub fn compare_css_rules(
        original_css: &str,
        generated_css: &str,
    ) -> Result<f64> {
        // 解析CSS规则集
        // 比较选择器覆盖和属性定义
    }
    
    /// 综合对比
    pub fn compare_all(
        original_html: &str,
        original_css: &str,
        original_js: &str,
        generated_html: &str,
        generated_css: &str,
        generated_js: &str,
        key_functions: &[String],
    ) -> Result<SemanticComparisonResult> {
        let dom_sim = Self::compare_dom_structure(original_html, generated_html)?;
        let js_sim = Self::compare_js_functions(original_js, generated_js, key_functions)?;
        let css_sim = Self::compare_css_rules(original_css, generated_css)?;
        
        // 加权综合
        let overall = dom_sim * 0.4 + js_sim.avg_score() * 0.4 + css_sim * 0.2;
        
        Ok(SemanticComparisonResult { /* ... */ })
    }
}
```

**B2. 集成到 LearningQuality**

```rust
// crates/browerai-learning/src/learning_quality.rs

pub struct LearningQuality {
    // ... 现有字段 ...
    
    /// 🆕 语义对比结果
    pub semantic_comparison: Option<SemanticComparisonResult>,
    
    /// 🆕 原始和生成代码的相似度
    pub code_equivalence_score: f64,
}

impl LearningQuality {
    pub fn evaluate_with_comparison(
        traces: &ExecutionTrace,
        workflows: &WorkflowExtractionResult,
        original_html: &str,
        original_css: &str,
        original_js: &str,
        generated_html: &str,
        generated_css: &str,
        generated_js: &str,
    ) -> Result<Self> {
        // ... 现有评估 ...
        
        // 🆕 添加语义对比
        let comparison = SemanticComparator::compare_all(
            original_html, original_css, original_js,
            generated_html, generated_css, generated_js,
            &Self::extract_key_functions(workflows),
        )?;
        
        // 将对比结果融入综合评分
        let with_equivalence_score = (
            func_coverage * 0.3 +
            workflow_completeness * 0.3 +
            func_preserved * 0.2 +
            comparison.overall_similarity * 0.2  // 🆕
        );
        
        Ok(LearningQuality {
            semantic_comparison: Some(comparison),
            code_equivalence_score: comparison.overall_similarity,
            overall_score: with_equivalence_score,
            // ... 其他字段 ...
        })
    }
}
```

---

### Phase C: 反馈优化层 (Feedback Optimization Layer)

#### 目标
基于验证和对比结果，优化后续的代码生成

#### 实现

**C1. 新增: `ComparisonFeedback` 模块**

```rust
// crates/browerai-learning/src/comparison_feedback.rs

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparisonFeedback {
    /// 验证反馈
    pub verification_feedback: CodeVerificationFeedback,
    
    /// 语义对比反馈
    pub comparison_feedback: SemanticComparisonFeedback,
    
    /// 推荐的改进方向
    pub improvement_strategies: Vec<ImprovementStrategy>,
    
    /// 学习置信度 (0-1)
    /// 基于: 验证评分 + 对比相似度 + 完整性
    pub learning_confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ImprovementStrategy {
    /// 重新提取工作流（可能遗漏了重要交互）
    ReextractWorkflows,
    
    /// 增加函数追踪深度
    IncreaseTraceDepth,
    
    /// 改进代码生成策略（当前生成质量低）
    ImproveCodeGeneration {
        /// 具体的生成规则建议
        rules: Vec<String>,
    },
    
    /// 需要手动审查某些函数
    ManualReviewRequired {
        functions: Vec<String>,
    },
    
    /// 使用更高级的模型重新学习
    UseAdvancedModel,
}

pub struct ComparisonFeedbackGenerator;

impl ComparisonFeedbackGenerator {
    pub fn generate(
        verification: &CodeVerificationResult,
        comparison: &SemanticComparisonResult,
        workflows: &WorkflowExtractionResult,
    ) -> Result<ComparisonFeedback> {
        let verification_fb = Self::analyze_verification(verification);
        let comparison_fb = Self::analyze_comparison(comparison);
        let strategies = Self::recommend_strategies(&verification_fb, &comparison_fb, workflows);
        
        let confidence = (
            verification.verification_score * 0.4 +
            comparison.overall_similarity * 0.6
        );
        
        Ok(ComparisonFeedback {
            verification_feedback: verification_fb,
            comparison_feedback: comparison_fb,
            improvement_strategies: strategies,
            learning_confidence: confidence,
        })
    }
    
    fn recommend_strategies(
        verification_fb: &CodeVerificationFeedback,
        comparison_fb: &SemanticComparisonFeedback,
        workflows: &WorkflowExtractionResult,
    ) -> Vec<ImprovementStrategy> {
        let mut strategies = vec![];
        
        // 基于反馈推荐改进方向
        if comparison_fb.missing_function_count > workflows.workflows.len() / 3 {
            strategies.push(ImprovementStrategy::ReextractWorkflows);
        }
        
        if verification_fb.parse_error_count > 5 {
            strategies.push(ImprovementStrategy::ImproveCodeGeneration {
                rules: vec!["增加类型注解".to_string()],
            });
        }
        
        if comparison_fb.missing_functions.len() > 10 {
            strategies.push(ImprovementStrategy::ManualReviewRequired {
                functions: comparison_fb.missing_functions.clone(),
            });
        }
        
        strategies
    }
}
```

**C2. 集成到 DualSandboxLearner**

```rust
// crates/browerai-learning/src/dual_sandbox_learner.rs

pub struct DualSandboxLearningResult {
    // ... 现有字段 ...
    
    /// 🆕 对比和验证反馈
    pub feedback: Option<ComparisonFeedback>,
}

impl DualSandboxLearner {
    pub async fn learn_and_generate(&self, url: &str) -> Result<DualSandboxLearningResult> {
        // ... Step 1-4 ...
        
        // 🆕 生成对比和验证反馈
        let feedback = ComparisonFeedbackGenerator::generate(
            &verification,
            &comparison,
            &workflows,
        )?;
        
        log::info!(
            "📊 学习置信度: {:.1}%, 建议改进: {:?}",
            feedback.learning_confidence * 100.0,
            feedback.improvement_strategies
        );
        
        // ... 返回结果，包含反馈 ...
        Ok(DualSandboxLearningResult {
            feedback: Some(feedback),
            // ... 其他字段 ...
        })
    }
}
```

---

### Phase D: 对比学习主入口 (Comparative Learning API)

#### 目标
为用户提供简洁的"对比学习"高级API

#### 实现

**D1. 新增: `ComparativeLearner` 接口**

```rust
// crates/browerai-learning/src/comparative_learner.rs

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparativeLearningReport {
    /// 追踪信息
    pub execution_traces: ExecutionTrace,
    
    /// 提取的工作流
    pub workflows: WorkflowExtractionResult,
    
    /// 代码质量评估（包含对比）
    pub quality: LearningQuality,
    
    /// 生成的代码
    pub generated: GeneratedCode,
    
    /// 验证结果
    pub verification: CodeVerificationResult,
    
    /// 语义对比
    pub comparison: SemanticComparisonResult,
    
    /// 反馈和改进建议
    pub feedback: ComparisonFeedback,
    
    /// 总体学习评分 (0-100)
    /// 综合所有方面的综合评分
    pub overall_learning_score: u32,
}

pub struct ComparativeLearner;

impl ComparativeLearner {
    /// 🎯 主要API：对比学习一个网站
    /// 
    /// # 参数
    /// - `original_html/css/js`: 原始网站代码
    /// - `url`: 网站URL（用于追踪）
    /// 
    /// # 返回
    /// 详细的对比学习报告，包含：
    /// 1. 执行追踪
    /// 2. 工作流提取
    /// 3. 代码验证
    /// 4. 语义对比
    /// 5. 质量评估
    /// 6. 改进建议
    pub async fn learn_and_compare(
        original_html: String,
        original_css: String,
        original_js: String,
        url: &str,
    ) -> Result<ComparativeLearningReport> {
        log::info!("🔄 开始对比学习: {}", url);
        
        // Step 1: 执行追踪
        let traces = trace_execution(url, &original_html, &original_js).await?;
        
        // Step 2: 工作流提取
        let workflows = WorkflowExtractor::extract_workflows(&traces)?;
        
        // Step 3: 代码生成
        let learner = DualSandboxLearner::new()?;
        let learning_result = learner.learn_and_generate_with_original(
            url,
            &original_html,
            &original_css,
            &original_js,
        ).await?;
        
        // Step 4: 代码验证
        let verification = CodeVerifier::verify_all(
            &learning_result.generated_html,
            &learning_result.generated_css,
            &learning_result.generated_js,
        )?;
        
        // Step 5: 语义对比
        let comparison = SemanticComparator::compare_all(
            &original_html, &original_css, &original_js,
            &learning_result.generated_html,
            &learning_result.generated_css,
            &learning_result.generated_js,
            &Self::extract_key_functions(&workflows),
        )?;
        
        // Step 6: 质量评估（包含对比）
        let quality = LearningQuality::evaluate_with_comparison(
            &traces,
            &workflows,
            &original_html, &original_css, &original_js,
            &learning_result.generated_html,
            &learning_result.generated_css,
            &learning_result.generated_js,
        )?;
        
        // Step 7: 反馈生成
        let feedback = ComparisonFeedbackGenerator::generate(
            &verification,
            &comparison,
            &workflows,
        )?;
        
        // 综合评分
        let overall_score = (
            quality.overall_score * 0.35 +
            verification.verification_score * 0.25 +
            comparison.overall_similarity * 0.25 +
            (1.0 - feedback.learning_confidence) * 0.15
        ) * 100.0;
        
        log::info!(
            "✅ 对比学习完成: 总体评分 {:.0}/100",
            overall_score
        );
        
        Ok(ComparativeLearningReport {
            execution_traces: traces,
            workflows,
            quality,
            generated: learning_result.generated,
            verification,
            comparison,
            feedback,
            overall_learning_score: overall_score as u32,
        })
    }
    
    /// 批量对比学习多个网站
    pub async fn batch_learn_and_compare(
        websites: Vec<WebsiteCode>,
    ) -> Result<Vec<ComparativeLearningReport>> {
        let mut reports = vec![];
        for website in websites {
            let report = Self::learn_and_compare(
                website.html,
                website.css,
                website.js,
                &website.url,
            ).await?;
            reports.push(report);
        }
        Ok(reports)
    }
}
```

---

## 3. 实现计划

### 第一阶段：基础验证层（1-2天）
- [ ] 创建 `code_verifier.rs` 模块
  - 实现HTML验证（html5ever解析）
  - 实现CSS验证（cssparser解析）
  - 实现JS验证（boa_parser解析 + 沙箱执行）
- [ ] 集成到 `DualSandboxLearner`
- [ ] 编写单元测试

### 第二阶段：语义对比层（1-2天）
- [ ] 创建 `semantic_comparator.rs` 模块
  - 实现DOM树对比
  - 实现JS函数对比
  - 实现CSS规则对比
- [ ] 集成到 `LearningQuality`
- [ ] 编写单元测试

### 第三阶段：反馈优化层（1天）
- [ ] 创建 `comparison_feedback.rs` 模块
  - 实现反馈分析
  - 实现改进建议生成
- [ ] 集成到 `DualSandboxLearner`
- [ ] 编写单元测试

### 第四阶段：高级API（0.5天）
- [ ] 创建 `comparative_learner.rs` 模块
  - 实现统一的学习API
  - 实现批量学习接口
- [ ] 编写示例和文档
- [ ] 编写集成测试

### 第五阶段：测试和优化（1天）
- [ ] 运行所有测试
- [ ] 性能优化
- [ ] 文档完善
- [ ] GitHub提交

---

## 4. 关键改进点总结

| 维度 | 当前状态 | 增强后 | 收益 |
|------|--------|-------|------|
| **代码质量** | 无验证 | 自动验证HTML/CSS/JS | ✅ 保证生成代码可执行 |
| **学习评估** | 启发式 | 语义对比评估 | ✅ 更准确的学习质量测量 |
| **反馈机制** | 无 | 完整的反馈和改进建议 | ✅ 支持迭代改进 |
| **用户体验** | 低级API | 高级ComparativeLearner API | ✅ 易用的统一接口 |
| **可信度** | 一般 | 综合置信度评分 | ✅ 明确的可信度指标 |

---

## 5. 文件结构更新

```
crates/browerai-learning/src/
  ├── lib.rs                          (更新module声明)
  ├── dual_sandbox_learner.rs         (集成验证和对比)
  ├── learning_quality.rs             (添加对比结果)
  ├── workflow_extractor.rs           (现有，无修改)
  ├── v8_tracer.rs                    (现有，无修改)
  ├── code_verifier.rs                ✨ 新增（Phase A）
  ├── semantic_comparator.rs          ✨ 新增（Phase B）
  ├── comparison_feedback.rs          ✨ 新增（Phase C）
  ├── comparative_learner.rs          ✨ 新增（Phase D）
  └── ...其他模块
```

---

## 6. 测试策略

### 单元测试
- `code_verifier_tests.rs`: HTML/CSS/JS验证
- `semantic_comparator_tests.rs`: 对比逻辑
- `comparison_feedback_tests.rs`: 反馈生成

### 集成测试
- `comparative_learning_tests.rs`: 完整流程
- `batch_learning_tests.rs`: 批量学习

### 使用示例
```rust
// 示例：对比学习一个网站
let report = ComparativeLearner::learn_and_compare(
    original_html,
    original_css,
    original_js,
    "https://example.com"
).await?;

println!("学习评分: {}/100", report.overall_learning_score);
println!("缺失函数: {:?}", report.comparison.missing_features);
println!("改进建议: {:?}", report.feedback.improvement_strategies);
```

---

## 7. 成功指标

- ✅ 代码验证覆盖 100% 的生成代码
- ✅ 语义对比准确度 ≥ 85%
- ✅ 学习置信度评分准确
- ✅ 所有测试通过（659+ 现有测试 + 新增测试）
- ✅ 预提交检查通过所有 11 个阶段
- ✅ 文档完善且易理解
