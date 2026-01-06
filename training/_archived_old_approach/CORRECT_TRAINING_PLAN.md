# BrowerAI 正确训练方案

## ❌ 错误的训练目标（已停止）

之前的训练方向**完全偏离了BrowerAI的核心目标**：
- ❌ 框架分类（React/Vue/Angular识别）
- ❌ 网站分类（新闻/电商/教育分类）

**问题**：这些是**分类任务**，不是BrowerAI需要的**解析/渲染能力**。

---

## ✅ 正确的训练目标（符合BrowerAI定位）

根据项目定义：
> BrowerAI是一个**AI驱动的浏览器**，使用机器学习模型来**自主解析和渲染HTML/CSS/JS**，而不是传统的硬编码解析器。

### 核心训练任务

#### 1. **HTML验证模型** (已集成到 `src/parser/html.rs`)

**输入**：HTML字符串（tokenized）
- 格式：`[1, 100]` 张量，f32类型
- 示例：`<div class="container"><p>Hello</p></div>`

**输出**：2个值
1. **有效性** (validity): 0-1之间的浮点数，>0.5表示有效
2. **复杂度** (complexity): 0-1之间的浮点数

**训练数据需求**：
```python
{
    "html": "<div class='container'><p>Hello World</p></div>",
    "validity": 1.0,      # 有效HTML
    "complexity": 0.3     # 简单结构
}

{
    "html": "<div><p>Unclosed tag",
    "validity": 0.2,      # 无效HTML
    "complexity": 0.1
}
```

**现有集成代码**：`src/ai/integration.rs` 的 `HtmlModelIntegration::validate_structure()`

---

#### 2. **JS反混淆模型** (已集成到 `src/ai/integration.rs`)

**输入**：混淆的JavaScript代码
- 格式：Token IDs `[1, 60]` 张量，i64类型
- 词汇表大小：160个token
- 特殊token：`<PAD>` (0), `<SOS>` (1), `<EOS>` (2), `<UNK>` (3)

**输出**：反混淆的JavaScript token IDs
- 格式：`[1, 60]` 张量，i64类型

**训练数据需求**（Seq2Seq任务）：
```python
{
    "obfuscated": "var _0x1234=['hello'];function _0x5678(){return _0x1234[0]}",
    "deobfuscated": "function getMessage() { return 'hello'; }"
}

{
    "obfuscated": "var a=b;b=c;c=a;",
    "deobfuscated": "var temp = a; a = b; b = c; c = temp;"
}
```

**现有集成代码**：`src/ai/integration.rs` 的 `JsDeobfuscatorIntegration::deobfuscate()`

**Tokenizer**：已实现 `JsTokenizer` (160个词汇)
- 关键字：function, var, let, const, if, else, return, etc.
- 操作符：=, +, -, *, /, ==, !=, etc.
- 符号：{, }, [, ], (, ), ;, ,
- 变量名：var0-9, tmp0-9, val0-9, data0-9, result0-9, item0-9, a-z

---

#### 3. **CSS优化模型** (已集成到 `src/parser/css.rs`)

**输入**：CSS规则字符串
- 格式：tokenized CSS

**输出**：优化后的CSS规则列表

**训练数据需求**：
```python
{
    "original_css": ".container { margin: 0px; padding: 0px; width: 100%; }",
    "optimized_css": ".container { margin: 0; padding: 0; width: 100%; }"
}

{
    "original_css": "div { color: red; } div { background: blue; }",
    "optimized_css": "div { color: red; background: blue; }"
}
```

**现有集成代码**：`src/ai/integration.rs` 的 `CssModelIntegration::optimize_rules()`

---

#### 4. **代码理解模型** (用于智能渲染决策)

**输入**：35维特征向量
- HTML特征（10维）：标签数量、深度、class数量、id数量等
- CSS特征（10维）：规则数量、选择器类型、布局方式等
- JS特征（10维）：函数数量、变量数量、复杂度等
- 混合特征（5维）：资源大小、加载时间等

**输出**：10维类别logits

**训练数据需求**：
```python
{
    "features": [
        # HTML特征
        150,    # 标签数量
        5,      # DOM深度
        50,     # class数量
        10,     # id数量
        # ... 31个特征
    ],
    "category": 2,  # 0-9的分类
    "rendering_strategy": "progressive"  # 渲染策略
}
```

**现有集成代码**：`src/ai/integration.rs` 的 `CodeUnderstandingIntegration::classify()`

---

## 🚀 实施计划

### Phase 1: JS反混淆训练 (最高优先级)

**原因**：
- 已有完整的集成代码
- 词汇表已定义（160个token）
- 输入输出格式明确
- 实际应用价值最高

**任务**：
1. ✅ 收集混淆/反混淆JS代码对（从爬取的977个网站中提取）
2. ✅ 实现Seq2Seq模型（Encoder-Decoder LSTM）
3. ✅ 训练模型（输入60 tokens → 输出60 tokens）
4. ✅ 导出ONNX模型到 `models/local/js_deobfuscator_v1.onnx`
5. ✅ 测试集成：调用 `JsDeobfuscatorIntegration::deobfuscate()`

### Phase 2: HTML验证训练

**任务**：
1. ✅ 收集有效/无效HTML样本（从977个网站）
2. ✅ 标注有效性和复杂度
3. ✅ 实现分类模型（输入100 tokens → 输出2个值）
4. ✅ 导出ONNX到 `models/local/html_validator_v1.onnx`
5. ✅ 测试集成：调用 `HtmlModelIntegration::validate_structure()`

### Phase 3: CSS优化训练

**任务**：
1. ✅ 收集原始CSS和优化后的CSS对
2. ✅ 实现优化模型
3. ✅ 导出ONNX到 `models/local/css_optimizer_v1.onnx`
4. ✅ 测试集成：调用 `CssModelIntegration::optimize_rules()`

### Phase 4: 代码理解训练

**任务**：
1. ✅ 从爬取的网站中提取35维特征
2. ✅ 标注渲染策略和类别
3. ✅ 训练特征分类模型
4. ✅ 导出ONNX到 `models/local/code_understanding_v1.onnx`
5. ✅ 测试集成：调用 `CodeUnderstandingIntegration::classify()`

---

## 📋 训练数据提取方案

### 从已爬取的977个网站中提取

#### 1. JS反混淆数据

```python
# 从 data/crawled/large_urls/*.json 中提取
for website in crawled_websites:
    for js_file in website['pages']['main']['js_files']:
        js_content = js_file['content']
        
        # 检测是否混淆
        if is_obfuscated(js_content):
            # 使用规则或工具生成反混淆版本
            deobfuscated = simple_deobfuscate(js_content)
            
            yield {
                'obfuscated': js_content[:500],  # 截取前500字符
                'deobfuscated': deobfuscated[:500]
            }
```

**混淆检测规则**：
- 短变量名比例 > 80%（如 `_0x1234`, `a`, `b`, `c`）
- 十六进制编码数组存在
- eval/Function构造器调用
- 字符串混淆（\x编码、Unicode编码）

#### 2. HTML验证数据

```python
for website in crawled_websites:
    html = website['pages']['main']['html']
    
    # 使用html5ever验证
    validity = validate_html(html)  # 0-1之间
    
    # 计算复杂度
    complexity = calculate_complexity(html)  # 基于标签数、深度、嵌套
    
    yield {
        'html': html[:5000],  # 截取前5000字符
        'validity': validity,
        'complexity': complexity
    }
```

#### 3. CSS优化数据

```python
for website in crawled_websites:
    for css_file in website['pages']['main']['css_files']:
        original = css_file['content']
        
        # 使用cssnano或类似工具优化
        optimized = optimize_css(original)
        
        if original != optimized:
            yield {
                'original': original[:2000],
                'optimized': optimized[:2000]
            }
```

#### 4. 代码理解特征

```python
for website in crawled_websites:
    html = website['pages']['main']['html']
    css_files = website['pages']['main']['css_files']
    js_files = website['pages']['main']['js_files']
    
    features = extract_features(html, css_files, js_files)
    # 返回35维向量
    
    category = website.get('category', 'unknown')
    framework = website.get('metadata', {}).get('framework', 'Unknown')
    
    yield {
        'features': features,  # 35维
        'category': category_to_id(category),  # 0-9
        'framework': framework_to_id(framework)  # 0-19
    }
```

---

## 🎯 总结

### 关键区别

| 错误方向 | 正确方向 |
|---------|---------|
| 框架分类（React/Vue/Angular） | HTML结构验证 |
| 网站分类（新闻/电商/教育） | JS反混淆（混淆→清晰） |
| 学习"是什么" | 学习"怎么做" |
| 分类器（Classifier） | 转换器（Transformer/Parser） |

### 实际应用价值

**错误方向的应用**：
- 识别网站使用React → 仅供统计分析
- 识别网站是电商 → 对渲染没有帮助

**正确方向的应用**：
- JS反混淆 → 让浏览器理解混淆的代码，提高兼容性
- HTML验证 → 快速判断HTML质量，优化解析策略
- CSS优化 → 减少CSS规则冗余，提高渲染性能
- 代码理解 → 智能选择渲染策略（懒加载/预渲染/增量渲染）

---

## 下一步行动

1. **立即停止**：框架分类/网站分类训练 ✅ 已完成
2. **数据提取**：从977个网站中提取JS混淆/反混淆对
3. **模型训练**：实现Seq2Seq JS反混淆模型
4. **ONNX导出**：生成 `js_deobfuscator_v1.onnx`
5. **集成测试**：在 `src/parser/js.rs` 中测试反混淆功能

**需要用户确认**：
- 是否同意这个训练方向？
- 优先级是否正确（JS反混淆 > HTML验证 > CSS优化）？
- 是否需要调整训练数据提取策略？
