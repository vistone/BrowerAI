# 🔧 BrowerAI 项目维护指南

本指南提供日常开发和维护的最佳实践，帮助保持代码结构的整洁和模块化。

---

## 1️⃣ 新功能开发流程

### 添加新的 Python 模块

**场景**：需要添加新的机器学习功能（如新的特征提取器）

```
步骤1：确定所属模块
├─ 如果是框架检测相关 → training/detectors/
├─ 如果是网站爬虫相关 → training/crawlers/
├─ 如果是模型训练相关 → training/trainers/
├─ 如果是数据处理相关 → training/data_tools/
└─ 如果不确定 → 在 training/utils/ 中，或创建新模块

步骤2：创建文件
├─ 在对应目录创建 my_feature.py
├─ 在模块的 __init__.py 中导出主类/函数
└─ 遵循该模块的命名约定

步骤3：更新导入
├─ 在使用该模块的文件中引入：
│  from training.detectors.my_feature import MyFeatureClass
└─ 避免相对导入（使用绝对导入）

步骤4：添加文档
├─ 在模块的 docstring 中说明功能
├─ 如果是复杂功能，在 docs/guides/ 中添加集成指南
└─ 在模块的 README（如果有）中更新说明
```

**示例**：添加新的框架检测器

```python
# training/detectors/my_detector.py
"""新型框架检测器，使用 XYZ 算法"""

class MyFrameworkDetector:
    """检测 MyFramework 使用情况"""
    
    def detect(self, code: str) -> float:
        """返回检测置信度 [0, 1]"""
        pass

# training/detectors/__init__.py
from .my_detector import MyFrameworkDetector

# __all__ = [..., 'MyFrameworkDetector']
```

### 添加新的测试

**场景**：为新的反混淆策略添加测试

```
步骤1：确定测试分类
├─ 反混淆相关 → tests/deobfuscation/
├─ JS 处理相关 → tests/js/
├─ 端到端测试 → tests/e2e/
├─ 框架检测相关 → tests/framework/
├─ AI 模型相关 → tests/ai/
└─ 其他集成 → tests/integration/

步骤2：创建测试文件
├─ 遵循命名: test_*.rs 或 *_tests.rs
├─ 在对应目录创建文件
└─ 确保与现有测试命名风格一致

步骤3：更新 tests/mod.rs
├─ 在对应分类模块中添加 mod test_my_feature;
└─ 保持模块按字母顺序排列（便于查找）

步骤4：编写和验证
├─ 运行: cargo test --test category_tests
├─ 或运行特定测试: cargo test --test my_feature_tests
└─ 检查覆盖率: cargo tarpaulin --out Html
```

**示例**：添加反混淆测试

```rust
// tests/deobfuscation/my_strategy_tests.rs
#[cfg(test)]
mod tests {
    use browerai::deobfuscation::MyStrategy;

    #[test]
    fn test_decode_simple_pattern() {
        let strategy = MyStrategy::new();
        let result = strategy.decode("obfuscated_code");
        assert_eq!(result, "expected_code");
    }
}

// tests/deobfuscation/mod.rs
mod my_strategy_tests;  // 添加这行
```

### 添加新的文档

**场景**：需要说明如何集成新的模块

```
步骤1：确定文档分类
├─ 架构相关 → docs/architecture/
├─ 快速开始/指南 → docs/guides/
├─ 快速参考 → docs/references/
├─ 反混淆技术 → docs/deobfuscation/
├─ 测试相关 → docs/testing/
├─ 学习资源 → docs/learning/
└─ 已完成项目 → docs/archived/

步骤2：创建文档
├─ 命名: FEATURE_NAME.md 或 FEATURE_INTEGRATION.md
├─ 遵循 Markdown 规范
└─ 在文件顶部添加简短描述

步骤3：更新导航
├─ 更新该目录的 README.md（如果有）
├─ 在 docs/README.md 的目录索引中添加链接
└─ 如果是重要指南，在 docs/maintenance/STRUCTURE.md 中引用

步骤4：多语言支持
├─ 如果需要多语言，复制到 docs/en/ 或 docs/zh-CN/
└─ 保持目录结构一致
```

**示例**：添加集成指南

```markdown
# docs/guides/MY_FEATURE_INTEGRATION.md

## 功能概述
描述新功能的用途和优势

## 安装和配置
说明如何集成到现有系统

## API 文档
给出代码示例

## 故障排查
常见问题和解决方案
```

---

## 2️⃣ 定期清理计划

### 每月清理清单

```bash
# 1. 检查 archived/ 中有没有需要删除的内容
find docs/archived -name "*.md" -type f | wc -l
# 如果数量过多，检查是否有真正需要保留的

# 2. 检查 legacy/ 脚本是否有可以删除的
ls -lh training/scripts/legacy/
# 评估是否这些脚本功能已完全被新脚本取代

# 3. 检查未使用的依赖
cargo tree --unused
cargo deny check advisories

# 4. 检查测试覆盖率
cargo tarpaulin --out Html  # 生成 HTML 覆盖率报告
```

### 清理工作流

```bash
#!/bin/bash
# scripts/cleanup_check.sh - 月度清理检查

echo "📋 BrowerAI 月度清理检查"
echo "========================="

echo ""
echo "1️⃣  docs/archived/ 统计："
archived_count=$(find docs/archived -name "*.md" -type f | wc -l)
echo "   📄 文档数: $archived_count"
echo "   💡 建议: 如果超过 10 个，检查是否有可删除的"

echo ""
echo "2️⃣  training/scripts/legacy/ 统计："
legacy_count=$(ls -1 training/scripts/legacy/*.py 2>/dev/null | wc -l)
echo "   📄 脚本数: $legacy_count"
echo "   💡 建议: 检查 training/scripts/legacy/README.md 中的迁移指南"

echo ""
echo "3️⃣  整体项目大小："
du -sh training/ docs/ tests/ | sort -h
echo "   💡 建议: 如果训练数据过大，考虑使用 .gitignore"

echo ""
echo "✅ 清理检查完成！"
```

### 删除过时文档的标准

删除 `docs/archived/` 或 `docs/maintenance/` 中的文件时，确保：

- ✅ **功能已迁移**：该文档描述的功能已在其他文档或代码中说明
- ✅ **无历史参考价值**：不是重要的项目历史记录
- ✅ **不影响维护**：删除后不会对项目理解产生负面影响
- ⚠️ **保留以下内容**：
  - 重要的架构决策记录
  - 已完成阶段的总结（可供参考）
  - 项目结构说明（STRUCTURE.md）

**安全删除流程**：

```bash
# 1. 备份旧文件（可选但推荐）
mkdir -p backups/
cp docs/archived/OLD_FILE.md backups/

# 2. 检查文件是否被引用
grep -r "OLD_FILE" docs/ crates/ training/ --include="*.md" --include="*.rs" --include="*.py"

# 3. 如果没有引用，安全删除
rm docs/archived/OLD_FILE.md

# 4. 提交更改
git add docs/archived/
git commit -m "chore: remove obsolete document OLD_FILE.md"
```

---

## 3️⃣ 模块化开发最佳实践

### Python 包结构检查

```bash
#!/bin/bash
# scripts/check_python_structure.sh

echo "🐍 检查 Python 包结构"

# 检查所有模块都有 __init__.py
for dir in training/detectors training/crawlers training/trainers training/obfuscation training/pipelines training/generators training/evaluation training/optimization training/onnx training/metrics training/services training/utils training/scripts/data_tools training/scripts/export; do
    if [ -d "$dir" ]; then
        if [ ! -f "$dir/__init__.py" ]; then
            echo "⚠️  $dir 缺少 __init__.py"
        else
            echo "✅ $dir"
        fi
    fi
done

# 检查导入语法
python3 -m py_compile training/**/*.py
if [ $? -eq 0 ]; then
    echo "✅ 所有 Python 文件语法正确"
fi
```

### Rust 模块检查

```bash
#!/bin/bash
# scripts/check_rust_structure.sh

echo "🦀 检查 Rust 模块结构"

# 检查所有 tests 都在 mod.rs 中声明
cargo check --workspace
if [ $? -eq 0 ]; then
    echo "✅ 所有 Rust 模块编译正确"
fi

# 检查测试发现
cargo test --no-run
echo "✅ 所有测试被正确发现"
```

### 导入路径规范

**✅ 推荐做法**：

```python
# 使用绝对导入
from training.detectors.high_precision_detector import HighPrecisionDetector
from training.pipelines.complete_system import CompleteSystem

# 模块内使用相对导入（仅在大型模块内）
# from ..utils import helper_function  # 谨慎使用
```

**❌ 避免**：

```python
# 不要使用相对路径
import sys
sys.path.insert(0, '../detectors')
from high_precision_detector import HighPrecisionDetector

# 不要混合相对和绝对
from detectors import HighPrecisionDetector  # 不清晰
```

---

## 4️⃣ 常见维护场景

### 场景 A：需要重命名一个模块

```bash
# 1. 重命名目录
mv training/old_module_name training/new_module_name

# 2. 更新所有导入
grep -r "from training.old_module_name" . --include="*.py" --include="*.rs"
# 手动替换为 from training.new_module_name

# 3. 更新 __init__.py
# 在 training/__init__.py 和模块的 __init__.py 中更新导出

# 4. 验证
python3 -c "from training.new_module_name import *; print('✅')"
cargo check --workspace
```

### 场景 B：需要合并两个相似的模块

```bash
# 1. 分析要保留的模块结构
ls -la training/module1/
ls -la training/module2/

# 2. 将 module2 的内容合并到 module1
cp training/module2/*.py training/module1/
# 解决命名冲突（如有）

# 3. 更新导入（所有引用从 module2 指向 module1）
grep -r "from training.module2" . --include="*.py"
# 批量替换

# 4. 清理
rm -rf training/module2

# 5. 验证和测试
cargo test --workspace
python3 -c "from training.module1 import *"
```

### 场景 C：需要将功能从 legacy 迁移到主模块

```bash
# 1. 分析 legacy 脚本的功能
head -50 training/scripts/legacy/old_script.py

# 2. 识别可复用的部分，复制到新位置
# 例如：从 train_paired_website_generator.py 复制功能到 generators/

# 3. 更新新实现
# - 使用新的导入路径
# - 遵循新模块的编码风格
# - 添加新的功能扩展

# 4. 创建迁移文档
# 在 docs/maintenance/STRUCTURE.md 中记录迁移细节

# 5. 更新 legacy/README.md
# 标记脚本为"已完全迁移"

# 6. 测试
cargo test --workspace
```

---

## 5️⃣ 快速命令参考

```bash
# 📊 项目统计
echo "总文件数："
find . -type f -name "*.py" -o -name "*.rs" -o -name "*.md" | wc -l

echo "Python 模块数："
find training -maxdepth 1 -type d | wc -l

echo "测试分类数："
ls -d tests/*/ | wc -l

echo "文档主题数："
ls -d docs/*/ | wc -l

# 🔍 查找和分析
# 查找所有 TODO 注释
grep -r "TODO\|FIXME\|XXX" training/ tests/ crates/ --include="*.py" --include="*.rs"

# 查找未使用的导入
grep -r "^import\|^from" training/ --include="*.py" | grep -v "# noqa"

# 检查代码行数
find training -name "*.py" -exec wc -l {} + | tail -1
find crates -name "*.rs" -exec wc -l {} + | tail -1

# 🧪 运行测试
# 运行所有测试
cargo test --workspace

# 运行特定分类
cargo test --test deobfuscation
cargo test --test e2e
cargo test --test framework

# 生成覆盖率报告
cargo tarpaulin --out Html --output-dir coverage

# 🐍 Python 检查
# 检查语法
python3 -m py_compile training/**/*.py

# 检查导入
python3 -c "from training.detectors import *; print('✅')"

# 代码风格检查（需安装 black/pylint）
black --check training/
pylint training/
```

---

## 6️⃣ 维护日志示例

创建 `docs/maintenance/MAINTENANCE_LOG.md` 记录定期维护：

```markdown
# 维护日志

## 2026年2月
### 2月1日
- ✅ 月度清理检查完成
- 📊 docs/archived/ 8 个文档（正常）
- 📊 training/scripts/legacy/ 5 个脚本（正常）
- 🔄 建议：保留所有文件，继续监控

### 2月15日
- ✅ 新增 `training/feature_extraction/` 模块
- 📝 新增文档：docs/guides/FEATURE_EXTRACTION_GUIDE.md
- 🧪 新增测试分类：tests/feature_extraction/
- ✅ 所有导入路径已更新

## 2026年3月
...
```

---

## 📝 总结

维护这个项目结构的关键：

1. **一致性**：新功能遵循现有的模块划分
2. **及时性**：定期检查和清理过时内容
3. **文档化**：任何结构变更都要更新相关文档
4. **自动化**：使用脚本检查和验证结构完整性
5. **最小化**：避免创建不必要的新目录，优先复用现有模块

让项目结构保持清晰、整洁、易于维护！ 🚀
