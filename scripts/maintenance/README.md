# 🔧 BrowerAI 维护脚本

这个目录包含用于项目结构维护和验证的实用脚本。

## 📋 可用脚本

### 1. `monthly_check.sh` - 月度清理检查

定期运行此脚本以检查项目结构的健康状态。

**功能**：
- ✅ 检查文档目录结构整洁性
- ✅ 验证 Python 模块的 `__init__.py` 完整性
- ✅ 统计训练脚本和遗留脚本数量
- ✅ 检查测试分类结构
- ✅ 验证 Python 代码语法
- ✅ 提供维护建议

**用法**：
```bash
bash scripts/maintenance/monthly_check.sh
```

**输出示例**：
```
📋 BrowerAI 月度项目维护检查
==============================

1️⃣  文档结构检查
   docs/ 根目录 .md 文件数（应仅有 README.md）:
   📄 1 个
   ✅ 正常

2️⃣  Python 模块结构检查
   training/ 模块总数:
   📦 19 个模块
   ...

✅ 月度检查完成！
```

### 2. `validate_structure.sh` - 结构完整性验证

用于验证项目结构是否完整正确，在提交重大更改前运行。

**功能**：
- ✅ 验证目录结构（根目录文件应最少化）
- ✅ 检查必要文件存在
- ✅ 验证所有 Python 模块有 `__init__.py`
- ✅ 检查所有测试分类目录
- ✅ 检查所有文档分类目录
- ✅ 验证 Python 包可导入
- ✅ 验证代码无语法错误
- ✅ （可选）验证 Cargo 编译

**用法**：
```bash
bash scripts/maintenance/validate_structure.sh
```

**输出示例**：
```
🔍 BrowerAI 项目结构验证
========================

1️⃣  目录结构检查
   ✅ docs/ 根目录仅有 README.md
   ✅ training/ 根目录仅有 __init__.py
   ✅ tests/ 根目录仅有 mod.rs
   ...

📊 验证结果
================================
通过: 48 / 48 项检查

✅ 项目结构完整正确！
```

## 🚀 使用场景

### 场景 1：添加新功能后的验证
```bash
# 添加新模块后验证
bash scripts/maintenance/validate_structure.sh
```

### 场景 2：月度维护
```bash
# 定期检查项目状态
bash scripts/maintenance/monthly_check.sh
```

### 场景 3：重构前检查
```bash
# 在进行大规模重构前先验证当前状态
bash scripts/maintenance/validate_structure.sh

# 重构完成后再次验证
bash scripts/maintenance/validate_structure.sh
```

## 📅 推荐的维护日程

| 频率 | 任务 | 脚本 |
|------|------|------|
| **每天** | 代码提交前验证 | `validate_structure.sh` |
| **每周** | 文档和模块检查 | `monthly_check.sh` |
| **每月** | 完整项目审查 | `monthly_check.sh` |
| **按需** | 大型重构验证 | `validate_structure.sh` |

## 🔧 自定义和扩展

### 修改检查项目

编辑脚本文件，在对应的 `check()` 调用中添加或修改条件。

例如，添加新的检查：
```bash
check "自定义检查名称" "[ -f path/to/file ]"
```

### 集成到 CI/CD

可以在 GitHub Actions 或其他 CI 系统中集成：
```yaml
- name: Validate project structure
  run: bash scripts/maintenance/validate_structure.sh
```

### 添加更多脚本

遵循现有脚本的模式，在 `scripts/maintenance/` 中创建新脚本。

## ❓ 常见问题

**Q: 脚本无法执行？**
A: 确保脚本有执行权限：`chmod +x scripts/maintenance/*.sh`

**Q: 脚本显示某些检查失败？**
A: 查看输出信息，按照建议进行相应的修复，然后重新运行脚本。

**Q: 如何在特定目录运行脚本？**
A: 修改脚本中的路径，或从项目根目录运行脚本。

**Q: 脚本可以自动修复问题吗？**
A: 目前脚本仅进行检查和报告，问题修复需要手动执行。如果需要自动修复，可以扩展脚本功能。

## 📚 相关文档

- [MAINTENANCE_GUIDE.md](../docs/maintenance/MAINTENANCE_GUIDE.md) - 项目维护指南
- [ORGANIZATION_SUMMARY.md](../docs/maintenance/ORGANIZATION_SUMMARY.md) - 整理总结
- [STRUCTURE.md](../docs/maintenance/STRUCTURE.md) - 项目结构说明

---

**保持项目整洁，从定期维护开始！** 🚀
