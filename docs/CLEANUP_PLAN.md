# 🧹 BrowerAI 根目录清理执行计划

**执行日期**: 2026-02-17  
**优先级**: 高  
**预计时间**: 30分钟

---

## 📋 清理策略

### 第1组: 保留在根目录 (7文件)

| 文件 | 操作 | 备注 |
|------|------|------|
| `README.md` | ✅ 保留 | 项目主入口 |
| `CHANGELOG.md` | ✅ 保留 | 版本变更记录 |
| `CONTRIBUTING.md` | ✅ 保留 | 贡献指南 |
| `LICENSE` | ✅ 保留 | 许可证 |
| `QUICK_START.md` | ⚡ 新建 | 聚合快速开始 |
| `DEVELOPMENT_GUIDE.md` | ⚡ 新建 | 聚合开发指南 |
| `PROJECT_STRUCTURE.md` | ⚡ 新建 | 项目结构说明 |

---

## 第2组: 移动到 /docs/guides/ (12文件)

部署和开发相关指南：

```
DEPLOYMENT_QUICKSTART.md         → docs/guides/DEPLOYMENT_QUICKSTART.md
DEPLOYMENT_CHECKLIST.md          → docs/guides/DEPLOYMENT_CHECKLIST.md
GITHUB_DEPLOYMENT_GUIDE.md       → docs/guides/GITHUB_DEPLOYMENT_GUIDE.md
QUICK_START_CARD.md              → docs/guides/QUICK_START_CARD.md
QUICK_START_REAL_SYSTEM.md       → docs/guides/REAL_SYSTEM_SETUP.md
REAL_DATA_LEARNING_GUIDE.md      → docs/guides/REAL_DATA_LEARNING.md
REAL_SYSTEM_BUILD_PLAN.md        → docs/guides/REAL_SYSTEM_BUILD.md
REAL_SYSTEM_EXECUTION_GUIDE.md   → docs/guides/REAL_SYSTEM_EXECUTION.md

注: 这些需要整合或保留为单独的指南文件
```

---

## 第3组: 移动到 /docs/phases/ - 周报告总结 (30+文件)

周报告和阶段总结（每周一个文件）：

```
WEEK2_EXECUTION_SUMMARY.md                 → docs/phases/WEEK2_SUMMARY.md
WEEK3_COMPLETE_SUMMARY.md                  → docs/phases/WEEK3_SUMMARY.md
WEEK4_COMPLETION_REPORT.md                 → docs/phases/WEEK4_SUMMARY.md
WEEK4_EXECUTION_SUMMARY.md                 (合并到WEEK4_SUMMARY.md)
WEEK4_DEPLOYMENT_PLAN.md                   (合并到WEEK4_SUMMARY.md)
WEEK4_PHASE1_COMPLETION.md                 (合并到WEEK4_SUMMARY.md)
WEEK4_PHASE1_PROGRESS.md                   (合并到WEEK4_SUMMARY.md)
WEEK4_PHASE2_PLAN.md                       (合并到WEEK4_SUMMARY.md)
WEEK4_QUICK_REFERENCE.md                   (合并或删除)

WEEK5_EXECUTION_PLAN.md                    → docs/phases/WEEK5_SUMMARY.md (新建)
WEEK5_IMMEDIATE_ACTIONS.md                 (合并)
WEEK5_PROGRESS_REPORT.md                   (合并)

WEEK6_ANALYSIS_SUMMARY.md                  → docs/phases/WEEK6_SUMMARY.md (新建/合并)
WEEK6_API_SPEC.md                          → docs/api/SPECIFICATIONS.md
WEEK6_ARCHITECTURE_ANALYSIS.md             → docs/architecture/ANALYSIS.md
WEEK6_COMPLETE_SUMMARY.md                  (合并)
WEEK6_COMPLETION_REPORT.md                 (合并)
WEEK6_COMPLETION_SUMMARY.md                (删除 - 重复)
WEEK6_DOCS_GUIDE.md                        (合并或移到archived)
WEEK6_FINAL_REPORT.md                      (合并)
WEEK6_IMPLEMENTATION_GUIDE.md              → docs/guides/IMPLEMENTATION.md
WEEK6_INTEGRATION_COMPLETE.md              (合并)
WEEK6_INTEGRATION_DESIGN.md                → docs/architecture/INTEGRATION_DESIGN.md
WEEK6_INTEGRATION_GUIDE.md                 → docs/guides/INTEGRATION.md
WEEK6_PROJECT_INDEX.md                     → docs/README.md (更新)
WEEK6_PYTHON_COMPLETE_REPORT.md            (合并)
WEEK6_PYTHON_IMPLEMENTATION.md             → docs/development/PYTHON_SETUP.md
WEEK6_QUICK_REFERENCE.md                   (合并到指南)
WEEK6_QUICK_START_PYTHON.md                → docs/guides/PYTHON_QUICK_START.md
WEEK6_REAL_DATA_LEARNING_REPORT.md         (合并)
WEEK6_VERIFICATION_CHECKLIST.md            (合并到guides/TESTING.md)

类似方式处理WEEK7_*和WEEK8_*文件
```

---

## 第4组: 移动到 /docs/archived/ (35+文件)

临时报告和一次性分析（归档存储）：

```
PROJECT_FINAL_STATUS.md                    → docs/archived/PROJECT_FINAL_STATUS.md
COMPREHENSIVE_TEST_AND_SUBMISSION_REPORT.md → docs/archived/TEST_SUBMISSION_REPORT.md
DATA_STATISTICS_REPORT.md                  → docs/archived/DATA_STATISTICS.md
DEPLOYMENT_PROGRESS.md                     → docs/archived/DEPLOYMENT_PROGRESS.md
LEARNING_PROGRESS.md                       → docs/archived/LEARNING_PROGRESS.md
QUICK_REFERENCE_REAL_DATA.md               → docs/archived/REAL_DATA_REFERENCE.md

WEEK7_FINAL_COMPLETION.md                  → docs/phases/WEEK7_SUMMARY.md (可选)
WEEK7_INTEGRATION_EXECUTION_REPORT.md      (合并)
WEEK7_INTEGRATION_QUICK_REFERENCE.md       (合并)
WEEK7_INTEGRATION_SUMMARY.md               (合并)
WEEK7_INTEGRATION_TESTING.md               (合并)
WEEK7_NEXT_STEPS.md                        (合并)

WEEK8_CHECKLIST.md                         (合并到week8)
WEEK8_COMPLETE_SUMMARY.md                  → docs/phases/WEEK8_SUMMARY.md
WEEK8_COMPREHENSIVE_SUMMARY.md             (删除 - 重复)
WEEK8_FINAL_SUMMARY.md                     (删除 - 重复)
WEEK8_LAUNCH_SUMMARY.md                    (合并)
WEEK8_OVERVIEW.md                          (合并)
WEEK8_PHASE_A_EXECUTION_REPORT.md          → docs/archived/WEEK8/PHASE_A_EXECUTION.md
WEEK8_PHASE_A_PLAN.md                      (合并)
WEEK8_PHASE_B_EXECUTION_REPORT.md          → docs/archived/WEEK8/PHASE_B_EXECUTION.md
WEEK8_PHASE_B_PLAN.md                      (合并)
WEEK8_PHASE_C_EXECUTION_REPORT.md          → docs/archived/WEEK8/PHASE_C_EXECUTION.md
WEEK8_PHASE_C_PLAN.md                      (合并)
WEEK8_PHASE_D_EXECUTION_REPORT.md          → docs/archived/WEEK8/PHASE_D_EXECUTION.md
WEEK8_PHASE_D_PLAN.md                      (合并)
WEEK8_PHASE_E_COMPLETE.md                  (删除 - 重复)
WEEK8_PHASE_E_COMPLETION_REPORT.md         → docs/archived/WEEK8/PHASE_E_EXECUTION.md
WEEK8_PHASE_E_DELIVERY_SUMMARY.md          (删除 - 重复)
WEEK8_PHASE_E_EXECUTION_CHECKLIST.md       (合并)
WEEK8_PHASE_E_EXECUTION_REPORT.md          (删除 - 重复)
WEEK8_PHASE_E_PLAN.md                      (合并)
WEEK8_PHASE_E_START_NOW.md                 (删除 - 过期)
WEEK8_PRE_LAUNCH_CHECK.md                  (合并)
WEEK8_QUICK_START.md                       (删除 - 重复)
WEEK8_READY_TO_LAUNCH.md                   (删除 - 过期)
```

---

## 第5组: 删除的文件 (26+文件)

明显重复或过期的文件：

```
❌ 删除 WEEK*_*COMPLETE*.md 中的重复文件
❌ 删除 WEEK*_*FINAL*.md 中的重复文件
❌ 删除 WEEK*_*SUMMARY*.md 中有多个重复副本的
❌ 删除 *_READY_TO_*.md 和 *_START_NOW*.md (临时文件)
❌ 删除 *_QUICK_REFERENCE*.md (多个，应该合并到指南)
❌ 删除所有 *_PLAN.md (内容应并入WEEK_*_SUMMARY.md)
❌ 删除所有 *_PROGRESS_REPORT*.md 副本

具体列表:
- WEEK8_COMPREHENSIVE_SUMMARY.md (replaced by WEEK8_COMPLETE_SUMMARY.md)
- WEEK8_FINAL_SUMMARY.md (same as above)
- WEEK8_PHASE_E_COMPLETE.md (replaced by WEEK8_PHASE_E_COMPLETION_REPORT.md)
- WEEK8_PHASE_E_DELIVERY_SUMMARY.md (same)
- WEEK8_PHASE_E_EXECUTION_REPORT.md (archived version exists)
- WEEK8_PHASE_E_START_NOW.md (临时文件)
- WEEK8_READY_TO_LAUNCH.md (临时文件)
- WEEK8_QUICK_START.md (replaced by DEPLOYMENT_QUICKSTART.md)
- WEEK6_COMPLETION_SUMMARY.md (same as WEEK6_COMPLETE_SUMMARY.md)
- ...等多个其他重复文件
```

---

## 📊 执行统计

```
当前状态:
├── 根目录md文件: 83个
├── docs目录文件: 364个
└── 总计: 447个

目标状态:
├── 根目录md文件: 7个 (↓ 91%)
│   ├── README.md
│   ├── CHANGELOG.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE
│   ├── QUICK_START.md (新)
│   ├── DEVELOPMENT_GUIDE.md (新)
│   └── PROJECT_STRUCTURE.md (新)
│
├── docs目录文件: ~250个 (合理数量)
│   ├── 有组织的子目录
│   ├── 移除重复
│   └── 统一的索引系统
│
└── 总计: ~257个 (↓ 42%)

删除/合并文件: 190+个
保留文件: 257个
```

---

## 🔄 执行步骤

### Step 1: 备份 (安全起见)

```bash
# 创建备份
tar czf ~/browerai_backup_$(date +%Y%m%d).tar.gz /home/stone/BrowerAI/*.md

# 或Git已有完整历史，可直接操作
```

### Step 2: 创建目录结构

```bash
# 确保所有必要的子目录存在
mkdir -p /home/stone/BrowerAI/docs/{guides,api,architecture,development,learning,integration,maintenance,references,phases,archived}
```

### Step 3: 移动文件

**操作这些文件**：
```bash
# 移动到 guides/
mv DEPLOYMENT_QUICKSTART.md docs/guides/
mv DEPLOYMENT_CHECKLIST.md docs/guides/
mv GITHUB_DEPLOYMENT_GUIDE.md docs/guides/
# ... 等等

# 移动到 phases/
mv WEEK*_SUMMARY.md docs/phases/
# ... 等等

# 移动到 archived/
mv PROJECT_FINAL_STATUS.md docs/archived/
# ... 等等
```

### Step 4: 整合文档

- 检查并合并重复的指南
- 更新所有内部链接
- 创建新的索引文档

### Step 5: 删除重复文件

```bash
# 删除明显重复的文件
rm -f /home/stone/BrowerAI/WEEK8_COMPREHENSIVE_SUMMARY.md
rm -f /home/stone/BrowerAI/WEEK8_FINAL_SUMMARY.md
# ... 等等
```

### Step 6: 验证和提交

```bash
# 检查根目录
ls -la /home/stone/BrowerAI/*.md  # 应该只有7个

# 检查docs结构
find /home/stone/BrowerAI/docs -name "*.md" | wc -l

# Git提交
git add -A
git commit -m "refactor: reorganize documentation structure

- Consolidate 83 root md files into 7 core files
- Organize docs into logical subdirectories
- Remove duplicate and outdated reports
- Establish documentation standards

See docs/PROJECT_STANDARDS.md for guidelines"
```

---

## ✅ 完成检查清单

清理完成后验证：

- [ ] 根目录只有7个md文件
- [ ] 所有指南在 `/docs/guides/`
- [ ] 所有API文档在 `/docs/api/`
- [ ] 所有架构文档在 `/docs/architecture/`
- [ ] 周报告在 `/docs/phases/`
- [ ] 历史报告在 `/docs/archived/`
- [ ] 所有内部链接正确
- [ ] 所有索引文件已更新
- [ ] docs/README.md是完整的导航
- [ ] 没有孤立的文件

---

## ⚠️ 注意事项

**重要**: 所有操作前，确保：
1. Git工作目录干净 (`git status` 显示clean)
2. 所有重要文件已提交到GitHub
3. 有备份副本（虽然Git已有完整历史）
4. 操作完成后运行测试，确保没有关键文档丢失

**回滚方案**: 如果出现问题
```bash
git reset --hard HEAD
# 恢复到之前的状态
```

---

**由架构清理团队制定**  
**预计执行时间**: 30-45分钟  
**风险等级**: 低 (仅重组，不删除源代码)
