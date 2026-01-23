#!/usr/bin/env python3
"""清理过期的markdown文档"""

import os
import glob
from pathlib import Path

os.chdir('/home/stone/BrowerAI')

# 需要保留的文件
KEEP_FILES = {
    'README.md',
    'CONTRIBUTING.md',
    'CHANGELOG.md',
}

# 删除模式
DELETE_PATTERNS = [
    'BULK_CSS_*.md',
    'CSS_LEARNING_*.md',
    'CSS_*.md',
    'DUAL_SANDBOX_*.md',
    'COMPLETE_*_GUIDE.md',
    'ANALYSIS_*.md',
    'COMPREHENSIVE_*.md',
    'LARGE_SCALE_*.md',
    'INTENT_OPTIMIZATION_*.md',
    'DATA_QUALITY_*.md',
    'ECOMMERCE_*.md',
    'EXPERIMENT_*.md',
    'IMPLEMENTATION_*.md',
    'INFERENCE_*.md',
    'COMPILATION_*.md',
    'COMPARATIVE_*.md',
    'CRATES_IO_*.md',
    'DEPLOYMENT_GUIDE.md',
    'DESIGN_*.md',
    'DOCUMENTATION_*.md',
    'EXECUTION_*.md',
    'FINAL_*.md',
    'FOLLOWUP_*.md',
    'GETTING_STARTED.md',
    'GLOBAL_DEPLOYMENT_*.md',
]

print("=== 文档清理 ===\n")

deleted_count = 0
deleted_files = []

for pattern in DELETE_PATTERNS:
    for file_path in glob.glob(pattern):
        # 检查是否在保留列表中
        if Path(file_path).name not in KEEP_FILES:
            try:
                os.remove(file_path)
                deleted_count += 1
                deleted_files.append(file_path)
                print(f"✅ 已删除: {file_path}")
            except Exception as e:
                print(f"❌ 删除失败 {file_path}: {e}")

print(f"\n🧹 清理完成: 删除了 {deleted_count} 个文件")

# 列出保留的文件
print("\n📄 保留的markdown文件:")
for md_file in sorted(glob.glob('*.md')):
    size = os.path.getsize(md_file) / 1024
    print(f"   {md_file} ({size:.1f} KB)")

# 统计
remaining = len(list(glob.glob('*.md')))
print(f"\n📊 剩余markdown文件: {remaining}个")
