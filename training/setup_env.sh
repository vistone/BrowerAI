#!/bin/bash
# BrowerAI 训练环境快速设置脚本

set -e

echo "🚀 BrowerAI 训练环境设置"
echo "========================"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安装！"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python 版本: $PYTHON_VERSION"

# 创建虚拟环境（推荐）
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 创建 Python 虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo ""
echo "🔌 激活虚拟环境..."
source venv/bin/activate

# 升级 pip
echo ""
echo "⬆️  升级 pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "📥 安装训练依赖（这可能需要几分钟）..."
pip install -r requirements.txt

# 尝试安装可选依赖（失败不影响使用）
echo ""
echo "📦 安装可选依赖（失败可忽略）..."
pip install onnx-simplifier 2>/dev/null || echo "⚠️  onnx-simplifier 安装失败（可选，不影响训练）"

echo ""
echo "✅ 环境设置完成！"
echo ""
echo "📋 下一步:"
echo "  1. 激活环境: source venv/bin/activate"
echo "  2. 收集数据: cd ../.. && cargo run -- --learn https://example.com"
echo "  3. 训练模型: cd training/scripts && python train_html_complexity.py"
echo "  4. 验证模型: python validate_model.py ../models/html_complexity_v1.onnx"
echo ""
echo "💡 提示: 确保至少收集 100+ 个反馈样本再开始训练"
