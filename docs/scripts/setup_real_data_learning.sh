#!/bin/bash

# 真实数据学习系统快速安装脚本
# 这是完整的、非Demo的系统
# 用于学习全球JS框架的混淆和反混淆

set -e

echo ""
echo "=============================================================="
echo "🌍 BrowerAI 真实数据学习系统"
echo "=============================================================="
echo ""

# 检查操作系统
if [[ ! "$OSTYPE" =~ ^linux ]]; then
    echo "❌ 仅支持Linux系统"
    exit 1
fi

# ==================== 阶段1: 环境准备 ====================

echo "📋 阶段1: 环境准备"
echo ""

# 1.1 安装Node.js
echo "1.1 检查Node.js..."
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js未安装,请手动安装:"
    echo "   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
    echo "   sudo apt-get install -y nodejs"
else
    echo "✅ Node.js已存在: $(node --version)"
fi

# 1.2 检查JavaScript混淆工具
echo ""
echo "1.2 检查JavaScript混淆工具..."
if ! command -v npx &> /dev/null; then
    echo "⚠️  NPX未安装,请先安装Node.js"
else
    echo "✅ NPX已安装"
    npx terser --version 2>/dev/null || echo "   (Terser将在首次使用时自动安装)"
    npx uglify-js --version 2>/dev/null || echo "   (UglifyJS将在首次使用时自动安装)"
fi

# 1.3 检查Python依赖
echo ""
echo "1.3 检查Python依赖..."

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3未安装"
    exit 1
fi

# PyTorch (GPU支持)
echo "   检查PyTorch..."
python3 << 'PYEOF'
try:
    import torch
    print(f"✅ PyTorch已安装: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ GPU支持: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  未检测到GPU支持")
except ImportError:
    print("⚠️  PyTorch未安装")
    print("   运行: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
PYEOF

# 1.4 验证GPU
echo ""
echo "1.4 验证GPU配置..."
python3 << 'PYEOF'
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   显存: {props.total_memory / 1e9:.2f}GB")
        print(f"   CUDA计算能力: {props.major}.{props.minor}")
    else:
        print("⚠️  未检测到GPU,将使用CPU(会很慢)")
except:
    pass
PYEOF

echo ""
echo "✅ 阶段1完成: 环境检查就绪"
echo ""

# ==================== 提示用户后续步骤 ====================

echo "=============================================================="
echo "🚀 快速开始指南"
echo "=============================================================="
echo ""
echo "完整步骤请查看: COMPLETE_REAL_DATA_EXECUTION_PLAN.md"
echo ""
echo "快速命令:"
echo ""
echo "1️⃣  下载GitHub框架源代码 (需要GitHub Token):"
echo "   export GITHUB_TOKEN='ghp_your_token_here'"
echo "   python3 training/github_framework_crawler.py \$GITHUB_TOKEN"
echo ""
echo "2️⃣  下载NPM包:"
echo "   python3 training/npm_package_crawler.py"
echo ""
echo "3️⃣  应用真实混淆工具:"
echo "   python3 training/real_code_obfuscator.py"
echo ""
echo "4️⃣  训练GPU框架检测模型:"
echo "   python3 training/gpu_framework_detector.py"
echo ""
echo "=============================================================="
