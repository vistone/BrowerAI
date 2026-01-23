#!/bin/bash

# 真实数据学习系统 - 交互式启动菜单
# 按步骤指导用户完成整个系统

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  🌍 BrowerAI 真实数据JavaScript框架学习系统                   ║"
echo "║  Real Data Learning System for Global JS Frameworks           ║"
echo "║                                                               ║"
echo "║  ✅ 真实数据 (GitHub + NPM)                                   ║"
echo "║  ✅ 真实混淆 (Terser + UglifyJS)                             ║"
echo "║  ✅ 真实GPU训练 (PyTorch Transformer)                        ║"
echo "║                                                               ║"
echo "║  不是Demo。全部都是真实的。                                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_menu() {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "📋 主菜单"
    echo "────────────────────────────────────────────────────────────────"
    echo ""
    echo "  ${BLUE}0${NC} - 显示完整指南"
    echo "  ${BLUE}1${NC} - 检查环境"
    echo "  ${BLUE}2${NC} - 获取GitHub Token"
    echo "  ${BLUE}3${NC} - 下载GitHub框架源代码 (8小时)"
    echo "  ${BLUE}4${NC} - 下载NPM包 (10小时)"
    echo "  ${BLUE}5${NC} - 应用真实混淆工具 (2-3天)"
    echo "  ${BLUE}6${NC} - 训练GPU框架检测模型 (1小时)"
    echo "  ${BLUE}7${NC} - 查看数据统计"
    echo "  ${BLUE}8${NC} - 监控GPU使用"
    echo "  ${BLUE}9${NC} - 清理数据"
    echo "  ${BLUE}q${NC} - 退出"
    echo ""
}

# 选项0: 显示完整指南
show_guide() {
    echo ""
    echo "${GREEN}完整执行步骤${NC}"
    echo ""
    echo "第0步: 检查环境 (已自动完成)"
    echo "  bash setup_real_data_learning.sh"
    echo ""
    echo "第1步: 获取GitHub Token (5分钟)"
    echo "  1. 访问: https://github.com/settings/tokens"
    echo "  2. 点击 'Generate new token (classic)'"
    echo "  3. 名称: BrowerAI_Data"
    echo "  4. 勾选: repo, read:user"
    echo "  5. 复制令牌"
    echo ""
    echo "第2步: 下载GitHub框架 (6-8小时)"
    echo "  export GITHUB_TOKEN='ghp_your_token_here'"
    echo "  python3 training/github_framework_crawler.py \$GITHUB_TOKEN"
    echo ""
    echo "第3步: 下载NPM包 (6-10小时)"
    echo "  python3 training/npm_package_crawler.py"
    echo ""
    echo "第4步: 应用混淆工具 (2-3天)"
    echo "  python3 training/real_code_obfuscator.py"
    echo ""
    echo "第5步: 训练GPU模型 (1小时)"
    echo "  python3 training/gpu_framework_detector.py"
    echo ""
    echo "详细文档: COMPLETE_REAL_DATA_EXECUTION_PLAN.md"
    echo ""
}

# 选项1: 检查环境
check_env() {
    echo ""
    echo "${GREEN}🔍 检查环境...${NC}"
    echo ""
    
    bash setup_real_data_learning.sh
    
    echo ""
    echo "${GREEN}✅ 环境检查完成${NC}"
    echo ""
}

# 选项2: GitHub Token
get_github_token() {
    echo ""
    echo "${GREEN}获取GitHub Token${NC}"
    echo ""
    echo "1️⃣  访问: https://github.com/settings/tokens"
    echo "2️⃣  点击: 'Generate new token (classic)'"
    echo ""
    echo "配置步骤:"
    echo "  • Token名称: BrowerAI_Data_Collector"
    echo "  • 过期: 无(按需)"
    echo "  • 权限:"
    echo "    ✓ repo (完整控制)"
    echo "    ✓ read:user (读取用户)"
    echo ""
    echo "3️⃣  复制生成的令牌"
    echo "4️⃣  在终端中设置:"
    echo ""
    read -p "粘贴你的GitHub Token (或回车跳过): " github_token
    
    if [ -n "$github_token" ]; then
        export GITHUB_TOKEN="$github_token"
        echo ""
        echo "${GREEN}✅ Token已设置${NC}"
        echo "使用方式:"
        echo "  python3 training/github_framework_crawler.py \$GITHUB_TOKEN"
    else
        echo ""
        echo "${YELLOW}⚠️  Token未设置。使用token-less模式(限制60请求/小时)${NC}"
        echo "建议使用Token以提高速率限制到5000请求/小时"
    fi
    echo ""
}

# 选项3: 下载GitHub框架
download_github() {
    echo ""
    echo "${GREEN}📥 开始下载GitHub框架源代码${NC}"
    echo ""
    echo "覆盖框架:"
    echo "  • React, Vue, Angular, Svelte, Preact (5个前端框架)"
    echo "  • Next.js, Nuxt, Remix, Gatsby, SvelteKit (5个全栈框架)"
    echo "  • Express, Koa, Fastify, Hapi, NestJS, Loopback (6个后端框架)"
    echo "  • Webpack, Vite, Parcel, Esbuild, Jest (4个工具)"
    echo ""
    echo "预期结果:"
    echo "  📂 输出目录: real_data/github_frameworks/"
    echo "  💾 数据大小: 500MB - 1GB"
    echo "  ⏱️  耗时: 6-8小时"
    echo ""
    
    read -p "确认开始下载? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "已取消"
        return
    fi
    
    # 检查Token
    if [ -z "$GITHUB_TOKEN" ]; then
        read -p "请输入GitHub Token: " GITHUB_TOKEN
        export GITHUB_TOKEN="$GITHUB_TOKEN"
    fi
    
    echo ""
    echo "${YELLOW}开始下载...${NC}"
    python3 training/github_framework_crawler.py "$GITHUB_TOKEN"
    
    echo ""
    echo "${GREEN}✅ GitHub框架下载完成${NC}"
    echo ""
}

# 选项4: 下载NPM包
download_npm() {
    echo ""
    echo "${GREEN}📥 开始下载NPM包${NC}"
    echo ""
    echo "覆盖包:"
    echo "  • 35+个主要JavaScript包"
    echo "  • 包括React、Vue、Angular、Webpack等"
    echo ""
    echo "预期结果:"
    echo "  📂 输出目录: real_data/npm_packages/"
    echo "  💾 数据大小: 1GB - 2GB"
    echo "  ⏱️  耗时: 6-10小时"
    echo ""
    
    read -p "确认开始下载? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "已取消"
        return
    fi
    
    echo ""
    echo "${YELLOW}开始下载...${NC}"
    python3 training/npm_package_crawler.py
    
    echo ""
    echo "${GREEN}✅ NPM包下载完成${NC}"
    echo ""
}

# 选项5: 应用混淆
apply_obfuscation() {
    echo ""
    echo "${GREEN}🔐 应用真实混淆工具${NC}"
    echo ""
    echo "使用工具:"
    echo "  • Terser - 生产级JavaScript最小化器"
    echo "  • UglifyJS - 完整的JavaScript优化工具"
    echo ""
    echo "预期结果:"
    echo "  📂 输出目录: real_data/obfuscated_code/"
    echo "  📊 训练对: 1000+个"
    echo "  💾 数据大小: 2-3GB"
    echo "  ⏱️  耗时: 2-3天"
    echo ""
    
    read -p "确认开始混淆? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "已取消"
        return
    fi
    
    echo ""
    echo "${YELLOW}检查依赖...${NC}"
    npx terser --version
    npx uglify-js --version
    
    echo ""
    echo "${YELLOW}开始混淆...${NC}"
    python3 training/real_code_obfuscator.py
    
    echo ""
    echo "${GREEN}✅ 混淆完成${NC}"
    echo ""
}

# 选项6: 训练模型
train_model() {
    echo ""
    echo "${GREEN}🤖 GPU框架检测模型训练${NC}"
    echo ""
    echo "模型配置:"
    echo "  • 架构: Transformer编码器"
    echo "  • 框架检测: 24个JavaScript框架"
    echo "  • 优化: GTX 1060专用优化"
    echo ""
    echo "预期结果:"
    echo "  📂 输出: models/local/"
    echo "  📊 准确率: 75-85%"
    echo "  ⏱️  耗时: 30-60分钟"
    echo "  💾 模型大小: 18MB"
    echo ""
    
    read -p "确认开始训练? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "已取消"
        return
    fi
    
    echo ""
    echo "${YELLOW}开始训练...${NC}"
    echo "(可以在另一个终端运行: watch -n 1 nvidia-smi 监控GPU)"
    echo ""
    
    python3 training/gpu_framework_detector.py
    
    echo ""
    echo "${GREEN}✅ 训练完成${NC}"
    echo "模型已保存:"
    echo "  • PyTorch: models/local/framework_detector_gpu.pt"
    echo "  • ONNX: models/local/framework_detector_gpu.onnx"
    echo ""
}

# 选项7: 查看数据统计
show_stats() {
    echo ""
    echo "${GREEN}📊 数据统计${NC}"
    echo ""
    
    # GitHub框架
    if [ -d "real_data/github_frameworks" ]; then
        github_count=$(find real_data/github_frameworks -name "*.js" -o -name "*.ts" 2>/dev/null | wc -l)
        github_size=$(du -sh real_data/github_frameworks 2>/dev/null | cut -f1)
        echo "✅ GitHub框架:"
        echo "   文件数: $github_count"
        echo "   大小: $github_size"
    else
        echo "❌ GitHub框架: 未下载"
    fi
    
    echo ""
    
    # NPM包
    if [ -d "real_data/npm_packages" ]; then
        npm_count=$(ls real_data/npm_packages 2>/dev/null | wc -l)
        npm_size=$(du -sh real_data/npm_packages 2>/dev/null | cut -f1)
        echo "✅ NPM包:"
        echo "   包数: $npm_count"
        echo "   大小: $npm_size"
    else
        echo "❌ NPM包: 未下载"
    fi
    
    echo ""
    
    # 混淆代码
    if [ -d "real_data/obfuscated_code" ]; then
        if [ -f "real_data/obfuscated_code/training_pairs.jsonl" ]; then
            pair_count=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
            echo "✅ 混淆代码对:"
            echo "   对数: $pair_count"
        fi
        
        if [ -f "real_data/obfuscated_code/statistics.json" ]; then
            echo "   统计: real_data/obfuscated_code/statistics.json"
        fi
        
        obf_size=$(du -sh real_data/obfuscated_code 2>/dev/null | cut -f1)
        echo "   大小: $obf_size"
    else
        echo "❌ 混淆代码: 未生成"
    fi
    
    echo ""
    
    # 模型
    if [ -f "models/local/framework_detector_gpu.pt" ]; then
        model_size=$(ls -lh models/local/framework_detector_gpu.pt | awk '{print $5}')
        echo "✅ 训练的模型:"
        echo "   PyTorch模型: $model_size"
    else
        echo "❌ 模型: 未训练"
    fi
    
    if [ -f "models/local/framework_detector_gpu.onnx" ]; then
        onnx_size=$(ls -lh models/local/framework_detector_gpu.onnx | awk '{print $5}')
        echo "   ONNX模型: $onnx_size"
    fi
    
    echo ""
    
    # 总大小
    if [ -d "real_data" ]; then
        total_size=$(du -sh real_data 2>/dev/null | cut -f1)
        echo "📊 总数据大小: $total_size"
    fi
    
    echo ""
}

# 选项8: 监控GPU
monitor_gpu() {
    echo ""
    echo "${GREEN}🖥️  GPU实时监控${NC}"
    echo ""
    echo "监控命令: watch -n 1 nvidia-smi"
    echo "按 Ctrl+C 停止"
    echo ""
    
    watch -n 1 nvidia-smi
}

# 选项9: 清理数据
cleanup_data() {
    echo ""
    echo "${YELLOW}⚠️  清理数据${NC}"
    echo ""
    echo "这将删除所有生成的数据!"
    echo ""
    
    read -p "确认删除? (输入 'yes' 确认): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "已取消"
        return
    fi
    
    echo ""
    echo "删除GitHub框架..."
    rm -rf real_data/github_frameworks
    
    echo "删除NPM包..."
    rm -rf real_data/npm_packages
    
    echo "删除混淆代码..."
    rm -rf real_data/obfuscated_code
    
    echo "删除训练的模型..."
    rm -f models/local/framework_detector_gpu.pt
    rm -f models/local/framework_detector_gpu.onnx
    
    echo ""
    echo "${GREEN}✅ 清理完成${NC}"
    echo ""
}

# 主菜单循环
while true; do
    show_menu
    read -p "选择操作 (0-9, q): " choice
    
    case $choice in
        0) show_guide ;;
        1) check_env ;;
        2) get_github_token ;;
        3) download_github ;;
        4) download_npm ;;
        5) apply_obfuscation ;;
        6) train_model ;;
        7) show_stats ;;
        8) monitor_gpu ;;
        9) cleanup_data ;;
        q|Q) 
            echo ""
            echo "${GREEN}👋 再见!${NC}"
            echo "继续使用? 运行: bash interactive_real_data_learning.sh"
            echo ""
            exit 0
            ;;
        *)
            echo "${RED}❌ 无效选项${NC}"
            ;;
    esac
done
