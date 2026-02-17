#!/bin/bash
# Week 6 加强学习系统 - 快速启动脚本
# ===========================================
# 
# 用法:
#   ./train_week6.sh              # 运行完整管道
#   ./train_week6.sh collect      # 仅采集数据
#   ./train_week6.sh train        # 仅训练
#   ./train_week6.sh --gpu cuda:0 # 指定GPU

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# 检查Python环境
check_python() {
    print_step "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python3未安装"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "   Python版本: $PYTHON_VERSION"
}

# 检查依赖
check_dependencies() {
    print_step "检查依赖..."
    
    # 检查必需包
    REQUIRED_PACKAGES=("json" "logging" "pathlib" "datetime")
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            print_warning "$package 未安装，尝试安装..."
        fi
    done
    
    # 检查PyTorch
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        echo -e "   ${GREEN}✓${NC} PyTorch: $TORCH_VERSION"
        
        # 检查CUDA
        if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
            CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
            if [ "$CUDA_AVAILABLE" = "True" ]; then
                echo -e "   ${GREEN}✓${NC} CUDA 可用"
            else
                echo -e "   ${YELLOW}⚠${NC} CUDA 不可用，将使用 CPU"
            fi
        fi
    else
        print_warning "PyTorch 未安装"
        echo "   建议安装: pip install torch"
    fi
}

# 检查GPU
check_gpu() {
    print_step "检查GPU..."
    
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'   GPU 设备数: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem = props.total_memory / 1e9
        print(f'   GPU {i}: {props.name} ({mem:.1f} GB)')
else:
    print('   GPU 不可用，将使用 CPU')
" 2>/dev/null || echo "   无法检查GPU"
}

# 创建输出目录
create_directories() {
    print_step "创建输出目录..."
    
    mkdir -p data/week6_unified_learning/{raw_data,obfuscation_samples,gpu_training/checkpoints}
    mkdir -p data/real_codes
    echo "   ✓ 目录创建完成"
}

# 运行完整管道
run_full_pipeline() {
    print_header "Week 6 加强学习系统 - 完整管道"
    
    check_python
    check_dependencies
    check_gpu
    create_directories
    
    echo ""
    print_step "启动统一学习管道..."
    echo ""
    
    python3 training/scripts/unified_learning_pipeline.py \
        --mode full \
        --gpu "${GPU_DEVICE:-auto}" \
        --samples "${NUM_SAMPLES:-500}" \
        --epochs "${NUM_EPOCHS:-100}"
}

# 仅采集数据
run_collect_only() {
    print_header "Week 6 数据采集"
    
    check_python
    create_directories
    
    echo ""
    print_step "采集真实数据..."
    echo ""
    
    python3 training/scripts/unified_learning_pipeline.py --mode collect
}

# 仅生成混淆样本
run_generate_only() {
    print_header "Week 6 混淆样本生成"
    
    check_python
    create_directories
    
    echo ""
    print_step "生成混淆样本..."
    echo ""
    
    python3 training/scripts/unified_learning_pipeline.py --mode generate
}

# 仅训练
run_train_only() {
    print_header "Week 6 GPU加速训练"
    
    check_python
    check_gpu
    create_directories
    
    echo ""
    print_step "启动GPU训练..."
    echo ""
    
    python3 training/scripts/unified_learning_pipeline.py \
        --mode train \
        --gpu "${GPU_DEVICE:-auto}" \
        --epochs "${NUM_EPOCHS:-100}"
}

# 真实数据学习（全流程）
run_real_learning() {
    print_header "Week 6 真实数据学习系统"

    check_python
    check_dependencies
    check_gpu
    create_directories

    echo ""
    print_step "运行真实数据学习（采集 → 混淆 → 训练）..."
    echo ""

    python3 training/scripts/real_learning.py \
        --collect-dir "${REAL_COLLECT_DIR:-crates}" \
        --techniques "${REAL_TECHNIQUES:-4}" \
        --epochs "${NUM_EPOCHS:-50}" \
        --batch-size "${REAL_BATCH_SIZE:-32}"
}

# 检查GPU环境
check_gpu_environment() {
    print_header "GPU环境检查"
    
    check_python
    check_gpu
    
    echo ""
    print_step "详细检查..."
    echo ""
    
    python3 training/scripts/gpu_unified_training.py --check-gpu
}

# 打印帮助
print_help() {
    echo ""
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  full      - 运行完整管道 (默认)"
    echo "  collect   - 仅采集真实数据"
    echo "  generate  - 仅生成混淆样本"
    echo "  train     - 仅进行GPU训练"
    echo "  real      - 真实数据学习全流程"
    echo "  check-gpu - 检查GPU环境"
    echo "  help      - 显示本帮助"
    echo ""
    echo "选项:"
    echo "  --gpu GPU_ID      - 指定GPU设备 (默认: auto)"
    echo "  --samples NUM     - 混淆样本数 (默认: 500)"
    echo "  --epochs NUM      - 训练轮数 (默认: 100)"
    echo "  --collect-dir DIR - 真实学习采集目录 (默认: crates)"
    echo "  --techniques NUM  - 真实学习混淆技术数 (默认: 4)"
    echo "  --batch-size NUM  - 真实学习批大小 (默认: 32)"
    echo ""
    echo "示例:"
    echo "  ./train_week6.sh                    # 完整管道"
    echo "  ./train_week6.sh collect            # 仅采集"
    echo "  ./train_week6.sh train --gpu cuda:0 # GPU训练"
    echo "  ./train_week6.sh train --epochs 200 # 自定义轮数"
    echo "  ./train_week6.sh real               # 真实数据学习"
    echo "  ./train_week6.sh real --collect-dir crates --techniques 6 --epochs 50 --batch-size 64"
    echo ""
}

# 主函数
main() {
    # 解析参数
    COMMAND="full"
    GPU_DEVICE="auto"
    NUM_SAMPLES=500
    NUM_EPOCHS=100
    REAL_COLLECT_DIR="crates"
    REAL_TECHNIQUES=4
    REAL_BATCH_SIZE=32
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu)
                GPU_DEVICE="$2"
                shift 2
                ;;
            --samples)
                NUM_SAMPLES="$2"
                shift 2
                ;;
            --epochs)
                NUM_EPOCHS="$2"
                shift 2
                ;;
            --collect-dir)
                REAL_COLLECT_DIR="$2"
                shift 2
                ;;
            --techniques)
                REAL_TECHNIQUES="$2"
                shift 2
                ;;
            --batch-size)
                REAL_BATCH_SIZE="$2"
                shift 2
                ;;
            collect|generate|train|check-gpu|help|full)
                COMMAND="$1"
                shift
                ;;
            real)
                COMMAND="$1"
                shift
                ;;
            *)
                print_error "未知参数: $1"
                print_help
                exit 1
                ;;
        esac
    done
    
    # 执行命令
    case $COMMAND in
        full)
            run_full_pipeline
            ;;
        collect)
            run_collect_only
            ;;
        generate)
            run_generate_only
            ;;
        train)
            run_train_only
            ;;
        real)
            run_real_learning
            ;;
        check-gpu)
            check_gpu_environment
            ;;
        help)
            print_help
            ;;
        *)
            print_error "未知命令: $COMMAND"
            print_help
            exit 1
            ;;
    esac
    
    # 打印完成消息
    echo ""
    print_header "✅ 任务完成"
}

# 运行主函数
main "$@"
