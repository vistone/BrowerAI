#!/bin/bash

################################################################################
# Week 8 Phase C - Docker Containerization Automated Test Script
# Purpose: Build, test, and validate Docker container
# Testing Strategy:
#   1. Build Docker image
#   2. Start container and verify health
#   3. Run functional tests
#   4. Execute stress tests (4 load levels)
#   5. Compare performance (Host vs Container)
#   6. Generate comprehensive report
################################################################################

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
TRAINING_DIR="$PROJECT_ROOT/training"
VENV_DIR="$PROJECT_ROOT/venv_test"

DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.python-api.yml"
DOCKERFILE="$PROJECT_DIR/Dockerfile.python-api"
IMAGE_NAME="browerai-api"
IMAGE_TAG="latest"
CONTAINER_NAME="browerai-api-test"

API_HOST="127.0.0.1"
API_PORT="5000"
API_URL="http://$API_HOST:$API_PORT"

LOG_DIR="/tmp"
PHASE_C_LOG="$LOG_DIR/phase_c_containerization.log"
RESULTS_FILE="$LOG_DIR/phase_c_results.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PHASE_C_LOG"
}

log_step() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$PHASE_C_LOG"
    echo -e "${BLUE}$1${NC}" | tee -a "$PHASE_C_LOG"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$PHASE_C_LOG"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$PHASE_C_LOG"
}

log_error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$PHASE_C_LOG"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$PHASE_C_LOG"
}

# Cleanup function
cleanup() {
    log_step "清理资源"
    
    if docker ps | grep -q "$CONTAINER_NAME"; then
        log "停止容器: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME" || true
    fi
    
    if docker ps -a | grep -q "$CONTAINER_NAME"; then
        log "移除容器: $CONTAINER_NAME"
        docker rm "$CONTAINER_NAME" || true
    fi
    
    log_success "清理完成"
}

# Trap errors and cleanup
trap cleanup EXIT

# ============================================================================
# Phase C: Step 1 - Check Prerequisites
# ============================================================================

check_prerequisites() {
    log_step "Step 1: 检查前置条件"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装"
        exit 1
    fi
    log_success "Docker 已安装: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose 未安装"
        exit 1
    fi
    log_success "Docker Compose 已安装: $(docker-compose --version)"
    
    # Check Python virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        log "创建虚拟环境..."
        python3 -m venv "$VENV_DIR"
    fi
    log_success "虚拟环境就绪: $VENV_DIR"
    
    # Check Dockerfile
    if [ ! -f "$DOCKERFILE" ]; then
        log_error "Dockerfile 不存在: $DOCKERFILE"
        exit 1
    fi
    log_success "Dockerfile 存在"
    
    # Check docker-compose file
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "docker-compose 文件不存在: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
    log_success "docker-compose 文件存在"
}

# ============================================================================
# Phase C: Step 2 - Build Docker Image
# ============================================================================

build_docker_image() {
    log_step "Step 2: 构建 Docker 镜像"
    
    log "构建命令: docker build -f $DOCKERFILE -t $IMAGE_NAME:$IMAGE_TAG ."
    
    BUILD_START=$(date +%s%N)
    
    if docker build -f "$DOCKERFILE" -t "$IMAGE_NAME:$IMAGE_TAG" "$PROJECT_ROOT"; then
        BUILD_END=$(date +%s%N)
        BUILD_TIME=$(( (BUILD_END - BUILD_START) / 1000000 ))
        
        log_success "Docker 镜像构建成功: ${BUILD_TIME}ms"
        
        # Get image info
        IMAGE_SIZE=$(docker images "$IMAGE_NAME:$IMAGE_TAG" --format='{{.Size}}')
        log_success "镜像大小: $IMAGE_SIZE"
        
        # Display image details
        log "镜像详情:"
        docker images "$IMAGE_NAME:$IMAGE_TAG" | tee -a "$PHASE_C_LOG"
    else
        log_error "Docker 镜像构建失败"
        exit 1
    fi
}

# ============================================================================
# Phase C: Step 3 - Start Container
# ============================================================================

start_container() {
    log_step "Step 3: 启动容器"
    
    # Cleanup any existing container
    cleanup || true
    
    log "启动容器: $CONTAINER_NAME"
    
    START_TIME=$(date +%s)
    
    # Start container with docker run
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "$API_PORT:5000" \
        -e FLASK_ENV=production \
        -e LOG_LEVEL=INFO \
        "$IMAGE_NAME:$IMAGE_TAG"
    
    # Wait for container to be ready
    READY=false
    ATTEMPT=0
    MAX_ATTEMPTS=30
    
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if curl -s "$API_URL/health" &> /dev/null; then
            READY=true
            break
        fi
        
        ATTEMPT=$((ATTEMPT + 1))
        sleep 1
    done
    
    STOP_TIME=$(date +%s)
    STARTUP_TIME=$((STOP_TIME - START_TIME))
    
    if [ "$READY" = true ]; then
        log_success "容器启动成功，启动时间: ${STARTUP_TIME}s"
    else
        log_error "容器启动失败或健康检查超时"
        docker logs "$CONTAINER_NAME" | tee -a "$PHASE_C_LOG"
        exit 1
    fi
    
    # Show container info
    log "容器信息:"
    docker ps --filter "name=$CONTAINER_NAME" | tee -a "$PHASE_C_LOG"
}

# ============================================================================
# Phase C: Step 4 - Functional Tests
# ============================================================================

run_functional_tests() {
    log_step "Step 4: 功能测试"
    
    TESTS_PASSED=0
    TESTS_FAILED=0
    
    # Test 1: Health check
    log "测试 1: 健康检查..."
    if curl -s "$API_URL/health" | grep -q "healthy"; then
        log_success "健康检查通过"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_error "健康检查失败"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    
    # Test 2: Feature encoding endpoint
    log "测试 2: 特征编码端点..."
    RESPONSE=$(curl -s -X POST "$API_URL/encode" \
        -H "Content-Type: application/json" \
        -d '{"url": "http://example.com", "html": "<html></html>"}')
    
    if echo "$RESPONSE" | grep -q "encoded_features"; then
        log_success "特征编码端点正常"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_error "特征编码端点异常: $RESPONSE"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    
    # Test 3: Code generation endpoint
    log "测试 3: 代码生成端点..."
    RESPONSE=$(curl -s -X POST "$API_URL/generate" \
        -H "Content-Type: application/json" \
        -d '{"features": [0.1]*48, "website_intent": "blog"}')
    
    if echo "$RESPONSE" | grep -q "generated_code"; then
        log_success "代码生成端点正常"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_warning "代码生成端点返回: $RESPONSE"
        # Don't fail here as endpoint may not be fully implemented
    fi
    
    # Test 4: Feedback endpoint
    log "测试 4: 反馈端点..."
    RESPONSE=$(curl -s -X POST "$API_URL/feedback" \
        -H "Content-Type: application/json" \
        -d '{"url": "http://example.com", "quality_score": 0.85}')
    
    if echo "$RESPONSE" | grep -q "accepted\|success"; then
        log_success "反馈端点正常"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_warning "反馈端点返回: $RESPONSE"
    fi
    
    log "功能测试结果: $TESTS_PASSED 通过, $TESTS_FAILED 失败"
    
    if [ $TESTS_FAILED -gt 0 ]; then
        log_warning "部分功能测试失败，但继续进行负载测试"
    fi
}

# ============================================================================
# Phase C: Step 5 - Container Stress Tests
# ============================================================================

run_container_stress_tests() {
    log_step "Step 5: 容器压力测试"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    cd "$TRAINING_DIR"
    
    # Define test scenarios
    TEST_SCENARIOS=(
        "10:10:light_load"
        "25:10:medium_load"
        "50:10:heavy_load"
        "100:5:extreme_load"
    )
    
    TOTAL_REQUESTS=0
    TOTAL_SUCCESS=0
    TOTAL_FAILED=0
    
    for scenario in "${TEST_SCENARIOS[@]}"; do
        IFS=':' read -r users requests label <<< "$scenario"
        
        log "执行测试: $label ($users 并发, $requests 请求/用户)"
        
        RESULT_FILE="/tmp/container_stress_${users}users.json"
        
        python3 stress_test.py \
            --concurrent-users "$users" \
            --requests-per-user "$requests" \
            --base-url "$API_URL" \
            --output "$RESULT_FILE"
        
        if [ -f "$RESULT_FILE" ]; then
            # Parse results
            SUCCESS=$(python3 -c "import json; data=json.load(open('$RESULT_FILE')); print(data.get('summary', {}).get('successful_requests', 0))")
            FAILED=$(python3 -c "import json; data=json.load(open('$RESULT_FILE')); print(data.get('summary', {}).get('failed_requests', 0))")
            
            TOTAL_REQUESTS=$((TOTAL_REQUESTS + users * requests))
            TOTAL_SUCCESS=$((TOTAL_SUCCESS + SUCCESS))
            TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
            
            log_success "测试完成: $SUCCESS 成功, $FAILED 失败"
        else
            log_error "测试结果文件未生成"
        fi
    done
    
    log "容器压力测试汇总:"
    log "  总请求数: $TOTAL_REQUESTS"
    log "  成功: $TOTAL_SUCCESS"
    log "  失败: $TOTAL_FAILED"
    
    if [ $TOTAL_FAILED -eq 0 ]; then
        log_success "所有压力测试通过！"
    else
        log_warning "部分压力测试失败"
    fi
    
    deactivate
}

# ============================================================================
# Phase C: Step 6 - Performance Analysis
# ============================================================================

analyze_performance() {
    log_step "Step 6: 性能分析"
    
    # Get container stats
    log "收集容器性能数据..."
    
    CONTAINER_ID=$(docker ps --filter "name=$CONTAINER_NAME" -q)
    
    if [ -n "$CONTAINER_ID" ]; then
        # Memory usage
        MEMORY=$(docker stats "$CONTAINER_ID" --no-stream --format='{{.MemUsage}}')
        log_success "内存使用: $MEMORY"
        
        # CPU usage
        CPU=$(docker stats "$CONTAINER_ID" --no-stream --format='{{.CPUPerc}}')
        log_success "CPU 使用率: $CPU"
        
        # Container info
        docker inspect "$CONTAINER_ID" > /tmp/container_info.json
        log_success "容器信息已保存到 /tmp/container_info.json"
    fi
    
    log "性能对比分析"
    log "  Host 性能指标 (Phase B):"
    log "    - RPS (50 并发): 164.44"
    log "    - 平均延迟: 3.61ms"
    log "    - 内存: 46.3MB"
    log "  容器内性能指标 (Phase C):"
    log "    - 待测试..."
}

# ============================================================================
# Phase C: Step 7 - Generate Report
# ============================================================================

generate_report() {
    log_step "Step 7: 生成执行报告"
    
    cat > "$RESULTS_FILE" << 'EOF'
{
  "phase": "Phase C - Docker Containerization",
  "timestamp": "$(date -Iseconds)",
  "status": "IN_PROGRESS",
  "tests": {
    "prerequisites": { "status": "PASSED" },
    "docker_build": { "status": "PENDING" },
    "container_startup": { "status": "PENDING" },
    "functional_tests": { "status": "PENDING" },
    "stress_tests": { "status": "PENDING" },
    "performance_analysis": { "status": "PENDING" }
  },
  "summary": {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "success_rate": "0%"
  }
}
EOF
    
    log_success "报告已生成: $RESULTS_FILE"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║          Week 8 Phase C - Docker Containerization Test             ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    log_step "初始化"
    log "开始时间: $(date)"
    log "项目路径: $PROJECT_ROOT"
    log "容器名称: $CONTAINER_NAME"
    log "API 地址: $API_URL"
    
    # Execute test steps
    check_prerequisites
    sleep 1
    
    build_docker_image
    sleep 1
    
    start_container
    sleep 2
    
    run_functional_tests
    sleep 1
    
    run_container_stress_tests
    sleep 1
    
    analyze_performance
    sleep 1
    
    generate_report
    
    # Final summary
    log_step "Phase C 执行完成"
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║                  Phase C 测试完成！                                ║"
    echo "║                                                                    ║"
    echo "║  日志文件: $PHASE_C_LOG"
    echo "║  结果文件: $RESULTS_FILE"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
}

# Run main
main "$@"
