#!/bin/bash
# BrowerAI 监控系统验证脚本

set -e

echo "=================================="
echo "BrowerAI 监控系统验证"
echo "=================================="
echo ""

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

info() {
    echo -e "→ $1"
}

# 1. 检查 Docker 服务
echo "1. 检查 Docker 服务状态..."
if docker-compose ps | grep -q "browerai-api.*Up"; then
    success "browerai-api 运行中"
else
    error "browerai-api 未运行"
    warning "运行: docker-compose up -d"
fi

if docker-compose ps | grep -q "prometheus.*Up"; then
    success "prometheus 运行中"
else
    warning "prometheus 未运行（可选）"
fi

if docker-compose ps | grep -q "grafana.*Up"; then
    success "grafana 运行中"
else
    warning "grafana 未运行（可选）"
fi
echo ""

# 2. 检查 API 端点
echo "2. 检查 API 端点..."
if curl -s http://localhost:3000/api/health | grep -q "ok"; then
    success "/api/health 可访问"
else
    error "/api/health 不可访问"
fi

if curl -s http://localhost:3000/api/version | grep -q "version"; then
    success "/api/version 可访问"
else
    error "/api/version 不可访问"
fi

if curl -s http://localhost:3000/api/metrics | grep -q "browerai_"; then
    success "/api/metrics 可访问且有数据"
else
    error "/api/metrics 不可访问或无数据"
fi
echo ""

# 3. 生成测试流量
echo "3. 生成测试流量..."
info "发送 5 个请求到 /api/health"
for i in {1..5}; do
    curl -s http://localhost:3000/api/health > /dev/null
done
success "测试流量已生成"
echo ""

# 4. 验证 Metrics
echo "4. 验证 Metrics 数据..."
METRICS=$(curl -s http://localhost:3000/api/metrics)

if echo "$METRICS" | grep -q "browerai_http_requests_total"; then
    success "HTTP 请求计数器存在"
else
    error "HTTP 请求计数器缺失"
fi

if echo "$METRICS" | grep -q "browerai_http_request_duration_seconds"; then
    success "HTTP 延迟直方图存在"
else
    error "HTTP 延迟直方图缺失"
fi

if echo "$METRICS" | grep -q "browerai_css_cache"; then
    success "CSS 缓存指标存在"
else
    warning "CSS 缓存指标缺失（需要 ONNX feature）"
fi

if echo "$METRICS" | grep -q "browerai_ai_inference"; then
    success "AI 推理指标存在"
else
    warning "AI 推理指标缺失（需要 AI 请求）"
fi
echo ""

# 5. 检查 Prometheus（如果运行）
if docker-compose ps | grep -q "prometheus.*Up"; then
    echo "5. 检查 Prometheus..."
    if curl -s http://localhost:9090/-/healthy | grep -q "Prometheus"; then
        success "Prometheus 健康"
    else
        error "Prometheus 不健康"
    fi
    
    # 检查 targets
    TARGETS=$(curl -s http://localhost:9090/api/v1/targets)
    if echo "$TARGETS" | grep -q "browerai-api"; then
        success "Prometheus 正在抓取 browerai-api"
    else
        error "Prometheus 未配置 browerai-api target"
    fi
    echo ""
fi

# 6. 检查 Grafana（如果运行）
if docker-compose ps | grep -q "grafana.*Up"; then
    echo "6. 检查 Grafana..."
    if curl -s http://localhost:3001/api/health | grep -q "ok"; then
        success "Grafana 健康"
    else
        error "Grafana 不健康"
    fi
    echo ""
fi

# 7. 编译检查
echo "7. 编译检查..."
info "检查 API server 编译..."
if cargo check -p browerai-api-server 2>&1 | grep -q "Finished"; then
    success "browerai-api-server 编译通过"
else
    error "browerai-api-server 编译失败"
fi
echo ""

# 8. 测试运行
echo "8. 运行测试..."
info "运行 metrics 模块测试..."
if cargo test -p browerai-api-server metrics 2>&1 | grep -q "test result: ok"; then
    success "Metrics 测试通过"
else
    error "Metrics 测试失败"
fi
echo ""

# 9. 文档检查
echo "9. 文档完整性..."
DOCS=(
    "PHASE3_WEEK4_MONITORING_COMPLETION_REPORT.md"
    "PHASE3_MONITORING_QUICKSTART.md"
    "PHASE3_WEEK4_SUMMARY.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        success "$doc 存在"
    else
        error "$doc 缺失"
    fi
done
echo ""

# 10. 配置文件检查
echo "10. 配置文件完整性..."
CONFIGS=(
    "prometheus.yml"
    "grafana-dashboard.json"
    "grafana-provisioning/datasources/prometheus.yml"
    "grafana-provisioning/dashboards/dashboards.yml"
    "docker-compose.yml"
)

for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        success "$config 存在"
    else
        error "$config 缺失"
    fi
done
echo ""

# 总结
echo "=================================="
echo "验证完成！"
echo "=================================="
echo ""
echo "快速访问："
echo "  • API:        http://localhost:3000"
echo "  • Metrics:    http://localhost:3000/api/metrics"
echo "  • Prometheus: http://localhost:9090"
echo "  • Grafana:    http://localhost:3001 (admin/admin)"
echo ""
echo "下一步："
echo "  1. 访问 Grafana 查看 dashboard"
echo "  2. 生成更多测试流量"
echo "  3. 配置告警规则"
echo ""
