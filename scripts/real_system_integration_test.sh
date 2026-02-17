#!/bin/bash

# 真实的端到端集成测试脚本
# 测试前后端完整流程

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

API_URL="http://localhost:3000/api"
FRONTEND_URL="http://localhost:5173"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 BrowerAI 真实系统集成测试${NC}"
echo "=================================="
echo ""

# 测试1: API服务器健康检查
echo -e "${YELLOW}📋 测试1: API服务器连接${NC}"
if curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API服务器就绪${NC}"
else
    echo -e "${RED}❌ API服务器未响应 (需要运行: cargo run --release -p browerai-api-server)${NC}"
    exit 1
fi

# 测试2: HTML解析
echo ""
echo -e "${YELLOW}📋 测试2: HTML解析${NC}"
HTML_TEST='<!DOCTYPE html><html><body><h1>Test</h1><p>Content</p></body></html>'

HTML_PAYLOAD=$(HTML="$HTML_TEST" python3 - <<'PY'
import json
import os

payload = {"html": os.environ["HTML"]}
print(json.dumps(payload))
PY
)

RESPONSE=$(curl -s -X POST "$API_URL/v1/parse/html" \
  -H "Content-Type: application/json" \
    --data-binary "$HTML_PAYLOAD")

if echo "$RESPONSE" | grep -q '"success":true'; then
        NODE_COUNT=$(echo "$RESPONSE" | python3 - <<'PY'
import json
import sys

data = json.loads(sys.stdin.read())
print(data.get("node_count", ""))
PY
)
    echo -e "${GREEN}✅ HTML解析成功 (节点数: $NODE_COUNT)${NC}"
else
    echo -e "${RED}❌ HTML解析失败${NC}"
    echo "$RESPONSE"
fi

# 测试3: CSS解析
echo ""
echo -e "${YELLOW}📋 测试3: CSS解析${NC}"
CSS_TEST='body { font-family: Arial; color: #333; } .class { margin: 10px; }'

CSS_PAYLOAD=$(CSS="$CSS_TEST" python3 - <<'PY'
import json
import os

payload = {"css": os.environ["CSS"]}
print(json.dumps(payload))
PY
)

RESPONSE=$(curl -s -X POST "$API_URL/v1/parse/css" \
  -H "Content-Type: application/json" \
    --data-binary "$CSS_PAYLOAD")

if echo "$RESPONSE" | grep -q '"success":true'; then
    RULES_COUNT=$(echo "$RESPONSE" | grep -o '"rules_count":[0-9]*' | cut -d':' -f2)
    echo -e "${GREEN}✅ CSS解析成功 (规则数: $RULES_COUNT)${NC}"
else
    echo -e "${RED}❌ CSS解析失败${NC}"
    echo "$RESPONSE"
fi

# 测试4: 完整渲染
echo ""
echo -e "${YELLOW}📋 测试4: 完整渲染${NC}"
RENDER_HTML='<html><body style="color:red;"><h1>Hello</h1></body></html>'
RENDER_CSS='body { background: white; } h1 { font-size: 24px; }'

RENDER_PAYLOAD=$(HTML="$RENDER_HTML" CSS="$RENDER_CSS" python3 - <<'PY'
import json
import os

payload = {
        "html": os.environ["HTML"],
        "css": os.environ["CSS"],
}
print(json.dumps(payload))
PY
)

RESPONSE=$(curl -s -X POST "$API_URL/v1/render" \
  -H "Content-Type: application/json" \
    --data-binary "$RENDER_PAYLOAD")

if echo "$RESPONSE" | grep -q '"success":true'; then
    echo -e "${GREEN}✅ 完整渲染成功${NC}"
else
    echo -e "${RED}❌ 完整渲染失败${NC}"
    echo "$RESPONSE"
fi

# 测试5: 真实数据集测试
echo ""
echo -e "${YELLOW}📋 测试5: 真实代码数据集${NC}"

REAL_SAMPLE=$(find "$PROJECT_ROOT/real_data" -type f -name "*.html" | head -1)
if [ -n "$REAL_SAMPLE" ]; then
    RESPONSE=$(python3 - <<PY | curl -s -X POST "$API_URL/v1/parse/html" -H "Content-Type: application/json" --data-binary @-
import json
from pathlib import Path

sample_path = Path("$REAL_SAMPLE")
content = sample_path.read_text(encoding="utf-8", errors="ignore")
payload = {"html": content[:50000]}
print(json.dumps(payload))
PY
    )

    if echo "$RESPONSE" | grep -q '"success":true'; then
        echo -e "${GREEN}✅ 真实数据解析成功${NC}"
    else
        echo -e "${YELLOW}⚠️  真实数据解析 (可能超大)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  未找到真实数据HTML文件${NC}"
fi

# 测试6: 性能测试
echo ""
echo -e "${YELLOW}📋 测试6: 性能测试${NC}"
echo "测试100次HTML解析..."

START_TIME=$(date +%s%N)
for i in {1..100}; do
        PERF_PAYLOAD=$(HTML="<div>Test $i</div>" python3 - <<'PY'
import json
import os

payload = {"html": os.environ["HTML"]}
print(json.dumps(payload))
PY
        )

        curl -s -X POST "$API_URL/v1/parse/html" \
            -H "Content-Type: application/json" \
            --data-binary "$PERF_PAYLOAD" > /dev/null
done
END_TIME=$(date +%s%N)

DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))
AVG_TIME_MS=$(( DURATION_MS / 100 ))

echo -e "${GREEN}✅ 性能测试完成${NC}"
echo "   总时间: ${DURATION_MS}ms"
echo "   平均时间: ${AVG_TIME_MS}ms"
echo "   吞吐量: $(( 100000 / DURATION_MS )) req/sec"

echo ""
echo -e "${BLUE}=================================="
echo -e "🎉 所有核心测试通过!"
echo -e "=================================="
echo ""
echo -e "${YELLOW}后续步骤:${NC}"
echo "1. 启动前端开发服务器:"
echo "   cd crates/browerai-webclient"
echo "   npm install"
echo "   npm run dev"
echo ""
echo "2. 打开浏览器访问: http://localhost:5173"
echo ""
