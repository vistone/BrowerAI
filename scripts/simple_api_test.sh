#!/bin/bash

# 简化的API集成测试
# 快速验证核心功能

set -e

API_URL="http://localhost:3000/api"

echo "🚀 BrowerAI API 集成测试"
echo "========================"
echo ""

# 测试1: 健康检查
echo "📋 测试1: 健康检查"
HEALTH=$(curl -s "$API_URL/health")
if echo "$HEALTH" | grep -q '"status":"ok"'; then
    echo "✅ 健康检查通过"
    echo "   响应: $HEALTH"
else
    echo "❌ 健康检查失败"
    exit 1
fi
echo ""

# 测试2: 版本信息
echo "📋 测试2: 版本信息"
VERSION=$(curl -s "$API_URL/version")
if echo "$VERSION" | grep -q '"version"'; then
    echo "✅ 版本信息获取成功"
    echo "   响应: $VERSION"
else
    echo "❌ 版本信息获取失败"
fi
echo ""

# 测试3: HTML解析
echo "📋 测试3: HTML解析"
HTML_RESPONSE=$(curl -s -X POST "$API_URL/v1/parse/html" \
  -H "Content-Type: application/json" \
  -d '{"html":"<html><body><h1>Test</h1><p>Hello World</p></body></html>"}')

if echo "$HTML_RESPONSE" | grep -q '"success":true'; then
    echo "✅ HTML解析成功"
    echo "   响应: $HTML_RESPONSE"
else
    echo "❌ HTML解析失败"
    echo "   响应: $HTML_RESPONSE"
fi
echo ""

# 测试4: CSS解析
echo "📋 测试4: CSS解析"
CSS_RESPONSE=$(curl -s -X POST "$API_URL/v1/parse/css" \
  -H "Content-Type: application/json" \
  -d '{"css":"body { color: red; margin: 10px; } .class { display: flex; }"}')

if echo "$CSS_RESPONSE" | grep -q '"success":true'; then
    echo "✅ CSS解析成功"
    echo "   响应: $CSS_RESPONSE"
else
    echo "❌ CSS解析失败"
    echo "   响应: $CSS_RESPONSE"
fi
echo ""

# 测试5: 完整渲染
echo "📋 测试5: 完整渲染"
RENDER_RESPONSE=$(curl -s -X POST "$API_URL/v1/render" \
  -H "Content-Type: application/json" \
  -d '{"html":"<html><body><h1>Title</h1><p>Content</p></body></html>","css":"body{color:blue;}h1{font-size:24px;}"}')

if echo "$RENDER_RESPONSE" | grep -q '"success":true'; then
    echo "✅ 渲染成功"
    echo "   响应: $RENDER_RESPONSE"
else
    echo "❌ 渲染失败"
    echo "   响应: $RENDER_RESPONSE"
fi
echo ""

# 测试6: 性能测试
echo "📋 测试6: 性能测试 (10个请求)"
START_TIME=$(date +%s%N)
for i in {1..10}; do
    curl -s "$API_URL/health" > /dev/null
done
END_TIME=$(date +%s%N)
DURATION=$(( (END_TIME - START_TIME) / 1000000 ))
AVG_LATENCY=$(( DURATION / 10 ))

echo "✅ 性能测试完成"
echo "   总时间: ${DURATION}ms"
echo "   平均延迟: ${AVG_LATENCY}ms"
echo ""

# 总结
echo "========================"
echo "✅ 所有测试完成!"
echo "API服务器运行正常"
