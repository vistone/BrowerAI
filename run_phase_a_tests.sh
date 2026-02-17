#!/bin/bash

# Week 8 Phase A 测试执行脚本
# 启动 API 服务器并运行真实 HTTP 集成测试

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║        Week 8 Phase A - 真实 HTTP 集成测试执行                    ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

cd /home/stone/BrowerAI

# 激活虚拟环境
echo "📦 激活虚拟环境..."
source venv_test/bin/activate

# 启动 API 服务器
echo "🚀 启动 API 服务器..."
cd training
python api_server.py > /tmp/api_server.log 2>&1 &
API_PID=$!
echo "   API 服务器 PID: $API_PID"

# 等待服务器启动
echo "⏳ 等待 API 服务器就绪..."
for i in {1..10}; do
    if curl -s http://127.0.0.1:5000/api/v1/health > /dev/null 2>&1; then
        echo "✅ API 服务器已就绪！"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ API 服务器启动超时"
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
    echo "   等待中... ($i/10)"
    sleep 2
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  开始执行真实 HTTP 集成测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 运行集成测试
python real_http_integration_tests.py
TEST_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  测试完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 关闭 API 服务器
echo "🛑 关闭 API 服务器..."
kill $API_PID 2>/dev/null || true
wait $API_PID 2>/dev/null || true

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║          ✅ Week 8 Phase A 测试全部通过！✅                       ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 查看详细结果:"
    echo "   cat week8_test_results.json"
    echo ""
else
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║          ❌ 部分测试失败，请检查日志                             ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📋 检查服务器日志:"
    echo "   cat /tmp/api_server.log"
    echo ""
fi

exit $TEST_EXIT_CODE
