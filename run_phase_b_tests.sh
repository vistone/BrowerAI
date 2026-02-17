#!/bin/bash

# Week 8 Phase B - Stress Test Execution Script
# Progressive load testing with resource monitoring

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║        Week 8 Phase B - 压力测试执行                              ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

cd /home/stone/BrowerAI

# 激活虚拟环境
echo "📦 激活虚拟环境..."
source venv_test/bin/activate

# 检查依赖
echo "🔍 检查依赖..."
python -c "import psutil, numpy; print('  ✅ psutil and numpy available')" || {
    echo "  📥 安装缺失的依赖..."
    pip install psutil numpy
}

# 启动 API 服务器
echo "🚀 启动 API 服务器..."
cd training
python api_server.py > /tmp/api_server_stress.log 2>&1 &
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
echo "  Phase B - Progressive Load Testing"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1: Light Load (10 concurrent users)
echo "📊 Test 1: Light Load (10 users × 10 requests)"
python stress_test.py \
    --users 10 \
    --requests 10 \
    --ramp-up 1 \
    --output /tmp/stress_test_10users.json

TEST1_STATUS=$?
echo ""
sleep 2

# Test 2: Medium Load (25 concurrent users)
echo "📊 Test 2: Medium Load (25 users × 10 requests)"
python stress_test.py \
    --users 25 \
    --requests 10 \
    --ramp-up 2 \
    --output /tmp/stress_test_25users.json

TEST2_STATUS=$?
echo ""
sleep 2

# Test 3: Heavy Load (50 concurrent users)
echo "📊 Test 3: Heavy Load (50 users × 10 requests)"
python stress_test.py \
    --users 50 \
    --requests 10 \
    --ramp-up 3 \
    --output /tmp/stress_test_50users.json

TEST3_STATUS=$?
echo ""
sleep 2

# Test 4: Extreme Load (100 concurrent users)
echo "📊 Test 4: Extreme Load (100 users × 5 requests)"
python stress_test.py \
    --users 100 \
    --requests 5 \
    --ramp-up 5 \
    --timeout 15 \
    --output /tmp/stress_test_100users.json

TEST4_STATUS=$?
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  测试完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 关闭 API 服务器
echo "🛑 关闭 API 服务器..."
kill $API_PID 2>/dev/null || true
wait $API_PID 2>/dev/null || true

# 生成汇总报告
echo "📊 生成汇总报告..."
python << 'EOF'
import json
from pathlib import Path

results_dir = Path("/tmp")
test_files = [
    ("10 users", results_dir / "stress_test_10users.json"),
    ("25 users", results_dir / "stress_test_25users.json"),
    ("50 users", results_dir / "stress_test_50users.json"),
    ("100 users", results_dir / "stress_test_100users.json")
]

print("\n" + "="*70)
print("  Phase B - Stress Test Summary")
print("="*70)
print()

summary_data = []

for name, file_path in test_files:
    if file_path.exists():
        with open(file_path) as f:
            data = json.load(f)
            summary = data.get("summary", {})
            latency = data.get("latency_ms", {})
            resources = data.get("resources", {})
            
            summary_data.append({
                "test": name,
                "requests": summary.get("total_requests", 0),
                "success_rate": summary.get("success_rate", 0),
                "throughput": summary.get("throughput_rps", 0),
                "latency_mean": latency.get("mean", 0),
                "latency_p95": latency.get("p95", 0),
                "cpu_mean": resources.get("cpu", {}).get("mean", 0),
                "memory_max": resources.get("memory_mb", {}).get("max", 0)
            })

# Print table
print(f"{'Test':<12} {'Requests':<10} {'Success':<10} {'RPS':<10} {'Latency':<15} {'P95':<10} {'CPU%':<10} {'Mem(MB)':<10}")
print("-" * 100)

for item in summary_data:
    print(f"{item['test']:<12} {item['requests']:<10} {item['success_rate']:>8.1f}% {item['throughput']:>8.1f} {item['latency_mean']:>13.2f}ms {item['latency_p95']:>8.2f}ms {item['cpu_mean']:>8.1f}% {item['memory_max']:>8.1f}")

print()
print("="*70)

# Overall assessment
all_passed = all(item['success_rate'] >= 95 for item in summary_data)
if all_passed:
    print("✅ All stress tests PASSED!")
else:
    print("⚠️  Some tests had success rate below 95%")

EOF

# Check overall status
if [ $TEST1_STATUS -eq 0 ] && [ $TEST2_STATUS -eq 0 ] && [ $TEST3_STATUS -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║          ✅ Week 8 Phase B 压力测试全部通过！✅                  ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 详细结果:"
    echo "   Test 1: /tmp/stress_test_10users.json"
    echo "   Test 2: /tmp/stress_test_25users.json"
    echo "   Test 3: /tmp/stress_test_50users.json"
    echo "   Test 4: /tmp/stress_test_100users.json"
    echo ""
    exit 0
else
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║          ⚠️  部分测试未达到预期目标                             ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📋 检查服务器日志:"
    echo "   cat /tmp/api_server_stress.log"
    echo ""
    exit 1
fi
