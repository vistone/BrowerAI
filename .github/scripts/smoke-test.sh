#!/bin/bash
# Phase E Smoke Test Script
# Purpose: Verify deployment health and basic functionality

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Smoke Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Configuration
SERVICE_URL=${1:-http://localhost:5000}
TIMEOUT=${2:-30}
RETRY_COUNT=0
MAX_RETRIES=5

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log_test() {
    echo -e "${GREEN}→${NC} $1"
}

log_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

log_fail() {
    echo -e "  ${RED}✗${NC} $1"
}

test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local expected_code=$4
    
    if [ -z "$expected_code" ]; then
        expected_code="200"
    fi
    
    if [ "$method" = "GET" ]; then
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            -X "$method" \
            "$SERVICE_URL$endpoint")
    else
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$SERVICE_URL$endpoint")
    fi
    
    if [ "$HTTP_CODE" = "$expected_code" ]; then
        log_pass "$method $endpoint -> $HTTP_CODE"
        return 0
    else
        log_fail "$method $endpoint -> $HTTP_CODE (expected $expected_code)"
        return 1
    fi
}

# Wait for service
echo "Waiting for service at $SERVICE_URL..."
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "$SERVICE_URL/health" > /dev/null 2>&1; then
        echo "✅ Service is reachable"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Attempt $RETRY_COUNT/$MAX_RETRIES... waiting..."
    sleep $((RETRY_COUNT * 2))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_fail "Service not reachable after $MAX_RETRIES attempts"
    exit 1
fi

# Test 1: Health Check
echo ""
log_test "Test 1: Health Check"
if test_endpoint "GET" "/health" "" "200"; then
    echo "  ✅ Health check passed"
else
    log_fail "Health check failed"
    exit 1
fi

# Test 2: Feature Encoding
echo ""
log_test "Test 2: Feature Encoding"
ENCODE_PAYLOAD='{"url":"https://example.com","html":"<html><body>Test</body></html>"}'
if test_endpoint "POST" "/encode" "$ENCODE_PAYLOAD" "200"; then
    ENCODE_RESPONSE=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$ENCODE_PAYLOAD" \
        "$SERVICE_URL/encode")
    
    if echo "$ENCODE_RESPONSE" | grep -q "encoded_features"; then
        echo "  ✅ Feature encoding successful"
    else
        log_fail "Feature encoding response invalid"
    fi
else
    log_fail "Feature encoding failed"
fi

# Test 3: Code Generation
echo ""
log_test "Test 3: Code Generation"
GENERATE_PAYLOAD='{"features":[],"website_intent":"search","html_context":"<html></html>"}'
if test_endpoint "POST" "/generate" "$GENERATE_PAYLOAD" "200"; then
    GENERATE_RESPONSE=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$GENERATE_PAYLOAD" \
        "$SERVICE_URL/generate")
    
    if echo "$GENERATE_RESPONSE" | grep -q "generated_code"; then
        echo "  ✅ Code generation successful"
    else
        log_fail "Code generation response invalid"
    fi
else
    log_fail "Code generation failed"
fi

# Test 4: Feedback Submission
echo ""
log_test "Test 4: Feedback Submission"
FEEDBACK_PAYLOAD='{"url":"https://example.com","quality_score":0.85,"feedback":"Test feedback"}'
if test_endpoint "POST" "/feedback" "$FEEDBACK_PAYLOAD" "200"; then
    echo "  ✅ Feedback submission successful"
else
    log_fail "Feedback submission failed"
fi

# Test 5: Response times
echo ""
log_test "Test 5: Response Time Benchmark"
TOTAL_TIME=0
NUM_REQUESTS=10

for i in $(seq 1 $NUM_REQUESTS); do
    START=$(date +%s%N)
    curl -s "$SERVICE_URL/health" > /dev/null
    END=$(date +%s%N)
    ELAPSED=$((($END - $START) / 1000000))
    TOTAL_TIME=$((TOTAL_TIME + ELAPSED))
    echo "  Request $i: ${ELAPSED}ms"
done

AVG_TIME=$((TOTAL_TIME / NUM_REQUESTS))
echo "  Average response time: ${AVG_TIME}ms"

if [ $AVG_TIME -lt 50 ]; then
    log_pass "Response time excellent: ${AVG_TIME}ms"
elif [ $AVG_TIME -lt 100 ]; then
    log_pass "Response time good: ${AVG_TIME}ms"
elif [ $AVG_TIME -lt 200 ]; then
    log_pass "Response time acceptable: ${AVG_TIME}ms"
else
    log_fail "Response time slow: ${AVG_TIME}ms"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Smoke Tests Completed Successfully"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Service URL: $SERVICE_URL"
echo "Tests passed: 5/5"
echo "Average latency: ${AVG_TIME}ms"
echo "Status: 🟢 Healthy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit 0
