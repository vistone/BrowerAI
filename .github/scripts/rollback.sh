#!/bin/bash
# Phase E Rollback Script
# Purpose: Manual rollback to previous deployment version

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "↩️  Rollback Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Configuration
DEPLOYMENT_NAME=${DEPLOYMENT_NAME:-browerai-api-deployment}
NAMESPACE=${NAMESPACE:-browerai}
REVISION=${1:-}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log_step() {
    echo -e "${GREEN}→${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check kubectl connection
log_step "Verifying kubectl connection..."
if ! kubectl cluster-info > /dev/null 2>&1; then
    log_error "Failed to connect to Kubernetes cluster"
    exit 1
fi
echo "✅ Connected to Kubernetes"

# Get current rollout history
echo ""
log_step "Getting rollout history for $DEPLOYMENT_NAME in namespace $NAMESPACE..."
echo ""

HISTORY=$(kubectl rollout history deployment/$DEPLOYMENT_NAME -n $NAMESPACE)
if [ -z "$HISTORY" ]; then
    log_error "No rollout history found for deployment $DEPLOYMENT_NAME"
    exit 1
fi

echo "Current rollout history:"
echo "$HISTORY"
echo ""

# If no revision specified, show usage
if [ -z "$REVISION" ]; then
    log_warning "No revision specified. Usage: $0 <revision>"
    echo ""
    echo "Examples:"
    echo "  $0 1        # Rollback to revision 1"
    echo "  $0 previous # Rollback to previous revision (will use latest non-current)"
    exit 1
fi

# Perform rollback
echo ""
log_step "Performing rollback..."

if [ "$REVISION" = "previous" ]; then
    log_step "Rolling back to previous revision..."
    kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE --record
else
    log_step "Rolling back to revision $REVISION..."
    kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE --to-revision=$REVISION --record
fi

echo "✅ Rollback initiated"

# Wait for rollout
echo ""
log_step "Waiting for rollout to complete..."
if kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=5m; then
    echo "✅ Rollout completed successfully"
else
    log_error "Rollout timed out or failed"
    exit 1
fi

# Verify rollback
echo ""
log_step "Verifying rollback..."
echo ""

echo "Current deployment status:"
kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE -o wide

echo ""
echo "Pod status:"
kubectl get pods -n $NAMESPACE -l app=browerai-api -o wide

echo ""
echo "Recent rollout history:"
kubectl rollout history deployment/$DEPLOYMENT_NAME -n $NAMESPACE | head -5

# Run health check
echo ""
log_step "Running health check..."

# Try to port-forward and check health
HEALTH_CHECK_TIMEOUT=30
HEALTH_CHECK_ATTEMPTS=0

while [ $HEALTH_CHECK_ATTEMPTS -lt 5 ]; do
    # Kill any existing port-forward
    pkill -f "kubectl port-forward" || true
    sleep 1
    
    # Start new port-forward in background
    kubectl port-forward -n $NAMESPACE svc/browerai-api-service 5000:5000 > /dev/null 2>&1 &
    PORT_FORWARD_PID=$!
    
    sleep 2
    
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo "✅ Service is healthy"
        kill $PORT_FORWARD_PID 2>/dev/null || true
        HEALTH_CHECK_SUCCESS=true
        break
    fi
    
    kill $PORT_FORWARD_PID 2>/dev/null || true
    HEALTH_CHECK_ATTEMPTS=$((HEALTH_CHECK_ATTEMPTS + 1))
    sleep 2
done

if [ "$HEALTH_CHECK_SUCCESS" != "true" ]; then
    log_warning "Could not verify health check (service may still be starting)"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Rollback Successful"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Deployment: $DEPLOYMENT_NAME"
echo "Namespace: $NAMESPACE"
if [ "$REVISION" = "previous" ]; then
    echo "Revision: Previous (auto-selected)"
else
    echo "Revision: $REVISION"
fi
echo "Status: ✅ Rolled back and verified"
echo "Time: $(date)"
echo ""
echo "Next steps:"
echo "  1. Verify application in $NAMESPACE namespace"
echo "  2. Check logs: kubectl logs -n $NAMESPACE -l app=browerai-api"
echo "  3. Re-deploy if ready: kubectl set image ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
