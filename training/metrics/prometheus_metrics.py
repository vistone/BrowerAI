#!/usr/bin/env python3
"""
Prometheus监控模块 - 框架检测API的性能和可靠性指标
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

# 创建独立的registry,避免与Flask-Prometheus冲突
REGISTRY = CollectorRegistry()

# ==== 请求指标 ====

request_count = Counter(
    'framework_detector_requests_total',
    '框架检测API总请求数',
    ['endpoint', 'method', 'status'],
    registry=REGISTRY
)

request_duration_seconds = Histogram(
    'framework_detector_request_duration_seconds',
    '请求响应时间分布(秒)',
    ['endpoint', 'method'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0),
    registry=REGISTRY
)

# ==== 检测指标 ====

detection_count = Counter(
    'framework_detection_total',
    '框架检测总数',
    ['framework', 'confidence_level'],  # confidence_level: high/medium/low
    registry=REGISTRY
)

detection_accuracy = Gauge(
    'framework_detection_accuracy',
    '检测准确率(%)',
    ['framework'],
    registry=REGISTRY
)

# ==== 错误指标 ====

error_count = Counter(
    'framework_detector_errors_total',
    '错误总数',
    ['error_type', 'endpoint'],
    registry=REGISTRY
)

# ==== 性能指标 ====

model_load_time = Gauge(
    'framework_detector_model_load_seconds',
    '模型加载耗时(秒)',
    ['model_type'],
    registry=REGISTRY
)

detector_inference_time = Histogram(
    'framework_detector_inference_seconds',
    '检测器推理耗时(秒)',
    ['detector_type'],  # rule_based, deep_learning
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
    registry=REGISTRY
)

# ==== 系统指标 ====

api_availability = Gauge(
    'framework_api_availability_percent',
    'API可用性(%)',
    registry=REGISTRY
)

active_requests = Gauge(
    'framework_detector_active_requests',
    '当前活跃请求数',
    registry=REGISTRY
)

detected_frameworks_gauge = Gauge(
    'framework_detected_frameworks',
    '检测到的框架数',
    ['framework'],
    registry=REGISTRY
)

# ==== 缓存指标 ====

cache_hit_total = Counter(
    'framework_detector_cache_hits_total',
    '缓存命中总数',
    registry=REGISTRY
)

cache_miss_total = Counter(
    'framework_detector_cache_misses_total',
    '缓存未命中总数',
    registry=REGISTRY
)

cache_size_bytes = Gauge(
    'framework_detector_cache_size_bytes',
    '缓存大小(字节)',
    registry=REGISTRY
)

# ==== 装饰器 ====

def track_request(endpoint: str, method: str = 'POST'):
    """追踪请求的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            active_requests.inc()
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                # 检查是否是成功响应
                if isinstance(result, dict) and 'success' in result:
                    status = 'success' if result.get('success') else 'error'
                return result
            except Exception as e:
                status = 'error'
                error_type = type(e).__name__
                error_count.labels(error_type=error_type, endpoint=endpoint).inc()
                logger.error(f"请求错误 {endpoint}: {error_type} - {str(e)}")
                raise
            finally:
                duration = time.time() - start_time
                request_duration_seconds.labels(endpoint=endpoint, method=method).observe(duration)
                request_count.labels(endpoint=endpoint, method=method, status=status).inc()
                active_requests.dec()
                
                # 记录慢请求
                if duration > 1.0:
                    logger.warning(f"慢请求告警 {endpoint}: {duration:.2f}s")
        
        return wrapper
    return decorator


def track_detection(detector_type: str = 'rule_based'):
    """追踪检测操作的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                framework, confidence = func(*args, **kwargs)
                
                # 记录检测
                if framework:
                    detection_count.labels(
                        framework=framework,
                        confidence_level='high' if confidence > 0.8 else 'medium' if confidence > 0.5 else 'low'
                    ).inc()
                    
                    # 更新框架计数器
                    detected_frameworks_gauge.labels(framework=framework).inc()
                
                return framework, confidence
            finally:
                duration = time.time() - start_time
                detector_inference_time.labels(detector_type=detector_type).observe(duration)
        
        return wrapper
    return decorator


# ==== 监控统计类 ====

class MetricsCollector:
    """收集和报告监控指标"""
    
    def __init__(self):
        self.total_requests = 0
        self.total_detections = 0
        self.total_errors = 0
        self.start_time = time.time()
    
    def get_metrics(self) -> dict:
        """获取当前指标摘要"""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.total_requests,
            'total_detections': self.total_detections,
            'total_errors': self.total_errors,
            'avg_uptime_hours': uptime / 3600,
        }
    
    def reset(self):
        """重置指标(用于新会话)"""
        self.total_requests = 0
        self.total_detections = 0
        self.total_errors = 0
        self.start_time = time.time()


# 全局收集器实例
metrics_collector = MetricsCollector()


def export_metrics() -> str:
    """导出Prometheus格式的指标"""
    return generate_latest(REGISTRY).decode('utf-8')


if __name__ == '__main__':
    print("Prometheus监控模块已加载")
    print(f"\n注册指标数: {len(REGISTRY.collect())}")
    print("\n可用指标:")
    for collector in REGISTRY.collect():
        for metric in collector.samples:
            print(f"  - {metric.name}")
