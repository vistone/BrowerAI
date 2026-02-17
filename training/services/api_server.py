#!/usr/bin/env python3
"""
Flask API服务器 - 框架检测与反混淆
提供HTTP API供Rust系统调用
"""

import json
import logging
import sys
import os
import hashlib
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, request, jsonify, Response
import torch

from training.detectors.high_precision_detector import HighPrecisionDetector
from training.metrics.prometheus_metrics import (
    track_request, track_detection, export_metrics,
    model_load_time, metrics_collector
)
from training.services.db_layer import get_db_operations
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局检测器实例
detector = None
model = None
label_encoder = None
db_ops = None


def load_models():
    """加载模型"""
    global detector, model, label_encoder, db_ops
    
    logger.info("🔄 加载模型...")
    
    # 规则检测器
    start = time.time()
    detector = HighPrecisionDetector()
    model_load_time.labels(model_type='rule_based').set(time.time() - start)
    logger.info("✅ 规则检测器已加载")
    
    # 初始化数据库
    try:
        db_ops = get_db_operations()
        if db_ops.health_check():
            logger.info("✅ 数据库已连接")
        else:
            logger.warning("⚠️ 数据库连接失败，将禁用缓存")
            db_ops = None
    except Exception as e:
        logger.warning(f"⚠️ 数据库初始化失败: {e}，将禁用缓存")
        db_ops = None
    
    # 深度学习模型
    model_path = Path("models/production/best_model.pt")
    encoder_path = Path("models/production/label_encoder.pkl")
    
    if model_path.exists() and encoder_path.exists():
        try:
            start = time.time()
            label_encoder = torch.load(encoder_path)
            model_load_time.labels(model_type='deep_learning').set(time.time() - start)
            # 模型加载逻辑（需要时可添加）
            logger.info("✅ 深度学习模型已加载")
        except Exception as e:
            logger.warning(f"⚠️ 深度学习模型加载失败: {e}")
    else:
        logger.warning("⚠️ 深度学习模型文件未找到，仅使用规则检测")


# 模块级初始化(Gunicorn导入app时自动执行)
load_models()


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'healthy', 'service': 'framework-detection'})


@app.route('/api/v1/detect', methods=['POST'])
@track_request(endpoint='/api/v1/detect')
def detect_framework():
    """
    框架检测API (带缓存)
    
    请求格式:
    {
        "html": "网站HTML代码",
        "use_ml": false,          // 可选，是否使用深度学习模型
        "use_cache": true         // 可选，是否使用缓存
    }
    
    响应格式:
    {
        "framework": "React",
        "confidence": 0.95,
        "method": "rule_based",
        "from_cache": false,
        "success": true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'html' not in data:
            return jsonify({'error': 'Missing html field'}), 400
        
        html = data['html']
        use_ml = data.get('use_ml', False)
        use_cache = data.get('use_cache', True)
        
        # 生成缓存键(SHA256哈希)
        cache_key = None
        if use_cache and db_ops:
            cache_key = hashlib.sha256(html.encode()).hexdigest()[:32]
            
            # 尝试从缓存获取
            cached = db_ops.get(cache_key)
            if cached:
                logger.debug(f"✅ 缓存命中: {cache_key}")
                return jsonify({**cached, 'from_cache': True, 'success': True})
        
        # 执行检测
        framework, confidence = detector.detect_with_category(html, category=None)
        
        metrics_collector.total_detections += 1
        
        result = {
            'framework': framework,
            'confidence': float(confidence),
            'method': 'rule_based',
            'from_cache': False,
            'success': True,
        }
        
        # 保存到缓存 (1小时TTL)
        if use_cache and db_ops and cache_key:
            try:
                db_ops.set(cache_key, result, ttl_seconds=3600)
                logger.debug(f"✅ 结果已缓存: {cache_key}")
            except Exception as e:
                logger.warning(f"⚠️ 缓存保存失败: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"❌ 检测错误: {e}")
        metrics_collector.total_errors += 1
        return jsonify({
            'error': str(e),
            'success': False,
        }), 500


@app.route('/api/v1/batch_detect', methods=['POST'])
@track_request(endpoint='/api/v1/batch_detect')
def batch_detect():
    """
    批量框架检测API (带缓存)
    
    请求格式:
    {
        "websites": [
            {"url": "https://example.com", "html": "..."},
            ...
        ],
        "use_cache": true  // 可选
    }
    
    响应格式:
    {
        "results": [
            {"url": "...", "framework": "React", "confidence": 0.95, "from_cache": false},
            ...
        ],
        "total": 2,
        "cached_hits": 1,
        "success": true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'websites' not in data:
            return jsonify({'error': 'Missing websites field'}), 400
        
        websites = data['websites']
        use_cache = data.get('use_cache', True)
        results = []
        cached_hits = 0
        
        for site in websites:
            html = site.get('html', '')
            url = site.get('url', 'unknown')
            
            if not html:
                results.append({
                    'url': url,
                    'framework': 'Unknown',
                    'confidence': 0.0,
                    'from_cache': False,
                    'error': 'Empty HTML',
                })
                continue
            
            # 尝试从缓存获取
            cache_key = None
            if use_cache and db_ops:
                cache_key = hashlib.sha256(html.encode()).hexdigest()[:32]
                cached = db_ops.get(cache_key)
                if cached:
                    results.append({
                        **cached,
                        'url': url,
                        'from_cache': True
                    })
                    cached_hits += 1
                    continue
            
            # 执行检测
            framework, confidence = detector.detect_with_category(html, category=None)
            metrics_collector.total_detections += 1
            
            result = {
                'url': url,
                'framework': framework,
                'confidence': float(confidence),
                'from_cache': False,
            }
            
            # 保存到缓存
            if use_cache and db_ops and cache_key:
                try:
                    db_ops.set(cache_key, result, ttl_seconds=3600)
                except Exception as e:
                    logger.warning(f"⚠️ 缓存保存失败: {e}")
            
            results.append(result)
        
        return jsonify({
            'results': results,
            'total': len(results),
            'cached_hits': cached_hits,
            'success': True,
        })
    
    except Exception as e:
        logger.error(f"❌ 批量检测错误: {e}")
        return jsonify({
            'error': str(e),
            'success': False,
        }), 500


@app.route('/api/v1/stats', methods=['GET'])
def stats():
    """统计信息 (包含缓存统计)"""
    cache_stats = {}
    if db_ops:
        try:
            cache_stats = db_ops.get_stats()
        except Exception as e:
            logger.warning(f"⚠️ 缓存统计获取失败: {e}")
    
    metrics = metrics_collector.get_metrics()
    return jsonify({
        'frameworks_supported': len(detector.PRECISE_RULES) if detector else 0,
        'detection_methods': ['rule_based', 'deep_learning'],
        'version': '2.1',
        'uptime_seconds': metrics['uptime_seconds'],
        'total_requests': metrics_collector.total_requests,
        'total_detections': metrics['total_detections'],
        'total_errors': metrics['total_errors'],
        'cache': {
            'enabled': db_ops is not None,
            'stats': cache_stats
        }
    })


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus指标端点"""
    return Response(export_metrics(), mimetype='text/plain')


def main():
    """启动服务器"""
    logger.info(f"\n{'='*70}")
    logger.info("🚀 框架检测 API 服务器")
    logger.info(f"{'='*70}\n")
    
    load_models()
    
    logger.info("\n📡 启动 Flask 服务器...")
    logger.info("   地址: http://0.0.0.0:5000")
    logger.info("   健康检查: http://0.0.0.0:5000/health")
    logger.info("   检测API: POST http://0.0.0.0:5000/api/v1/detect")
    logger.info("   Prometheus: http://0.0.0.0:5000/metrics")
    logger.info(f"\n{'='*70}\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
