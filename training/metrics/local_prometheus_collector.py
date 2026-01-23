#!/usr/bin/env python3
"""
Prometheus本地监控系统模拟器
不需要安装Prometheus二进制，直接从API收集指标
"""

import json
import logging
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class LocalPrometheusCollector:
    """本地Prometheus指标收集器"""
    
    def __init__(self, api_url: str = "http://localhost:5000", collect_interval: int = 5):
        """
        初始化收集器
        
        Args:
            api_url: API服务器地址
            collect_interval: 收集间隔(秒)
        """
        self.api_url = api_url
        self.collect_interval = collect_interval
        self.metrics = []
        self.metrics_file = Path("./metrics_history.json")
        self.running = False
        self.collector_thread = None
        
        # 加载历史数据
        self._load_metrics()
    
    def _load_metrics(self):
        """加载历史指标"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                logger.info(f"✅ 加载{len(self.metrics)}条历史指标")
            except Exception as e:
                logger.warning(f"⚠️ 加载历史指标失败: {e}")
    
    def _save_metrics(self):
        """保存指标到文件"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics[-100:], f, indent=2)  # 只保存最近100条
        except Exception as e:
            logger.warning(f"⚠️ 保存指标失败: {e}")
    
    def collect_metrics(self):
        """从API收集当前指标"""
        try:
            response = requests.get(f"{self.api_url}/api/v1/stats", timeout=3)
            response.raise_for_status()
            
            data = response.json()
            
            metric = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            self.metrics.append(metric)
            
            # 打印当前指标
            logger.info(f"📊 指标收集:")
            logger.info(f"   总请求: {data.get('total_requests', 0)}")
            logger.info(f"   总检测: {data.get('total_detections', 0)}")
            logger.info(f"   总错误: {data.get('total_errors', 0)}")
            
            cache = data.get('cache', {})
            if cache.get('enabled'):
                stats = cache.get('stats', {})
                logger.info(f"   缓存条目: {stats.get('total_entries', 0)}")
                logger.info(f"   缓存命中: {stats.get('total_hits', 0)}")
                logger.info(f"   缓存大小: {stats.get('total_size_bytes', 0)} bytes")
            
            # 定期保存
            if len(self.metrics) % 10 == 0:
                self._save_metrics()
            
            return True
        
        except Exception as e:
            logger.error(f"❌ 指标收集失败: {e}")
            return False
    
    def start(self):
        """启动收集线程"""
        if self.running:
            logger.warning("⚠️ 收集器已运行")
            return
        
        self.running = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        logger.info(f"🚀 指标收集器已启动 (间隔: {self.collect_interval}s)")
    
    def _collection_loop(self):
        """收集循环"""
        while self.running:
            self.collect_metrics()
            time.sleep(self.collect_interval)
    
    def stop(self):
        """停止收集"""
        if not self.running:
            return
        
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)
        
        self._save_metrics()
        logger.info("✅ 指标收集器已停止")
    
    def get_metrics(self) -> List[Dict]:
        """获取所有收集的指标"""
        return self.metrics
    
    def get_summary(self) -> Dict:
        """获取指标摘要"""
        if not self.metrics:
            return {}
        
        latest = self.metrics[-1]['data']
        
        # 计算趋势
        if len(self.metrics) >= 2:
            prev = self.metrics[-2]['data']
            detections_delta = latest.get('total_detections', 0) - prev.get('total_detections', 0)
            errors_delta = latest.get('total_errors', 0) - prev.get('total_errors', 0)
        else:
            detections_delta = 0
            errors_delta = 0
        
        return {
            'latest_timestamp': self.metrics[-1]['timestamp'],
            'total_metrics_collected': len(self.metrics),
            'total_requests': latest.get('total_requests', 0),
            'total_detections': latest.get('total_detections', 0),
            'detections_since_last': detections_delta,
            'total_errors': latest.get('total_errors', 0),
            'errors_since_last': errors_delta,
            'cache_enabled': latest.get('cache', {}).get('enabled', False),
            'cache_entries': latest.get('cache', {}).get('stats', {}).get('total_entries', 0),
            'cache_hits': latest.get('cache', {}).get('stats', {}).get('total_hits', 0),
            'uptime_seconds': latest.get('uptime_seconds', 0),
        }
    
    def print_summary(self):
        """打印摘要"""
        summary = self.get_summary()
        
        if not summary:
            logger.info("⚠️ 暂无指标数据")
            return
        
        logger.info("\n" + "="*70)
        logger.info("📈 Prometheus本地收集器 - 摘要")
        logger.info("="*70)
        logger.info(f"最后更新: {summary['latest_timestamp']}")
        logger.info(f"收集总数: {summary['total_metrics_collected']}")
        logger.info(f"\n检测指标:")
        logger.info(f"  总请求: {summary['total_requests']}")
        logger.info(f"  总检测: {summary['total_detections']} (+{summary['detections_since_last']})")
        logger.info(f"  总错误: {summary['total_errors']} (+{summary['errors_since_last']})")
        
        if summary['cache_enabled']:
            logger.info(f"\n缓存指标:")
            logger.info(f"  已启用: ✅")
            logger.info(f"  缓存条目: {summary['cache_entries']}")
            logger.info(f"  总命中: {summary['cache_hits']}")
        else:
            logger.info(f"\n缓存: ❌ 已禁用")
        
        logger.info(f"\n系统:")
        logger.info(f"  运行时间: {summary['uptime_seconds']:.2f}秒")
        logger.info("="*70 + "\n")


def main():
    """主函数"""
    collector = LocalPrometheusCollector(
        api_url="http://localhost:5000",
        collect_interval=5
    )
    
    try:
        collector.start()
        
        logger.info("\n💡 本地Prometheus收集器运行中")
        logger.info("   按 Ctrl+C 停止")
        logger.info("   指标保存在: metrics_history.json\n")
        
        # 每30秒打印一次摘要
        while True:
            time.sleep(30)
            collector.print_summary()
    
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  停止信号")
    
    finally:
        collector.stop()
        collector.print_summary()


if __name__ == '__main__':
    main()
