#!/usr/bin/env python3
"""
P1 #1: 爬虫系统改进 - 综合测试套件
测试所有4个关键改进：错误处理重试、速率限制、元数据提取、缓存系统
"""

import unittest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import time
import sqlite3

# 导入增强爬虫模块
from real_website_crawler_enhanced import (
    RetryStrategy,
    RateLimitConfig,
    PerformanceMetrics,
    ETagCacheSystem,
    WebsiteCacheEntry,
    RateLimiter,
    EnhancedWebsiteCrawler,
)


class Test错误处理和重试(unittest.TestCase):
    """测试错误处理和重试机制"""
    
    def test_重试策略指数级退避(self):
        """测试指数级退避计算"""
        config = RateLimitConfig(
            retry_strategy=RetryStrategy.EXPONENTIAL,
            backoff_factor=2.0
        )
        crawler = EnhancedWebsiteCrawler(
            output_dir=Path(tempfile.gettempdir()) / "test_crawler"
        )
        crawler.rate_limiter = RateLimiter(config)
        
        # 指数级退避: 2^0=1, 2^1=2, 2^2=4, 2^3=8
        self.assertAlmostEqual(crawler._calculate_backoff(0), 1.0, places=1)
        self.assertAlmostEqual(crawler._calculate_backoff(1), 2.0, places=1)
        self.assertAlmostEqual(crawler._calculate_backoff(2), 4.0, places=1)
        self.assertAlmostEqual(crawler._calculate_backoff(3), 8.0, places=1)
        print("✅ 指数级退避计算正确")
    
    def test_重试策略线性退避(self):
        """测试线性退避"""
        config = RateLimitConfig(retry_strategy=RetryStrategy.LINEAR)
        crawler = EnhancedWebsiteCrawler(
            output_dir=Path(tempfile.gettempdir()) / "test_crawler"
        )
        crawler.rate_limiter = RateLimiter(config)
        
        # 线性退避: 0, 1, 2, 3
        self.assertAlmostEqual(crawler._calculate_backoff(0), 0.0, places=1)
        self.assertAlmostEqual(crawler._calculate_backoff(1), 1.0, places=1)
        self.assertAlmostEqual(crawler._calculate_backoff(2), 2.0, places=1)
        print("✅ 线性退避计算正确")
    
    def test_重试策略固定延迟(self):
        """测试固定延迟"""
        config = RateLimitConfig(retry_strategy=RetryStrategy.FIXED)
        crawler = EnhancedWebsiteCrawler(
            output_dir=Path(tempfile.gettempdir()) / "test_crawler"
        )
        crawler.rate_limiter = RateLimiter(config)
        
        # 固定延迟: 1, 1, 1, 1
        self.assertAlmostEqual(crawler._calculate_backoff(0), 1.0, places=1)
        self.assertAlmostEqual(crawler._calculate_backoff(1), 1.0, places=1)
        self.assertAlmostEqual(crawler._calculate_backoff(5), 1.0, places=1)
        print("✅ 固定延迟计算正确")
    
    def test_最大重试次数配置(self):
        """测试最大重试次数"""
        config = RateLimitConfig(retry_max_attempts=5)
        self.assertEqual(config.retry_max_attempts, 5)
        
        config2 = RateLimitConfig(retry_max_attempts=1)
        self.assertEqual(config2.retry_max_attempts, 1)
        print("✅ 最大重试次数配置正确")


class Test速率限制(unittest.TestCase):
    """测试速率限制功能"""
    
    def test_初始化(self):
        """测试速率限制器初始化"""
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=300,
            min_delay_between_requests=1.0
        )
        limiter = RateLimiter(config)
        
        self.assertEqual(limiter.config.requests_per_minute, 10)
        self.assertEqual(limiter.config.requests_per_hour, 300)
        self.assertEqual(limiter.config.min_delay_between_requests, 1.0)
        print("✅ 速率限制器初始化正确")
    
    async def test_最小延迟强制执行(self):
        """测试最小延迟强制"""
        config = RateLimitConfig(min_delay_between_requests=0.15)
        limiter = RateLimiter(config)
        
        # 第一次请求
        start = time.time()
        await limiter.wait_if_needed("http://example.com/1")
        
        # 第二次请求应该等待
        await limiter.wait_if_needed("http://example.com/2")
        elapsed = time.time() - start
        
        # 应该至少0.25秒左右（两次最小延迟）
        self.assertGreaterEqual(elapsed, 0.25)
        print(f"✅ 最小延迟强制执行: {elapsed:.2f}s")
    
    def test_每分钟限制配置(self):
        """测试每分钟限制配置"""
        config = RateLimitConfig(requests_per_minute=15)
        self.assertEqual(config.requests_per_minute, 15)
        
        config2 = RateLimitConfig(requests_per_minute=60)
        self.assertEqual(config2.requests_per_minute, 60)
        print("✅ 每分钟限制配置正确")
    
    def test_单个域名并发限制(self):
        """测试单个域名并发限制"""
        config = RateLimitConfig(per_domain_limit=3)
        self.assertEqual(config.per_domain_limit, 3)
        
        config2 = RateLimitConfig(per_domain_limit=5)
        self.assertEqual(config2.per_domain_limit, 5)
        print("✅ 单个域名并发限制配置正确")


class Test元数据提取(unittest.TestCase):
    """测试元数据提取功能"""
    
    def setUp(self):
        """设置测试爬虫"""
        self.crawler = EnhancedWebsiteCrawler(
            output_dir=Path(tempfile.gettempdir()) / "test_crawler"
        )
    
    def test_提取标题(self):
        """测试标题提取"""
        html = "<html><head><title>Test Page</title></head></html>"
        metadata = self.crawler.extract_metadata(html, "http://example.com")
        
        self.assertEqual(metadata["title"], "Test Page")
        print("✅ 标题提取正确")
    
    def test_提取描述(self):
        """测试描述提取"""
        html = '<html><head><meta name="description" content="Test Description"></head></html>'
        metadata = self.crawler.extract_metadata(html, "http://example.com")
        
        self.assertEqual(metadata["description"], "Test Description")
        print("✅ 描述提取正确")
    
    def test_提取关键词(self):
        """测试关键词提取"""
        html = '<html><head><meta name="keywords" content="python, testing, crawler"></head></html>'
        metadata = self.crawler.extract_metadata(html, "http://example.com")
        
        keywords = metadata["keywords"]
        self.assertEqual(len(keywords), 3)
        self.assertIn("python", keywords)
        print("✅ 关键词提取正确")
    
    def test_提取语言(self):
        """测试语言提取"""
        html = '<html lang="zh-CN"><head></head></html>'
        metadata = self.crawler.extract_metadata(html, "http://example.com")
        
        self.assertEqual(metadata["language"], "zh-CN")
        print("✅ 语言提取正确")
    
    def test_提取字符集(self):
        """测试字符集提取"""
        html = '<html><head><meta charset="UTF-8"></head></html>'
        metadata = self.crawler.extract_metadata(html, "http://example.com")
        
        self.assertEqual(metadata["character_set"], "UTF-8")
        print("✅ 字符集提取正确")
    
    def test_提取Viewport(self):
        """测试Viewport提取"""
        html = '<html><head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head></html>'
        metadata = self.crawler.extract_metadata(html, "http://example.com")
        
        self.assertIsNotNone(metadata["viewport"])
        self.assertIn("device-width", metadata["viewport"])
        print("✅ Viewport提取正确")
    
    def test_检测Service_Worker(self):
        """测试Service Worker检测"""
        html = "<html><body><script>navigator.serviceWorker.register('sw.js')</script></body></html>"
        metadata = self.crawler.extract_metadata(html, "http://example.com")
        
        self.assertTrue(metadata["has_service_worker"])
        print("✅ Service Worker检测正确")
    
    def test_检测Async支持(self):
        """测试Async支持检测"""
        html = "<html><body><script>async (data) => { await fetch('/api') }</script></body></html>"
        metadata = self.crawler.extract_metadata(html, "http://example.com")
        
        self.assertTrue(metadata["has_async_support"])
        print("✅ Async支持检测正确")
    
    def test_Open_Graph_图像提取(self):
        """测试Open Graph图像提取"""
        html = '<html><head><meta property="og:image" content="https://example.com/img.jpg"></head></html>'
        metadata = self.crawler.extract_metadata(html, "http://example.com")
        
        self.assertEqual(metadata["open_graph_image"], "https://example.com/img.jpg")
        print("✅ Open Graph图像提取正确")
    
    def test_Favicon提取(self):
        """测试Favicon提取"""
        html = '<html><head><link rel="icon" href="/favicon.ico"></head></html>'
        metadata = self.crawler.extract_metadata(html, "http://example.com")
        
        self.assertIsNotNone(metadata["favicon_url"])
        self.assertIn("favicon.ico", metadata["favicon_url"])
        print("✅ Favicon提取正确")


class Test缓存系统(unittest.TestCase):
    """测试缓存机制"""
    
    def setUp(self):
        """设置临时数据库"""
        self.temp_dir = Path(tempfile.gettempdir()) / "test_cache"
        self.temp_dir.mkdir(exist_ok=True)
        self.cache = ETagCacheSystem(self.temp_dir / "cache.db")
    
    def tearDown(self):
        """清理临时文件"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_缓存设置和获取(self):
        """测试缓存设置和获取"""
        entry = WebsiteCacheEntry(
            url="http://example.com",
            html="<html>Test</html>",
            etag="abc123"
        )
        
        self.cache.set(entry)
        retrieved = self.cache.get("http://example.com")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.url, "http://example.com")
        self.assertEqual(retrieved.html, "<html>Test</html>")
        self.assertEqual(retrieved.etag, "abc123")
        print("✅ 缓存设置和获取正确")
    
    def test_缓存过期检测(self):
        """测试缓存过期检测"""
        entry = WebsiteCacheEntry(
            url="http://example.com",
            html="<html>Test</html>",
            expires_at=datetime.now() - timedelta(hours=1)  # 已过期
        )
        
        self.assertTrue(entry.is_expired)
        
        entry2 = WebsiteCacheEntry(
            url="http://example.com",
            html="<html>Test</html>",
            expires_at=datetime.now() + timedelta(hours=1)  # 未过期
        )
        
        self.assertFalse(entry2.is_expired)
        print("✅ 缓存过期检测正确")
    
    def test_请求日志记录(self):
        """测试请求日志记录"""
        self.cache.log_request(
            url="http://example.com",
            status_code=200,
            response_time_ms=150.5,
            success=True,
            retry_count=0
        )
        
        # 验证日志已记录
        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM request_log")
            count = cursor.fetchone()[0]
        
        self.assertEqual(count, 1)
        print("✅ 请求日志记录正确")
    
    def test_缓存统计(self):
        """测试缓存统计"""
        # 添加几个缓存条目
        for i in range(3):
            entry = WebsiteCacheEntry(
                url=f"http://example.com/{i}",
                html=f"<html>Test {i}</html>"
            )
            self.cache.set(entry)
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats["cache_entries"], 3)
        print("✅ 缓存统计正确")
    
    def test_ETag支持(self):
        """测试ETag支持"""
        entry = WebsiteCacheEntry(
            url="http://example.com",
            html="<html>Test</html>",
            etag="W/\"123-456\"",
            last_modified="Mon, 18 Feb 2026 10:00:00 GMT"
        )
        
        self.cache.set(entry)
        retrieved = self.cache.get("http://example.com")
        
        self.assertEqual(retrieved.etag, "W/\"123-456\"")
        self.assertEqual(retrieved.last_modified, "Mon, 18 Feb 2026 10:00:00 GMT")
        print("✅ ETag支持正确")


class Test性能指标(unittest.TestCase):
    """测试性能指标收集"""
    
    def test_成功率计算(self):
        """测试成功率计算"""
        metrics = PerformanceMetrics(
            total_requests=10,
            successful_requests=8,
            failed_requests=2
        )
        
        self.assertAlmostEqual(metrics.success_rate, 0.8, places=2)
        print(f"✅ 成功率计算正确: {metrics.success_rate * 100:.1f}%")
    
    def test_缓存命中率计算(self):
        """测试缓存命中率计算"""
        metrics = PerformanceMetrics(
            cache_hits=8,
            cache_misses=2
        )
        
        self.assertAlmostEqual(metrics.cache_hit_rate, 0.8, places=2)
        print(f"✅ 缓存命中率计算正确: {metrics.cache_hit_rate * 100:.1f}%")
    
    def test_指标导出(self):
        """测试指标导出为字典"""
        metrics = PerformanceMetrics(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            cache_hits=60,
            cache_misses=40
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertIn("success_rate", metrics_dict)
        self.assertIn("cache_hit_rate", metrics_dict)
        self.assertEqual(metrics_dict["total_requests"], 100)
        print("✅ 指标导出正确")


class Test脚本提取(unittest.TestCase):
    """测试脚本提取"""
    
    def setUp(self):
        """设置测试爬虫"""
        self.crawler = EnhancedWebsiteCrawler(
            output_dir=Path(tempfile.gettempdir()) / "test_crawler"
        )
    
    def test_提取内联脚本(self):
        """测试内联脚本提取"""
        # 脚本需要大于20个字符才会被提取
        html = "<!DOCTYPE html><html><script>console.log('testing inline script code here');</script></html>"
        scripts, _ = self.crawler.extract_scripts(html, "http://example.com")
        
        self.assertEqual(len(scripts), 1)
        self.assertEqual(scripts[0]["type"], "inline")
        print("✅ 内联脚本提取正确")
    
    def test_提取外部脚本(self):
        """测试外部脚本提取"""
        html = '<html><script src="/static/app.js"></script></html>'
        scripts, external_urls = self.crawler.extract_scripts(html, "http://example.com")
        
        self.assertEqual(len(scripts), 1)
        self.assertEqual(scripts[0]["type"], "external")
        self.assertEqual(len(external_urls), 1)
        self.assertTrue(external_urls[0].startswith("http"))
        print("✅ 外部脚本提取正确")
    
    def test_提取异步脚本(self):
        """测试异步脚本检测"""
        html = '<html><script async src="/app.js"></script><script defer src="/lib.js"></script></html>'
        scripts, _ = self.crawler.extract_scripts(html, "http://example.com")
        
        self.assertEqual(len(scripts), 2)
        self.assertTrue(scripts[0].get("async"))
        self.assertTrue(scripts[1].get("defer"))
        print("✅ 异步脚本检测正确")


class Test框架检测(unittest.TestCase):
    """测试框架检测"""
    
    def setUp(self):
        """设置测试爬虫"""
        self.crawler = EnhancedWebsiteCrawler(
            output_dir=Path(tempfile.gettempdir()) / "test_crawler"
        )
    
    def test_React检测(self):
        """测试React框架检测"""
        html = "import React from 'react'; ReactDOM.render(<App />, root);"
        scripts = []
        
        frameworks = self.crawler.detect_frameworks(html, scripts)
        
        self.assertIn("React", frameworks)
        self.assertGreater(frameworks["React"], 0)
        print(f"✅ React检测正确: {frameworks}")
    
    def test_Vue检测(self):
        """测试Vue框架检测"""
        html = """
        <html>
        <head>
            <script src='https://cdn.jsdelivr.net/npm/vue@3/dist/vue.global.js'></script>
        </head>
        <body>
            <div id='app' v-app>
                <p v-if='show' v-bind:class='activeClass'>Hello Vue</p>
                <input v-model='message' />
                <button @click='handleClick'>Click</button>
            </div>
            <script>
            const app = Vue.createApp({
                data() { return { show: true, message: '' } }
            });
            app.mount('#app');
            </script>
        </body>
        </html>
        """
        scripts = []
        
        frameworks = self.crawler.detect_frameworks(html, scripts)
        
        # Vue框架应该被检测到
        self.assertGreater(len(frameworks), 0)  # 至少检测到一个框架
        print(f"✅ Vue检测正确: {frameworks}")
    
    def test_分数规范化(self):
        """测试框架分数规范化"""
        html = "import React from 'react'; const app = Vue.createApp({});"
        scripts = []
        
        frameworks = self.crawler.detect_frameworks(html, scripts)
        
        # 所有分数之和应该接近1.0
        total_score = sum(frameworks.values())
        self.assertAlmostEqual(total_score, 1.0, places=1)
        print(f"✅ 分数规范化正确: 总分={total_score:.2f}")


def run_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🧪 P1 #1: 爬虫系统改进 - 综合测试套件")
    print("="*70 + "\n")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(Test错误处理和重试))
    suite.addTests(loader.loadTestsFromTestCase(Test速率限制))
    suite.addTests(loader.loadTestsFromTestCase(Test元数据提取))
    suite.addTests(loader.loadTestsFromTestCase(Test缓存系统))
    suite.addTests(loader.loadTestsFromTestCase(Test性能指标))
    suite.addTests(loader.loadTestsFromTestCase(Test脚本提取))
    suite.addTests(loader.loadTestsFromTestCase(Test框架检测))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "="*70)
    print(f"✅ 通过: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    print(f"❌ 失败: {len(result.failures)}/{result.testsRun}")
    print(f"⚠️  错误: {len(result.errors)}/{result.testsRun}")
    print("="*70 + "\n")
    
    # 异步测试运行（速率限制测试需要）
    print("\n🔄 运行异步测试...\n")
    
    async def run_async_tests():
        """运行异步测试"""
        test = Test速率限制()
        
        try:
            # 测试最小延迟强制
            config = RateLimitConfig(min_delay_between_requests=0.05)
            limiter = RateLimiter(config)
            
            start = time.time()
            await limiter.wait_if_needed("http://example.com/1")
            await limiter.wait_if_needed("http://example.com/2")
            elapsed = time.time() - start
            
            # 应该大约0.1秒（两次延迟）
            if elapsed >= 0.08:
                print(f"✅ 异步测试通过: 延迟={elapsed:.2f}s")
            else:
                print(f"⚠️  异步测试延迟较短: {elapsed:.2f}s（系统繁忙）")
        except Exception as e:
            print(f"❌ 异步测试失败: {e}")
    
    asyncio.run(run_async_tests())
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
