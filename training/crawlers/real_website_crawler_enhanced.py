#!/usr/bin/env python3
"""
增强级别的真实网站爬虫 - P1改进
包含：错误处理重试、速率限制、元数据提取、缓存系统
"""

import asyncio
import aiohttp
import logging
import sqlite3
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import re
import json
from collections import defaultdict, deque
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """重试策略"""
    EXPONENTIAL = "exponential"    # 指数退避: 1s -> 2s -> 4s -> 8s
    LINEAR = "linear"              # 线性退避: 1s -> 2s -> 3s -> 4s
    FIXED = "fixed"                # 固定延迟: 2s -> 2s -> 2s


class CrawlPriority(Enum):
    """爬虫优先级"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    requests_per_minute: int = 10       # 每分钟最多10个请求
    requests_per_hour: int = 300        # 每小时最多300个请求
    min_delay_between_requests: float = 1.0  # 相邻请求最小延迟(秒)
    per_domain_limit: int = 5           # 单个域名最多5个并发连接
    retry_max_attempts: int = 3         # 最多重试次数
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    timeout_seconds: int = 30           # 请求超时(秒)
    backoff_factor: float = 2.0         # 退避因子


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_time_seconds: float = 0.0
    average_response_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total
    
    def to_dict(self) -> Dict:
        """转为字典"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "retried_requests": self.retried_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "average_response_time_ms": round(self.average_response_time * 1000, 2),
            "success_rate": round(self.success_rate * 100, 2),
            "cache_hit_rate": round(self.cache_hit_rate * 100, 2),
        }


@dataclass
class WebsiteCacheEntry:
    """网站缓存条目"""
    url: str
    html: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    content_hash: str = ""
    cached_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        return datetime.now() > self.expires_at


@dataclass
class EnhancedWebsiteData:
    """增强的网站数据"""
    url: str
    domain: str
    html: str
    scripts: List[Dict] = field(default_factory=list)
    css: List[Dict] = field(default_factory=list)
    detected_frameworks: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    timestamp: str = ""
    success: bool = True
    error: Optional[str] = None
    retry_count: int = 0
    response_time_ms: float = 0.0
    from_cache: bool = False
    
    # 增强的元数据
    title: Optional[str] = None
    description: Optional[str] = None
    favicon_url: Optional[str] = None
    open_graph_image: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    language: Optional[str] = None
    character_set: Optional[str] = None
    viewport: Optional[str] = None
    async_support: bool = False
    service_worker: bool = False


class ETagCacheSystem:
    """基于SQLite的缓存系统，支持ETag和Last-Modified"""
    
    def __init__(self, db_path: Path = Path("real_data/cache.db")):
        """初始化缓存"""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webpage_cache (
                    url TEXT PRIMARY KEY,
                    html TEXT,
                    etag TEXT,
                    last_modified TEXT,
                    content_hash TEXT,
                    cached_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT,
                    status_code INTEGER,
                    response_time_ms REAL,
                    success BOOLEAN,
                    timestamp TIMESTAMP,
                    retry_count INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires ON webpage_cache(expires_at)
            """)
            
            conn.commit()
    
    def get(self, url: str) -> Optional[WebsiteCacheEntry]:
        """获取缓存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT html, etag, last_modified, content_hash, metadata FROM webpage_cache WHERE url = ?",
                (url,)
            )
            row = cursor.fetchone()
            
            if row:
                html, etag, last_modified, content_hash, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                return WebsiteCacheEntry(
                    url=url,
                    html=html,
                    etag=etag,
                    last_modified=last_modified,
                    content_hash=content_hash,
                    metadata=metadata
                )
        
        return None
    
    def set(self, entry: WebsiteCacheEntry, ttl_days: int = 7):
        """设置缓存"""
        expires_at = datetime.now() + timedelta(days=ttl_days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO webpage_cache 
                (url, html, etag, last_modified, content_hash, cached_at, expires_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.url,
                entry.html,
                entry.etag,
                entry.last_modified,
                entry.content_hash,
                datetime.now().isoformat(),
                expires_at.isoformat(),
                json.dumps(entry.metadata)
            ))
            conn.commit()
    
    def log_request(self, url: str, status_code: int, response_time_ms: float, 
                   success: bool, retry_count: int = 0):
        """记录请求"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO request_log (url, status_code, response_time_ms, success, timestamp, retry_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (url, status_code, response_time_ms, success, datetime.now().isoformat(), retry_count))
            conn.commit()
    
    def cleanup_expired(self):
        """清理过期缓存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM webpage_cache WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            )
            deleted = cursor.rowcount
            conn.commit()
        
        logger.info(f"🧹 清理了 {deleted} 个过期缓存条目")
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        with sqlite3.connect(self.db_path) as conn:
            # 缓存大小
            cursor = conn.execute("SELECT COUNT(*) FROM webpage_cache")
            cache_entries = cursor.fetchone()[0]
            
            # 今日请求
            cursor = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE DATE(timestamp) = DATE('now')"
            )
            today_requests = cursor.fetchone()[0]
            
            # 成功率
            cursor = conn.execute(
                "SELECT COUNT(CASE WHEN success=1 THEN 1 END), COUNT(*) FROM request_log"
            )
            success_count, total_count = cursor.fetchone()
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        return {
            "cache_entries": cache_entries,
            "today_requests": today_requests,
            "success_rate": round(success_rate, 2),
        }


class RateLimiter:
    """基于时间窗口的速率限制器"""
    
    def __init__(self, config: RateLimitConfig):
        """初始化速率限制"""
        self.config = config
        self.per_minute_requests = deque(maxlen=60)  # 最近60秒的请求时间
        self.per_hour_requests = deque(maxlen=3600)   # 最近3600秒的请求时间
        self.per_domain_requests = defaultdict(deque)  # 按域名统计
        self.last_request_time = 0
    
    async def wait_if_needed(self, url: str):
        """如果需要则等待"""
        domain = urlparse(url).netloc
        now = time.time()
        
        # 清理过期的请求记录
        while self.per_minute_requests and self.per_minute_requests[0] < now - 60:
            self.per_minute_requests.popleft()
        
        while self.per_hour_requests and self.per_hour_requests[0] < now - 3600:
            self.per_hour_requests.popleft()
        
        # 检查每分钟限制
        if len(self.per_minute_requests) >= self.config.requests_per_minute:
            wait_time = 60 - (now - self.per_minute_requests[0])
            if wait_time > 0:
                logger.warning(f"⏸️  达到每分钟限制, 等待 {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # 检查每小时限制
        if len(self.per_hour_requests) >= self.config.requests_per_hour:
            wait_time = 3600 - (now - self.per_hour_requests[0])
            if wait_time > 0:
                logger.warning(f"⏸️  达到每小时限制, 等待 {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # 检查最小延迟
        elapsed = now - self.last_request_time
        if elapsed < self.config.min_delay_between_requests:
            wait_time = self.config.min_delay_between_requests - elapsed
            await asyncio.sleep(wait_time)
        
        # 记录这次请求
        self.per_minute_requests.append(time.time())
        self.per_hour_requests.append(time.time())
        self.last_request_time = time.time()


class EnhancedWebsiteCrawler:
    """增强级别的网站爬虫 - 包含所有P1改进"""
    
    # 网站列表（与原版本相同）
    WEBSITES = {
        "React框架": [
            "https://www.airbnb.com",
            "https://www.netflix.com",
            "https://www.facebook.com",
            "https://www.uber.com",
            "https://www.spotify.com",
        ],
        "Vue框架": [
            "https://www.alibaba.com",
            "https://www.xiaomi.com",
            "https://laravel.com",
            "https://www.booking.com",
            "https://www.figma.com",
        ],
        "Angular框架": [
            "https://mail.google.com",
            "https://drive.google.com",
            "https://analytics.google.com",
            "https://www.forbes.com",
            "https://www.bankofamerica.com",
        ],
        "Next.js/Nuxt": [
            "https://www.vercel.com",
            "https://www.nextjs.org",
            "https://www.nuxtjs.org",
            "https://www.twitch.tv",
            "https://www.docker.com",
        ],
        "Express/Node": [
            "https://www.nodejs.org",
            "https://www.npmjs.com",
            "https://www.github.com",
            "https://www.stackoverflow.com",
        ],
    }
    
    def __init__(self, 
                 output_dir: Path = Path("real_data/websites_enhanced"),
                 cache_dir: Path = Path("real_data/cache"),
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """初始化增强爬虫"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_file = self.output_dir / "websites_data.jsonl"
        self.stats_file = self.output_dir / "crawl_stats.json"
        
        # 缓存系统
        self.cache = ETagCacheSystem(cache_dir / "cache.db")
        
        # 速率限制
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        
        # 性能指标
        self.metrics = PerformanceMetrics()
        
        # 统计信息
        self.stats = {
            "total_attempted": 0,
            "total_success": 0,
            "total_failed": 0,
            "cache_hits": 0,
            "frameworks_found": defaultdict(int),
            "start_time": datetime.now().isoformat(),
            "retry_distribution": defaultdict(int),
            "urls_by_framework": defaultdict(list),
        }
    
    def _calculate_backoff(self, retry_count: int) -> float:
        """计算退避时间"""
        config = self.rate_limiter.config
        
        if config.retry_strategy == RetryStrategy.EXPONENTIAL:
            return (config.backoff_factor ** retry_count)
        elif config.retry_strategy == RetryStrategy.LINEAR:
            return float(retry_count)
        else:  # FIXED
            return 1.0
    
    async def fetch_website(self, 
                          session: aiohttp.ClientSession, 
                          url: str,
                          retry_count: int = 0) -> Tuple[Optional[str], Dict]:
        """爬取网站（带重试、缓存、ETag支持）"""
        
        # 速率限制
        await self.rate_limiter.wait_if_needed(url)
        
        # 检查缓存
        cached = self.cache.get(url)
        if cached and not cached.is_expired:
            logger.info(f"💾 缓存命中: {url}")
            self.metrics.cache_hits += 1
            self.stats["cache_hits"] += 1
            return cached.html, {"from_cache": True, "etag": cached.etag}
        
        self.metrics.cache_misses += 1
        
        try:
            headers = {
                'User-Agent': 'BrowserAI-Enhanced-Crawler/1.0 (+https://github.com/vistone/BrowerAI)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
            }
            
            # 添加ETag/Last-Modified（如果有）
            if cached:
                if cached.etag:
                    headers['If-None-Match'] = cached.etag
                if cached.last_modified:
                    headers['If-Modified-Since'] = cached.last_modified
            
            start_time = time.time()
            
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.rate_limiter.config.timeout_seconds),
                ssl=False,
                allow_redirects=True
            ) as response:
                response_time = (time.time() - start_time) * 1000  # 毫秒
                
                # 304 Not Modified - 使用缓存
                if response.status == 304 and cached:
                    logger.info(f"✅ 304 Not Modified: {url}")
                    self.metrics.cache_hits += 1
                    return cached.html, {"from_cache": True, "status": 304}
                
                # 成功获取
                if response.status == 200:
                    html = await response.text()
                    
                    # 保存到缓存
                    entry = WebsiteCacheEntry(
                        url=url,
                        html=html,
                        etag=response.headers.get('ETag'),
                        last_modified=response.headers.get('Last-Modified'),
                        content_hash=hashlib.md5(html.encode()).hexdigest(),
                        metadata={
                            "content_type": response.headers.get('Content-Type'),
                            "server": response.headers.get('Server'),
                            "response_time_ms": response_time,
                        }
                    )
                    self.cache.set(entry)
                    
                    # 记录请求
                    self.cache.log_request(url, response.status, response_time, True, retry_count)
                    self.metrics.successful_requests += 1
                    self.metrics.average_response_time = (
                        (self.metrics.average_response_time * (self.metrics.successful_requests - 1) + response_time) /
                        self.metrics.successful_requests
                    )
                    
                    logger.info(f"✅ {url} ({response_time:.0f}ms)")
                    return html, {"status": 200, "response_time_ms": response_time}
                
                # 其他状态码
                logger.warning(f"⚠️  {url}: 状态 {response.status}")
                self.cache.log_request(url, response.status, response_time, False, retry_count)
                
        except asyncio.TimeoutError:
            logger.warning(f"⏱️  超时: {url}")
            self.metrics.failed_requests += 1
            self.cache.log_request(url, 0, 0, False, retry_count)
            
            # 重试逻辑
            if retry_count < self.rate_limiter.config.retry_max_attempts:
                backoff = self._calculate_backoff(retry_count)
                logger.info(f"🔄 重试 {url} (尝试 {retry_count + 1}/{self.rate_limiter.config.retry_max_attempts})，等待 {backoff:.1f}s")
                await asyncio.sleep(backoff)
                self.metrics.retried_requests += 1
                self.stats["retry_distribution"]["timeout"] += 1
                return await self.fetch_website(session, url, retry_count + 1)
        
        except Exception as e:
            logger.error(f"❌ 错误: {url}: {e}")
            self.metrics.failed_requests += 1
            self.cache.log_request(url, 0, 0, False, retry_count)
            
            # 重试逻辑
            if retry_count < self.rate_limiter.config.retry_max_attempts:
                backoff = self._calculate_backoff(retry_count)
                logger.info(f"🔄 重试 {url} (尝试 {retry_count + 1}/{self.rate_limiter.config.retry_max_attempts})，等待 {backoff:.1f}s")
                await asyncio.sleep(backoff)
                self.metrics.retried_requests += 1
                self.stats["retry_distribution"]["error"] += 1
                return await self.fetch_website(session, url, retry_count + 1)
        
        return None, {}
    
    def extract_metadata(self, html: str, url: str) -> Dict:
        """提取丰富的元数据"""
        metadata = {
            "title": None,
            "description": None,
            "keywords": [],
            "favicon_url": None,
            "open_graph_image": None,
            "language": None,
            "character_set": None,
            "viewport": None,
            "has_service_worker": False,
            "has_async_support": False,
        }
        
        # 标题
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # 描述
        description_match = re.search(
            r'<meta\s+name="description"\s+content="([^"]+)"|<meta\s+content="([^"]+)"\s+name="description"',
            html, re.IGNORECASE
        )
        if description_match:
            metadata["description"] = description_match.group(1) or description_match.group(2)
        
        # 关键词
        keywords_match = re.search(
            r'<meta\s+name="keywords"\s+content="([^"]+)|<meta\s+content="([^"]+)"\s+name="keywords"',
            html, re.IGNORECASE
        )
        if keywords_match:
            keywords_str = keywords_match.group(1) or keywords_match.group(2)
            metadata["keywords"] = [k.strip() for k in keywords_str.split(',')]
        
        # Favicon
        favicon_match = re.search(
            r'<link[^>]*rel="icon"[^>]*href="([^"]+)"|<link[^>]*href="([^"]+)"[^>]*rel="icon"',
            html, re.IGNORECASE
        )
        if favicon_match:
            favicon_url = favicon_match.group(1) or favicon_match.group(2)
            metadata["favicon_url"] = urljoin(url, favicon_url)
        
        # Open Graph 图像
        og_image_match = re.search(
            r'<meta\s+property="og:image"\s+content="([^"]+)"|<meta\s+content="([^"]+)"\s+property="og:image"',
            html, re.IGNORECASE
        )
        if og_image_match:
            metadata["open_graph_image"] = og_image_match.group(1) or og_image_match.group(2)
        
        # 语言
        lang_match = re.search(r'<html[^>]*lang="([^"]+)"', html, re.IGNORECASE)
        if lang_match:
            metadata["language"] = lang_match.group(1)
        
        # 字符集
        charset_match = re.search(r'<meta[^>]*charset="([^"]+)"|<meta[^>]*charset=([^\s>]+)', html, re.IGNORECASE)
        if charset_match:
            metadata["character_set"] = charset_match.group(1) or charset_match.group(2)
        
        # Viewport
        viewport_match = re.search(
            r'<meta\s+name="viewport"\s+content="([^"]+)"|<meta\s+content="([^"]+)"\s+name="viewport"',
            html, re.IGNORECASE
        )
        if viewport_match:
            metadata["viewport"] = viewport_match.group(1) or viewport_match.group(2)
        
        # Service Worker
        if re.search(r'serviceWorker|SW\(|\.register\(', html, re.IGNORECASE):
            metadata["has_service_worker"] = True
        
        # Async/Await 支持检查
        if re.search(r'async\s+\(.*?\)\s*=>', html):
            metadata["has_async_support"] = True
        
        return metadata
    
    def extract_scripts(self, html: str, url: str) -> Tuple[List[Dict], List[str]]:
        """提取脚本（与原版本相同）"""
        scripts = []
        external_urls = []
        
        # 内联脚本
        inline_pattern = r'<script[^>]*>([^<]*)<\/script>'
        for match in re.finditer(inline_pattern, html, re.IGNORECASE | re.DOTALL):
            content = match.group(1).strip()
            if content and len(content) > 20:
                scripts.append({
                    "type": "inline",
                    "content": content[:10000],
                    "size": len(content),
                    "async": "async" in match.group().lower(),
                    "defer": "defer" in match.group().lower(),
                })
        
        # 外部脚本
        src_pattern = r'<script[^>]*src=["\']([^"\']+)["\']'
        for match in re.finditer(src_pattern, html, re.IGNORECASE):
            src = match.group(1)
            absolute_url = urljoin(url, src)
            external_urls.append(absolute_url)
            scripts.append({
                "type": "external",
                "src": src,
                "absolute_url": absolute_url,
                "size": 0,
                "async": "async" in match.group().lower(),
                "defer": "defer" in match.group().lower(),
            })
        
        return scripts, external_urls
    
    def extract_css(self, html: str, url: str) -> List[Dict]:
        """提取CSS（与原版本相同）"""
        css_list = []
        
        # 内联样式
        style_pattern = r'<style[^>]*>([^<]*)<\/style>'
        for match in re.finditer(style_pattern, html, re.IGNORECASE | re.DOTALL):
            content = match.group(1).strip()
            if content:
                css_list.append({
                    "type": "inline",
                    "size": len(content),
                    "media": match.group().lower().get('media', 'all'),
                })
        
        # 外部CSS
        link_pattern = r'<link[^>]*href=["\']([^"\']+)["\'][^>]*rel=["\']stylesheet["\']'
        for match in re.finditer(link_pattern, html, re.IGNORECASE):
            href = match.group(1)
            css_list.append({
                "type": "external",
                "href": href,
                "absolute_url": urljoin(url, href),
            })
        
        return css_list
    
    def detect_frameworks(self, html: str, scripts: List[Dict]) -> Dict[str, float]:
        """框架检测（与原版本相同）"""
        detected = {}
        
        full_content = html + " " + " ".join([s.get("content", "") for s in scripts if "content" in s])
        full_content_lower = full_content.lower()
        
        # React
        react_patterns = [
            r'react(?:dom)?\.render',
            r'ReactDOM\.render',
            r'ReactDOMClient\.createRoot',
            r'from\s+["\']react["\']',
            r'import\s+React\s+from',
        ]
        react_score = sum(1 for p in react_patterns if re.search(p, full_content_lower)) / len(react_patterns)
        if react_score > 0:
            detected["React"] = react_score
        
        # Vue
        vue_patterns = [
            r'Vue\.createApp',
            r'new Vue\(',
            r'from\s+["\']vue["\']',
            r'v-bind|v-model|@click',
        ]
        vue_score = sum(1 for p in vue_patterns if re.search(p, full_content_lower)) / len(vue_patterns)
        if vue_score > 0:
            detected["Vue"] = vue_score
        
        # Angular
        angular_patterns = [
            r'angular\.module',
            r'@angular',
            r'ng-app|ng-model|ng-repeat',
        ]
        angular_score = sum(1 for p in angular_patterns if re.search(p, full_content_lower)) / len(angular_patterns)
        if angular_score > 0:
            detected["Angular"] = angular_score
        
        # 规范化分数
        if detected:
            total = sum(detected.values())
            detected = {k: v / total for k, v in detected.items()}
        
        return detected
    
    async def crawl_website(self, session: aiohttp.ClientSession, url: str, category: str) -> Optional[EnhancedWebsiteData]:
        """爬取单个网站的完整增强数据"""
        self.stats["total_attempted"] += 1
        self.metrics.total_requests += 1
        
        logger.info(f"🌐 爬取 {url}...")
        
        start_time = time.time()
        html, fetch_info = await self.fetch_website(session, url)
        response_time = (time.time() - start_time) * 1000
        
        if not html:
            self.stats["total_failed"] += 1
            return EnhancedWebsiteData(
                url=url,
                domain=urlparse(url).netloc,
                html="",
                detected_frameworks={},
                metadata={"category": category},
                timestamp=datetime.now().isoformat(),
                success=False,
                error="Failed to fetch HTML",
                response_time_ms=response_time,
            )
        
        # 提取数据
        scripts, external_urls = self.extract_scripts(html, url)
        css = self.extract_css(html, url)
        frameworks = self.detect_frameworks(html, scripts)
        
        # 提取增强的元数据
        metadata = self.extract_metadata(html, url)
        
        self.stats["total_success"] += 1
        
        # 更新框架统计
        for framework in frameworks:
            self.stats["frameworks_found"][framework] += 1
            self.stats["urls_by_framework"][framework].append(url)
        
        logger.info(f"✅ {url} -> {frameworks} ({response_time:.0f}ms)")
        
        return EnhancedWebsiteData(
            url=url,
            domain=urlparse(url).netloc,
            html=html[:50000],
            scripts=scripts,
            css=css,
            detected_frameworks=frameworks,
            metadata={
                "category": category,
                "script_count": len(scripts),
                "css_count": len(css),
                "html_size": len(html),
                **fetch_info,
            },
            timestamp=datetime.now().isoformat(),
            success=True,
            response_time_ms=response_time,
            from_cache=fetch_info.get('from_cache', False),
            title=metadata.get('title'),
            description=metadata.get('description'),
            favicon_url=metadata.get('favicon_url'),
            open_graph_image=metadata.get('open_graph_image'),
            keywords=metadata.get('keywords', []),
            language=metadata.get('language'),
            character_set=metadata.get('character_set'),
            viewport=metadata.get('viewport'),
            service_worker=metadata.get('has_service_worker', False),
            async_support=metadata.get('has_async_support', False),
        )
    
    async def crawl_all_websites(self, max_workers: int = 5):
        """批量爬取所有网站"""
        logger.info(f"🚀 启动爬虫，最多 {max_workers} 个并发连接")
        
        connector = aiohttp.TCPConnector(
            limit=max_workers,
            limit_per_host=self.rate_limiter.config.per_domain_limit,
            ssl=False
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            
            for category, urls in self.WEBSITES.items():
                for url in urls:
                    tasks.append(self.crawl_website(session, url, category))
            
            # 并发执行
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 清理过期缓存
        self.cache.cleanup_expired()
        
        # 保存结果
        self.save_results(results)
        self.save_stats()
        
        return results
    
    def save_results(self, results: List):
        """保存爬取结果"""
        logger.info(f"💾 保存数据到 {self.data_file}...")
        
        with open(self.data_file, 'w', encoding='utf-8') as f:
            for result in results:
                if isinstance(result, EnhancedWebsiteData):
                    f.write(json.dumps(asdict(result), ensure_ascii=False, default=str) + '\n')
        
        logger.info(f"✅ 保存 {len(results)} 个网站数据")
    
    def save_stats(self):
        """保存统计信息"""
        self.stats["end_time"] = datetime.now().isoformat()
        self.stats["frameworks_found"] = dict(self.stats["frameworks_found"])
        self.stats["urls_by_framework"] = dict(self.stats["urls_by_framework"])
        self.stats["retry_distribution"] = dict(self.stats["retry_distribution"])
        self.stats["performance_metrics"] = self.metrics.to_dict()
        self.stats["cache_stats"] = self.cache.get_stats()
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        self._print_final_stats()
    
    def _print_final_stats(self):
        """打印最终统计"""
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 爬虫运行统计:")
        logger.info(f"  总尝试: {self.stats['total_attempted']}")
        logger.info(f"  成功: {self.stats['total_success']}")
        logger.info(f"  失败: {self.stats['total_failed']}")
        logger.info(f"  缓存命中: {self.stats['cache_hits']}")
        logger.info(f"  检测到的框架: {dict(self.stats['frameworks_found'])}")
        
        metrics = self.stats.get('performance_metrics', {})
        logger.info(f"\n⚡ 性能指标:")
        logger.info(f"  成功率: {metrics.get('success_rate', 0):.1f}%")
        logger.info(f"  平均响应时间: {metrics.get('average_response_time_ms', 0):.0f}ms")
        logger.info(f"  缓存命中率: {metrics.get('cache_hit_rate', 0):.1f}%")
        logger.info(f"  重试请求: {metrics.get('retried_requests', 0)}")
        
        cache_stats = self.stats.get('cache_stats', {})
        logger.info(f"\n💾 缓存统计:")
        logger.info(f"  缓存条目: {cache_stats.get('cache_entries', 0)}")
        logger.info(f"  今日请求: {cache_stats.get('today_requests', 0)}")
        logger.info(f"  缓存成功率: {cache_stats.get('success_rate', 0):.1f}%")
        
        logger.info(f"{'='*70}\n")


async def main():
    """主函数"""
    config = RateLimitConfig(
        requests_per_minute=10,
        min_delay_between_requests=1.0,
        retry_max_attempts=3,
    )
    
    crawler = EnhancedWebsiteCrawler(rate_limit_config=config)
    await crawler.crawl_all_websites(max_workers=5)


if __name__ == "__main__":
    asyncio.run(main())
