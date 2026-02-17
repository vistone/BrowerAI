#!/usr/bin/env python3
"""
真实网站爬取引擎 - 生产级系统
批量爬取真实网站，收集框架检测训练数据
"""

import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import hashlib
from urllib.parse import urljoin, urlparse
import re
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WebsiteData:
    """网站采样数据"""
    url: str
    domain: str
    html: str
    scripts: List[Dict]  # [{"src": "...", "content": "...", "type": "inline|external"}]
    css: List[Dict]
    detected_frameworks: Dict[str, float]  # {"react": 0.95, "vue": 0.05}
    metadata: Dict
    timestamp: str
    success: bool = True
    error: Optional[str] = None


class RealWebsiteCrawler:
    """真实网站爬取器 - 生产系统"""
    
    # 流行网站列表 - 按行业分类
    WEBSITES = {
        "React框架": [
            "https://www.airbnb.com",
            "https://www.netflix.com",
            "https://www.facebook.com",
            "https://www.instagram.com",
            "https://www.uber.com",
            "https://www.spotify.com",
            "https://www.slack.com",
            "https://www.stripe.com",
            "https://www.dropbox.com",
            "https://www.asana.com",
        ],
        "Vue框架": [
            "https://www.alibaba.com",
            "https://www.xiaomi.com",
            "https://www.bilibili.com",
            "https://laravel.com",
            "https://www.weibo.com",
            "https://www.booking.com",
            "https://www.douyin.com",
            "https://www.bytedance.com",
            "https://grammarly.com",
            "https://www.figma.com",
        ],
        "Angular框架": [
            "https://mail.google.com",
            "https://drive.google.com",
            "https://analytics.google.com",
            "https://www.forbes.com",
            "https://weather.com",
            "https://www.bankofamerica.com",
            "https://www.xbox.com",
            "https://www.ibm.com",
            "https://www.vmware.com",
            "https://www.sap.com",
        ],
        "Next.js/Nuxt": [
            "https://www.vercel.com",
            "https://www.nextjs.org",
            "https://www.nuxtjs.org",
            "https://www.hulu.com",
            "https://www.twitch.tv",
            "https://www.docker.com",
            "https://www.hashicorp.com",
        ],
        "Express/Node": [
            "https://www.nodejs.org",
            "https://www.npmjs.com",
            "https://www.github.com",
            "https://api.github.com",
            "https://www.stackoverflow.com",
            "https://www.heroku.com",
        ],
        "jQuery/传统": [
            "https://www.wikipedia.org",
            "https://www.wordpress.com",
            "https://www.amazon.com",
            "https://www.ebay.com",
            "https://www.cnn.com",
            "https://www.bbc.com",
        ],
        "Svelte/其他": [
            "https://www.svelte.dev",
            "https://www.remix.run",
            "https://www.astro.build",
            "https://www.qwik.builder.io",
        ],
    }
    
    def __init__(self, output_dir: Path = Path("real_data/websites")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_file = self.output_dir / "websites_data.jsonl"
        self.stats_file = self.output_dir / "crawl_stats.json"
        
        self.stats = {
            "total_attempted": 0,
            "total_success": 0,
            "total_failed": 0,
            "frameworks_found": defaultdict(int),
            "start_time": datetime.now().isoformat(),
            "urls_by_framework": defaultdict(list),
        }
        
    async def fetch_website(self, session: aiohttp.ClientSession, url: str, timeout: int = 10) -> Optional[str]:
        """爬取单个网站"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout),
                                   ssl=False, allow_redirects=True) as response:
                if response.status == 200:
                    return await response.text()
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ 超时: {url}")
        except Exception as e:
            logger.warning(f"❌ 获取失败 {url}: {e}")
        return None
    
    def extract_scripts(self, html: str, url: str) -> Tuple[List[Dict], List[str]]:
        """提取内联和外部脚本"""
        scripts = []
        external_urls = []
        
        # 提取内联脚本
        inline_pattern = r'<script[^>]*>([^<]*)<\/script>'
        for match in re.finditer(inline_pattern, html, re.IGNORECASE | re.DOTALL):
            content = match.group(1).strip()
            if content and len(content) > 20:  # 忽略短脚本
                scripts.append({
                    "type": "inline",
                    "content": content[:10000],  # 限制大小
                    "size": len(content)
                })
        
        # 提取外部脚本URL
        src_pattern = r'<script[^>]*src=["\']([^"\']+)["\']'
        for match in re.finditer(src_pattern, html, re.IGNORECASE):
            src = match.group(1)
            # 转为绝对URL
            absolute_url = urljoin(url, src)
            external_urls.append(absolute_url)
            scripts.append({
                "type": "external",
                "src": src,
                "absolute_url": absolute_url,
                "size": 0
            })
        
        return scripts, external_urls
    
    def extract_css(self, html: str, url: str) -> List[Dict]:
        """提取CSS信息"""
        css_list = []
        
        # 内联样式
        style_pattern = r'<style[^>]*>([^<]*)<\/style>'
        for match in re.finditer(style_pattern, html, re.IGNORECASE | re.DOTALL):
            content = match.group(1).strip()
            if content:
                css_list.append({
                    "type": "inline",
                    "size": len(content)
                })
        
        # 外部样式
        link_pattern = r'<link[^>]*href=["\']([^"\']+)["\'][^>]*rel=["\']stylesheet["\']'
        for match in re.finditer(link_pattern, html, re.IGNORECASE):
            href = match.group(1)
            css_list.append({
                "type": "external",
                "href": href,
                "absolute_url": urljoin(url, href)
            })
        
        return css_list
    
    def detect_frameworks(self, html: str, scripts: List[Dict]) -> Dict[str, float]:
        """框架检测 - 基于特征"""
        detected = {}
        
        # 合并内容用于分析
        full_content = html + " " + " ".join([s.get("content", "") for s in scripts if "content" in s])
        full_content_lower = full_content.lower()
        
        # React特征
        react_patterns = [
            r'react(?:dom)?\.render',
            r'ReactDOM\.render',
            r'ReactDOMClient\.createRoot',
            r'from\s+["\']react["\']',
            r'import\s+React\s+from',
            r'__REACT_',
            r'_react_jsx',
            r'react-app-',
        ]
        react_score = sum(1 for p in react_patterns if re.search(p, full_content_lower)) / len(react_patterns)
        if react_score > 0:
            detected["React"] = react_score
        
        # Vue特征
        vue_patterns = [
            r'Vue\.createApp',
            r'new Vue\(',
            r'from\s+["\']vue["\']',
            r'<template>.*?</template>',
            r'v-bind',
            r'v-model',
            r'@click',
            r'v-if',
        ]
        vue_score = sum(1 for p in vue_patterns if re.search(p, full_content_lower)) / len(vue_patterns)
        if vue_score > 0:
            detected["Vue"] = vue_score
        
        # Angular特征
        angular_patterns = [
            r'angular\.module',
            r'ng-app',
            r'ng-model',
            r'ng-repeat',
            r'@angular',
            r'from\s+["\']@angular',
            r'Component\s*\(',
        ]
        angular_score = sum(1 for p in angular_patterns if re.search(p, full_content_lower)) / len(angular_patterns)
        if angular_score > 0:
            detected["Angular"] = angular_score
        
        # Svelte特征
        svelte_patterns = [
            r'<script>.*?</script>',  # Svelte components
            r'let\s+\w+\s*=\s*',
            r'reactive\(',
            r'from\s+["\']svelte["\']',
        ]
        svelte_score = sum(1 for p in svelte_patterns if re.search(p, full_content_lower)) / len(svelte_patterns)
        if svelte_score > 0:
            detected["Svelte"] = svelte_score
        
        # Express/Node特征
        express_patterns = [
            r'app\.get\(',
            r'app\.post\(',
            r'require\(["\']express["\']',
            r'from\s+["\']express["\']',
        ]
        express_score = sum(1 for p in express_patterns if re.search(p, full_content_lower)) / len(express_patterns)
        if express_score > 0:
            detected["Express"] = express_score
        
        # jQuery特征
        jquery_patterns = [
            r'jQuery|(?<!\w)\$\(',
            r'\$\(.*?\)\.(?:on|click|bind)',
            r'jquery',
        ]
        jquery_score = sum(1 for p in jquery_patterns if re.search(p, full_content_lower)) / len(jquery_patterns)
        if jquery_score > 0:
            detected["jQuery"] = jquery_score
        
        # 规范化分数
        if detected:
            total = sum(detected.values())
            detected = {k: v / total for k, v in detected.items()}
        
        return detected
    
    async def fetch_external_scripts(self, session: aiohttp.ClientSession, urls: List[str], limit: int = 5):
        """并发获取外部脚本内容"""
        tasks = [self.fetch_website(session, url, timeout=5) for url in urls[:limit]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        external_scripts = []
        for url, result in zip(urls[:limit], results):
            if isinstance(result, str):
                external_scripts.append({
                    "type": "external",
                    "src": url,
                    "content": result[:10000],
                    "size": len(result)
                })
        
        return external_scripts
    
    async def crawl_website(self, session: aiohttp.ClientSession, url: str, category: str) -> Optional[WebsiteData]:
        """爬取单个网站的完整数据"""
        self.stats["total_attempted"] += 1
        
        logger.info(f"🌐 爬取 {url}...")
        html = await self.fetch_website(session, url)
        
        if not html:
            self.stats["total_failed"] += 1
            return WebsiteData(
                url=url,
                domain=urlparse(url).netloc,
                html="",
                scripts=[],
                css=[],
                detected_frameworks={},
                metadata={"category": category},
                timestamp=datetime.now().isoformat(),
                success=False,
                error="Failed to fetch HTML"
            )
        
        # 提取数据
        scripts, external_urls = self.extract_scripts(html, url)
        css = self.extract_css(html, url)
        
        # 获取外部脚本内容
        external_scripts = await self.fetch_external_scripts(session, external_urls)
        scripts.extend(external_scripts)
        
        # 框架检测
        frameworks = self.detect_frameworks(html, scripts)
        
        self.stats["total_success"] += 1
        
        # 更新统计
        for framework in frameworks:
            self.stats["frameworks_found"][framework] += 1
            self.stats["urls_by_framework"][framework].append(url)
        
        logger.info(f"✅ {url} -> {frameworks}")
        
        return WebsiteData(
            url=url,
            domain=urlparse(url).netloc,
            html=html[:50000],  # 限制大小
            scripts=scripts,
            css=css,
            detected_frameworks=frameworks,
            metadata={
                "category": category,
                "script_count": len(scripts),
                "css_count": len(css),
                "html_size": len(html),
            },
            timestamp=datetime.now().isoformat(),
            success=True
        )
    
    async def crawl_all_websites(self, max_workers: int = 5):
        """批量爬取所有网站"""
        connector = aiohttp.TCPConnector(limit=max_workers, ssl=False)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            
            for category, urls in self.WEBSITES.items():
                for url in urls:
                    tasks.append(self.crawl_website(session, url, category))
            
            # 并发执行
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 保存结果
        self.save_results(results)
        self.save_stats()
        
        return results
    
    def save_results(self, results: List):
        """保存爬取结果"""
        logger.info(f"💾 保存数据到 {self.data_file}...")
        
        with open(self.data_file, 'w', encoding='utf-8') as f:
            for result in results:
                if isinstance(result, WebsiteData):
                    f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
        
        logger.info(f"✅ 保存 {len(results)} 个网站数据")
    
    def save_stats(self):
        """保存统计信息"""
        self.stats["end_time"] = datetime.now().isoformat()
        self.stats["frameworks_found"] = dict(self.stats["frameworks_found"])
        self.stats["urls_by_framework"] = dict(self.stats["urls_by_framework"])
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 爬取统计:")
        logger.info(f"  总尝试: {self.stats['total_attempted']}")
        logger.info(f"  成功: {self.stats['total_success']}")
        logger.info(f"  失败: {self.stats['total_failed']}")
        logger.info(f"  检测到的框架: {self.stats['frameworks_found']}")
        logger.info(f"{'='*60}\n")


async def main():
    """主函数"""
    crawler = RealWebsiteCrawler()
    await crawler.crawl_all_websites(max_workers=5)


if __name__ == "__main__":
    asyncio.run(main())
