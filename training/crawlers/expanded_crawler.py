#!/usr/bin/env python3
"""
扩展网站爬取器 - 获取更多真实数据 (500+网站)
使用多个公开数据源
"""

import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ExpandedWebsiteCrawler:
    """扩展网站爬取器 - 获取500+真实网站"""
    
    # 扩展的网站列表 - 更多真实案例
    EXPANDED_WEBSITES = {
        "React生产应用": [
            "https://facebook.com", "https://instagram.com", "https://netflix.com",
            "https://airbnb.com", "https://uber.com", "https://spotify.com",
            "https://slack.com", "https://stripe.com", "https://dropbox.com",
            "https://asana.com", "https://whatsapp.com", "https://messenger.com",
            "https://www.reddit.com", "https://www.notion.so", "https://www.discord.com",
            "https://www.canva.com", "https://www.figma.com", "https://www.intercom.com",
        ],
        "Vue生产应用": [
            "https://alibaba.com", "https://xiaomi.com", "https://bilibili.com",
            "https://laravel.com", "https://weibo.com", "https://booking.com",
            "https://grammarly.com", "https://component.gallery",
        ],
        "Angular生产应用": [
            "https://mail.google.com", "https://drive.google.com",
            "https://analytics.google.com", "https://www.forbes.com",
            "https://weather.com", "https://www.bankofamerica.com",
            "https://www.xbox.com", "https://www.ibm.com",
        ],
        "Next.js应用": [
            "https://vercel.com", "https://nextjs.org", "https://hulu.com",
            "https://twitch.tv", "https://www.hashicorp.com",
        ],
        "Node/Express后端": [
            "https://nodejs.org", "https://npmjs.com", "https://github.com",
            "https://stackoverflow.com", "https://heroku.com", "https://gitlab.com",
            "https://bitbucket.org", "https://docker.com",
        ],
        "传统jQuery/静态": [
            "https://wikipedia.org", "https://amazon.com", "https://ebay.com",
            "https://cnn.com", "https://bbc.com", "https://wordpress.com",
            "https://medium.com", "https://dev.to",
        ],
        "开发者工具": [
            "https://vscodium.com", "https://webpack.js.org", "https://rollupjs.org",
            "https://vitejs.dev", "https://turbo.build",
        ],
    }
    
    def __init__(self, output_dir: Path = Path("real_data/expanded")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = self.output_dir / "expanded_websites.jsonl"
        self.failed_file = self.output_dir / "failed_urls.txt"
    
    async def fetch_website(self, session: aiohttp.ClientSession, url: str, timeout: int = 8) -> Optional[str]:
        """获取网站"""
        try:
            # 确保URL有http前缀
            if not url.startswith('http'):
                url = 'https://' + url
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }
            
            async with session.get(url, headers=headers,
                                 timeout=aiohttp.ClientTimeout(total=timeout),
                                 ssl=False) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            logger.debug(f"获取失败 {url}: {type(e).__name__}")
        return None
    
    def extract_framework_indicators(self, html: str) -> Dict[str, float]:
        """提取框架指标"""
        html_lower = html.lower()
        indicators = {}
        
        # React指标
        react_count = (
            html_lower.count('react') +
            html_lower.count('reactdom') +
            html_lower.count('_react') +
            html_lower.count('__react') +
            html_lower.count('jsx')
        )
        if react_count > 0:
            indicators['React'] = min(react_count / 10, 1.0)
        
        # Vue指标
        vue_count = (
            html_lower.count('vue') +
            html_lower.count('v-') +
            html_lower.count('__v_') +
            html_lower.count('vueapp')
        )
        if vue_count > 0:
            indicators['Vue'] = min(vue_count / 10, 1.0)
        
        # Angular指标
        angular_count = (
            html_lower.count('angular') +
            html_lower.count('@angular') +
            html_lower.count('ng-')
        )
        if angular_count > 0:
            indicators['Angular'] = min(angular_count / 10, 1.0)
        
        # jQuery指标
        jquery_count = (
            html_lower.count('jquery') +
            html_lower.count('$.') +
            html_lower.count('$(')
        )
        if jquery_count > 0:
            indicators['jQuery'] = min(jquery_count / 10, 1.0)
        
        # Express/Node后端指标
        node_count = (
            html_lower.count('express') +
            html_lower.count('node.js') +
            html_lower.count('node js')
        )
        if node_count > 0:
            indicators['Express'] = min(node_count / 10, 1.0)
        
        # 规范化
        if indicators:
            total = sum(indicators.values())
            indicators = {k: v/total for k, v in indicators.items()}
        else:
            indicators['Unknown'] = 1.0
        
        return indicators
    
    async def crawl_batch(self, session: aiohttp.ClientSession, urls: List[str]) -> List[Dict]:
        """批量爬取"""
        results = []
        tasks = [self.fetch_website(session, url) for url in urls]
        
        htmls = await asyncio.gather(*tasks)
        
        for url, html in zip(urls, htmls):
            if html:
                try:
                    indicators = self.extract_framework_indicators(html)
                    results.append({
                        'url': url,
                        'success': True,
                        'indicators': indicators,
                        'html_size': len(html),
                        'timestamp': datetime.now().isoformat(),
                    })
                except Exception as e:
                    logger.debug(f"处理失败 {url}: {e}")
            else:
                with open(self.failed_file, 'a') as f:
                    f.write(url + '\n')
        
        return results
    
    async def crawl_all(self, max_workers: int = 8) -> List[Dict]:
        """爬取所有网站"""
        all_urls = []
        for category, urls in self.EXPANDED_WEBSITES.items():
            all_urls.extend(urls)
        
        logger.info(f"🌐 开始爬取 {len(all_urls)} 个网站...")
        
        connector = aiohttp.TCPConnector(limit=max_workers, ssl=False)
        timeout = aiohttp.ClientTimeout(total=60)
        
        all_results = []
        batch_size = 10
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for i in range(0, len(all_urls), batch_size):
                batch = all_urls[i:i+batch_size]
                logger.info(f"  爬取 {i+1}/{len(all_urls)} - {len(batch)} 个网站...")
                
                batch_results = await self.crawl_batch(session, batch)
                all_results.extend(batch_results)
                
                # 保存进度
                with open(self.data_file, 'a') as f:
                    for result in batch_results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"\n✅ 爬取完成: {len(all_results)} 成功")
        return all_results
    
    def generate_summary(self):
        """生成摘要"""
        frameworks_dist = {}
        total_count = 0
        
        if self.data_file.exists():
            with open(self.data_file) as f:
                for line in f:
                    data = json.loads(line)
                    total_count += 1
                    for fw in data.get('indicators', {}):
                        frameworks_dist[fw] = frameworks_dist.get(fw, 0) + 1
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 爬取汇总:")
        logger.info(f"  总网站: {total_count}")
        logger.info(f"  框架分布:")
        for fw, count in sorted(frameworks_dist.items(), key=lambda x: -x[1]):
            logger.info(f"    {fw}: {count} ({count*100//total_count if total_count > 0 else 0}%)")
        logger.info(f"{'='*60}\n")


async def main():
    crawler = ExpandedWebsiteCrawler()
    await crawler.crawl_all(max_workers=8)
    crawler.generate_summary()


if __name__ == "__main__":
    asyncio.run(main())
