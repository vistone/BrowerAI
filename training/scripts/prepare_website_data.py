"""
Website Data Collection and Preparation

Crawls complete websites to create training data for holistic learning.
Captures:
- Complete HTML structure
- All CSS files in loading order
- All JS files in loading order
- Dependency graphs between files
- Metadata (framework, build tool, company style)
- Device adaptation strategies

Usage:
    python scripts/prepare_website_data.py --output data/websites --num-sites 100
    python scripts/prepare_website_data.py --config configs/website_learner.yaml
"""

import sys
import os
import json
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import re

import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebsiteCrawler:
    """Crawls complete websites for holistic learning with DEPTH support"""
    
    def __init__(self, max_files: int = 50, max_depth: int = 3, max_pages: int = 10):
        self.max_files = max_files
        self.max_depth = max_depth  # 爬取深度
        self.max_pages = max_pages  # 每个网站最多爬取的页面数
        self.session = None
        self.visited_urls = set()  # 已访问URL集合
        
        # Framework detection patterns
        self.framework_patterns = {
            'React': [
                r'react\.min\.js',
                r'react-dom',
                r'_jsx\(',
                r'React\.createElement',
                r'/__react'
            ],
            'Vue': [
                r'vue\.js',
                r'vue\.min\.js',
                r'new Vue\(',
                r'v-if=',
                r'v-for='
            ],
            'Angular': [
                r'angular\.js',
                r'angular\.min\.js',
                r'ng-app',
                r'ng-controller',
                r'\[ngFor\]'
            ],
            'jQuery': [
                r'jquery\.js',
                r'jquery\.min\.js',
                r'\$\(',
                r'jQuery\('
            ],
            'Svelte': [
                r'svelte',
                r'\.svelte\.js'
            ],
            'Next.js': [
                r'_next/',
                r'next\.config\.js',
                r'__NEXT_DATA__'
            ],
            'Nuxt': [
                r'_nuxt/',
                r'nuxt\.js'
            ],
            'Bootstrap': [
                r'bootstrap\.css',
                r'bootstrap\.min\.css',
                r'class="container"',
                r'class="row"'
            ],
            'Tailwind': [
                r'tailwind\.css',
                r'@tailwind',
                r'class=".*?(flex|grid|bg-|text-)'
            ],
            'Material-UI': [
                r'@material-ui',
                r'@mui'
            ]
        }
        
        # Build tool patterns
        self.build_tool_patterns = {
            'Webpack': [
                r'webpack',
                r'webpackJsonp',
                r'__webpack'
            ],
            'Vite': [
                r'/@vite/',
                r'vite\.config'
            ],
            'Rollup': [
                r'rollup',
                r'\.rollup\.'
            ],
            'Parcel': [
                r'parcel',
                r'\.parcel-'
            ],
            'esbuild': [
                r'esbuild'
            ],
            'Browserify': [
                r'browserify'
            ]
        }
        
        # Company patterns (examples)
        self.company_patterns = {
            'Google': [
                r'google-analytics',
                r'googletagmanager',
                r'gstatic\.com',
                r'googleapis\.com'
            ],
            'Amazon': [
                r'cloudfront\.net',
                r'aws\.amazon\.com',
                r'amazon-adsystem'
            ],
            'Facebook': [
                r'facebook\.com/plugins',
                r'fbcdn\.net',
                r'connect\.facebook\.net'
            ],
            'Shopify': [
                r'cdn\.shopify\.com',
                r'Shopify\.theme',
                r'shopify-analytics'
            ]
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch(self, url: str) -> str:
        """Fetch URL content with robust encoding handling and fast failure"""
        try:
            # 使用更短的超时：总超时10秒，连接超时5秒
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            async with self.session.get(url, timeout=timeout) as response:
                # 快速跳过非200状态
                if response.status != 200:
                    logger.debug(f"Skip {url}: status {response.status}")
                    return ""
                
                try:
                    # Try to decode as UTF-8 first
                    return await response.text(encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        # Fallback to Latin-1 (covers most Western encodings)
                        content = await response.read()
                        return content.decode('latin-1', errors='ignore')
                    except Exception as decode_error:
                        # Last resort: ignore errors
                        logger.debug(f"Encoding error for {url}, using fallback: {decode_error}")
                        return await response.text(errors='ignore')
        except asyncio.TimeoutError:
            logger.debug(f"Timeout (fast skip): {url}")
            return ""
        except Exception as e:
            # 快速记录并跳过，不打印ERROR级别日志（太多噪音）
            logger.debug(f"Skip {url}: {e}")
        return ""
    
    def extract_css_files(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract CSS files in loading order"""
        css_files = []
        
        for idx, link in enumerate(soup.find_all('link', rel='stylesheet')):
            href = link.get('href')
            if href:
                url = urljoin(base_url, href)
                css_files.append({
                    'path': href,
                    'url': url,
                    'order': idx
                })
        
        # Inline styles
        for idx, style in enumerate(soup.find_all('style')):
            css_files.append({
                'path': f'inline_{idx}.css',
                'content': style.string or '',
                'order': len(css_files)
            })
        
        return css_files
    
    def extract_js_files(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract JS files in loading order"""
        js_files = []
        
        for idx, script in enumerate(soup.find_all('script')):
            src = script.get('src')
            
            if src:
                # External script
                url = urljoin(base_url, src)
                
                # Determine loading type
                js_type = 'blocking'
                if script.get('async') is not None:
                    js_type = 'async'
                elif script.get('defer') is not None:
                    js_type = 'defer'
                
                js_files.append({
                    'path': src,
                    'url': url,
                    'type': js_type,
                    'order': idx
                })
            else:
                # Inline script
                content = script.string or ''
                if content.strip():
                    js_files.append({
                        'path': f'inline_{idx}.js',
                        'content': content,
                        'type': 'blocking',
                        'order': len(js_files)
                    })
        
        return js_files
    
    def extract_dependencies(self, js_content: str, css_content: str) -> List[Dict[str, str]]:
        """Extract file dependencies from JS and CSS"""
        dependencies = []
        
        # JS import patterns
        import_patterns = [
            r'import .+ from [\'"](.+?)[\'"]',
            r'require\([\'"](.+?)[\'"]\)',
            r'import\([\'"](.+?)[\'"]\)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, js_content)
            for match in matches:
                dependencies.append({
                    'from': 'main.js',
                    'to': match,
                    'type': 'import'
                })
        
        # CSS @import patterns
        css_import_pattern = r'@import [\'"](.+?)[\'"]'
        matches = re.findall(css_import_pattern, css_content)
        for match in matches:
            dependencies.append({
                'from': 'main.css',
                'to': match,
                'type': 'import'
            })
        
        return dependencies
    
    def detect_framework(self, html: str, js_content: str, css_content: str) -> str:
        """Detect frontend framework"""
        combined = html + js_content + css_content
        
        for framework, patterns in self.framework_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return framework
        
        return 'Unknown'
    
    def detect_build_tool(self, js_content: str) -> str:
        """Detect build tool"""
        for tool, patterns in self.build_tool_patterns.items():
            for pattern in patterns:
                if re.search(pattern, js_content, re.IGNORECASE):
                    return tool
        
        return 'Unknown'
    
    def detect_company_style(self, html: str, js_content: str) -> str:
        """Detect company-specific patterns"""
        combined = html + js_content
        
        for company, patterns in self.company_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return company
        
        return 'Unknown'
    
    def detect_responsive(self, html: str, css_content: str) -> bool:
        """Detect responsive design"""
        # Check for viewport meta tag
        if '<meta name="viewport"' in html.lower():
            return True
        
        # Check for media queries
        if '@media' in css_content:
            return True
        
        return False
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract internal links from page for depth crawling"""
        links = []
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href:
                continue
            
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            # Only keep same-domain links
            if parsed.netloc == base_domain:
                # Remove fragments and query params for deduplication
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if clean_url not in self.visited_urls and len(self.visited_urls) < self.max_pages:
                    links.append(clean_url)
        
        return links[:self.max_pages]  # Limit number of links
    
    async def crawl_page(self, url: str, base_url: str) -> Dict[str, Any]:
        """Crawl a single page and return its data"""
        html = await self.fetch(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract resources (lightweight for sub-pages)
        css_files = self.extract_css_files(soup, url)[:5]  # Limit for sub-pages
        js_files = self.extract_js_files(soup, url)[:5]
        
        # Only extract inline content for sub-pages to save time
        css_files = [f for f in css_files if 'inline' in f.get('path', '')]
        js_files = [f for f in js_files if 'inline' in f.get('path', '')]
        
        return {
            'url': url,
            'html': html[:50000],  # Smaller limit for sub-pages
            'css_files': css_files,
            'js_files': js_files,
            'title': soup.title.string if soup.title else '',
            'links': self.extract_links(soup, base_url)
        }
    
    async def crawl_website_with_depth(self, url: str, category: str = 'unknown') -> Dict[str, Any]:
        """Crawl a complete website with DEPTH - multiple pages"""
        logger.info(f"深度爬取 {url} (最大深度: {self.max_depth}, 最大页面: {self.max_pages})...")
        
        self.visited_urls.clear()
        self.visited_urls.add(url)
        
        # 1. Crawl main page (detailed)
        html = await self.fetch(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract resources from main page
        css_files = self.extract_css_files(soup, url)
        js_files = self.extract_js_files(soup, url)
        
        # Fetch external CSS/JS content
        for css_file in css_files[:self.max_files]:
            if 'url' in css_file and 'content' not in css_file:
                css_file['content'] = await self.fetch(css_file['url'])
        
        for js_file in js_files[:self.max_files]:
            if 'url' in js_file and 'content' not in js_file:
                js_file['content'] = await self.fetch(js_file['url'])
        
        # Combine content for analysis
        all_css = ' '.join([f.get('content', '') for f in css_files])
        all_js = ' '.join([f.get('content', '') for f in js_files])
        
        # 2. Extract links for depth crawling
        internal_links = self.extract_links(soup, url)
        
        # 3. Crawl sub-pages (breadth-first, limited depth)
        sub_pages = []
        current_level_urls = internal_links[:self.max_pages-1]  # Reserve 1 for main page
        
        for depth in range(1, self.max_depth + 1):
            if not current_level_urls or len(self.visited_urls) >= self.max_pages:
                break
            
            next_level_urls = []
            for sub_url in current_level_urls[:self.max_pages - len(self.visited_urls)]:
                if sub_url in self.visited_urls:
                    continue
                
                self.visited_urls.add(sub_url)
                page_data = await self.crawl_page(sub_url, url)
                
                if page_data:
                    sub_pages.append(page_data)
                    # Collect links for next level
                    if depth < self.max_depth:
                        next_level_urls.extend(page_data.get('links', []))
            
            current_level_urls = next_level_urls
        
        # Extract dependencies
        dependencies = self.extract_dependencies(all_js, all_css)
        
        # Detect metadata
        framework = self.detect_framework(html, all_js, all_css)
        build_tool = self.detect_build_tool(all_js)
        company = self.detect_company_style(html, all_js)
        responsive = self.detect_responsive(html, all_css)
        
        # Build website data with depth info
        website_data = {
            'url': url,
            'category': category,
            'depth': len(sub_pages) + 1,  # Including main page
            'pages': {
                'main': {
                    'url': url,
                    'html': html[:100000],
                    'css_files': [
                        {
                            'path': f['path'],
                            'content': f.get('content', '')[:50000],
                            'order': f['order']
                        }
                        for f in css_files[:10]
                    ],
                    'js_files': [
                        {
                            'path': f['path'],
                            'content': f.get('content', '')[:50000],
                            'order': f['order'],
                            'type': f['type']
                        }
                        for f in js_files[:10]
                    ],
                },
                'sub_pages': sub_pages
            },
            'dependencies': dependencies[:100],
            'metadata': {
                'framework': framework,
                'build_tool': build_tool,
                'company': company,
                'responsive': responsive,
                'total_pages': len(sub_pages) + 1
            }
        }
        
        logger.info(f"已爬取 {url} - 页面数: {len(sub_pages)+1}, Framework: {framework}, Build: {build_tool}")
        return website_data
    
    async def crawl_website(self, url: str, category: str = 'unknown') -> Dict[str, Any]:
        """Crawl a complete website (wrapper that calls depth crawler)"""
        # Use depth crawler by default
        return await self.crawl_website_with_depth(url, category)


async def crawl_websites(urls: List[tuple], output_file: str, max_depth: int = 3, max_pages: int = 10, concurrency: int = 10):
    """Crawl multiple websites with depth and save to JSONL - HIGH CONCURRENCY VERSION
    
    Args:
        urls: List of (url, category) tuples
        output_file: Output JSONL file path
        max_depth: Maximum crawl depth per website
        max_pages: Maximum pages per website
        concurrency: Number of concurrent crawling tasks (default: 10)
    """
    
    async with WebsiteCrawler(max_files=50, max_depth=max_depth, max_pages=max_pages) as crawler:
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrency)
        total = len(urls)
        
        async def crawl_with_semaphore(url: str, category: str):
            """Wrapper to crawl with semaphore control"""
            async with semaphore:
                try:
                    website_data = await crawler.crawl_website(url, category)
                    return website_data
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    return None
        
        # Create all tasks
        logger.info(f"🚀 启动高并发爬取: {total} 个网站, 并发数={concurrency}")
        tasks = [crawl_with_semaphore(url, category) for url, category in urls]
        
        # Open file for incremental writing (avoid data loss on interrupt)
        success_count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            # Execute with progress bar
            with tqdm(total=total, desc="高并发爬取") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    if result:
                        # 💾 实时保存每个成功的结果（防止中断丢失数据）
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()  # 强制写入磁盘
                        success_count += 1
                    pbar.update(1)
        
        logger.info(f"✅ 完成！成功爬取 {success_count}/{total} 个网站")
        logger.info(f"💾 数据已保存到 {output_file}")


def load_urls_from_file(urls_file: str) -> List[tuple]:
    """
    从文件加载URL列表
    
    Args:
        urls_file: URL文件路径，每行格式: url,category
        
    Returns:
        List of (url, category) tuples
    """
    urls = []
    with open(urls_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments and empty lines
                parts = line.split(',')
                url = parts[0].strip()
                category = parts[1].strip() if len(parts) > 1 else 'unknown'
                urls.append((url, category))
    return urls


def get_example_urls() -> List[tuple]:
    """Get example URLs for each category"""
    return [
        # Ecommerce
        ('https://www.amazon.com', 'ecommerce'),
        ('https://www.ebay.com', 'ecommerce'),
        ('https://www.etsy.com', 'ecommerce'),
        
        # News
        ('https://www.cnn.com', 'news'),
        ('https://www.bbc.com', 'news'),
        ('https://www.reuters.com', 'news'),
        
        # Education
        ('https://www.coursera.org', 'education'),
        ('https://www.khanacademy.org', 'education'),
        ('https://www.edx.org', 'education'),
        
        # Entertainment
        ('https://www.youtube.com', 'entertainment'),
        ('https://www.netflix.com', 'entertainment'),
        ('https://www.twitch.tv', 'entertainment'),
        
        # Social
        ('https://twitter.com', 'social'),
        ('https://www.linkedin.com', 'social'),
        ('https://www.reddit.com', 'social'),
        
        # Documentation
        ('https://developer.mozilla.org', 'documentation'),
        ('https://docs.python.org', 'documentation'),
        ('https://reactjs.org', 'documentation'),
        
        # Tools
        ('https://github.com', 'tools'),
        ('https://stackoverflow.com', 'tools'),
        ('https://www.figma.com', 'tools'),
    ]


def main():
    parser = argparse.ArgumentParser(description="Prepare website data for holistic learning")
    parser.add_argument(
        "--output",
        type=str,
        default="data/websites/websites_train.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        default=None,
        help="File containing URLs (one per line, format: url,category)"
    )
    parser.add_argument(
        "--num-sites",
        type=int,
        default=None,
        help="Number of sites to crawl (uses example URLs)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum crawl depth (default: 3)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum pages per website (default: 10)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent crawling tasks (default: 10, max recommended: 50)"
    )
    
    args = parser.parse_args()
    
    # Get URLs
    if args.urls_file:
        urls = load_urls_from_file(args.urls_file)
    else:
        urls = get_example_urls()
        if args.num_sites:
            urls = urls[:args.num_sites]
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Crawl websites with depth and high concurrency
    logger.info(f"开始深度爬取: 深度={args.depth}, 最大页面={args.max_pages}, 并发数={args.concurrency}")
    asyncio.run(crawl_websites(urls, args.output, max_depth=args.depth, max_pages=args.max_pages, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
