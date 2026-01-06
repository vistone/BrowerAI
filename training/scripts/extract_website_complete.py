#!/usr/bin/env python3
"""
网站级别数据提取
从爬取的网站中提取完整的 HTML+CSS+JS
为端到端网站生成模型准备训练数据
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_website_data(data_file: Path, output_file: Path, min_size: int = 1000):
    """
    提取完整网站数据
    
    输出格式:
    {
        "website_id": "example_com",
        "url": "https://example.com",
        "original": {
            "html": "...",  # 主页面HTML（5000字符）
            "css": "...",   # 所有CSS合并（2000字符）
            "js": "...",    # 所有JS合并（2000字符）
        },
        "metadata": {
            "dom_depth": 8,
            "element_count": 150,
            "css_rules": 50,
            "js_functions": 10
        }
    }
    """
    logger.info(f"Loading websites from {data_file}")
    
    websites = []
    with open(data_file, 'r') as f:
        for line in f:
            try:
                site = json.loads(line.strip())
                websites.append(site)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(websites)} websites")
    
    # 提取完整数据
    extracted = []
    for site in websites:
        try:
            # 提取主页面
            main_page = site.get('pages', {}).get('main', {})
            html = main_page.get('html', '')
            
            # 合并所有CSS
            css_files = main_page.get('css_files', [])
            css_content = '\n'.join([
                f['content'] for f in css_files if f.get('content')
            ])
            
            # 合并所有JS
            js_files = main_page.get('js_files', [])
            js_content = '\n'.join([
                f['content'] for f in js_files if f.get('content')
            ])
            
            # 检查大小
            total_size = len(html) + len(css_content) + len(js_content)
            if total_size < min_size:
                continue
            
            # 创建website_id
            url = site.get('url', '')
            website_id = url.replace('https://', '').replace('http://', '').replace('/', '_').replace('.', '_')
            
            # 简单的元数据统计
            metadata = {
                'dom_depth': html.count('<') // 20,  # 粗略估计
                'element_count': html.count('<'),
                'css_rules': css_content.count('{'),
                'js_functions': js_content.count('function') + js_content.count('=>'),
                'total_size': total_size
            }
            
            extracted.append({
                'website_id': website_id[:50],  # 限制长度
                'url': url,
                'original': {
                    'html': html[:5000],  # 限制HTML到5000字符
                    'css': css_content[:2000],  # 限制CSS到2000字符
                    'js': js_content[:2000],  # 限制JS到2000字符
                },
                'metadata': metadata
            })
            
        except Exception as e:
            logger.warning(f"Error extracting {site.get('url', 'unknown')}: {e}")
            continue
    
    logger.info(f"Extracted {len(extracted)} valid websites")
    
    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for item in extracted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved to {output_file}")
    
    # 统计
    total_html = sum(len(item['original']['html']) for item in extracted)
    total_css = sum(len(item['original']['css']) for item in extracted)
    total_js = sum(len(item['original']['js']) for item in extracted)
    
    logger.info(f"""
数据统计:
  - 网站数量: {len(extracted)}
  - HTML总量: {total_html / 1024:.1f} KB
  - CSS总量: {total_css / 1024:.1f} KB
  - JS总量: {total_js / 1024:.1f} KB
  - 平均网站大小: {(total_html + total_css + total_js) / len(extracted) / 1024:.1f} KB
""")


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    input_file = data_dir / 'websites' / '1000_sites.jsonl'
    output_file = data_dir / 'website_complete.jsonl'
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    extract_website_data(input_file, output_file, min_size=1000)


if __name__ == '__main__':
    main()
