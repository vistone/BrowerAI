#!/usr/bin/env python3
"""
Week 6 框架样本采集脚本
目标: 扩展框架覆盖从 6 个到 12+ 个
"""

import asyncio
import aiohttp
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 框架官网和示例网站
FRAMEWORK_URLS = {
    # 已有框架
    'react': [
        'https://reactjs.org',
        'https://react.dev',
        'https://react-bootstrap.github.io',
    ],
    'vue': [
        'https://vuejs.org',
        'https://vue-loader.vuejs.org',
        'https://vueuse.org',
    ],
    'angular': [
        'https://angular.io',
        'https://material.angular.io',
        'https://angular.schule',
    ],
    'jquery': [
        'https://jquery.com',
        'https://plugins.jquery.com',
        'https://api.jquery.com',
    ],
    'svelte': [
        'https://svelte.dev',
        'https://kit.svelte.dev',
    ],
    'express': [
        'https://expressjs.com',
    ],
    
    # 新增框架
    'ember': [
        'https://emberjs.com',
        'https://guides.emberjs.com',
        'https://ember-cli.com',
    ],
    'backbone': [
        'https://backbonejs.org',
        'https://marionettejs.com',
    ],
    'alpine': [
        'https://alpinejs.dev',
    ],
    'htmx': [
        'https://htmx.org',
    ],
    'nextjs': [
        'https://nextjs.org',
        'https://next-auth.js.org',
    ],
    'nuxt': [
        'https://nuxt.com',
        'https://nuxt.com/docs',
    ],
}

class FrameworkSampleCollector:
    """框架样本采集器"""
    
    def __init__(self, output_dir='data/week6_samples', frameworks=None, samples_per_framework=100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frameworks = frameworks or list(FRAMEWORK_URLS.keys())
        self.samples_per_framework = samples_per_framework
        self.collected = []
        # 扩展框架检测关键词
        self.framework_indicators = {
            'react': ['React', 'react-dom', 'next.js', 'gatsby', '__REACT_'],
            'vue': ['Vue', 'vue.js', 'Vuex', '__VUE', 'v-'],
            'angular': ['Angular', '@angular', 'ng-', '\\[ngIf\\]'],
            'jquery': ['jQuery', 'jquery', '$\\.fn'],
            'svelte': ['Svelte', 'svelte.js'],
            'express': ['Express', 'express.js'],
            'ember': ['Ember', 'ember.js', '@ember'],
            'backbone': ['Backbone', 'backbone.js'],
            'alpine': ['Alpine', 'alpine.js'],
            'htmx': ['HTMX', 'htmx.org', 'hx-'],
            'nextjs': ['Next.js', 'nextjs', '__NEXT', '_next'],
            'nuxt': ['Nuxt', 'nuxt.js', '__NUXT'],
        }
    
    async def fetch_url(self, session, url, framework):
        """异步获取单个 URL"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    sample = {
                        'url': url,
                        'framework': framework,
                        'html': html,
                        'html_size': len(html),
                        'html_hash': hashlib.md5(html.encode()).hexdigest(),
                        'detected_framework': self._detect_framework(html),
                        'metadata': {
                            'status_code': response.status,
                            'content_type': response.headers.get('Content-Type', ''),
                        }
                    }
                    
                    logger.info(f"✅ {framework}: {url} ({len(html)} bytes)")
                    return sample
                else:
                    logger.warning(f"⚠️  {framework}: {url} - HTTP {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.warning(f"⏱️  {framework}: {url} - Timeout")
            return None
        except Exception as e:
            logger.warning(f"❌ {framework}: {url} - {str(e)[:50]}")
            return None
    
    def _detect_framework(self, html: str) -> str:
        """改进的框架检测 - 多关键词匹配"""
        scores = {}
        html_lower = html.lower()
        
        # 对每个框架计算得分
        for framework, keywords in self.framework_indicators.items():
            score = 0
            for keyword in keywords:
                # 精确匹配和模糊匹配
                pattern = keyword.lower().replace('\\\\', '')
                if pattern in html_lower:
                    score += 2
                # 检查前缀
                if html_lower.count(pattern) > 0:
                    score += len(pattern)
            scores[framework] = score
        
        # 返回得分最高的框架，或返回 unknown
        if scores and max(scores.values()) > 0:
            best_framework = None
            for fw, sc in scores.items():
                if best_framework is None or sc > scores[best_framework]:
                    best_framework = fw
            return best_framework if best_framework else 'unknown'
        
        return 'unknown'
            return 'jquery'
        
        # Svelte
        if 'svelte' in html_lower:
            return 'svelte'
        
        # Express
        if 'express' in html_lower:
            return 'express'
        
        # Ember
        if 'ember' in html_lower:
            return 'ember'
        
        # Backbone
        if 'backbone' in html_lower:
            return 'backbone'
        
        # Alpine
        if 'alpine' in html_lower or 'x-data' in html_lower:
            return 'alpine'
        
        # htmx
        if 'htmx' in html_lower or 'hx-get' in html_lower:
            return 'htmx'
        
        # Next.js
        if 'next.js' in html_lower or '__next' in html_lower:
            return 'nextjs'
        
        # Nuxt
        if 'nuxt' in html_lower:
            return 'nuxt'
        
        return 'unknown'
    
    async def collect_framework_samples(self, framework: str):
        """采集单个框架的样本"""
        logger.info(f"\n📥 开始采集框架: {framework}")
        
        urls = FRAMEWORK_URLS.get(framework, [])
        if not urls:
            logger.warning(f"⚠️  没有为 {framework} 配置 URL")
            return []
        
        samples = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url, framework) for url in urls[:self.samples_per_framework]]
            results = await asyncio.gather(*tasks)
            samples = [r for r in results if r is not None]
        
        logger.info(f"✅ {framework}: 采集 {len(samples)}/{len(urls)} 个样本")
        return samples
    
    async def collect_all_frameworks(self):
        """采集所有框架样本"""
        logger.info("="*80)
        logger.info("🚀 Week 6 框架样本采集开始")
        logger.info(f"目标框架: {', '.join(self.frameworks)}")
        logger.info(f"每框架样本: {self.samples_per_framework}")
        logger.info("="*80)
        
        for framework in self.frameworks:
            samples = await self.collect_framework_samples(framework)
            self.collected.extend(samples)
        
        logger.info("\n" + "="*80)
        logger.info(f"✅ 采集完成: {len(self.collected)} 个样本")
        logger.info("="*80)
        
        # 统计
        framework_stats = {}
        for sample in self.collected:
            fw = sample.get('detected_framework', 'unknown')
            framework_stats[fw] = framework_stats.get(fw, 0) + 1
        
        logger.info("\n📊 框架分布:")
        for fw, count in sorted(framework_stats.items(), key=lambda x: -x[1]):
            percentage = (count / len(self.collected)) * 100 if self.collected else 0
            logger.info(f"  {fw:15} {count:4} ({percentage:5.1f}%)")
        
        return self.collected
    
    def save_samples(self):
        """保存样本到文件"""
        if not self.collected:
            logger.warning("⚠️  没有采集到样本")
            return
        
        # 保存到 JSONL 格式
        output_file = self.output_dir / 'framework_samples.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.collected:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"\n💾 样本保存到: {output_file}")
        logger.info(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 保存摘要
        summary = {
            'total_samples': len(self.collected),
            'frameworks': list(set(s.get('framework') for s in self.collected)),
            'framework_counts': {
                fw: sum(1 for s in self.collected if s.get('detected_framework') == fw)
                for fw in set(s.get('detected_framework') for s in self.collected)
            },
            'total_size_bytes': sum(s.get('html_size', 0) for s in self.collected),
        }
        
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 摘要保存到: {summary_file}")

async def main():
    parser = argparse.ArgumentParser(description='Week 6 框架样本采集')
    parser.add_argument('--frameworks', type=str, 
                       help='逗号分隔的框架列表 (默认: 全部)')
    parser.add_argument('--samples-per-framework', type=int, default=100,
                       help='每个框架的样本数 (默认: 100)')
    parser.add_argument('--output', type=str, default='data/week6_samples',
                       help='输出目录 (默认: data/week6_samples)')
    
    args = parser.parse_args()
    
    frameworks = None
    if args.frameworks:
        frameworks = [fw.strip() for fw in args.frameworks.split(',')]
    
    collector = FrameworkSampleCollector(
        output_dir=args.output,
        frameworks=frameworks,
        samples_per_framework=args.samples_per_framework
    )
    
    await collector.collect_all_frameworks()
    collector.save_samples()
    
    logger.info("\n✅ 采集任务完成!")

if __name__ == '__main__':
    asyncio.run(main())
