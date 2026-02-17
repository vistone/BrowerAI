#!/usr/bin/env python3
"""
Week 6 框架样本采集脚本 v2 - 生产规模版本
目标: 采集 600+ 个框架样本 (50个/框架)
优化: 改进框架检测, 支持大规模采集, 内存效率
"""

import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from typing import Dict, List
import hashlib
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 12 个框架的官网和示例网站库
FRAMEWORK_URLS = {
    'react': [
        'https://react.dev',
        'https://reactjs.org',
        'https://create-react-app.dev',
        'https://nextjs.org',
        'https://remix.run',
    ],
    'vue': [
        'https://vuejs.org',
        'https://nuxt.com',
        'https://vueuse.org',
        'https://headlessui.com',
    ],
    'angular': [
        'https://angular.io',
        'https://material.angular.io',
        'https://ngrx.io',
    ],
    'jquery': [
        'https://jquery.com',
        'https://jqueryui.com',
        'https://plugins.jquery.com',
    ],
    'svelte': [
        'https://svelte.dev',
        'https://kit.svelte.dev',
        'https://sveltekit.js.org',
    ],
    'express': [
        'https://expressjs.com',
        'https://www.fastify.io',
        'https://koa.js.org',
    ],
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
        'https://livewire.laravel.com',
    ],
    'htmx': [
        'https://htmx.org',
        'https://htmx.org/examples',
    ],
    'nextjs': [
        'https://nextjs.org',
        'https://next-auth.js.org',
        'https://prisma.io',
    ],
    'nuxt': [
        'https://nuxt.com',
        'https://nuxt.com/docs',
        'https://nuxtjs.org',
    ],
}

class LargeScaleFrameworkCollector:
    """大规模框架样本采集器 - 内存优化版本"""
    
    def __init__(self, output_dir='data/week6_samples_production', 
                 samples_per_framework=50, batch_size=5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_framework = samples_per_framework
        self.batch_size = batch_size
        
        # 改进的框架检测关键词库
        self.framework_keywords = {
            'react': {'React', 'react-dom', 'next.js', 'remix', '__REACT'},
            'vue': {'Vue', 'vue@', 'nuxt', 'vueuse', '__VUE'},
            'angular': {'Angular', '@angular', 'ng-', 'rxjs'},
            'jquery': {'jQuery', 'jquery.', 'jquery-'},
            'svelte': {'Svelte', 'svelte.dev', 'kit.svelte'},
            'express': {'Express', 'expressjs', 'fastify', 'koa'},
            'ember': {'Ember', 'ember.js', '@ember'},
            'backbone': {'Backbone', 'backbone.js', 'marionette'},
            'alpine': {'Alpine', 'alpinejs', 'livewire'},
            'htmx': {'HTMX', 'htmx.org', 'hx-'},
            'nextjs': {'Next.js', 'nextjs', '__NEXT', '_next'},
            'nuxt': {'Nuxt', 'nuxt.js', '__NUXT', 'nuxtjs'},
        }
    
    async def fetch_url(self, session: aiohttp.ClientSession, url: str, 
                       framework: str, timeout: int = 20):
        """异步获取单个 URL"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    detected_fw = self._detect_framework(html)
                    
                    return {
                        'url': url,
                        'framework': framework,
                        'html_size': len(html),
                        'html_hash': hashlib.md5(html.encode()).hexdigest()[:8],
                        'detected': detected_fw,
                        'status': 200
                    }
                else:
                    logger.warning(f"HTTP {resp.status}: {url}")
                    return None
        except asyncio.TimeoutError:
            logger.debug(f"Timeout: {url}")
            return None
        except Exception as e:
            logger.debug(f"Error: {url[:50]}... - {str(e)[:40]}")
            return None
    
    def _detect_framework(self, html: str) -> str:
        """改进的框架检测 - 多关键词评分"""
        html_lower = html.lower()
        scores = {}
        
        # 对每个框架计算得分
        for framework, keywords in self.framework_keywords.items():
            score = 0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # 精确匹配
                if keyword_lower in html_lower:
                    score += 10
                # 子字符串匹配
                if html_lower.count(keyword_lower) > 0:
                    score += html_lower.count(keyword_lower)
            scores[framework] = score
        
        # 返回得分最高的框架
        if scores and max(scores.values()) > 0:
            return max(scores, key=lambda x: scores[x])
        return 'unknown'
    
    async def collect_batch(self, session: aiohttp.ClientSession, 
                           framework: str, urls: List[str]) -> List[Dict]:
        """批量采集框架样本"""
        tasks = []
        for url in urls[:self.samples_per_framework]:
            task = self.fetch_url(session, url, framework)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    async def collect_all(self, frameworks: List[str] = None):
        """采集所有框架样本"""
        if frameworks is None:
            frameworks = list(FRAMEWORK_URLS.keys())
        
        frameworks = [f for f in frameworks if f in FRAMEWORK_URLS]
        
        all_samples = []
        
        async with aiohttp.ClientSession() as session:
            for framework in frameworks:
                urls = FRAMEWORK_URLS.get(framework, [])
                
                logger.info(f"📥 采集 {framework}...")
                samples = await self.collect_batch(session, framework, urls)
                
                logger.info(f"  ✅ {framework}: {len(samples)} 个样本")
                all_samples.extend(samples)
                
                # 流式保存，释放内存
                self._stream_save(samples, framework)
                
                # 批量间隔
                await asyncio.sleep(0.5)
        
        return all_samples
    
    def _stream_save(self, samples: List[Dict], framework: str):
        """流式保存样本"""
        output_file = self.output_dir / f"{framework}_samples.jsonl"
        
        with open(output_file, 'a') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    def merge_results(self):
        """合并所有采集结果到单个文件"""
        output_file = self.output_dir / "all_samples.jsonl"
        total = 0
        
        for jsonl_file in self.output_dir.glob("*_samples.jsonl"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    with open(output_file, 'a') as out:
                        out.write(line)
                    total += 1
        
        logger.info(f"\n✅ 合并完成: {total} 个样本")
        logger.info(f"📁 文件: {output_file}")
        
        # 生成统计信息
        return self._generate_summary(total)
    
    def _generate_summary(self, total: int) -> Dict:
        """生成采集统计"""
        summary = {
            'total_samples': total,
            'frameworks': list(FRAMEWORK_URLS.keys()),
            'framework_counts': {},
            'total_size_mb': 0.0
        }
        
        # 统计每个框架
        for framework in FRAMEWORK_URLS.keys():
            jsonl_file = self.output_dir / f"{framework}_samples.jsonl"
            if jsonl_file.exists():
                count = 0
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        count += 1
                summary['framework_counts'][framework] = count
        
        # 保存统计
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 统计信息已保存: {summary_file}")
        
        return summary

async def main():
    parser = argparse.ArgumentParser(description='框架样本采集工具')
    parser.add_argument('--samples-per-framework', type=int, default=50,
                       help='每个框架采集的样本数 (default: 50)')
    parser.add_argument('--frameworks', type=str, default=None,
                       help='指定框架 (逗号分隔, 默认全部)')
    parser.add_argument('--output-dir', type=str, default='data/week6_samples_production',
                       help='输出目录')
    
    args = parser.parse_args()
    
    frameworks = None
    if args.frameworks:
        frameworks = [f.strip() for f in args.frameworks.split(',')]
    
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║  Week 6 Phase 2 Step 4 - 大规模框架样本采集            ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    
    collector = LargeScaleFrameworkCollector(
        output_dir=args.output_dir,
        samples_per_framework=args.samples_per_framework
    )
    
    # 采集样本
    logger.info(f"🚀 开始采集 ({args.samples_per_framework} 个/框架)...")
    samples = await collector.collect_all(frameworks)
    
    # 合并结果
    summary = collector.merge_results()
    
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║  采集完成！                                            ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"\n📊 统计信息:")
    print(f"  总样本数: {summary['total_samples']}")
    print(f"  框架覆盖: {len(summary['framework_counts'])} 个")
    for fw, count in summary['framework_counts'].items():
        pct = (count / summary['total_samples'] * 100) if summary['total_samples'] > 0 else 0
        print(f"    {fw:15} {count:3} 个 ({pct:5.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())
