#!/usr/bin/env python3
"""
大规模网站批量爬取脚本

用于爬取1000+网站数据，支持断点续传、错误重试、进度保存
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging

# 导入现有的爬虫模块
sys.path.insert(0, str(Path(__file__).parent))
from prepare_website_data import WebsiteCrawler, get_example_urls

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('large_crawl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LargeScaleCrawler:
    """大规模网站爬取器"""
    
    def __init__(self, urls_file: Path, output_dir: Path, 
                 batch_size: int = 50, max_depth: int = 2, max_pages: int = 5):
        self.urls_file = urls_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_depth = max_depth
        self.max_pages = max_pages
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 进度跟踪
        self.progress_file = output_dir / "crawl_progress.json"
        self.completed_urls = self.load_progress()
        
    def load_progress(self) -> set:
        """加载已完成的URL（断点续传）"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"📂 加载进度: 已完成 {len(data['completed'])} 个网站")
                    return set(data['completed'])
            except Exception as e:
                logger.warning(f"⚠️ 无法加载进度文件: {e}")
        return set()
    
    def save_progress(self):
        """保存爬取进度"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'completed': list(self.completed_urls),
                    'total': len(self.completed_urls),
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"❌ 无法保存进度: {e}")
    
    def load_urls(self) -> list:
        """加载URL列表"""
        urls = []
        with open(self.urls_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    url = parts[0]
                    category = parts[1] if len(parts) > 1 else 'unknown'
                    
                    # 跳过已完成的URL
                    if url not in self.completed_urls:
                        urls.append((url, category))
        
        logger.info(f"📋 待爬取: {len(urls)} 个网站 (已完成: {len(self.completed_urls)})")
        return urls
    
    async def crawl_batch(self, batch: list, batch_num: int, total_batches: int):
        """爬取一批网站"""
        batch_file = self.output_dir / f"batch_{batch_num:04d}.jsonl"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始批次 {batch_num}/{total_batches} ({len(batch)} 个网站)")
        logger.info(f"输出文件: {batch_file}")
        logger.info(f"{'='*60}\n")
        
        # 创建爬虫
        crawler = WebsiteCrawler(
            max_files=50,
            max_depth=self.max_depth,
            max_pages=self.max_pages
        )
        
        # 爬取结果
        results = []
        
        # 使用 tqdm 显示进度
        for url, category in tqdm(batch, desc=f"批次 {batch_num}"):
            try:
                logger.info(f"🌐 爬取: {url}")
                data = await crawler.crawl_website_with_depth(url, category)
                
                if data:
                    results.append(data)
                    self.completed_urls.add(url)
                    
                    # 实时统计
                    pages = data.get('depth', 1)
                    framework = data.get('metadata', {}).get('framework', 'Unknown')
                    logger.info(f"✅ 完成: {url} - {pages}页, {framework}")
                else:
                    logger.warning(f"⚠️ 空数据: {url}")
                    
            except Exception as e:
                logger.error(f"❌ 失败: {url} - {e}")
                continue
            
            # 每10个网站保存一次进度
            if len(results) % 10 == 0:
                self.save_progress()
        
        # 保存批次结果
        if results:
            with open(batch_file, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"\n✅ 批次 {batch_num} 完成: {len(results)}/{len(batch)} 个网站")
        
        # 保存进度
        self.save_progress()
        
        return len(results)
    
    async def crawl_all(self):
        """爬取所有网站"""
        urls = self.load_urls()
        
        if not urls:
            logger.info("🎉 所有网站已爬取完成！")
            return
        
        # 分批处理
        total_batches = (len(urls) + self.batch_size - 1) // self.batch_size
        total_crawled = 0
        
        logger.info(f"\n📊 爬取计划:")
        logger.info(f"  - 总网站数: {len(urls)}")
        logger.info(f"  - 批次数量: {total_batches}")
        logger.info(f"  - 每批大小: {self.batch_size}")
        logger.info(f"  - 深度设置: {self.max_depth}")
        logger.info(f"  - 最大页面: {self.max_pages}")
        logger.info(f"  - 预计页面: ~{len(urls) * 3} 页")
        logger.info(f"\n")
        
        for i in range(0, len(urls), self.batch_size):
            batch = urls[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            try:
                count = await self.crawl_batch(batch, batch_num, total_batches)
                total_crawled += count
                
                # 批次间休息（避免被封IP）
                if batch_num < total_batches:
                    logger.info(f"😴 休息 30 秒...")
                    await asyncio.sleep(30)
                    
            except Exception as e:
                logger.error(f"❌ 批次 {batch_num} 错误: {e}")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 爬取完成!")
        logger.info(f"  - 成功爬取: {total_crawled} 个网站")
        logger.info(f"  - 总计完成: {len(self.completed_urls)} 个网站")
        logger.info(f"  - 输出目录: {self.output_dir}")
        logger.info(f"{'='*60}\n")
    
    def merge_batches(self, output_file: Path):
        """合并所有批次文件"""
        logger.info(f"\n📦 合并批次文件到: {output_file}")
        
        batch_files = sorted(self.output_dir.glob("batch_*.jsonl"))
        total_sites = 0
        total_pages = 0
        
        with open(output_file, 'w', encoding='utf-8') as out:
            for batch_file in batch_files:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
                            data = json.loads(line)
                            total_sites += 1
                            total_pages += data.get('depth', 1)
        
        logger.info(f"✅ 合并完成:")
        logger.info(f"  - 网站总数: {total_sites}")
        logger.info(f"  - 页面总数: {total_pages}")
        logger.info(f"  - 平均深度: {total_pages/total_sites:.1f} 页/站")
        logger.info(f"  - 输出文件: {output_file}")
        
        return total_sites, total_pages


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="大规模网站批量爬取")
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=Path("data/large_urls.txt"),
        help="URL列表文件"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/websites/large_scale"),
        help="输出目录"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="每批爬取的网站数量"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="爬取深度"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="每个网站最大页面数"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="只合并已有的批次文件"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/websites/large_train.jsonl"),
        help="合并后的输出文件"
    )
    
    args = parser.parse_args()
    
    # 创建爬虫
    crawler = LargeScaleCrawler(
        urls_file=args.urls_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_depth=args.depth,
        max_pages=args.max_pages
    )
    
    if args.merge:
        # 只合并文件
        crawler.merge_batches(args.output)
    else:
        # 爬取 + 合并
        await crawler.crawl_all()
        crawler.merge_batches(args.output)
    
    logger.info("\n✨ 任务完成!")


if __name__ == "__main__":
    asyncio.run(main())
