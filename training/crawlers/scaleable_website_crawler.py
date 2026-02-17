#!/usr/bin/env python3
"""
可扩展网站爬取器 - 目标500+真实网站
来源: Umbrella Top-1M 公共榜单
输出: training/real_data/scaleable/scaleable_websites.jsonl
"""

import argparse
import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import sys
import io
import zipfile

import aiohttp

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from training.detectors.high_precision_detector import HighPrecisionDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


TOP_SITES_URLS = [
    "https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip",
    "https://raw.githubusercontent.com/zakird/crux-top-lists/main/data/global/current.csv",
    "https://raw.githubusercontent.com/zakird/crux-top-lists/master/data/global/current.csv",
    "https://raw.githubusercontent.com/umami-software/top-websites/master/top-websites.csv",
]


class ScaleableWebsiteCrawler:
    """从公开榜单批量爬取真实网站"""

    def __init__(
        self,
        output_dir: Path = Path("training/real_data/scaleable"),
        list_dir: Path = Path("training/real_data/top_sites"),
        sample_size: int = 2000,
        target_success: int = 500,
        max_workers: int = 12,
        timeout: int = 12,
        min_html: int = 800,
        confidence_threshold: float = 0.6,
    ):
        self.output_dir = output_dir
        self.list_dir = list_dir
        self.sample_size = sample_size
        self.target_success = target_success
        self.max_workers = max_workers
        self.timeout = timeout
        self.min_html = min_html
        self.confidence_threshold = confidence_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.list_dir.mkdir(parents=True, exist_ok=True)

        self.list_file = self.list_dir / "top-1m.csv"
        self.data_file = self.output_dir / "scaleable_websites.jsonl"
        self.failed_file = self.output_dir / "failed_urls.txt"

        self.detector = HighPrecisionDetector()

    def load_existing_domains(self) -> Tuple[set, int]:
        """读取已有数据以支持断点续爬"""
        if not self.data_file.exists():
            return set(), 0

        existing = set()
        count = 0
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        domain = record.get("domain") or record.get("url", "").split("//", 1)[-1].split("/", 1)[0]
                        if domain:
                            existing.add(domain)
                            count += 1
                    except Exception:
                        continue
        except Exception:
            return set(), 0

        return existing, count

    async def download_top_list(self, session: aiohttp.ClientSession) -> None:
        """下载榜单文件"""
        if self.list_file.exists() and self.list_file.stat().st_size > 1024:
            logger.info("✅ 榜单已存在，跳过下载")
            return

        logger.info("⬇️  下载 Top-1M 榜单...")
        last_error = None
        for url in TOP_SITES_URLS:
            for attempt in range(1, 4):
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as resp:
                        resp.raise_for_status()
                        content = await resp.read()

                    if url.endswith(".zip") or content[:2] == b"PK":
                        with zipfile.ZipFile(io.BytesIO(content)) as zf:
                            csv_candidates = [n for n in zf.namelist() if n.endswith(".csv")]
                            if not csv_candidates:
                                raise RuntimeError("zip中未找到CSV")
                            with zf.open(csv_candidates[0]) as csv_file:
                                self.list_file.write_bytes(csv_file.read())
                    else:
                        self.list_file.write_bytes(content)

                    logger.info(f"✅ 下载成功: {url}")
                    logger.info(f"✅ 保存榜单: {self.list_file}")
                    return
                except Exception as exc:
                    last_error = exc
                    logger.warning(f"⚠️  下载失败({attempt}/3): {type(exc).__name__}")

        if self.list_file.exists() and self.list_file.stat().st_size > 1024:
            logger.info("✅ 使用已存在榜单文件")
            return

        raise RuntimeError(f"下载榜单失败: {last_error}")

    def load_domains(self) -> List[str]:
        """读取榜单域名"""
        domains = []
        with open(self.list_file, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    domains.append(row[1].strip())
                if len(domains) >= self.sample_size:
                    break
        return domains

    async def fetch_html(self, session: aiohttp.ClientSession, domain: str) -> Tuple[str, Optional[str]]:
        """尝试https/http获取HTML"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        for scheme in ("https", "http"):
            url = f"{scheme}://{domain}"
            try:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    allow_redirects=True,
                    ssl=False,
                ) as resp:
                    if resp.status != 200:
                        continue
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                        continue
                    html = await resp.text(errors="ignore")
                    return url, html
            except Exception:
                continue

        return f"https://{domain}", None

    def classify_html(self, html: str) -> Tuple[str, float]:
        """使用高精度检测器自动标注"""
        framework, confidence = self.detector.detect_with_category(html, category=None)
        if confidence < self.confidence_threshold:
            return "Unknown", confidence
        return framework, confidence

    async def crawl(self) -> None:
        """主爬取流程"""
        connector = aiohttp.TCPConnector(limit=self.max_workers, ssl=False)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            await self.download_top_list(session)
            domains = self.load_domains()

            existing_domains, existing_success = self.load_existing_domains()
            if existing_domains:
                domains = [d for d in domains if d not in existing_domains]
                logger.info(f"✅ 已有数据 {existing_success} 条，跳过 {len(existing_domains)} 个域名")

            if existing_success >= self.target_success:
                logger.info("✅ 已达到目标成功数量，直接结束")
                return

            logger.info(f"🌐 计划爬取 {len(domains)} 个网站...")
            semaphore = asyncio.Semaphore(self.max_workers)

            async def bound_fetch(domain: str):
                async with semaphore:
                    return await self.fetch_html(session, domain)

            tasks = [asyncio.create_task(bound_fetch(domain)) for domain in domains]

            success = existing_success
            failed = 0

            write_mode = "a" if self.data_file.exists() else "w"
            fail_mode = "a" if self.failed_file.exists() else "w"

            with open(self.data_file, write_mode, encoding="utf-8") as out, open(
                self.failed_file, fail_mode, encoding="utf-8"
            ) as fail:
                for i, task in enumerate(asyncio.as_completed(tasks), 1):
                    url, html = await task

                    if not html or len(html) < self.min_html:
                        failed += 1
                        fail.write(url + "\n")
                        continue

                    category, confidence = self.classify_html(html)

                    record = {
                        "url": url,
                        "domain": url.split("//", 1)[-1].split("/", 1)[0],
                        "html": html[:50000],
                        "html_size": len(html),
                        "category": category,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat(),
                    }

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    success += 1

                    if i % 50 == 0:
                        logger.info(f"✅ 进度 {i}/{len(domains)} | 成功 {success} | 失败 {failed}")

                    if success >= self.target_success:
                        logger.info("✅ 已达到目标成功数量，提前结束")
                        for pending in tasks:
                            if not pending.done():
                                pending.cancel()
                        break

            logger.info("\n🎉 爬取完成!")
            logger.info(f"  成功: {success}")
            logger.info(f"  失败: {failed}")
            logger.info(f"  输出: {self.data_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaleable website crawler")
    parser.add_argument("--size", type=int, default=2000, help="最大域名数量 (默认2000)")
    parser.add_argument("--target-success", type=int, default=500, help="目标成功数量 (默认500)")
    parser.add_argument("--workers", type=int, default=12, help="并发数 (默认12)")
    parser.add_argument("--timeout", type=int, default=12, help="超时秒数 (默认12)")
    parser.add_argument("--min-html", type=int, default=800, help="最小HTML长度 (默认800)")
    parser.add_argument("--confidence", type=float, default=0.6, help="置信度阈值 (默认0.6)")
    return parser.parse_args()


def main():
    args = parse_args()
    crawler = ScaleableWebsiteCrawler(
        sample_size=args.size,
        target_success=args.target_success,
        max_workers=args.workers,
        timeout=args.timeout,
        min_html=args.min_html,
        confidence_threshold=args.confidence,
    )
    asyncio.run(crawler.crawl())


if __name__ == "__main__":
    main()
