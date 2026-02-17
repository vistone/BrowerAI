#!/usr/bin/env python3
"""
🕷️ 真实混淆JS样本爬虫

从真实网站收集混淆的JavaScript代码用于训练
"""

import requests
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import logging
from bs4 import BeautifulSoup
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ObfuscatedJSCrawler:
    """混淆JS爬虫"""
    
    def __init__(self, output_dir: str = "data/obfuscated_js"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.collected_samples = []
        
        logger.info(f"✓ 爬虫初始化")
        logger.info(f"  输出目录: {self.output_dir}")
    
    def is_obfuscated(self, js_code: str) -> Dict[str, Any]:
        """检测JS是否被混淆"""
        
        # 混淆特征
        features = {
            'hex_strings': len(re.findall(r"'\\x[0-9a-fA-F]{2}", js_code)),
            'unicode_strings': len(re.findall(r'\\u[0-9a-fA-F]{4}', js_code)),
            '_0x_vars': len(re.findall(r'_0x[a-f0-9]{4}', js_code)),
            'string_fromcharcode': len(re.findall(r'String\.fromCharCode', js_code)),
            'eval_calls': len(re.findall(r'\beval\s*\(', js_code)),
            'atob_calls': len(re.findall(r'\batob\s*\(', js_code)),
            'debugger_statements': len(re.findall(r'\bdebugger\b', js_code)),
            'jsfuck_pattern': bool(re.search(r'^\[[\!\+\[\]]{10,}', js_code)),
            'aaencode_pattern': bool(re.search(r'ﾟωﾟ|ﾟДﾟ', js_code)),
            'sojson_pattern': bool(re.search(r"sojson\.v\d+|'\|'\.split", js_code)),
            'packer_pattern': bool(re.search(r'eval\(function\(p,a,c,k,e,d\)', js_code)),
        }
        
        # 计算混淆分数
        score = sum([
            features['hex_strings'] * 2,
            features['unicode_strings'] * 2,
            features['_0x_vars'] * 3,
            features['string_fromcharcode'] * 2,
            features['eval_calls'] * 3,
            features['atob_calls'] * 2,
            features['debugger_statements'] * 2,
            features['jsfuck_pattern'] * 10,
            features['aaencode_pattern'] * 10,
            features['sojson_pattern'] * 10,
            features['packer_pattern'] * 10,
        ])
        
        # 检测混淆类型
        obfuscation_types = []
        if features['hex_strings'] > 5 or features['unicode_strings'] > 5:
            obfuscation_types.append('string_encoding')
        if features['_0x_vars'] > 3:
            obfuscation_types.append('javascript-obfuscator')
        if features['jsfuck_pattern']:
            obfuscation_types.append('jsfuck')
        if features['aaencode_pattern']:
            obfuscation_types.append('aaencode')
        if features['sojson_pattern']:
            obfuscation_types.append('sojson')
        if features['packer_pattern']:
            obfuscation_types.append('packer')
        
        is_obf = score > 5 or len(obfuscation_types) > 0
        
        return {
            'is_obfuscated': is_obf,
            'score': score,
            'features': features,
            'obfuscation_types': obfuscation_types,
        }
    
    def crawl_website(self, url: str) -> List[Dict[str, Any]]:
        """爬取单个网站的JS文件"""
        
        logger.info(f"\n🕷️ 爬取网站: {url}")
        samples = []
        
        try:
            # 获取HTML
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 1. 提取内联JS
            inline_scripts = soup.find_all('script', src=False)
            logger.info(f"  找到 {len(inline_scripts)} 个内联脚本")
            
            for i, script in enumerate(inline_scripts):
                if script.string:
                    js_code = script.string.strip()
                    if len(js_code) > 100:  # 过滤太短的代码
                        analysis = self.is_obfuscated(js_code)
                        
                        if analysis['is_obfuscated']:
                            sample = {
                                'url': url,
                                'type': 'inline',
                                'index': i,
                                'code': js_code,
                                'hash': hashlib.md5(js_code.encode()).hexdigest(),
                                **analysis
                            }
                            samples.append(sample)
                            logger.info(f"    ✅ 内联脚本 #{i} - 分数:{analysis['score']}, 类型:{analysis['obfuscation_types']}")
            
            # 2. 提取外部JS文件
            external_scripts = soup.find_all('script', src=True)
            logger.info(f"  找到 {len(external_scripts)} 个外部脚本")
            
            for i, script in enumerate(external_scripts[:10]):  # 限制最多10个
                js_url = urljoin(url, script['src'])
                
                try:
                    js_response = self.session.get(js_url, timeout=10)
                    js_response.raise_for_status()
                    js_code = js_response.text
                    
                    if len(js_code) > 100:
                        analysis = self.is_obfuscated(js_code)
                        
                        if analysis['is_obfuscated']:
                            sample = {
                                'url': url,
                                'type': 'external',
                                'src': js_url,
                                'code': js_code,
                                'hash': hashlib.md5(js_code.encode()).hexdigest(),
                                **analysis
                            }
                            samples.append(sample)
                            logger.info(f"    ✅ 外部脚本: {js_url[:60]}... - 分数:{analysis['score']}")
                    
                    time.sleep(0.5)  # 礼貌延时
                
                except Exception as e:
                    logger.warning(f"    ⚠ 无法获取 {js_url}: {e}")
        
        except Exception as e:
            logger.error(f"  ❌ 爬取失败: {e}")
        
        return samples
    
    def crawl_multiple_websites(self, urls: List[str]) -> None:
        """爬取多个网站"""
        
        logger.info("="*80)
        logger.info(f"🕷️ 开始批量爬取 ({len(urls)} 个网站)")
        logger.info("="*80)
        
        all_samples = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"\n[{i}/{len(urls)}] 处理: {url}")
            
            samples = self.crawl_website(url)
            all_samples.extend(samples)
            self.collected_samples.extend(samples)
            
            logger.info(f"  本站收集: {len(samples)} 个混淆样本")
            
            time.sleep(1)  # 站点间延时
        
        # 保存结果
        self.save_samples(all_samples)
        
        # 统计
        logger.info("\n" + "="*80)
        logger.info("✅ 爬取完成")
        logger.info("="*80)
        logger.info(f"\n总计:")
        logger.info(f"  网站数: {len(urls)}")
        logger.info(f"  混淆样本: {len(all_samples)}")
        logger.info(f"  平均每站: {len(all_samples) / len(urls):.1f}")
        
        # 按类型统计
        type_counts = {}
        for sample in all_samples:
            for obf_type in sample.get('obfuscation_types', []):
                type_counts[obf_type] = type_counts.get(obf_type, 0) + 1
        
        logger.info(f"\n混淆类型分布:")
        for obf_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {obf_type}: {count}")
    
    def save_samples(self, samples: List[Dict[str, Any]]) -> None:
        """保存样本到文件"""
        
        # JSONL格式
        output_file = self.output_dir / 'obfuscated_samples.jsonl'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                # 移除代码内容,只保存元数据(代码太大)
                meta = {
                    'url': sample['url'],
                    'type': sample['type'],
                    'hash': sample['hash'],
                    'score': sample['score'],
                    'features': sample['features'],
                    'obfuscation_types': sample['obfuscation_types'],
                    'code_length': len(sample['code']),
                }
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        
        logger.info(f"\n💾 元数据已保存: {output_file}")
        
        # 保存完整样本(单独文件)
        for i, sample in enumerate(samples):
            sample_file = self.output_dir / f"sample_{sample['hash'][:8]}.js"
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(sample['code'])
        
        logger.info(f"💾 代码文件已保存: {len(samples)} 个")


def main():
    """主函数"""
    
    # 创建爬虫
    crawler = ObfuscatedJSCrawler()
    
    # 测试网站列表(替换为真实网站)
    test_urls = [
        # 'https://example.com',  # 示例
        # 注意: 实际使用时需要遵守robots.txt和网站使用条款
    ]
    
    if not test_urls:
        logger.warning("⚠ 未配置爬取URL,使用演示数据")
        
        # 演示数据
        demo_samples = [
            {
                'url': 'https://example-demo.com',
                'type': 'inline',
                'code': r"var _0x1234 = '\x48\x65\x6c\x6c\x6f'; console['log'](_0x1234);",
                'hash': 'demo001',
                'is_obfuscated': True,
                'score': 15,
                'features': {},
                'obfuscation_types': ['javascript-obfuscator', 'string_encoding'],
            },
            {
                'url': 'https://example-demo.com',
                'type': 'external',
                'code': r"var msg = '\u0048\u0065\u006c\u006c\u006f';",
                'hash': 'demo002',
                'is_obfuscated': True,
                'score': 12,
                'features': {},
                'obfuscation_types': ['string_encoding'],
            },
        ]
        
        crawler.save_samples(demo_samples)
        
        logger.info("\n📊 演示模式:")
        logger.info(f"  生成样本: {len(demo_samples)}")
        logger.info(f"  输出目录: {crawler.output_dir}")
    else:
        crawler.crawl_multiple_websites(test_urls)
    
    # 使用说明
    logger.info("\n📖 使用说明:")
    logger.info("  1. 在test_urls列表中添加目标网站")
    logger.info("  2. 确保遵守网站robots.txt和使用条款")
    logger.info("  3. 爬虫会自动检测和收集混淆JS")
    logger.info("  4. 结果保存在 data/obfuscated_js/ 目录")
    
    logger.info("\n🎯 下一步:")
    logger.info("  1. 人工审核收集的样本")
    logger.info("  2. 构建混淆-原始代码配对数据集")
    logger.info("  3. 用于训练反混淆模型")


if __name__ == '__main__':
    main()
