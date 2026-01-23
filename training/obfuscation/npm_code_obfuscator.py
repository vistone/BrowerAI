#!/usr/bin/env python3
"""
NPM真实代码混淆器 - 从NPM包中提取真实JavaScript代码并混淆
专注于网络获取的真实数据，而非生成数据
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NPMCodeObfuscator:
    """从NPM包中提取并混淆真实JavaScript代码"""
    
    def __init__(self):
        self.npm_dir = Path('real_data/npm_packages')
        self.output_dir = Path('real_data/obfuscated_code')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_files': 0,
            'successfully_obfuscated': 0,
            'failed': 0,
        }
    
    def obfuscate_with_terser(self, code: str) -> Optional[str]:
        """使用Terser混淆"""
        try:
            with open('/tmp/input.js', 'w', encoding='utf-8', errors='ignore') as f:
                f.write(code)
            
            result = subprocess.run(
                ['npx', 'terser', '/tmp/input.js', '--compress', '--mangle'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
            return None
        except:
            return None
    
    def obfuscate_with_uglifyjs(self, code: str) -> Optional[str]:
        """使用UglifyJS混淆"""
        try:
            with open('/tmp/input.js', 'w', encoding='utf-8', errors='ignore') as f:
                f.write(code)
            
            result = subprocess.run(
                ['npx', 'uglifyjs', '/tmp/input.js', '--compress', '--mangle'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
            return None
        except:
            return None
    
    def get_npm_js_files(self, max_files: int = 2000) -> List[Path]:
        """从NPM包中获取JavaScript文件"""
        if not self.npm_dir.exists():
            logger.error(f"NPM目录不存在: {self.npm_dir}")
            return []
        
        # 扫描所有JS/TS文件
        js_files = list(self.npm_dir.glob('**/*.js'))
        js_files.extend(self.npm_dir.glob('**/*.ts'))
        
        # 过滤：排除测试、node_modules、压缩文件
        filtered_files = [
            f for f in js_files
            if 'node_modules' not in str(f)
            and 'test' not in str(f).lower()
            and 'spec' not in str(f).lower()
            and '.min.' not in str(f)
            and 'dist' not in str(f)
            and f.stat().st_size > 200  # 至少200字节
            and f.stat().st_size < 100000  # 最大100KB
        ]
        
        logger.info(f"找到 {len(filtered_files)} 个真实NPM JavaScript文件")
        
        # 限制数量
        return filtered_files[:max_files]
    
    def process_file(self, file_path: Path) -> List[Dict]:
        """处理单个文件，生成混淆对"""
        pairs = []
        
        try:
            # 读取原始代码
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_code = f.read()
            
            if len(original_code) < 200:
                return pairs  # 跳过太小的文件
            
            # 提取包名
            rel_path = file_path.relative_to(self.npm_dir)
            package_name = str(rel_path).split('/')[0]
            
            # Terser混淆
            obfuscated_terser = self.obfuscate_with_terser(original_code)
            if obfuscated_terser and len(obfuscated_terser) > 50:
                pairs.append({
                    'original': original_code,
                    'obfuscated': obfuscated_terser,
                    'obfuscator': 'terser',
                    'source_file': str(file_path),
                    'package': package_name,
                    'original_size': len(original_code),
                    'obfuscated_size': len(obfuscated_terser),
                    'compression_ratio': len(obfuscated_terser) / len(original_code),
                })
                self.stats['successfully_obfuscated'] += 1
            
            # UglifyJS混淆
            obfuscated_uglify = self.obfuscate_with_uglifyjs(original_code)
            if obfuscated_uglify and len(obfuscated_uglify) > 50:
                pairs.append({
                    'original': original_code,
                    'obfuscated': obfuscated_uglify,
                    'obfuscator': 'uglifyjs',
                    'source_file': str(file_path),
                    'package': package_name,
                    'original_size': len(original_code),
                    'obfuscated_size': len(obfuscated_uglify),
                    'compression_ratio': len(obfuscated_uglify) / len(original_code),
                })
                self.stats['successfully_obfuscated'] += 1
        
        except Exception as e:
            self.stats['failed'] += 1
        
        return pairs
    
    def generate_training_pairs(self, max_files: int = 2000):
        """生成训练对"""
        print("\n" + "="*70)
        print("🌍 从NPM真实包中提取JavaScript代码并混淆")
        print("="*70)
        
        # 获取NPM文件
        npm_files = self.get_npm_js_files(max_files)
        self.stats['total_files'] = len(npm_files)
        
        if not npm_files:
            print("❌ 没有找到NPM JavaScript文件")
            return
        
        print(f"✅ 找到 {len(npm_files)} 个真实NPM文件\n")
        
        # 处理文件
        all_pairs = []
        for i, file_path in enumerate(npm_files, 1):
            if i % 50 == 0:
                print(f"进度: {i}/{len(npm_files)} ({self.stats['successfully_obfuscated']} 成功)")
            
            pairs = self.process_file(file_path)
            all_pairs.extend(pairs)
        
        # 保存训练对
        if all_pairs:
            output_file = self.output_dir / 'training_pairs.jsonl'
            with open(output_file, 'w', encoding='utf-8') as f:
                for pair in all_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
            print(f"\n✅ 成功生成 {len(all_pairs)} 个训练对")
            print(f"📁 保存到: {output_file}")
        else:
            print("\n❌ 没有生成任何训练对")
        
        # 统计
        print("\n" + "="*70)
        print("📊 处理统计")
        print("="*70)
        print(f"总文件数: {self.stats['total_files']}")
        print(f"成功混淆: {self.stats['successfully_obfuscated']}")
        print(f"失败: {self.stats['failed']}")
        print(f"训练对: {len(all_pairs)}")
        print("="*70)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 NPM真实代码混淆器")
    print("从网络获取的真实NPM包中提取JavaScript代码")
    print("="*70)
    
    # 检查NPM目录
    npm_dir = Path('real_data/npm_packages')
    if not npm_dir.exists():
        print(f"\n❌ NPM目录不存在: {npm_dir}")
        print("   请先运行: python3 training/npm_package_crawler.py")
        return
    
    # 创建混淆器
    obfuscator = NPMCodeObfuscator()
    
    # 生成训练对（处理2000个文件）
    obfuscator.generate_training_pairs(max_files=2000)


if __name__ == '__main__':
    main()
