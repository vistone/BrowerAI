#!/usr/bin/env python3
"""
高性能NPM代码混淆器
- 并行处理文件
- 智能缓存
- 动态选择混淆引擎
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastNpmObfuscator:
    """高性能NPM代码混淆器"""
    
    def __init__(self, npm_dir: Path = Path('real_data/npm_packages'),
                 output_file: Path = Path('real_data/obfuscated_code/training_pairs.jsonl'),
                 max_files: int = 5000, max_workers: int = 4,
                 multiplier: int = 1):
        self.npm_dir = npm_dir
        self.output_file = output_file
        self.max_files = max_files
        self.max_workers = max_workers
        self.multiplier = max(1, multiplier)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 验证工具
        self.has_terser = self._check_tool('terser')
        self.has_uglify = self._check_tool('uglifyjs')
        
        if not self.has_terser and not self.has_uglify:
            logger.error("❌ 未安装混淆工具！请运行:")
            logger.error("   npm install -g terser uglify-js")
            raise RuntimeError("混淆工具不可用")
        
        logger.info(f"✅ Terser: {'可用' if self.has_terser else '不可用'}")
        logger.info(f"✅ UglifyJS: {'可用' if self.has_uglify else '不可用'}")
    
    def _check_tool(self, tool_name: str) -> bool:
        """检查工具是否已安装"""
        try:
            result = subprocess.run(
                f"{tool_name} --version",
                shell=True,
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    def get_npm_js_files(self) -> List[Path]:
        """获取NPM包中的JS文件"""
        logger.info("🔍 扫描NPM包...")
        
        js_files = []
        exclude_patterns = {
            'node_modules', 'test', 'spec', '.min.js', 
            'dist', 'build', '.d.ts', 'lib/'
        }
        
        for js_file in self.npm_dir.rglob('*.js'):
            # 跳过排除目录
            if any(pat in str(js_file) for pat in exclude_patterns):
                continue
            
            # 检查文件大小
            size = js_file.stat().st_size
            if size < 200 or size > 100000:  # 200B ~ 100KB
                continue
            
            js_files.append(js_file)
            if len(js_files) >= self.max_files:
                break
        
        logger.info(f"✅ 找到 {len(js_files)} 个JavaScript文件")
        return js_files
    
    def obfuscate_with_terser(self, code: str) -> Optional[str]:
        """使用Terser混淆"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    f"terser {temp_file} -c -m --output /tmp/terser_out.js",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    with open('/tmp/terser_out.js', 'r', encoding='utf-8', errors='ignore') as f:
                        output = f.read().strip()
                    if output:
                        return output
            finally:
                os.unlink(temp_file)
                if os.path.exists('/tmp/terser_out.js'):
                    os.unlink('/tmp/terser_out.js')
        except:
            pass
        
        return None
    
    def obfuscate_with_uglify(self, code: str) -> Optional[str]:
        """使用UglifyJS混淆"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    f"uglifyjs {temp_file} -c -m -o /tmp/uglify_out.js",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    with open('/tmp/uglify_out.js', 'r', encoding='utf-8', errors='ignore') as f:
                        output = f.read().strip()
                    if output:
                        return output
            finally:
                os.unlink(temp_file)
                if os.path.exists('/tmp/uglify_out.js'):
                    os.unlink('/tmp/uglify_out.js')
        except:
            pass
        
        return None
    
    def process_file(self, file_path: Path) -> List[Dict]:
        """处理单个文件"""
        pairs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_code = f.read()
            
            if len(original_code) < 200:
                return pairs
            
            original_size = len(original_code)
            package_name = str(file_path.relative_to(self.npm_dir)).split('/')[0]
            
            for variant in range(self.multiplier):
                # 尝试Terser
                if self.has_terser:
                    obfuscated_terser = self.obfuscate_with_terser(original_code)
                    if obfuscated_terser and len(obfuscated_terser) > 50:
                        pairs.append({
                            'original': original_code,
                            'obfuscated': obfuscated_terser,
                            'obfuscator': 'terser',
                            'variant': variant,
                            'package': package_name,
                            'original_size': original_size,
                            'obfuscated_size': len(obfuscated_terser),
                            'compression_ratio': len(obfuscated_terser) / original_size
                        })
                
                # 尝试UglifyJS
                if self.has_uglify:
                    obfuscated_uglify = self.obfuscate_with_uglify(original_code)
                    if obfuscated_uglify and len(obfuscated_uglify) > 50:
                        pairs.append({
                            'original': original_code,
                            'obfuscated': obfuscated_uglify,
                            'obfuscator': 'uglifyjs',
                            'variant': variant,
                            'package': package_name,
                            'original_size': original_size,
                            'obfuscated_size': len(obfuscated_uglify),
                            'compression_ratio': len(obfuscated_uglify) / original_size
                        })
        
        except Exception as e:
            logger.debug(f"⚠️  处理失败: {file_path} - {e}")
        
        return pairs
    
    def generate_training_pairs(self):
        """生成训练对(并行)"""
        js_files = self.get_npm_js_files()
        
        logger.info(f"📝 使用 {self.max_workers} 个工作线程处理...")
        
        total_pairs = 0
        processed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_file, js_file): js_file 
                for js_file in js_files
            }
            
            with open(self.output_file, 'w') as output:
                for future in as_completed(futures):
                    js_file = futures[future]
                    processed += 1
                    
                    try:
                        pairs = future.result()
                        total_pairs += len(pairs)
                        
                        for pair in pairs:
                            output.write(json.dumps(pair) + '\n')
                        
                        if processed % 50 == 0:
                            logger.info(f"✅ 已处理 {processed}/{len(js_files)} 文件, "
                                       f"生成 {total_pairs} 个训练对")
                    
                    except Exception as e:
                        logger.error(f"❌ 处理失败: {js_file} - {e}")
        
        logger.info(f"✅ 完成! 生成 {total_pairs} 个训练对")
        logger.info(f"📁 输出文件: {self.output_file}")
        
        # 验证
        with open(self.output_file) as f:
            verify_count = sum(1 for _ in f)
        
        logger.info(f"✓ 验证: {verify_count} 行")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='高性能NPM代码混淆器')
    parser.add_argument('--npm-dir', type=Path, default=Path('real_data/npm_packages'))
    parser.add_argument('--output', type=Path, default=Path('real_data/obfuscated_code/training_pairs.jsonl'))
    parser.add_argument('--max-files', type=int, default=5000)
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--multiplier', type=int, default=1)
    return parser.parse_args()


def main():
    print("\n" + "="*70)
    print("🚀 高性能NPM代码混淆器")
    print("="*70)
    
    try:
        args = parse_args()
        obfuscator = FastNpmObfuscator(
            npm_dir=args.npm_dir,
            output_file=args.output,
            max_files=args.max_files,
            max_workers=args.max_workers,
            multiplier=args.multiplier
        )
        obfuscator.generate_training_pairs()
        
        print("\n" + "="*70)
        print("✅ 成功! 数据已准备好进行模型训练")
        print("="*70)
    
    except KeyboardInterrupt:
        print("\n❌ 被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
