#!/usr/bin/env python3
"""
纯Python代码混淆器 - 不依赖外部工具
使用正则表达式进行变量名混淆和代码压缩
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging
import random
import string

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PythonObfuscator:
    """纯Python代码混淆器"""
    
    def __init__(self, npm_dir: Path = Path('real_data/npm_packages'),
                 output_file: Path = Path('real_data/obfuscated_code/training_pairs.jsonl'),
                 max_files: int = 2000):
        self.npm_dir = npm_dir
        self.output_file = output_file
        self.max_files = max_files
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    def simple_obfuscate(self, code: str) -> str:
        """简单的代码混淆 - 变量名替换 + 空白移除"""
        try:
            # 1. 移除注释
            code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
            
            # 2. 变量名混淆 - 替换 const x = 为 const a = 
            var_map = {}
            var_counter = 0
            
            def replace_var(match):
                nonlocal var_counter
                var_name = match.group(1)
                if var_name not in var_map:
                    # 生成混淆名 a, b, c, ... z, aa, ab ...
                    if var_counter < 26:
                        var_map[var_name] = chr(97 + var_counter)
                    else:
                        var_map[var_name] = f"_{var_counter}"
                    var_counter += 1
                return f"const {var_map[var_name]}"
            
            code = re.sub(r'\bconst\s+(\w+)\s*=', replace_var, code)
            code = re.sub(r'\blet\s+(\w+)\s*=', replace_var, code)
            code = re.sub(r'\bvar\s+(\w+)\s*=', replace_var, code)
            
            # 3. 用映射表替换变量引用
            for old, new in var_map.items():
                code = re.sub(rf'\b{old}\b', new, code)
            
            # 4. 移除多余空白
            code = re.sub(r'\s+', ' ', code)
            code = code.strip()
            
            return code
        except:
            return code
    
    def alternative_obfuscate(self, code: str) -> str:
        """替代混淆方法 - 函数名混淆"""
        try:
            # 1. 移除注释
            code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
            
            # 2. 函数名混淆
            func_map = {}
            func_counter = 0
            
            def replace_func(match):
                nonlocal func_counter
                func_name = match.group(1)
                if func_name not in func_map:
                    func_map[func_name] = f"f{func_counter}"
                    func_counter += 1
                return f"function {func_map[func_name]}"
            
            code = re.sub(r'\bfunction\s+(\w+)\s*\(', replace_func, code)
            
            # 3. 替换函数调用
            for old, new in func_map.items():
                code = re.sub(rf'\b{old}\s*\(', f'{new}(', code)
            
            # 4. 数字转16进制
            code = re.sub(r'\b(\d+)\b', lambda m: hex(int(m.group(1))), code)
            
            # 5. 移除空白
            code = re.sub(r'\s+', ' ', code)
            
            return code
        except:
            return code
    
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
            
            # 第一种混淆方式
            obfuscated1 = self.simple_obfuscate(original_code)
            if obfuscated1 and len(obfuscated1) > 50 and obfuscated1 != original_code:
                pairs.append({
                    'original': original_code,
                    'obfuscated': obfuscated1,
                    'obfuscator': 'python_simple',
                    'package': package_name,
                    'original_size': original_size,
                    'obfuscated_size': len(obfuscated1),
                    'compression_ratio': len(obfuscated1) / original_size
                })
            
            # 第二种混淆方式
            obfuscated2 = self.alternative_obfuscate(original_code)
            if obfuscated2 and len(obfuscated2) > 50 and obfuscated2 != original_code:
                pairs.append({
                    'original': original_code,
                    'obfuscated': obfuscated2,
                    'obfuscator': 'python_alternative',
                    'package': package_name,
                    'original_size': original_size,
                    'obfuscated_size': len(obfuscated2),
                    'compression_ratio': len(obfuscated2) / original_size
                })
        
        except Exception as e:
            logger.debug(f"⚠️  处理失败: {file_path} - {e}")
        
        return pairs
    
    def generate_training_pairs(self):
        """生成训练对"""
        js_files = self.get_npm_js_files()
        
        logger.info(f"📝 处理文件 (单线程)...")
        
        total_pairs = 0
        processed = 0
        
        with open(self.output_file, 'w') as output:
            for js_file in js_files:
                processed += 1
                
                try:
                    pairs = self.process_file(js_file)
                    total_pairs += len(pairs)
                    
                    for pair in pairs:
                        output.write(json.dumps(pair, ensure_ascii=False) + '\n')
                    
                    if processed % 100 == 0:
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


def main():
    print("\n" + "="*70)
    print("🚀 纯Python代码混淆器 (无依赖)")
    print("="*70)
    
    try:
        obfuscator = PythonObfuscator(max_files=2000)
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
