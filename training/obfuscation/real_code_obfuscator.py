#!/usr/bin/env python3
"""
真实代码混淆生成器
使用真实的JavaScript混淆工具混淆源代码
生成用于GPU训练的真实混淆/反混淆对
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

class RealCodeObfuscator:
    """使用真实混淆工具"""
    
    def __init__(self):
        self.source_dir = Path('real_data/github_frameworks')
        self.output_dir = Path('real_data/obfuscated_code')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_pairs = []
        self.stats = {
            'total_files': 0,
            'successfully_obfuscated': 0,
            'failed': 0,
            'total_size': 0,
        }
    
    def obfuscate_with_terser(self, code: str) -> Optional[str]:
        """使用Terser混淆"""
        try:
            # 创建临时文件
            with open('/tmp/input.js', 'w') as f:
                f.write(code)
            
            # 执行Terser
            result = subprocess.run(
                ['npx', 'terser', '/tmp/input.js', '--compress', '--mangle'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                print(f"    ⚠️  Terser错误: {result.stderr[:100]}")
                return None
        except Exception as e:
            print(f"    ⚠️  异常: {str(e)}")
            return None
    
    def obfuscate_with_uglifyjs(self, code: str) -> Optional[str]:
        """使用UglifyJS混淆"""
        try:
            with open('/tmp/input.js', 'w') as f:
                f.write(code)
            
            result = subprocess.run(
                ['npx', 'uglify-js', '/tmp/input.js', '--compress', '--mangle'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return None
        except:
            return None
    
    def process_javascript_file(self, file_path: Path) -> List[Dict]:
        """处理单个JS文件"""
        pairs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_code = f.read()
            
            if len(original_code) < 50 or len(original_code) > 200000:
                return pairs  # 跳过太小或太大的文件（放宽限制）
            
            print(f"  📄 处理: {file_path.name} ({len(original_code)} bytes)")
            
            # 用Terser混淆
            obfuscated_terser = self.obfuscate_with_terser(original_code)
            if obfuscated_terser:
                pairs.append({
                    'original': original_code,
                    'obfuscated': obfuscated_terser,
                    'obfuscator': 'terser',
                    'source_file': str(file_path),
                    'original_size': len(original_code),
                    'obfuscated_size': len(obfuscated_terser),
                    'compression_ratio': len(obfuscated_terser) / len(original_code),
                })
                self.stats['successfully_obfuscated'] += 1
                print(f"    ✅ Terser混淆成功")
            
            # 用UglifyJS混淆
            obfuscated_uglify = self.obfuscate_with_uglifyjs(original_code)
            if obfuscated_uglify:
                pairs.append({
                    'original': original_code,
                    'obfuscated': obfuscated_uglify,
                    'obfuscator': 'uglifyjs',
                    'source_file': str(file_path),
                    'original_size': len(original_code),
                    'obfuscated_size': len(obfuscated_uglify),
                    'compression_ratio': len(obfuscated_uglify) / len(original_code),
                })
                self.stats['successfully_obfuscated'] += 1
                print(f"    ✅ UglifyJS混淆成功")
        
        except Exception as e:
            print(f"    ❌ 错误: {str(e)}")
            self.stats['failed'] += 1
        
        return pairs
    
    def generate_obfuscation_pairs(self) -> List[Dict]:
        """生成所有混淆对"""
        print("\n" + "="*70)
        print("🔄 生成真实混淆代码对")
        print("="*70)
        
        if not self.source_dir.exists():
            print(f"❌ 源目录不存在: {self.source_dir}")
            print("   请先运行: python3 training/github_framework_crawler.py")
            return []
        
        # 扫描所有JS/TS文件
        js_files = list(self.source_dir.glob('**/*.js')) + list(self.source_dir.glob('**/*.ts'))
        print(f"找到 {len(js_files)} 个JavaScript文件\n")
        
        self.stats['total_files'] = len(js_files)
        
        for js_file in js_files:
            print(f"🔄 处理: {js_file.relative_to(self.source_dir)}")
            pairs = self.process_javascript_file(js_file)
            self.training_pairs.extend(pairs)
        
        return self.training_pairs
    
    def save_training_data(self):
        """保存训练数据"""
        if not self.training_pairs:
            print("❌ 没有生成任何训练对!")
            return
        
        print(f"\n📝 保存 {len(self.training_pairs)} 个训练对...")
        
        # 保存为JSONL格式 (方便大文件处理)
        training_file = self.output_dir / 'training_pairs.jsonl'
        with open(training_file, 'w') as f:
            for pair in self.training_pairs:
                # 为了节省空间,可以只保存hash和指针
                record = {
                    'obfuscator': pair['obfuscator'],
                    'source_file': pair['source_file'],
                    'original_size': pair['original_size'],
                    'obfuscated_size': pair['obfuscated_size'],
                    'compression_ratio': pair['compression_ratio'],
                    'original_hash': hashlib.sha256(pair['original'].encode()).hexdigest()[:16],
                    'obfuscated_hash': hashlib.sha256(pair['obfuscated'].encode()).hexdigest()[:16],
                }
                f.write(json.dumps(record) + '\n')
        
        print(f"✅ 元数据保存到: {training_file}")
        
        # 也保存原始对用于小规模训练
        if len(self.training_pairs) <= 100:
            full_data_file = self.output_dir / 'training_pairs_full.json'
            with open(full_data_file, 'w') as f:
                json.dump(self.training_pairs, f, indent=2)
            print(f"✅ 完整数据保存到: {full_data_file}")
        
        # 保存统计信息
        stats_file = self.output_dir / 'statistics.json'
        stats = {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'total_pairs': len(self.training_pairs),
            'processing_stats': self.stats,
            'obfuscators': list(set(p['obfuscator'] for p in self.training_pairs)),
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 打印总结
        print("\n" + "="*70)
        print("📊 混淆代码生成总结")
        print("="*70)
        print(f"✅ 总对数: {len(self.training_pairs)}")
        print(f"✅ 成功混淆: {self.stats['successfully_obfuscated']}")
        print(f"❌ 失败: {self.stats['failed']}")
        print(f"📂 输出目录: {self.output_dir}")
        print("="*70)


def check_dependencies():
    """检查必需的工具"""
    print("🔍 检查依赖...")
    
    required_tools = ['node', 'npm', 'npx']
    missing = []
    
    for tool in required_tools:
        try:
            result = subprocess.run(
                [tool, '--version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"  ✅ {tool}: 已安装")
            else:
                missing.append(tool)
        except:
            missing.append(tool)
    
    if missing:
        print(f"\n❌ 缺少工具: {', '.join(missing)}")
        print("\n安装步骤:")
        print("  # Ubuntu/Debian")
        print("  curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -")
        print("  sudo apt-get install -y nodejs")
        print("\n  # 然后安装混淆工具")
        print("  npm install -g terser uglify-js")
        return False
    
    # 检查混淆工具
    print("\n🔍 检查JavaScript混淆工具...")
    tools_ok = True
    
    for tool in ['terser', 'uglify-js']:
        try:
            result = subprocess.run(
                ['npx', tool, '--version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"  ✅ {tool}: {result.stdout.decode().strip()}")
            else:
                tools_ok = False
        except:
            tools_ok = False
    
    if not tools_ok:
        print("\n⚠️  某些混淆工具未正确安装,自动安装中...")
        subprocess.run(['npm', 'install', '-g', 'terser', 'uglify-js'], check=False)
    
    return True


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🌍 真实JavaScript混淆代码生成器")
    print("="*70)
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败,请先安装Node.js和混淆工具")
        return
    
    # 生成混淆对
    obfuscator = RealCodeObfuscator()
    obfuscator.generate_obfuscation_pairs()
    obfuscator.save_training_data()


if __name__ == '__main__':
    main()
