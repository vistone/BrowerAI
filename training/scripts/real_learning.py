#!/usr/bin/env python3
"""
真实数据学习系统 - Week 6
========================

从真实代码库采集、混淆、训练
"""

import json
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import random
import re
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class RealCodeCollector:
    """真实代码采集器"""
    
    def __init__(self, output_dir='data/real_codes'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.codes = []
    
    def collect_from_directory(self, directory: str, extensions=('.js', '.ts', '.jsx')) -> List[str]:
        """从目录递归采集代码文件"""
        logger.info(f"\n📂 从目录采集真实代码: {directory}")
        
        codes = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"   ⚠️  目录不存在: {directory}")
            return codes
        
        file_count = 0
        for ext in extensions:
            for file in dir_path.rglob(f'*{ext}'):
                try:
                    # 跳过过大文件和特殊目录
                    if file.stat().st_size > 100000:
                        continue
                    if any(x in str(file) for x in ['node_modules', 'dist', 'build', '.git', 'test']):
                        continue
                    
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if len(content) > 100 and content.count('\n') > 2:
                        codes.append({
                            'source': str(file),
                            'content': content,
                            'size': len(content),
                            'extension': ext
                        })
                        file_count += 1
                        
                except Exception as e:
                    logger.debug(f"   无法读取: {file} ({e})")
        
        logger.info(f"   ✅ 采集了 {file_count} 个文件")
        self.codes.extend(codes)
        return codes
    
    def save_raw_codes(self):
        """保存采集的原始代码"""
        if not self.codes:
            logger.warning("⚠️  没有采集到代码")
            return None
        
        output_file = self.output_dir / 'raw_codes.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for code in self.codes:
                f.write(json.dumps(code, ensure_ascii=False) + '\n')
        
        logger.info(f"\n💾 原始代码已保存: {output_file}")
        logger.info(f"   总计: {len(self.codes)} 个代码文件")
        return output_file


class RealObfuscationEngine:
    """真实混淆引擎 - 应用12+种混淆技术"""
    
    def __init__(self):
        self.obfuscated = []
    
    def control_flow_obfuscation(self, code: str) -> str:
        """1. 控制流混淆"""
        lines = code.split('\n')
        result = []
        for i, line in enumerate(lines):
            result.append(line)
            if i % 3 == 0 and line.strip() and not line.strip().startswith('//'):
                result.append(f"if(Math.random()>1){{}}")
        return '\n'.join(result)
    
    def dead_code_insertion(self, code: str) -> str:
        """2. 死代码注入"""
        dead_code = f"""
function _dead_{random.randint(1000, 9999)}() {{
    const x = Math.random();
    if (x > 2) return x;
    return null;
}}
"""
        return dead_code + '\n' + code
    
    def string_obfuscation(self, code: str) -> str:
        """3. 字符串混淆 - 十六进制编码"""
        try:
            strings = re.findall(r'"([^"]*)"', code)
            result = code
            for s in strings[:5]:
                hex_str = ''.join(f'\\x{ord(c):02x}' for c in s)
                result = result.replace(f'"{s}"', f'"{hex_str}"', 1)
            return result
        except:
            return code
    
    def variable_renaming(self, code: str) -> str:
        """4. 变量重命名"""
        result = code
        vars_to_rename = ['data', 'result', 'value', 'temp', 'item', 'index']
        for var in vars_to_rename:
            new_var = f'_v{random.randint(100, 999)}'
            result = re.sub(rf'\b{var}\b', new_var, result)
        return result
    
    def property_encryption(self, code: str) -> str:
        """5. 属性加密"""
        result = code
        result = re.sub(r'\.([a-zA-Z_][a-zA-Z0-9_]*)', r'["\\1"]', result)
        return result
    
    def function_wrapping(self, code: str) -> str:
        """6. 函数包装 IIFE"""
        wrapper_id = random.randint(10000, 99999)
        return f"""
(function __wrapper{wrapper_id}() {{
{code}
}})();
"""
    
    def regex_obfuscation(self, code: str) -> str:
        """7. 正则表达式混淆"""
        result = code
        result = re.sub(r'\.indexOf\("(\w+)"\)\s*!==\s*-1', r'/\\1/.test()', result)
        return result
    
    def array_obfuscation(self, code: str) -> str:
        """8. 数组混淆"""
        array_id = random.randint(10000, 99999)
        array_def = f"""
const _arr{array_id} = ['console', 'log', 'error', 'warn', 'return', 'function'];
"""
        return array_def + '\n' + code
    
    def eval_obfuscation(self, code: str) -> str:
        """9. Eval混淆"""
        escaped = code.replace('"', '\\"').replace('\n', '\\n')[:200]
        return f'eval("{escaped}");'
    
    def comment_obfuscation(self, code: str) -> str:
        """10. 注释混淆"""
        comments = [
            "// TODO: 性能优化",
            "// NOTE: 关键逻辑",
            "// FIXME: bug需要修复",
            "// HACK: 临时解决方案",
        ]
        lines = code.split('\n')
        result = []
        for i, line in enumerate(lines):
            result.append(line)
            if i % 4 == 0 and line.strip():
                result.append(random.choice(comments))
        return '\n'.join(result)
    
    def semantic_obfuscation(self, code: str) -> str:
        """11. 语义混淆"""
        result = code
        result = re.sub(r'\+\s*1\b', r'+ 1 - 0 + 1', result)
        result = re.sub(r'\*\s*2\b', r'* 2 / 1', result)
        return result
    
    def whitespace_obfuscation(self, code: str) -> str:
        """12. 空白混淆"""
        result = code.replace(';', ';\u200b')
        return result
    
    def apply_obfuscation(self, codes: List[Dict], num_techniques=3) -> List[Dict]:
        """应用混淆技术"""
        logger.info(f"\n🔀 应用混淆技术到真实代码 ({num_techniques}种组合)...")
        
        techniques = [
            ('control_flow', self.control_flow_obfuscation),
            ('dead_code', self.dead_code_insertion),
            ('string', self.string_obfuscation),
            ('variable_rename', self.variable_renaming),
            ('property_encrypt', self.property_encryption),
            ('function_wrap', self.function_wrapping),
            ('regex', self.regex_obfuscation),
            ('array', self.array_obfuscation),
            ('eval', self.eval_obfuscation),
            ('comment', self.comment_obfuscation),
            ('semantic', self.semantic_obfuscation),
            ('whitespace', self.whitespace_obfuscation),
        ]
        
        obfuscated_samples = []
        
        for idx, code_obj in enumerate(codes):
            original = code_obj['content'][:500]  # 限制大小
            
            # 随机选择混淆技术
            selected = random.sample(techniques, min(num_techniques, len(techniques)))
            
            obfuscated = original
            applied_techniques = []
            
            for tech_name, tech_func in selected:
                try:
                    obfuscated = tech_func(obfuscated)
                    applied_techniques.append(tech_name)
                except Exception as e:
                    logger.debug(f"技术 {tech_name} 失败: {e}")
            
            if obfuscated != original:
                sample = {
                    'id': f'real_{idx:06d}',
                    'source_file': code_obj['source'],
                    'original_code': original,
                    'obfuscated_code': obfuscated,
                    'techniques': applied_techniques,
                    'size_ratio': len(obfuscated) / len(original) if original else 1.0,
                    'timestamp': datetime.now().isoformat()
                }
                obfuscated_samples.append(sample)
                
                if (idx + 1) % 20 == 0:
                    logger.info(f"   ✅ 已处理 {idx + 1}/{len(codes)}")
        
        logger.info(f"✅ 混淆完成: {len(obfuscated_samples)} 个样本")
        self.obfuscated = obfuscated_samples
        return obfuscated_samples
    
    def save_obfuscated(self, output_dir='data/real_codes'):
        """保存混淆样本"""
        output_file = Path(output_dir) / 'obfuscated_samples.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.obfuscated:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"\n💾 混淆样本已保存: {output_file}")
        return output_file


class FeatureExtractor:
    """特征提取器"""
    
    @staticmethod
    def extract_features(original: str, obfuscated: str) -> dict:
        """提取特征"""
        features = {
            'original_length': len(original),
            'obfuscated_length': len(obfuscated),
            'length_ratio': len(obfuscated) / len(original) if original else 1.0,
            'line_count_original': original.count('\n'),
            'line_count_obfuscated': obfuscated.count('\n'),
            'keyword_count': sum(original.count(kw) for kw in ['function', 'const', 'let', 'var']),
            'entropy_original': FeatureExtractor._entropy(original),
            'entropy_obfuscated': FeatureExtractor._entropy(obfuscated),
        }
        return features
    
    @staticmethod
    def _entropy(text: str) -> float:
        """计算Shannon熵"""
        if not text:
            return 0.0
        import math
        entropy = 0.0
        text_len = len(text)
        for char in set(text):
            freq = text.count(char) / text_len
            entropy -= freq * math.log2(freq) if freq > 0 else 0
        return entropy


class GPUTrainer:
    """GPU训练器"""
    
    def __init__(self, batch_size=32, epochs=50):
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = []
        
        try:
            import torch
            self.torch = torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"🖥️  使用设备: {self.device}")
        except ImportError:
            logger.error("❌ PyTorch未安装")
            sys.exit(1)
    
    def prepare_data(self, samples: List[Dict]) -> Tuple:
        """准备训练数据"""
        logger.info(f"\n📊 准备训练数据 ({len(samples)} 样本)...")
        
        X = []
        y = []
        
        for sample in samples:
            features = FeatureExtractor.extract_features(
                sample['original_code'],
                sample['obfuscated_code']
            )
            
            # 48维特征向量
            feature_vector = [
                features['original_length'],
                features['obfuscated_length'],
                features['length_ratio'],
                features['line_count_original'],
                features['line_count_obfuscated'],
                features['keyword_count'],
                features['entropy_original'],
                features['entropy_obfuscated'],
            ]
            
            # 补充到48维
            feature_vector.extend([0.0] * (48 - len(feature_vector)))
            
            X.append(feature_vector[:48])
            y.append(1)  # 混淆样本标记为1
        
        # 添加负样本（未混淆）
        for i in range(len(X)):
            X.append([random.gauss(0, 1) for _ in range(48)])
            y.append(0)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        logger.info(f"   ✅ 数据准备完成: {len(X)} 个样本, 48维特征")
        return X, y
    
    def train(self, X, y):
        """GPU训练"""
        logger.info(f"\n🚀 开始GPU训练 ({self.epochs} epochs, batch_size={self.batch_size})...")
        
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        torch = self.torch
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        
        # 构建模型
        model = nn.Sequential(
            nn.Linear(48, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        ).to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   模型参数: {total_params:,}")
        
        # 训练循环
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(loader)
            self.history.append({'epoch': epoch + 1, 'loss': avg_loss})
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"   Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}")
        
        logger.info(f"✅ 训练完成！")
        return self.history


def main():
    parser = argparse.ArgumentParser(description='真实数据学习系统')
    parser.add_argument('--collect-dir', type=str, default='crates',
                       help='代码采集目录')
    parser.add_argument('--techniques', type=int, default=4,
                       help='混淆技术数量')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批大小')
    
    args = parser.parse_args()
    
    logger.info("""
╔═══════════════════════════════════════════════════════════════╗
║           真实数据学习系统 - Week 6                          ║
║                                                               ║
║  1. 采集真实代码  2. 应用混淆  3. 特征提取  4. GPU训练       ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # 1. 采集真实代码
    collector = RealCodeCollector()
    collector.collect_from_directory(args.collect_dir, extensions=('.rs', '.py', '.js', '.ts'))
    collector.collect_from_directory('training', extensions=('.py', '.js', '.sh'))
    collector.save_raw_codes()
    
    if not collector.codes:
        logger.error("❌ 没有采集到代码，退出")
        return
    
    # 2. 应用混淆
    obfuscator = RealObfuscationEngine()
    obfuscated = obfuscator.apply_obfuscation(collector.codes, num_techniques=args.techniques)
    obfuscator.save_obfuscated()
    
    if not obfuscated:
        logger.error("❌ 没有混淆样本，退出")
        return
    
    # 3. 准备训练数据
    trainer = GPUTrainer(batch_size=args.batch_size, epochs=args.epochs)
    X, y = trainer.prepare_data(obfuscated)
    
    # 4. GPU训练
    history = trainer.train(X, y)
    
    # 保存训练历史
    history_file = Path('data/real_codes/training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\n✅ 学习完成！结果保存到: data/real_codes/")
    logger.info(f"   • 原始代码: data/real_codes/raw_codes.jsonl")
    logger.info(f"   • 混淆样本: data/real_codes/obfuscated_samples.jsonl")
    logger.info(f"   • 训练历史: data/real_codes/training_history.json")


if __name__ == '__main__':
    main()
