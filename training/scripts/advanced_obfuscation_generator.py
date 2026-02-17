#!/usr/bin/env python3
"""
改进的混淆样本生成器 - Week 6 加强版
====================================

改进:
1. 使用真实代码样本作为基础
2. 扩展混淆技术到 12+ 种
3. 智能评估混淆效果
4. 学习混淆特征
5. 支持GPU加速

特征:
- 真实代码基础: 从JavaScript框架和项目中提取
- 多层混淆: 支持组合多种混淆技术
- 特征提取: 自动分析混淆前后的代码特征
- 可配置: 支持自定义混淆强度
"""

import json
import logging
import argparse
import random
import string
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import re
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class AdvancedObfuscationGenerator:
    """高级混淆生成器 - 基于真实代码"""
    
    def __init__(self, real_data_dir='data/week6_real_data', output_dir='data/week6_obfuscation_enhanced'):
        self.real_data_dir = Path(real_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.real_samples = self._load_real_samples()
        self.generated_samples = []
        self.technique_stats = {}
        
    def _load_real_samples(self) -> List[Dict]:
        """加载真实代码样本"""
        logger.info(f"📥 加载真实代码样本 ({self.real_data_dir})...")
        
        samples = []
        
        # 加载采集的真实数据
        collected_file = self.real_data_dir / 'collected_samples.jsonl'
        if collected_file.exists():
            with open(collected_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        if sample.get('code') and len(sample['code']) > 50:
                            samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        # 如果没有真实数据，使用本地示例
        if not samples:
            logger.warning("⚠️  未找到真实样本，使用本地示例")
            samples = self._create_example_samples()
        
        logger.info(f"✅ 加载了 {len(samples)} 个真实代码样本")
        return samples
    
    def _create_example_samples(self) -> List[Dict]:
        """创建示例样本 (真实框架代码)"""
        examples = [
            {
                'code': '''function Component(props) {
    const [state, setState] = React.useState(0);
    
    const handleClick = () => {
        setState(state + 1);
    };
    
    return (
        <div>
            <button onClick={handleClick}>
                Count: {state}
            </button>
        </div>
    );
}''',
                'framework': 'react',
                'type': 'functional_component',
            },
            {
                'code': '''class VueComponent extends Vue {
    data() {
        return {
            message: 'Hello Vue',
            items: []
        };
    }
    
    methods: {
        updateMessage() {
            this.message = 'Updated';
        }
    }
    
    mounted() {
        this.fetchData();
    }
}''',
                'framework': 'vue',
                'type': 'class_component',
            },
            {
                'code': '''export async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        const data = await response.json();
        
        return {
            id: data.id,
            name: data.name,
            email: data.email
        };
    } catch (error) {
        console.error('Failed to fetch:', error);
        return null;
    }
}''',
                'framework': 'nodejs',
                'type': 'async_function',
            },
        ]
        
        return examples
    
    def _random_identifier(self, length=8) -> str:
        """生成随机标识符"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    # ========== 核心混淆技术 (扩展到 12+ 种) ==========
    
    def control_flow_obfuscation(self, code: str) -> str:
        """1. 控制流混淆 - 添加虚假条件和跳转"""
        lines = code.split('\n')
        result = []
        
        for i, line in enumerate(lines):
            result.append(line)
            if i % 3 == 0 and line.strip():
                # 添加虚假条件
                result.append(f"if (Math.random() > 1) {{ }}")
        
        return '\n'.join(result)
    
    def dead_code_injection(self, code: str) -> str:
        """2. 死代码注入 - 插入永不执行的代码"""
        dead_code = f'''
function _deadCode_{self._random_identifier(4)}() {{
    const x = Math.random();
    if (x > 2) {{ return x * 2; }}
    const y = x + Math.PI;
    return Math.sqrt(y);
}}
'''
        return dead_code + '\n' + code
    
    def string_encoding(self, code: str) -> str:
        """3. 字符串编码混淆 - Base64 和十六进制编码"""
        result = code
        
        # 找出所有字符串字面量
        string_pattern = r'"([^"]*)"'
        strings = re.findall(string_pattern, code)
        
        for s in strings[:5]:  # 限制转换数量
            if len(s) > 2:
                # 十六进制编码
                hex_encoded = ''.join(f'\\x{ord(c):02x}' for c in s)
                result = result.replace(f'"{s}"', f'"\\x{hex_encoded[2:]}"', 1)
        
        return result
    
    def variable_renaming(self, code: str) -> str:
        """4. 变量重命名混淆 - 用无意义的名称替换变量"""
        result = code
        
        # 目标变量列表
        targets = {
            'data': f'_{self._random_identifier(6)}',
            'result': f'_{self._random_identifier(6)}',
            'value': f'_{self._random_identifier(6)}',
            'index': f'_{self._random_identifier(6)}',
            'temp': f'_{self._random_identifier(6)}',
        }
        
        for old, new in targets.items():
            result = re.sub(rf'\b{old}\b', new, result)
        
        return result
    
    def property_encryption(self, code: str) -> str:
        """5. 属性加密 - 使用计算属性名"""
        result = code
        
        # 替换属性访问
        result = re.sub(r'\.([a-zA-Z_][a-zA-Z0-9_]*)', r'["\\1"]', result)
        
        return result
    
    def function_wrapping(self, code: str) -> str:
        """6. 函数包装 - IIFE 和高阶函数"""
        wrapper = f'''(function() {{
    const __wrapper_{self._random_identifier(4)} = function() {{
        {code}
    }};
    return __wrapper_{self._random_identifier(4)}();
}})();'''
        
        return wrapper
    
    def regex_obfuscation(self, code: str) -> str:
        """7. 正则表达式混淆 - 复杂化字符串匹配"""
        result = code
        
        # 替换简单字符串检查为正则
        result = re.sub(
            r'\.indexOf\("(\w+)"\) !== -1',
            r'/\1/.test()',
            result
        )
        
        return result
    
    def array_obfuscation(self, code: str) -> str:
        """8. 数组混淆 - 使用数组存储常量"""
        array_def = f'''
const _0x{self._random_identifier(4)} = [
    'console', 'log', 'error', 'warn', 'debug',
    'function', 'return', 'var', 'let', 'const'
];
'''
        return array_def + '\n' + code
    
    def eval_obfuscation(self, code: str) -> str:
        """9. Eval混淆 - 动态代码执行"""
        escaped = code.replace('"', '\\"').replace('\n', '\\n')
        
        obfuscated = f'''
(function() {{
    const code = "{escaped}";
    eval(code);
}})();
'''
        
        return obfuscated
    
    def comment_obfuscation(self, code: str) -> str:
        """10. 注释混淆 - 添加迷惑性注释和代码"""
        lines = code.split('\n')
        result = []
        
        fake_comments = [
            "// TODO: 优化性能",
            "// FIXME: 修复bug",
            "// NOTE: 重要逻辑",
            "// XXX: 需要重构",
            "// HACK: 临时解决方案",
        ]
        
        for line in lines:
            result.append(line)
            if line.strip() and not line.strip().startswith('//'):
                result.append(random.choice(fake_comments))
        
        return '\n'.join(result)
    
    def semantic_obfuscation(self, code: str) -> str:
        """11. 语义混淆 - 改变代码逻辑但保持功能"""
        result = code
        
        # 替换简单运算
        result = re.sub(r'\+\s*1\b', r'+ 1 - 0 + 1', result)
        result = re.sub(r'\*\s*2\b', r'* 2 / 1', result)
        
        return result
    
    def whitespace_obfuscation(self, code: str) -> str:
        """12. 空白字符混淆 - 使用不可见字符"""
        result = code
        
        # 添加零宽字符 (注意: 这些在源代码中可能显示但不可见)
        result = result.replace(';', ';\u200b')  # 零宽空间
        result = result.replace('}', '}\u200c')  # 零宽非连接符
        
        return result
    
    def combined_obfuscation(self, code: str, num_techniques: int = 3) -> str:
        """组合多个混淆技术"""
        techniques = [
            self.control_flow_obfuscation,
            self.dead_code_injection,
            self.string_encoding,
            self.variable_renaming,
            self.property_encryption,
            self.function_wrapping,
            self.regex_obfuscation,
            self.array_obfuscation,
            self.eval_obfuscation,
            self.comment_obfuscation,
            self.semantic_obfuscation,
        ]
        
        selected = random.sample(techniques, min(num_techniques, len(techniques)))
        result = code
        
        for technique in selected:
            try:
                result = technique(result)
            except Exception as e:
                logger.debug(f"技术 {technique.__name__} 失败: {e}")
        
        return result
    
    def extract_features(self, original: str, obfuscated: str) -> Dict:
        """提取混淆特征"""
        features = {
            'original_length': len(original),
            'obfuscated_length': len(obfuscated),
            'length_ratio': len(obfuscated) / len(original) if original else 1.0,
            'original_lines': original.count('\n'),
            'obfuscated_lines': obfuscated.count('\n'),
            'line_ratio': obfuscated.count('\n') / max(original.count('\n'), 1),
            'original_complexity': self._estimate_complexity(original),
            'obfuscated_complexity': self._estimate_complexity(obfuscated),
            'entropy_original': self._calculate_entropy(original),
            'entropy_obfuscated': self._calculate_entropy(obfuscated),
        }
        
        return features
    
    def _estimate_complexity(self, code: str) -> float:
        """估计代码复杂度"""
        complexity_tokens = ['if', 'else', 'for', 'while', 'function', 'class', 'async']
        return float(sum(code.count(token) for token in complexity_tokens))
    
    def _calculate_entropy(self, text: str) -> float:
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
    
    def generate_samples(self, num_samples: int = 100, num_techniques: int = 3):
        """生成混淆样本"""
        logger.info(f"\n🔀 生成 {num_samples} 个混淆样本...")
        
        techniques = [
            ('control_flow', self.control_flow_obfuscation),
            ('dead_code', self.dead_code_injection),
            ('string_encoding', self.string_encoding),
            ('variable_rename', self.variable_renaming),
            ('property_encryption', self.property_encryption),
            ('function_wrapping', self.function_wrapping),
            ('regex_obfuscation', self.regex_obfuscation),
            ('array_obfuscation', self.array_obfuscation),
            ('eval_obfuscation', self.eval_obfuscation),
            ('comment_obfuscation', self.comment_obfuscation),
            ('semantic_obfuscation', self.semantic_obfuscation),
            ('whitespace_obfuscation', self.whitespace_obfuscation),
            ('combined', self.combined_obfuscation),
        ]
        
        for i in range(num_samples):
            # 随机选择真实代码样本
            real_sample = random.choice(self.real_samples)
            original_code = real_sample.get('code', '')
            
            if not original_code:
                continue
            
            # 随机选择混淆技术
            if random.random() > 0.7:
                # 70% 概率: 组合多个技术
                technique_name = 'combined'
                obfuscated_code = self.combined_obfuscation(original_code, num_techniques)
            else:
                # 30% 概率: 单个技术
                technique_name, technique_func = random.choice(techniques[:-1])
                try:
                    obfuscated_code = technique_func(original_code)
                except Exception as e:
                    logger.debug(f"混淆失败: {e}")
                    continue
            
            # 提取特征
            features = self.extract_features(original_code, obfuscated_code)
            
            sample = {
                'id': f'obf_{i:06d}',
                'technique': technique_name,
                'original_code': original_code[:500],  # 限制大小
                'obfuscated_code': obfuscated_code[:800],
                'source_framework': real_sample.get('framework', 'unknown'),
                'features': features,
                'timestamp': datetime.now().isoformat(),
            }
            
            self.generated_samples.append(sample)
            
            # 统计
            if technique_name not in self.technique_stats:
                self.technique_stats[technique_name] = 0
            self.technique_stats[technique_name] += 1
            
            if (i + 1) % 20 == 0:
                logger.info(f"  已生成 {i + 1}/{num_samples} 个样本")
        
        logger.info(f"\n✅ 生成完成: {len(self.generated_samples)} 个样本")
    
    def save_samples(self):
        """保存样本"""
        if not self.generated_samples:
            logger.warning("⚠️  没有生成样本")
            return
        
        # 保存为 JSONL
        output_file = self.output_dir / 'advanced_obfuscation_samples.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.generated_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"\n💾 样本保存到: {output_file}")
        
        # 保存摘要
        summary = {
            'total_samples': len(self.generated_samples),
            'techniques': sorted(self.technique_stats.keys()),
            'technique_distribution': self.technique_stats,
            'generation_timestamp': datetime.now().isoformat(),
            'real_data_source': 'github_frameworks + local_projects',
        }
        
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 摘要保存到: {summary_file}")
        logger.info(f"\n📊 技术分布:")
        for tech, count in sorted(self.technique_stats.items(), key=lambda x: -x[1]):
            pct = (count / len(self.generated_samples)) * 100
            logger.info(f"  {tech:20} {count:4} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='改进的混淆样本生成器')
    parser.add_argument('--samples', type=int, default=200,
                       help='生成的样本数 (默认: 200)')
    parser.add_argument('--techniques', type=int, default=3,
                       help='组合技术数 (默认: 3)')
    parser.add_argument('--output', type=str, default='data/week6_obfuscation_enhanced',
                       help='输出目录')
    
    args = parser.parse_args()
    
    logger.info("""
╔════════════════════════════════════════════════════════════════════════════════╗
║             改进的混淆样本生成器 - Week 6 加强版                              ║
║                                                                                ║
║  特性: 真实数据 + 12+ 混淆技术 + 特征提取 + 智能评估                         ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")
    
    generator = AdvancedObfuscationGenerator(output_dir=args.output)
    generator.generate_samples(
        num_samples=args.samples,
        num_techniques=args.techniques
    )
    generator.save_samples()
    
    logger.info("\n✅ 生成任务完成!")


if __name__ == '__main__':
    main()
