#!/usr/bin/env python3
"""
Week 6 混淆检测样本生成脚本
目标: 扩展混淆技术从 4 种到 8+ 种
"""

import json
import logging
import argparse
import random
import string
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ObfuscationSampleGenerator:
    """混淆检测样本生成器"""
    
    def __init__(self, output_dir='data/week6_obfuscation', samples=50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples = samples
        self.generated = []
    
    def _random_name(self, length=8):
        """生成随机变量名"""
        return ''.join(random.choices(string.ascii_letters, k=length))
    
    # 已有的 4 种混淆技术
    
    def control_flow_obfuscation(self, code: str) -> str:
        """控制流混淆"""
        # 添加无用的条件分支
        obfuscated = f"""
if (true) {{
    if (false) {{
        console.log('never executed');
    }} else {{
        {code}
    }}
}}
"""
        return obfuscated
    
    def dead_code_insertion(self, code: str) -> str:
        """死代码插入"""
        dead_code = f"""
function unusedFunction{self._random_name(4)}() {{
    var x = Math.random();
    if (x > 1.5) {{ return x; }}
    return null;
}}
"""
        return dead_code + "\n" + code
    
    def string_encoding(self, code: str) -> str:
        """字符串编码混淆"""
        # 简单的 Base64 编码示例
        strings = ["hello", "world", "function", "return"]
        obfuscated = code
        for s in strings:
            if s in code:
                encoded = s.encode().hex()
                obfuscated = obfuscated.replace(
                    f'"{s}"',
                    f'String.fromCharCode(..."{encoded}".match(/.{{1,2}}/g).map(x=>parseInt(x,16)))'
                )
        return obfuscated
    
    def variable_renaming(self, code: str) -> str:
        """变量重命名混淆"""
        # 替换常见变量名
        mappings = {
            'value': self._random_name(6),
            'result': self._random_name(6),
            'data': self._random_name(6),
            'index': self._random_name(6),
        }
        obfuscated = code
        for old, new in mappings.items():
            obfuscated = obfuscated.replace(old, new)
        return obfuscated
    
    # 新增的 4 种混淆技术
    
    def property_encryption(self, code: str) -> str:
        """属性加密混淆"""
        # 使用计算属性名
        obfuscated = code.replace('obj.property', 'obj["prop" + "erty"]')
        obfuscated = obfuscated.replace('.method()', '["meth" + "od"]()')
        return obfuscated
    
    def function_wrapping(self, code: str) -> str:
        """函数包装混淆"""
        # 将代码包装在 IIFE 中
        wrapper_name = self._random_name(8)
        obfuscated = f"""
(function {wrapper_name}() {{
    var _0x{self._random_name(4)} = function() {{
        {code}
    }};
    return _0x{self._random_name(4)}();
}})();
"""
        return obfuscated
    
    def regex_obfuscation(self, code: str) -> str:
        """正则表达式混淆"""
        # 将简单字符串匹配替换为复杂正则
        obfuscated = code.replace(
            'str.indexOf("test") !== -1',
            '/t[e]s[t]/.test(str)'
        )
        obfuscated = obfuscated.replace(
            '"hello"',
            'String.raw`${"h"}${"e"}${"l"}${"l"}${"o"}`'
        )
        return obfuscated
    
    def array_obfuscation(self, code: str) -> str:
        """数组混淆"""
        # 使用数组索引访问
        obfuscated_prefix = f"""
var _0x{self._random_name(4)} = [
    'log', 'warn', 'error', 'info',
    'function', 'return', 'var', 'const'
];
"""
        obfuscated = obfuscated_prefix + "\n" + code
        obfuscated = obfuscated.replace('console.log', f'console[_0x{self._random_name(4)}[0]]')
        return obfuscated
    
    def generate_samples_for_technique(self, technique_name: str, technique_func):
        """为单个混淆技术生成样本"""
        logger.info(f"\n🔀 生成混淆技术样本: {technique_name}")
        
        # 基础代码模板
        base_codes = [
            'function add(a, b) { return a + b; }',
            'var value = 42; console.log(value);',
            'for (var i = 0; i < 10; i++) { console.log(i); }',
            'var obj = { property: "value", method: function() {} };',
            'function processData(data) { return data.map(x => x * 2); }',
        ]
        
        samples = []
        for i in range(self.samples):
            base_code = random.choice(base_codes)
            obfuscated_code = technique_func(base_code)
            
            sample = {
                'id': f'{technique_name}_{i}',
                'technique': technique_name,
                'original_code': base_code,
                'obfuscated_code': obfuscated_code,
                'code_size': len(obfuscated_code),
                'obfuscation_ratio': len(obfuscated_code) / len(base_code) if base_code else 1.0,
            }
            samples.append(sample)
        
        logger.info(f"✅ {technique_name}: 生成 {len(samples)} 个样本")
        return samples
    
    def generate_all_samples(self):
        """生成所有混淆技术样本"""
        logger.info("="*80)
        logger.info("🚀 Week 6 混淆检测样本生成开始")
        logger.info(f"每种技术样本数: {self.samples}")
        logger.info("="*80)
        
        techniques = {
            # 已有
            'control_flow': self.control_flow_obfuscation,
            'dead_code': self.dead_code_insertion,
            'string_encoding': self.string_encoding,
            'variable_rename': self.variable_renaming,
            # 新增
            'property_encryption': self.property_encryption,
            'function_wrapping': self.function_wrapping,
            'regex_obfuscation': self.regex_obfuscation,
            'array_obfuscation': self.array_obfuscation,
        }
        
        for technique_name, technique_func in techniques.items():
            samples = self.generate_samples_for_technique(technique_name, technique_func)
            self.generated.extend(samples)
        
        logger.info("\n" + "="*80)
        logger.info(f"✅ 生成完成: {len(self.generated)} 个样本")
        logger.info("="*80)
        
        # 统计
        technique_stats = {}
        for sample in self.generated:
            tech = sample.get('technique')
            technique_stats[tech] = technique_stats.get(tech, 0) + 1
        
        logger.info("\n📊 混淆技术分布:")
        for tech, count in sorted(technique_stats.items()):
            percentage = (count / len(self.generated)) * 100 if self.generated else 0
            logger.info(f"  {tech:20} {count:4} ({percentage:5.1f}%)")
        
        # 计算平均混淆比例
        avg_ratio = sum(s.get('obfuscation_ratio', 1.0) for s in self.generated) / len(self.generated)
        logger.info(f"\n平均混淆比例: {avg_ratio:.2f}x")
        
        return self.generated
    
    def save_samples(self):
        """保存样本到文件"""
        if not self.generated:
            logger.warning("⚠️  没有生成样本")
            return
        
        # 保存到 JSONL 格式
        output_file = self.output_dir / 'obfuscation_samples.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.generated:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"\n💾 样本保存到: {output_file}")
        logger.info(f"   文件大小: {output_file.stat().st_size / 1024:.2f} KB")
        
        # 保存摘要
        summary = {
            'total_samples': len(self.generated),
            'techniques': list(set(s.get('technique') for s in self.generated)),
            'technique_counts': {
                tech: sum(1 for s in self.generated if s.get('technique') == tech)
                for tech in set(s.get('technique') for s in self.generated)
            },
            'avg_obfuscation_ratio': sum(s.get('obfuscation_ratio', 1.0) for s in self.generated) / len(self.generated),
        }
        
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 摘要保存到: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Week 6 混淆检测样本生成')
    parser.add_argument('--techniques', type=str,
                       help='逗号分隔的混淆技术列表 (默认: 全部)')
    parser.add_argument('--samples', type=int, default=50,
                       help='每种技术的样本数 (默认: 50)')
    parser.add_argument('--output', type=str, default='data/week6_obfuscation',
                       help='输出目录 (默认: data/week6_obfuscation)')
    
    args = parser.parse_args()
    
    generator = ObfuscationSampleGenerator(
        output_dir=args.output,
        samples=args.samples
    )
    
    generator.generate_all_samples()
    generator.save_samples()
    
    logger.info("\n✅ 生成任务完成!")

if __name__ == '__main__':
    main()
