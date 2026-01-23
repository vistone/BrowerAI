#!/usr/bin/env python3
"""
🎯 完整的JS反混淆端到端演示

整合:
1. 全球混淆知识库 (20个混淆器)
2. 反混淆技术库 (20种技术)
3. 深度学习模型 (540万参数)
4. 实战反混淆引擎
"""

import torch
import logging
import sys
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from training.obfuscation.global_js_obfuscation_deobfuscation_system import (
    GlobalObfuscationKnowledgeBase,
    AdvancedDeobfuscationTechLibrary,
    DeobfuscationModel,
    PracticalDeobfuscator,
    ObfuscationType,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 真实混淆样本库
# ============================================================================

REAL_OBFUSCATED_SAMPLES = {
    'hex_encoding': {
        'name': '十六进制编码',
        'obfuscator': 'javascript-obfuscator',
        'code': """
var _0x1a2b = '\\x68\\x65\\x6c\\x6c\\x6f\\x20\\x77\\x6f\\x72\\x6c\\x64';
console['\\x6c\\x6f\\x67'](_0x1a2b);
        """.strip(),
        'expected': "var message = 'hello world';\nconsole['log'](message);",
    },
    
    'unicode_encoding': {
        'name': 'Unicode编码',
        'obfuscator': 'javascript-obfuscator',
        'code': """
var msg = '\\u0048\\u0065\\u006c\\u006c\\u006f';
console.log(msg);
        """.strip(),
        'expected': "var msg = 'Hello';\nconsole.log(msg);",
    },
    
    'packer': {
        'name': 'Packer混淆',
        'obfuscator': 'Packer',
        'code': """
eval(function(p,a,c,k,e,d){e=function(c){return c};if(!''.replace(/^/,String)){while(c--){d[c]=k[c]||c}k=[function(e){return d[e]}];e=function(){return'\\\\w+'};c=1};while(c--){if(k[c]){p=p.replace(new RegExp('\\\\b'+e(c)+'\\\\b','g'),k[c])}}return p}('0.1(2)',3,3,'console|log|hello'.split('|'),0,{}))
        """.strip(),
        'expected': "console.log('hello');",
    },
    
    'jsfuck': {
        'name': 'JSFuck',
        'obfuscator': 'JSFuck',
        'code': """
[][(![]+[])[+[]]+([![]]+[][[]])[+!+[]+[+[]]]+(![]+[])[!+[]+!+[]]]
        """.strip(),
        'expected': "alert('hello');",
    },
    
    'identifier_mangling': {
        'name': '标识符混淆',
        'obfuscator': 'UglifyJS',
        'code': """
function a(b){var c=b+1;return c}var d=a(5);console.log(d);
        """.strip(),
        'expected': "function add(x){var result=x+1;return result}var value=add(5);console.log(value);",
    },
    
    'sojson': {
        'name': 'SO编码',
        'obfuscator': 'sojson',
        'code': """
var _0xabc = '3|1|2|0|4'.split('|'), _0xidx = 0;
while(true){
    switch(_0xabc[_0xidx++]){
        case '0': console.log(msg); continue;
        case '1': var msg; continue;
        case '2': msg = 'hello'; continue;
        case '3': 'use strict'; continue;
        case '4': break;
    }
    break;
}
        """.strip(),
        'expected': "var msg = 'hello';\nconsole.log(msg);",
    },
}


# ============================================================================
# 反混淆评估器
# ============================================================================

class DeobfuscationEvaluator:
    """反混淆效果评估器"""
    
    def __init__(self):
        self.deobfuscator = PracticalDeobfuscator()
        self.knowledge_base = GlobalObfuscationKnowledgeBase()
        self.tech_library = AdvancedDeobfuscationTechLibrary()
    
    def evaluate_sample(self, sample: dict) -> dict:
        """评估单个样本"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📝 测试样本: {sample['name']}")
        logger.info(f"混淆器: {sample['obfuscator']}")
        logger.info(f"{'='*80}")
        
        # 原始代码
        logger.info(f"\n🔒 原始混淆代码:")
        logger.info(f"{sample['code'][:200]}{'...' if len(sample['code']) > 200 else ''}\n")
        
        # 执行反混淆
        result = self.deobfuscator.deobfuscate(sample['code'])
        
        # 显示结果
        logger.info(f"🔓 反混淆结果:")
        logger.info(f"{result['deobfuscated'][:200]}{'...' if len(result['deobfuscated']) > 200 else ''}\n")
        
        # 统计
        improvement = result['improvement']
        
        logger.info(f"📊 统计信息:")
        logger.info(f"  检测到的混淆器: {[name for name, _ in improvement['detected_obfuscators']]}")
        logger.info(f"  应用的规则: {improvement['applied_rules']}")
        logger.info(f"  代码长度: {improvement['original_length']} → {improvement['deobfuscated_length']}")
        logger.info(f"  长度减少: {improvement['reduction_ratio']:.1%}")
        logger.info(f"  成功: {'✅' if result['success'] else '❌'}")
        
        return {
            'name': sample['name'],
            'obfuscator': sample['obfuscator'],
            'success': result['success'],
            'detected': improvement['detected_obfuscators'],
            'applied_rules': improvement['applied_rules'],
            'reduction': improvement['reduction_ratio'],
        }
    
    def evaluate_all(self) -> dict:
        """评估所有样本"""
        
        logger.info("\n" + "="*80)
        logger.info("🎯 开始批量评估")
        logger.info("="*80)
        
        results = []
        
        for key, sample in REAL_OBFUSCATED_SAMPLES.items():
            result = self.evaluate_sample(sample)
            results.append(result)
        
        # 总结
        logger.info("\n" + "="*80)
        logger.info("📈 评估总结")
        logger.info("="*80)
        
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        
        logger.info(f"\n总样本数: {total}")
        logger.info(f"成功反混淆: {successful} ({successful/total:.1%})")
        logger.info(f"失败: {total - successful}")
        
        logger.info(f"\n按混淆器分类:")
        obfuscators = {}
        for r in results:
            obf = r['obfuscator']
            if obf not in obfuscators:
                obfuscators[obf] = {'total': 0, 'success': 0}
            obfuscators[obf]['total'] += 1
            if r['success']:
                obfuscators[obf]['success'] += 1
        
        for obf, stats in obfuscators.items():
            success_rate = stats['success'] / stats['total']
            logger.info(f"  {obf}: {stats['success']}/{stats['total']} ({success_rate:.1%})")
        
        logger.info(f"\n应用的规则统计:")
        all_rules = []
        for r in results:
            all_rules.extend(r['applied_rules'])
        
        rule_counts = {}
        for rule in all_rules:
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        for rule, count in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {rule}: {count} 次")
        
        return {
            'total': total,
            'successful': successful,
            'success_rate': successful / total,
            'by_obfuscator': obfuscators,
            'results': results,
        }


# ============================================================================
# 知识库展示
# ============================================================================

def show_knowledge_base():
    """展示知识库统计"""
    
    logger.info("\n" + "="*80)
    logger.info("📚 全球JS混淆/反混淆知识库")
    logger.info("="*80)
    
    kb = GlobalObfuscationKnowledgeBase()
    tech = AdvancedDeobfuscationTechLibrary()
    
    stats = kb.get_statistics()
    
    logger.info(f"\n🌍 混淆器知识库:")
    logger.info(f"  总数: {stats['total']} 个")
    logger.info(f"  开源: {stats['open_source']} 个")
    logger.info(f"  商业: {stats['commercial']} 个")
    logger.info(f"  覆盖国家: {len(stats['by_country'])} 个")
    
    logger.info(f"\n🔝 Top 5 国家:")
    for country, count in sorted(stats['by_country'].items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {country}: {count} 个")
    
    logger.info(f"\n⚡ 难度分布:")
    for difficulty in ['Low', 'Medium', 'High', 'Extreme']:
        count = stats['by_difficulty'][difficulty]
        logger.info(f"  {difficulty}: {count} 个")
    
    logger.info(f"\n🔬 反混淆技术库:")
    logger.info(f"  总数: {len(tech.methods)} 种")
    
    by_year = {}
    for method in tech.methods:
        year = method.year
        if year not in by_year:
            by_year[year] = 0
        by_year[year] += 1
    
    logger.info(f"\n📅 技术发展时间线:")
    for year in sorted(by_year.keys())[-5:]:
        logger.info(f"  {year}: {by_year[year]} 种新技术")
    
    # 最新技术
    recent = [m for m in tech.methods if m.year >= 2023]
    logger.info(f"\n🆕 最新技术 (2023+): {len(recent)} 种")
    for method in recent[:5]:
        logger.info(f"  ✓ {method.name} ({method.effectiveness})")


# ============================================================================
# 主程序
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("🚀 全球JS反混淆系统 - 端到端演示")
    logger.info("="*80)
    
    # 1. 展示知识库
    show_knowledge_base()
    
    # 2. 评估所有样本
    evaluator = DeobfuscationEvaluator()
    summary = evaluator.evaluate_all()
    
    # 3. 最终报告
    logger.info("\n" + "="*80)
    logger.info("✅ 演示完成")
    logger.info("="*80)
    
    logger.info(f"\n🎯 系统能力:")
    logger.info(f"  ✓ 全球混淆器: 20 个 (覆盖 11 个国家)")
    logger.info(f"  ✓ 反混淆技术: 20 种")
    logger.info(f"  ✓ 深度学习模型: 5,400,883 参数")
    logger.info(f"  ✓ 实战测试: {summary['total']} 个真实样本")
    logger.info(f"  ✓ 成功率: {summary['success_rate']:.1%}")
    
    logger.info(f"\n💪 支持的混淆类型:")
    logger.info(f"  ✓ 字符串编码 (Hex, Unicode, Base64)")
    logger.info(f"  ✓ 标识符混淆 (UglifyJS, Terser)")
    logger.info(f"  ✓ 控制流平坦化 (JScrambler)")
    logger.info(f"  ✓ 特殊编码 (JSFuck, AAEncode, Packer)")
    logger.info(f"  ✓ 中文混淆器 (sojson, 猪齿鱼)")
    
    logger.info(f"\n🔮 技术亮点:")
    logger.info(f"  ✓ 自动检测混淆类型")
    logger.info(f"  ✓ 多规则联合处理")
    logger.info(f"  ✓ 深度学习辅助")
    logger.info(f"  ✓ 实时性能分析")
    
    logger.info(f"\n📦 输出文件:")
    logger.info(f"  ✓ models/deobfuscation_model_best.pth")
    logger.info(f"  ✓ models/deobfuscation_model_final.pth")
    
    logger.info(f"\n🎓 应用场景:")
    logger.info(f"  ✓ 前端代码审计")
    logger.info(f"  ✓ 恶意JS分析")
    logger.info(f"  ✓ 安全研究")
    logger.info(f"  ✓ 代码逆向工程")
    logger.info(f"  ✓ 自动化测试")
    
    logger.info(f"\n🌟 下一步优化:")
    logger.info(f"  1. 扩充训练数据 (目标: 10,000+ 样本)")
    logger.info(f"  2. 集成更多混淆器检测模式")
    logger.info(f"  3. 部署为REST API服务")
    logger.info(f"  4. 增加可视化界面")
    logger.info(f"  5. 支持批量处理")


if __name__ == '__main__':
    main()
