#!/usr/bin/env python3
"""
高精度框架检测器 v2.0
基于真实网站HTML特征优化

目标: 90%+ 准确率
策略: 
1. 使用类别（category）作为强先验
2. 关键字密度计算
3. 多维特征融合
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class HighPrecisionDetector:
    """高精度框架检测器"""
    
    # 超精准规则库 - 基于真实网站特征分析
    PRECISE_RULES = {
        "React": {
            "keywords": ["react", "jsx", "usestate", "useeffect", "reactdom", "next.js", "_next"],
            "imports": ["from 'react'", "from \"react\"", "import react"],
            "api_patterns": [
                r"React\.createElement",
                r"ReactDOM\.render",
                r"useState\(",
                r"useEffect\(",
                r"useContext\(",
                r"/_next/",  # Next.js特征
            ],
            "threshold": 3,  # 降低阈值，包含Next.js
        },
        "Vue": {
            "keywords": ["vue", "v-bind", "v-model", "v-if", "v-for"],
            "imports": ["from 'vue'", "from \"vue\"", "import vue"],
            "api_patterns": [
                r"Vue\.createApp",
                r"new Vue\(",
                r"<template>",
                r"v-bind",
                r"v-model",
            ],
            "threshold": 5,
        },
        "Angular": {
            "keywords": ["angular", "@angular", "ng-app", "ng-controller"],
            "imports": ["@angular", "angular.module"],
            "api_patterns": [
                r"@Component",
                r"@Injectable",
                r"@NgModule",
                r"angular\.module",
            ],
            "threshold": 5,
        },
        "Express": {
            "keywords": ["express", "app.get", "app.post", "middleware"],
            "imports": ["require('express')", "from 'express'"],
            "api_patterns": [
                r"app\.get\(",
                r"app\.post\(",
                r"app\.use\(",
                r"express\(\)",
            ],
            "threshold": 3,
        },
        "jQuery": {
            "keywords": ["jquery", "$(", "$.ajax", "$.get"],
            "imports": ["jquery"],
            "api_patterns": [
                r"jQuery\(",
                r"\$\(",
                r"\.on\(",
                r"\.click\(",
            ],
            "threshold": 5,
        },
        "Svelte": {
            "keywords": ["svelte", "bind:", "on:", "$:", "sveltekit", "@sveltejs"],
            "imports": ["from '.svelte'", "from 'svelte'", "sveltekit"],
            "api_patterns": [
                r"svelte\.dev",
                r"sveltekit",
                r"\.svelte",
                r"@sveltejs/",
                r"__svelte",  # Svelte编译标记
            ],
            "threshold": 20,  # 大幅提高阈值，严格匹配
        },
    }
    
    def __init__(self):
        self.compile_patterns()
    
    def compile_patterns(self):
        """预编译正则表达式"""
        self.compiled_patterns = {}
        for framework, rules in self.PRECISE_RULES.items():
            self.compiled_patterns[framework] = [
                re.compile(pattern, re.IGNORECASE | re.DOTALL)
                for pattern in rules["api_patterns"]
            ]
    
    def detect_with_category(self, html: str, category: Optional[str] = None) -> Tuple[str, float]:
        """
        基于类别增强的检测
        
        如果提供了category，则使用强先验
        """
        html_lower = html.lower()
        scores = {}
        
        # 为所有框架计算分数
        for framework, rules in self.PRECISE_RULES.items():
            score = 0
            
            # 1. 关键字计数
            keyword_count = sum(html_lower.count(kw.lower()) for kw in rules["keywords"])
            score += keyword_count
            
            # 2. 导入语句检测
            for import_stmt in rules["imports"]:
                if import_stmt.lower() in html_lower:
                    score += 20  # 导入语句权重很高
            
            # 3. API模式匹配
            for pattern in self.compiled_patterns[framework]:
                matches = len(pattern.findall(html))
                score += matches * 5
            
            # 4. 阈值过滤
            if keyword_count < rules["threshold"]:
                score *= 0.1  # 严重降权
            
            scores[framework] = score
        
        # 如果有category，大幅提升对应框架的分数
        if category and category in scores:
            # 但只在该框架有基础证据时才提升
            if scores[category] > 0:
                scores[category] *= 5.0  # 5倍加权
        
        # 找出最高分
        if not scores or all(s == 0 for s in scores.values()):
            return "Unknown", 0.0
        
        best_framework = max(scores.items(), key=lambda x: x[1])
        
        # 计算置信度
        total_score = sum(scores.values())
        confidence = best_framework[1] / total_score if total_score > 0 else 0
        
        return best_framework[0], min(confidence, 1.0)
    
    def batch_detect(self, websites: List[Dict]) -> Dict:
        """批量检测并计算准确率"""
        results = defaultdict(list)
        correct = 0
        
        for site in websites:
            html = site.get('html', '')
            category = site.get('category')
            
            # 跳过Unknown标签或空HTML
            if not category or category == 'Unknown' or len(html) < 100:
                continue
            
            # 纯HTML检测，不使用category先验（避免0分被放大）
            detected_fw, confidence = self.detect_with_category(html, category=None)
            
            is_correct = detected_fw == category
            if is_correct:
                correct += 1
            
            results[detected_fw].append({
                'url': site.get('url'),
                'expected': category,
                'confidence': confidence,
                'correct': is_correct,
            })
        
        total = sum(len(matches) for matches in results.values())
        accuracy = correct / total * 100 if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'by_framework': dict(results),
        }


def main():
    """测试高精度检测器"""
    logger.info(f"\n{'='*70}")
    logger.info("🎯 高精度检测器 v2.0 - 目标90%+准确率")
    logger.info(f"{'='*70}\n")
    
    # 加载所有标注后的数据集
    websites = []
    data_files = [
        Path("training/real_data/annotated/final_annotated.jsonl"),
        Path("training/real_data/annotated/expanded_annotated.jsonl"),
        Path("training/real_data/annotated/websites_annotated.jsonl"),
    ]
    
    for data_file in data_files:
        if data_file.exists():
            with open(data_file) as f:
                for line in f:
                    websites.append(json.loads(line))
    
    logger.info(f"📊 测试数据: {len(websites)} 个网站\n")
    
    # 运行检测
    detector = HighPrecisionDetector()
    results = detector.batch_detect(websites)
    
    logger.info(f"🎯 检测结果:")
    logger.info(f"  总体准确率: {results['accuracy']:.2f}%")
    logger.info(f"  正确: {results['correct']}/{results['total']}\n")
    
    logger.info(f"  按框架分布:")
    for framework, matches in sorted(results['by_framework'].items(), key=lambda x: -len(x[1])):
        total = len(matches)
        correct = sum(1 for m in matches if m['correct'])
        logger.info(f"    {framework}: {correct}/{total} ({correct*100//total if total > 0 else 0}%)")
    
    # 分析错误
    logger.info(f"\n  错误分析:")
    for framework, matches in results['by_framework'].items():
        errors = [m for m in matches if not m['correct']]
        if errors:
            logger.info(f"    {framework}:")
            for err in errors[:3]:  # 只显示前3个
                logger.info(f"      {err['url']}: 期望={err['expected']}, 置信度={err['confidence']:.2f}")
    
    # 保存报告
    report_path = Path("models/production/HIGH_PRECISION_REPORT.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump({
            'version': '2.0',
            'accuracy': results['accuracy'],
            'total_samples': len(websites),
            'correct': results['correct'],
            'by_framework': {
                fw: {
                    'total': len(matches),
                    'correct': sum(1 for m in matches if m['correct']),
                    'accuracy': sum(1 for m in matches if m['correct']) / len(matches) * 100 if matches else 0,
                }
                for fw, matches in results['by_framework'].items()
            }
        }, f, indent=2)
    
    logger.info(f"\n📋 报告已保存: {report_path}")
    
    if results['accuracy'] >= 90:
        logger.info(f"\n✅ 目标达成！准确率 {results['accuracy']:.2f}% >= 90%")
    else:
        logger.info(f"\n⚠️  准确率 {results['accuracy']:.2f}% < 90%，需要继续优化")
    
    logger.info(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
