#!/usr/bin/env python3
"""
Week 6 Step 5 - 综合特征提取与增强
目标: 从 35 维扩展到 50+ 维
包括:
  1. 框架 HTML 特征 (从 framework_samples)
  2. 混淆代码特征 (从 obfuscation_samples)
  3. 交叉特征 (框架 × 混淆, 大小 × 复杂度)
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CombinedFeatureExtractor:
    """综合特征提取器"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_framework_features(self, html: str) -> Dict[str, float]:
        """从框架 HTML 中提取特征 (35 维)"""
        features = {}
        
        # ====== HTML 特征 (15 维) ======
        features['html_size'] = float(len(html))
        features['html_lines'] = float(html.count('\n'))
        
        # 标签分析
        tag_matches = re.findall(r'<([a-z][a-z0-9]*)', html, re.IGNORECASE)
        features['tag_count'] = float(len(tag_matches))
        features['unique_tags'] = float(len(set(tag_matches)))
        features['tag_diversity'] = float(len(set(tag_matches))) / max(1, len(tag_matches))
        
        # 属性分析
        attr_matches = re.findall(r'\s([a-z-]+)=["\']', html, re.IGNORECASE)
        features['attr_count'] = float(len(attr_matches))
        features['unique_attrs'] = float(len(set(attr_matches)))
        
        # 脚本和样式
        features['script_count'] = float(len(re.findall(r'<script', html, re.IGNORECASE)))
        features['style_count'] = float(len(re.findall(r'<style', html, re.IGNORECASE)))
        features['link_count'] = float(len(re.findall(r'<link', html, re.IGNORECASE)))
        
        # 其他标记
        features['entity_count'] = float(len(re.findall(r'&[a-z]+;', html, re.IGNORECASE)))
        features['comment_count'] = float(len(re.findall(r'<!--', html)))
        features['data_attr_count'] = float(len(re.findall(r'data-', html)))
        
        # 类和 ID
        features['class_count'] = float(len(re.findall(r'class=["\']([^"\']*)["\']', html)))
        features['id_count'] = float(len(re.findall(r'id=["\']([^"\']*)["\']', html)))
        
        # ====== JavaScript 特征 (10 维) ======
        script_pattern = r'<script[^>]*>(.*?)</script>'
        scripts = re.findall(script_pattern, html, re.DOTALL | re.IGNORECASE)
        js_code = '\n'.join(scripts)
        
        features['js_size'] = float(len(js_code))
        features['inline_js_blocks'] = float(len(scripts))
        
        # JS 声明
        js_decl = re.findall(r'(function|const|let|var)\s+\w+', js_code)
        features['js_declarations'] = float(len(js_decl))
        
        # 控制流密度
        control_flow = sum(len(re.findall(rf'\b{kw}\b', js_code)) 
                          for kw in ['if', 'for', 'while', 'switch', 'try'])
        features['control_flow_density'] = float(control_flow)
        
        # 混淆指标
        features['obfuscation_indicators'] = float(len(re.findall(r'_\w+|__\w+', js_code)))
        
        # 字符串密度
        strings = re.findall(r'["\']([^"\']{1,})["\']', js_code)
        features['string_count'] = float(len(strings))
        features['string_length_avg'] = float(sum(len(s) for s in strings) / max(1, len(strings)))
        
        # 正则表达式
        features['regex_count'] = float(len(re.findall(r'/[^/]+/', js_code)))
        
        # ====== CSS 特征 (10 维) ======
        style_pattern = r'<style[^>]*>(.*?)</style>'
        styles = re.findall(style_pattern, html, re.DOTALL | re.IGNORECASE)
        css_code = '\n'.join(styles)
        
        # 添加 style 属性
        style_attrs = re.findall(r'style=["\']([^"\']*)["\']', html)
        css_code += '\n'.join(style_attrs)
        
        features['css_size'] = float(len(css_code))
        features['style_blocks'] = float(len(styles))
        
        # 选择器分析
        selectors = re.findall(r'([.#\w\s>+~,\[\]=":\-\(\)]+)\s*\{', css_code)
        features['selector_count'] = float(len(selectors))
        
        # 复杂选择器
        complex_selectors = len([s for s in selectors if '>' in s or '+' in s or '~' in s])
        features['complex_selector_count'] = float(complex_selectors)
        
        # CSS 属性
        properties = re.findall(r'([a-z-]+):', css_code)
        features['property_count'] = float(len(properties))
        features['unique_properties'] = float(len(set(properties)))
        
        # 高级 CSS
        features['media_query_count'] = float(len(re.findall(r'@media', css_code)))
        features['animation_count'] = float(len(re.findall(r'@keyframes|animation:', css_code)))
        
        # ====== 复合特征 (3 维) ======
        features['complexity_score'] = (
            features['tag_count'] * 0.1 +
            features['js_size'] * 0.001 +
            features['css_size'] * 0.001
        )
        
        features['script_ratio'] = features['script_count'] / max(1, features['tag_count'])
        features['style_ratio'] = features['style_count'] / max(1, features['tag_count'])
        
        return features
    
    def extract_obfuscation_features(self, code: str) -> Dict[str, float]:
        """从混淆代码中提取特征 (8 维)"""
        features = {}
        
        # 混淆指标
        features['var_underscore_ratio'] = float(len(re.findall(r'_[a-zA-Z]', code)) / max(1, len(re.findall(r'\w+', code))))
        features['iife_count'] = float(len(re.findall(r'\(\s*function\s*\(', code)))
        features['hex_string_count'] = float(len(re.findall(r'0x[0-9a-fA-F]+', code)))
        features['obfuscated_array_count'] = float(len(re.findall(r'\[\s*["\'].*?["\']\s*\]', code)))
        
        # 代码复杂性
        features['bracket_ratio'] = float(code.count('[')) / max(1, len(code))
        features['paren_ratio'] = float(code.count('(')) / max(1, len(code))
        features['semicolon_density'] = float(code.count(';')) / max(1, len(code.split('\n')))
        
        # 熵分析
        if len(code) > 0:
            freq = {}
            for char in code:
                freq[char] = freq.get(char, 0) + 1
            entropy = 0.0
            for count in freq.values():
                p = count / len(code)
                entropy -= p * (p.bit_length() if p > 0 else 0) / 8
            features['code_entropy'] = float(entropy)
        else:
            features['code_entropy'] = 0.0
        
        return features
    
    def create_cross_features(self, framework_feats: Dict[str, float],
                            obfuscation_feats: Dict[str, float]) -> Dict[str, float]:
        """创建交叉特征 (6 维)"""
        features = {}
        
        # 框架 × 混淆 指标组合
        features['framework_obfuscation_interaction'] = (
            framework_feats.get('obfuscation_indicators', 0) *
            obfuscation_feats.get('code_entropy', 0)
        )
        
        # 大小 × 复杂度
        features['size_complexity_product'] = (
            framework_feats.get('html_size', 0) * framework_feats.get('complexity_score', 0) / 1e6
        )
        
        # JS × CSS 相互作用
        features['js_css_ratio'] = (
            framework_feats.get('js_size', 0) / max(1, framework_feats.get('css_size', 1))
        )
        
        # 代码密度
        total_size = (framework_feats.get('html_size', 0) +
                     framework_feats.get('js_size', 0) +
                     framework_feats.get('css_size', 0))
        features['code_density'] = (
            obfuscation_feats.get('code_entropy', 0) / max(1, total_size / 1000)
        )
        
        # 动态性指标
        features['dynamic_score'] = (
            framework_feats.get('script_count', 0) +
            obfuscation_feats.get('iife_count', 0) * 2 +
            framework_feats.get('control_flow_density', 0) / 10
        )
        
        # 混淆程度
        features['obfuscation_level'] = (
            obfuscation_feats.get('var_underscore_ratio', 0) * 10 +
            obfuscation_feats.get('hex_string_count', 0) +
            obfuscation_feats.get('code_entropy', 0) * 5
        )
        
        return features
    
    def extract_all_features(self, html: str, obfuscation_code: str = '') -> Dict[str, float]:
        """提取所有特征 (50+ 维)"""
        # 框架特征
        framework_feats = self.extract_framework_features(html)
        
        # 混淆特征 (如果有代码)
        if obfuscation_code:
            obfuscation_feats = self.extract_obfuscation_features(obfuscation_code)
        else:
            # 从 HTML 中的 JS 模拟混淆特征
            script_pattern = r'<script[^>]*>(.*?)</script>'
            scripts = re.findall(script_pattern, html, re.DOTALL | re.IGNORECASE)
            js_code = '\n'.join(scripts)
            obfuscation_feats = self.extract_obfuscation_features(js_code)
        
        # 交叉特征
        cross_feats = self.create_cross_features(framework_feats, obfuscation_feats)
        
        # 合并所有特征
        all_features = {}
        all_features.update(framework_feats)
        all_features.update(obfuscation_feats)
        all_features.update(cross_feats)
        
        return all_features

def process_all_samples():
    """处理所有样本并提取增强特征"""
    print("🔧 综合特征提取与增强...")
    
    extractor = CombinedFeatureExtractor()
    
    # 处理框架样本
    framework_path = Path("data/week6_samples/framework_samples.jsonl")
    obfuscation_path = Path("data/week6_obfuscation/obfuscation_samples.jsonl")
    output_path = Path("data/week6_enhanced_features/combined_features.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not framework_path.exists():
        print(f"⚠️  框架样本不存在: {framework_path}")
        return
    
    processed = 0
    
    # 加载混淆样本用于交叉特征
    obfuscation_samples = {}
    if obfuscation_path.exists():
        with open(obfuscation_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                technique = sample.get('technique', 'unknown')
                if technique not in obfuscation_samples:
                    obfuscation_samples[technique] = []
                obfuscation_samples[technique].append(sample.get('obfuscated_code', ''))
    
    # 处理框架样本
    features_dict = {}
    with open(framework_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            try:
                sample = json.loads(line)
                
                # 处理生产数据格式 (没有 html 字段)
                if 'html' not in sample and 'html_size' in sample:
                    # 跳过已处理的摘要行
                    continue
                    
                html = sample.get('html', '')
                
                if not html:
                    continue
                
                # 提取增强特征
                features_dict = extractor.extract_all_features(html)
                
                output = {
                    'framework': sample.get('framework'),
                    'url': sample.get('url'),
                    'features': features_dict,
                    'feature_count': len(features_dict),
                    'feature_dimensions': list(features_dict.keys())
                }
                
                outfile.write(json.dumps(output, ensure_ascii=False) + '\n')
                processed += 1
            except Exception as e:
                logger.debug(f"跳过行: {str(e)[:50]}")
                continue
    
    print(f"  ✅ 处理完成: {processed} 个样本")
    if features_dict:
        print(f"  📊 特征维度: {len(features_dict)} 维")
    print(f"  💾 输出: {output_path}")
    
    # 生成特征统计
    return generate_feature_statistics(output_path, len(features_dict) if features_dict else 0)

def generate_feature_statistics(data_path: Path, feature_dim: int):
    """生成特征统计"""
    print("\n📈 特征统计分析...")
    
    all_features = {}
    sample_count = 0
    
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            for feat_name, feat_value in data.get('features', {}).items():
                if feat_name not in all_features:
                    all_features[feat_name] = []
                all_features[feat_name].append(feat_value)
            sample_count += 1
    
    print(f"\n  总样本数: {sample_count}")
    print(f"  特征维度: {feature_dim}")
    
    print(f"\n  特征统计 (Top 20):")
    stats = []
    for feat_name, values in all_features.items():
        import statistics
        avg = statistics.mean(values)
        try:
            stdev = statistics.stdev(values) if len(values) > 1 else 0
        except:
            stdev = 0
        stats.append((feat_name, avg, stdev))
    
    for feat_name, avg, stdev in sorted(stats, key=lambda x: abs(x[1]), reverse=True)[:20]:
        print(f"    {feat_name:30} μ:{avg:12.4f}  σ:{stdev:10.4f}")
    
    # 保存统计
    summary = {
        'total_samples': sample_count,
        'feature_dimensions': feature_dim,
        'features_statistics': {name: {'mean': float(avg), 'stdev': float(stdev)}
                               for name, avg, stdev in stats}
    }
    
    summary_path = data_path.parent / "feature_statistics.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n  📊 统计信息已保存: {summary_path}")
    
    return summary

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  Week 6 Step 5 - 综合特征提取与增强                   ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    
    result = process_all_samples()
    
    if result:
        print("\n✅ 特征增强完成！")
        print(f"\n🚀 下一步: 使用增强特征重新训练模型")
        print("  python3 training/scripts/train_with_enhanced_features.py")

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent.parent)
    main()
