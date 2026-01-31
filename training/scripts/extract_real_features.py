#!/usr/bin/env python3
"""
Week 6 特征提取器 - 从真实 HTML/CSS/JS 中提取特征
处理已采集的框架样本和混淆样本
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

class RealFeatureExtractor:
    """从真实网页内容中提取特征"""
    
    def extract_html_features(self, html: str) -> Dict[str, float]:
        """从 HTML 提取特征"""
        features = {}
        
        # 基本指标
        features['html_size'] = len(html)
        features['html_lines'] = html.count('\n')
        
        # 标签分析
        tag_pattern = r'<([a-z][a-z0-9]*)'
        tags = re.findall(tag_pattern, html, re.IGNORECASE)
        features['tag_count'] = len(tags)
        features['unique_tags'] = len(set(tags))
        
        # 属性分析
        attr_pattern = r'\s([a-z-]+)=["\']'
        attrs = re.findall(attr_pattern, html, re.IGNORECASE)
        features['attr_count'] = len(attrs)
        features['unique_attrs'] = len(set(attrs))
        
        # 脚本分析
        script_count = len(re.findall(r'<script', html, re.IGNORECASE))
        style_count = len(re.findall(r'<style', html, re.IGNORECASE))
        link_count = len(re.findall(r'<link', html, re.IGNORECASE))
        features['script_count'] = script_count
        features['style_count'] = style_count
        features['link_count'] = link_count
        
        # 实体分析
        entity_count = len(re.findall(r'&[a-z]+;', html, re.IGNORECASE))
        features['entity_count'] = entity_count
        
        # 注释分析
        comment_count = len(re.findall(r'<!--', html))
        features['comment_count'] = comment_count
        
        # 数据属性
        data_attr = len(re.findall(r'data-', html))
        features['data_attr_count'] = data_attr
        
        # 类和 ID
        class_count = len(re.findall(r'class=["\']([^"\']*)["\']', html))
        id_count = len(re.findall(r'id=["\']([^"\']*)["\']', html))
        features['class_count'] = class_count
        features['id_count'] = id_count
        
        # 框架检测特征
        features['vue_indicators'] = len(re.findall(r'v-|@click|:bind', html))
        features['react_indicators'] = len(re.findall(r'class=|className', html))
        features['angular_indicators'] = len(re.findall(r'\[\w+\]|\(\w+\)|#\w+', html))
        
        return features
    
    def extract_javascript_features(self, html: str) -> Dict[str, float]:
        """从 HTML 中的 JS 代码提取特征"""
        features = {}
        
        # 提取 <script> 标签内的代码
        script_pattern = r'<script[^>]*>(.*?)</script>'
        scripts = re.findall(script_pattern, html, re.DOTALL | re.IGNORECASE)
        js_code = '\n'.join(scripts)
        
        features['js_size'] = len(js_code)
        features['inline_js_blocks'] = len(scripts)
        
        # 函数和变量
        func_pattern = r'function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+'
        features['js_declarations'] = len(re.findall(func_pattern, js_code))
        
        # 关键字密度
        keywords = ['if', 'for', 'while', 'switch', 'try', 'catch']
        features['control_flow_density'] = sum(len(re.findall(rf'\b{kw}\b', js_code)) for kw in keywords)
        
        # 混淆指标
        features['obfuscation_indicators'] = len(re.findall(r'_\w+|__\w+', js_code))
        
        # 字符串密度
        strings = re.findall(r'["\']([^"\']*)["\']', js_code)
        features['string_count'] = len(strings)
        features['string_length_avg'] = sum(len(s) for s in strings) / max(1, len(strings))
        
        # 正则表达式
        features['regex_count'] = len(re.findall(r'/[^/]+/', js_code))
        
        return features
    
    def extract_css_features(self, html: str) -> Dict[str, float]:
        """从 CSS 提取特征"""
        features = {}
        
        # 提取 <style> 标签和 style 属性
        style_pattern = r'<style[^>]*>(.*?)</style>'
        styles = re.findall(style_pattern, html, re.DOTALL | re.IGNORECASE)
        css_code = '\n'.join(styles)
        
        # 提取 style 属性
        style_attr = re.findall(r'style=["\']([^"\']*)["\']', html)
        css_code += '\n'.join(style_attr)
        
        features['css_size'] = len(css_code)
        features['style_blocks'] = len(styles)
        
        # 选择器分析
        selectors = re.findall(r'([.#\w\s>+~,\[\]=":\-\(\)]+)\s*\{', css_code)
        features['selector_count'] = len(selectors)
        
        # 复杂选择器
        complex_selectors = [s for s in selectors if '>' in s or '+' in s or '~' in s]
        features['complex_selector_count'] = len(complex_selectors)
        
        # 属性分析
        properties = re.findall(r'([a-z-]+):', css_code)
        features['property_count'] = len(properties)
        features['unique_properties'] = len(set(properties))
        
        # 媒体查询
        features['media_query_count'] = len(re.findall(r'@media', css_code))
        
        # 伪选择器
        features['pseudo_selector_count'] = len(re.findall(r'::[a-z-]+|:[a-z-]+', css_code))
        
        # 动画
        features['animation_count'] = len(re.findall(r'@keyframes|animation:', css_code))
        
        return features
    
    def extract_all_features(self, html: str) -> Dict[str, float]:
        """提取所有特征"""
        features = {}
        
        # 提取各类特征
        features.update(self.extract_html_features(html))
        features.update(self.extract_javascript_features(html))
        features.update(self.extract_css_features(html))
        
        # 添加交叉特征
        features['complexity_score'] = (
            features.get('tag_count', 0) * 0.1 +
            features.get('js_size', 0) * 0.001 +
            features.get('css_size', 0) * 0.001
        )
        
        return features

def process_framework_samples():
    """处理框架样本并提取特征"""
    print("📊 处理框架样本...")
    
    jsonl_path = Path("data/week6_samples/framework_samples.jsonl")
    output_path = Path("data/week6_features/framework_features.jsonl")
    
    if not jsonl_path.exists():
        print(f"  ⚠️  输入文件不存在: {jsonl_path}")
        return 0
    
    extractor = RealFeatureExtractor()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(jsonl_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            sample = json.loads(line)
            html = sample.get('html', '')
            
            if not html:
                continue
            
            features = extractor.extract_all_features(html)
            
            output = {
                'framework': sample.get('framework'),
                'url': sample.get('url'),
                'features': features,
                'feature_count': len(features)
            }
            
            outfile.write(json.dumps(output, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"  ✅ 处理完成: {count} 个样本")
    print(f"  💾 输出: {output_path}")
    
    return count

def analyze_feature_statistics():
    """分析特征统计信息"""
    print("\n📈 特征统计分析...")
    
    input_path = Path("data/week6_features/framework_features.jsonl")
    
    if not input_path.exists():
        print(f"  ⚠️  数据文件不存在: {input_path}")
        return
    
    all_features = {}
    sample_count = 0
    
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            for feat_name, feat_value in data.get('features', {}).items():
                if feat_name not in all_features:
                    all_features[feat_name] = []
                all_features[feat_name].append(feat_value)
            sample_count += 1
    
    print(f"\n  总样本数: {sample_count}")
    print(f"  特征维度: {len(all_features)}")
    
    print("\n  特征统计 (Top 15):")
    stats = []
    for feat_name, values in all_features.items():
        import statistics
        avg = statistics.mean(values)
        try:
            stdev = statistics.stdev(values)
        except:
            stdev = 0
        stats.append((feat_name, avg, stdev))
    
    for feat_name, avg, stdev in sorted(stats, key=lambda x: abs(x[1]), reverse=True)[:15]:
        print(f"    {feat_name:25} 均值: {avg:10.2f}  标准差: {stdev:10.2f}")

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  Week 6 特征提取 - 从真实网页数据中提取               ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    
    # 处理框架样本
    count = process_framework_samples()
    
    if count > 0:
        # 分析特征
        analyze_feature_statistics()
        
        print("\n✅ 特征提取完成!")
        print("\n🚀 下一步: 使用真实特征重新训练模型")
        print("  python3 training/scripts/train_with_real_features.py")
    else:
        print("\n⚠️  没有可处理的样本数据")

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent.parent)
    main()
