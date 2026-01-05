#!/usr/bin/env python3
"""
混淆检测分析工具
分析采集到的HTML/CSS/JS内容，识别混淆特征
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


class ObfuscationDetector:
    """混淆检测器"""
    
    def __init__(self):
        self.patterns = {
            # JS混淆特征
            'js_minified': re.compile(r'^[^\n]{1000,}', re.MULTILINE),  # 单行超长代码
            'js_obfuscated_vars': re.compile(r'\b(_|__|\\$|[a-z0-9]{1,2})\s*[=:]'),  # 短变量名
            'js_hex_string': re.compile(r'\\x[0-9a-fA-F]{2}'),  # 十六进制字符串
            'js_eval': re.compile(r'\beval\s*\('),  # eval使用
            'js_function_constructor': re.compile(r'\bFunction\s*\('),  # Function构造器
            'js_array_bracket': re.compile(r'\["[0-9a-zA-Z_]+"\]'),  # 数组下标访问
            'js_webpack': re.compile(r'__webpack_require__'),  # webpack打包
            'js_unicode_escape': re.compile(r'\\u[0-9a-fA-F]{4}'),  # Unicode转义
            
            # CSS混淆特征
            'css_minified': re.compile(r'^[^\n]{500,}', re.MULTILINE),  # 单行超长CSS
            'css_obfuscated_class': re.compile(r'\.[a-zA-Z0-9_]{20,}'),  # 超长类名
            'css_single_char_class': re.compile(r'\.[a-z0-9]{1,2}\{'),  # 单字符类名
            
            # HTML混淆特征
            'html_inline_js_long': re.compile(r'<script[^>]*>[^<]{1000,}</script>', re.IGNORECASE),
            'html_base64': re.compile(r'data:[^;]+;base64,'),  # base64内联
        }
        
        # 框架检测
        self.frameworks = {
            'react': re.compile(r'\breact[-.]|ReactDOM'),
            'vue': re.compile(r'\bvue[-.]|Vue\.'),
            'angular': re.compile(r'\bangular[-.]|ng-'),
            'jquery': re.compile(r'\bjquery[-.]|\$\(|jQuery'),
            'webpack': re.compile(r'__webpack'),
            'rollup': re.compile(r'__rollup'),
            'parcel': re.compile(r'__parcel'),
        }
    
    def analyze_js(self, code: str) -> Dict:
        """分析JS代码的混淆程度"""
        if not code or not code.strip():
            return {'has_code': False}
        
        features = {
            'has_code': True,
            'length': len(code),
            'lines': code.count('\n') + 1,
        }
        
        # 检测各种混淆特征
        for name, pattern in self.patterns.items():
            if name.startswith('js_'):
                matches = pattern.findall(code)
                features[name.replace('js_', '')] = len(matches) > 0
                features[name.replace('js_', '') + '_count'] = len(matches)
        
        # 检测框架
        detected_frameworks = []
        for fw, pattern in self.frameworks.items():
            if pattern.search(code):
                detected_frameworks.append(fw)
        features['frameworks'] = detected_frameworks
        
        # 计算混淆分数 (0-100)
        obfuscation_score = 0
        if features.get('minified'):
            obfuscation_score += 30
        if features.get('obfuscated_vars'):
            obfuscation_score += 20
        if features.get('hex_string'):
            obfuscation_score += 15
        if features.get('unicode_escape'):
            obfuscation_score += 15
        if features.get('eval'):
            obfuscation_score += 10
        if features.get('function_constructor'):
            obfuscation_score += 10
        
        features['obfuscation_score'] = min(obfuscation_score, 100)
        features['is_obfuscated'] = obfuscation_score > 30
        
        return features
    
    def analyze_css(self, code: str) -> Dict:
        """分析CSS代码的混淆程度"""
        if not code or not code.strip():
            return {'has_code': False}
        
        features = {
            'has_code': True,
            'length': len(code),
            'lines': code.count('\n') + 1,
        }
        
        # 检测CSS混淆特征
        for name, pattern in self.patterns.items():
            if name.startswith('css_'):
                matches = pattern.findall(code)
                features[name.replace('css_', '')] = len(matches) > 0
                features[name.replace('css_', '') + '_count'] = len(matches)
        
        # 计算混淆分数
        obfuscation_score = 0
        if features.get('minified'):
            obfuscation_score += 40
        if features.get('obfuscated_class'):
            obfuscation_score += 30
        if features.get('single_char_class'):
            obfuscation_score += 30
        
        features['obfuscation_score'] = min(obfuscation_score, 100)
        features['is_obfuscated'] = obfuscation_score > 30
        
        return features
    
    def analyze_html(self, code: str) -> Dict:
        """分析HTML代码"""
        if not code or not code.strip():
            return {'has_code': False}
        
        features = {
            'has_code': True,
            'length': len(code),
            'lines': code.count('\n') + 1,
        }
        
        # 检测内联JS和base64
        inline_scripts = self.patterns['html_inline_js_long'].findall(code)
        base64_data = self.patterns['html_base64'].findall(code)
        
        features['inline_js_count'] = len(inline_scripts)
        features['base64_count'] = len(base64_data)
        features['has_long_inline_js'] = len(inline_scripts) > 0
        features['has_base64'] = len(base64_data) > 0
        
        # 提取外部JS数量
        external_js = re.findall(r'<script[^>]*src=', code, re.IGNORECASE)
        features['external_js_count'] = len(external_js)
        
        # 提取外部CSS数量
        external_css = re.findall(r'<link[^>]*rel=["\']stylesheet', code, re.IGNORECASE)
        features['external_css_count'] = len(external_css)
        
        return features


def analyze_feedback_file(file_path: Path, detector: ObfuscationDetector) -> List[Dict]:
    """分析单个反馈文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    results = []
    for event in events:
        event_type = event.get('type')
        content = event.get('content')
        
        if not content:
            continue
        
        result = {
            'file': file_path.name,
            'type': event_type,
            'timestamp': event.get('timestamp'),
        }
        
        if event_type == 'html_parsing':
            result['analysis'] = detector.analyze_html(content)
            result['size'] = event.get('size')
        elif event_type == 'css_parsing':
            result['analysis'] = detector.analyze_css(content)
        elif event_type == 'js_parsing':
            result['analysis'] = detector.analyze_js(content)
        else:
            continue
        
        results.append(result)
    
    return results


def main():
    """主函数"""
    detector = ObfuscationDetector()
    data_dir = Path(__file__).parent.parent / 'data'
    
    # 分析所有反馈文件
    all_results = []
    feedback_files = sorted(data_dir.glob('feedback_*.json'))
    
    print(f"📊 分析 {len(feedback_files)} 个反馈文件...")
    
    for file_path in feedback_files:
        try:
            results = analyze_feedback_file(file_path, detector)
            all_results.extend(results)
        except Exception as e:
            print(f"❌ 处理 {file_path.name} 失败: {e}")
    
    # 统计结果
    print(f"\n✅ 共分析 {len(all_results)} 个事件\n")
    
    # 分类统计
    html_count = sum(1 for r in all_results if r['type'] == 'html_parsing')
    css_count = sum(1 for r in all_results if r['type'] == 'css_parsing')
    js_count = sum(1 for r in all_results if r['type'] == 'js_parsing')
    
    print(f"📝 事件类型分布:")
    print(f"   HTML: {html_count} 个")
    print(f"   CSS:  {css_count} 个")
    print(f"   JS:   {js_count} 个\n")
    
    # 混淆统计
    obfuscated_js = [r for r in all_results 
                     if r['type'] == 'js_parsing' and r['analysis'].get('has_code') 
                     and r['analysis'].get('is_obfuscated')]
    
    obfuscated_css = [r for r in all_results 
                      if r['type'] == 'css_parsing' and r['analysis'].get('has_code') 
                      and r['analysis'].get('is_obfuscated')]
    
    print(f"🔒 混淆检测结果:")
    print(f"   混淆JS:  {len(obfuscated_js)}/{js_count}")
    print(f"   混淆CSS: {len(obfuscated_css)}/{css_count}\n")
    
    # 框架统计
    framework_counts = {}
    for r in all_results:
        if r['type'] == 'js_parsing' and r['analysis'].get('has_code'):
            for fw in r['analysis'].get('frameworks', []):
                framework_counts[fw] = framework_counts.get(fw, 0) + 1
    
    if framework_counts:
        print(f"🛠️  检测到的框架/工具:")
        for fw, count in sorted(framework_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {fw}: {count} 次")
        print()
    
    # 显示混淆样本
    if obfuscated_js:
        print(f"🔍 混淆JS样本 (前3个):")
        for i, r in enumerate(obfuscated_js[:3], 1):
            analysis = r['analysis']
            print(f"\n   样本 {i}:")
            print(f"   - 文件: {r['file']}")
            print(f"   - 长度: {analysis['length']} 字符, {analysis['lines']} 行")
            print(f"   - 混淆分数: {analysis['obfuscation_score']}/100")
            print(f"   - 特征: ", end='')
            features = []
            if analysis.get('minified'): features.append('压缩')
            if analysis.get('obfuscated_vars'): features.append('混淆变量')
            if analysis.get('hex_string'): features.append('十六进制')
            if analysis.get('unicode_escape'): features.append('Unicode转义')
            if analysis.get('eval'): features.append('eval')
            if analysis.get('webpack'): features.append('webpack')
            print(', '.join(features))
    
    # 保存分析结果
    output_path = data_dir / 'obfuscation_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 分析结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
