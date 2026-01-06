#!/usr/bin/env python3
"""
代码语义特征提取器 - 轻量级实现
不依赖大型预训练模型，使用AST和启发式方法提取语义特征
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import hashlib


class CodeSemanticExtractor:
    """轻量级代码语义特征提取器"""
    
    def __init__(self):
        # 语义关键词词典
        self.html_semantic_tags = {
            'header', 'nav', 'main', 'article', 'section', 'aside', 
            'footer', 'figure', 'figcaption', 'details', 'summary'
        }
        
        self.js_semantic_patterns = {
            'event_listener': r'addEventListener\(',
            'promise': r'\.then\(|\.catch\(|new Promise\(',
            'async_await': r'async\s+function|await\s+',
            'class_definition': r'class\s+\w+',
            'arrow_function': r'=>',
            'destructuring': r'const\s*\{|\[\s*\w+\s*\]',
            'spread_operator': r'\.\.\.',
            'template_literal': r'`[^`]*`',
            'module_import': r'import\s+.*from',
            'module_export': r'export\s+(default|const|class|function)',
        }
        
        self.css_semantic_features = {
            'grid_layout': r'display:\s*grid|grid-template',
            'flex_layout': r'display:\s*flex|flex-direction',
            'responsive': r'@media',
            'animation': r'@keyframes|animation:',
            'transition': r'transition:',
            'variable': r'var\(--',
            'pseudo_class': r':[a-z-]+\(',
            'pseudo_element': r'::[a-z-]+',
        }
    
    def extract_html_semantic(self, html: str) -> Dict[str, Any]:
        """提取HTML语义特征"""
        features = {}
        
        # 语义标签使用率
        semantic_count = sum(html.lower().count(f'<{tag}') for tag in self.html_semantic_tags)
        all_tags = len(re.findall(r'<(\w+)', html))
        features['semantic_tag_ratio'] = semantic_count / max(all_tags, 1)
        
        # 文档结构
        features['has_header'] = '<header' in html.lower()
        features['has_nav'] = '<nav' in html.lower()
        features['has_main'] = '<main' in html.lower()
        features['has_footer'] = '<footer' in html.lower()
        features['has_article'] = '<article' in html.lower()
        
        # ARIA 可访问性
        aria_count = len(re.findall(r'aria-[\w-]+', html))
        features['aria_usage'] = aria_count / max(all_tags, 1)
        
        # 微数据/Schema.org
        features['has_microdata'] = 'itemscope' in html or 'itemtype' in html
        features['has_json_ld'] = 'application/ld+json' in html
        
        # 表单结构
        form_elements = ['input', 'select', 'textarea', 'button']
        features['form_complexity'] = sum(html.lower().count(f'<{el}') for el in form_elements)
        
        # 多媒体内容
        features['image_count'] = html.lower().count('<img')
        features['video_count'] = html.lower().count('<video')
        features['audio_count'] = html.lower().count('<audio')
        
        # 交互元素
        features['interactive_count'] = (
            html.count('onclick') + 
            html.count('onsubmit') +
            html.count('data-action')
        )
        
        return features
    
    def extract_js_semantic(self, js_code: str) -> Dict[str, Any]:
        """提取JavaScript语义特征"""
        features = {}
        
        # 代码模式检测
        for pattern_name, pattern in self.js_semantic_patterns.items():
            matches = len(re.findall(pattern, js_code))
            features[f'pattern_{pattern_name}'] = matches
        
        # 编程范式
        features['oop_score'] = (
            js_code.count('class ') +
            js_code.count('this.') +
            js_code.count('prototype.')
        )
        
        features['functional_score'] = (
            js_code.count('.map(') +
            js_code.count('.filter(') +
            js_code.count('.reduce(') +
            js_code.count('=>')
        )
        
        # 框架特征
        features['react_signals'] = (
            js_code.count('React.') +
            js_code.count('useState') +
            js_code.count('useEffect') +
            js_code.count('jsx')
        )
        
        features['vue_signals'] = (
            js_code.count('Vue.') +
            js_code.count('v-if') +
            js_code.count('v-for') +
            js_code.count('$emit')
        )
        
        features['jquery_signals'] = (
            js_code.count('$(') +
            js_code.count('jQuery(') +
            js_code.count('.ajax')
        )
        
        # 异步编程复杂度
        features['async_complexity'] = (
            features['pattern_promise'] +
            features['pattern_async_await'] * 2  # async/await权重更高
        )
        
        # 模块化程度
        features['modularity'] = (
            features['pattern_module_import'] +
            features['pattern_module_export']
        )
        
        # 代码质量指标
        features['variable_count'] = len(re.findall(r'\b(const|let|var)\b', js_code))
        features['function_count'] = len(re.findall(r'\bfunction\b|\bconst\s+\w+\s*=\s*\(', js_code))
        features['code_density'] = len(js_code.split()) / max(js_code.count('\n'), 1)
        
        return features
    
    def extract_css_semantic(self, css_code: str) -> Dict[str, Any]:
        """提取CSS语义特征"""
        features = {}
        
        # 布局模式
        for pattern_name, pattern in self.css_semantic_features.items():
            matches = len(re.findall(pattern, css_code, re.IGNORECASE))
            features[f'feature_{pattern_name}'] = matches
        
        # 选择器复杂度
        selectors = re.findall(r'([^{]+)\s*\{', css_code)
        if selectors:
            avg_selector_parts = sum(s.count(' ') + s.count('>') + s.count('+') for s in selectors) / len(selectors)
            features['selector_complexity'] = avg_selector_parts
        else:
            features['selector_complexity'] = 0
        
        # 颜色使用
        hex_colors = len(re.findall(r'#[0-9a-fA-F]{3,6}', css_code))
        rgb_colors = len(re.findall(r'rgba?\(', css_code))
        features['color_diversity'] = hex_colors + rgb_colors
        
        # CSS变量使用（现代化指标）
        features['uses_variables'] = '--' in css_code
        features['variable_count'] = css_code.count('var(--')
        
        # 响应式设计
        media_queries = len(re.findall(r'@media', css_code))
        features['responsive_design'] = media_queries
        
        # 动画和过渡
        features['animation_usage'] = (
            features['feature_animation'] +
            features['feature_transition']
        )
        
        # 浏览器前缀（兼容性指标）
        prefixes = ['-webkit-', '-moz-', '-ms-', '-o-']
        features['browser_prefix_count'] = sum(css_code.count(prefix) for prefix in prefixes)
        
        # 伪元素/伪类使用（高级特性）
        features['advanced_selectors'] = (
            features['feature_pseudo_class'] +
            features['feature_pseudo_element']
        )
        
        return features
    
    def compute_semantic_hash(self, features: Dict[str, Any]) -> str:
        """计算特征向量的语义哈希"""
        # 将特征转换为稳定的字符串表示
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()[:16]
    
    def extract_all(self, html: str, css: str, js: str) -> Dict[str, Any]:
        """提取完整页面的语义特征"""
        html_features = self.extract_html_semantic(html)
        css_features = self.extract_css_semantic(css)
        js_features = self.extract_js_semantic(js)
        
        # 组合特征
        combined = {
            'html': html_features,
            'css': css_features,
            'js': js_features,
        }
        
        # 计算整体语义向量
        all_numeric = []
        for category in combined.values():
            for value in category.values():
                if isinstance(value, (int, float)):
                    all_numeric.append(value)
                elif isinstance(value, bool):
                    all_numeric.append(1.0 if value else 0.0)
        
        combined['semantic_vector'] = all_numeric
        combined['semantic_hash'] = self.compute_semantic_hash(combined)
        combined['vector_dim'] = len(all_numeric)
        
        return combined


def main():
    """测试语义提取器"""
    import sys
    
    extractor = CodeSemanticExtractor()
    
    # 读取现有反馈数据测试
    data_dir = Path(__file__).parent.parent / 'data'
    feedback_files = list(data_dir.glob('feedback_*.json'))
    
    if not feedback_files:
        print("❌ 没有找到反馈数据文件")
        return 1
    
    print(f"📊 处理 {len(feedback_files)} 个反馈文件...")
    
    results = []
    processed = 0
    
    for feedback_file in feedback_files[:10]:  # 先测试10个
        try:
            with open(feedback_file) as f:
                data = json.load(f)
            
            # 数据可能是列表或字典
            events = data if isinstance(data, list) else data.get('events', [])
            
            # 提取内容
            html_content = ""
            css_content = ""
            js_content = ""
            url = "unknown"
            
            for event in events:
                event_type = event.get('type', event.get('event_type', ''))
                
                if event_type == 'html_parsing':
                    # 尝试从event中提取内容（如果有的话）
                    pass
                elif event_type == 'css_parsing':
                    pass
                elif event_type == 'js_parsing':
                    pass
            
            # 如果没有内容，使用提取的特征文件
            if not (html_content or css_content or js_content):
                # 跳过空文件
                continue
            
            features = extractor.extract_all(html_content, css_content, js_content)
            
            results.append({
                'file': feedback_file.name,
                'url': url,
                'features': features,
            })
            
            processed += 1
        
        except Exception as e:
            print(f"⚠️ 处理 {feedback_file.name} 失败: {e}")
    
    print(f"\n✅ 成功提取 {processed} 个样本的语义特征")
    
    if results:
        # 显示第一个样本的特征
        sample = results[0]
        print(f"\n📋 样本示例: {sample['url']}")
        print(f"   语义向量维度: {sample['features']['vector_dim']}")
        print(f"   语义哈希: {sample['features']['semantic_hash']}")
        print(f"\n   HTML特征:")
        for k, v in list(sample['features']['html'].items())[:5]:
            print(f"      {k}: {v}")
        print(f"\n   JS特征:")
        for k, v in list(sample['features']['js'].items())[:5]:
            print(f"      {k}: {v}")
        print(f"\n   CSS特征:")
        for k, v in list(sample['features']['css'].items())[:5]:
            print(f"      {k}: {v}")
        
        # 保存结果
        output_file = Path(__file__).parent.parent / 'features' / 'semantic_features.json'
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 语义特征已保存: {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
