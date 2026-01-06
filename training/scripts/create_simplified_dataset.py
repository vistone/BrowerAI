#!/usr/bin/env python3
"""
创建简化版本数据集
输入：原始网站代码
输出：简化版网站代码（去除冗余，优化结构）
"""

import json
import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import cssutils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 禁用cssutils的警告
cssutils.log.setLevel(logging.ERROR)


class WebsiteSimplifier:
    """网站代码简化器"""
    
    def __init__(self):
        self.class_counter = 0
        self.class_map = {}
    
    def simplify_html(self, html_code):
        """
        简化HTML代码
        - 移除注释
        - 移除多余空白
        - 缩短class名称
        - 移除data-*属性
        """
        try:
            soup = BeautifulSoup(html_code, 'html.parser')
            
            # 移除注释
            for comment in soup.find_all(string=lambda text: isinstance(text, type(soup))):
                comment.extract()
            
            # 简化class名称
            for tag in soup.find_all(class_=True):
                original_classes = tag.get('class', [])
                new_classes = []
                for cls in original_classes:
                    if cls not in self.class_map:
                        self.class_counter += 1
                        self.class_map[cls] = f'c{self.class_counter}'
                    new_classes.append(self.class_map[cls])
                tag['class'] = new_classes
            
            # 移除data-*属性
            for tag in soup.find_all():
                attrs_to_remove = [attr for attr in tag.attrs if attr.startswith('data-')]
                for attr in attrs_to_remove:
                    del tag[attr]
            
            # 转换回字符串
            simplified = str(soup)
            
            # 移除多余空白
            simplified = re.sub(r'\s+', ' ', simplified)
            simplified = re.sub(r'>\s+<', '><', simplified)
            
            return simplified.strip()
        
        except Exception as e:
            logger.warning(f"HTML simplification failed: {e}")
            return html_code
    
    def simplify_css(self, css_code):
        """
        简化CSS代码
        - 合并相同规则
        - 移除注释
        - 压缩空白
        - 更新class名称映射
        """
        try:
            sheet = cssutils.parseString(css_code)
            
            simplified_rules = []
            for rule in sheet:
                if rule.type == rule.STYLE_RULE:
                    # 更新选择器中的class名
                    selector_text = rule.selectorText
                    for original, simplified in self.class_map.items():
                        selector_text = selector_text.replace(f'.{original}', f'.{simplified}')
                    
                    # 保留规则
                    simplified_rules.append(f'{selector_text}{{{rule.style.cssText}}}')
            
            simplified = ''.join(simplified_rules)
            
            # 压缩空白
            simplified = re.sub(r'\s+', ' ', simplified)
            
            return simplified.strip()
        
        except Exception as e:
            logger.warning(f"CSS simplification failed: {e}")
            # 至少做基本的class名替换
            simplified = css_code
            for original, new in self.class_map.items():
                simplified = simplified.replace(f'.{original}', f'.{new}')
            return simplified
    
    def simplify_js(self, js_code):
        """
        简化JavaScript代码
        - 移除注释
        - 移除console.log
        - 压缩空白
        """
        try:
            # 移除单行注释
            simplified = re.sub(r'//.*$', '', js_code, flags=re.MULTILINE)
            
            # 移除多行注释
            simplified = re.sub(r'/\*.*?\*/', '', simplified, flags=re.DOTALL)
            
            # 移除console.log
            simplified = re.sub(r'console\.log\([^)]*\);?', '', simplified)
            
            # 压缩空白
            simplified = re.sub(r'\s+', ' ', simplified)
            
            return simplified.strip()
        
        except Exception as e:
            logger.warning(f"JS simplification failed: {e}")
            return js_code
    
    def simplify_website(self, website_data):
        """
        简化完整网站
        
        Args:
            website_data: {html, css, js, url}
        
        Returns:
            {original, simplified}
        """
        self.class_counter = 0
        self.class_map = {}
        
        # 原始代码（合并）
        original = website_data['html'] + '\n' + website_data['css'] + '\n' + website_data['js']
        
        # 简化各部分
        html_simplified = self.simplify_html(website_data['html'])
        css_simplified = self.simplify_css(website_data['css'])
        js_simplified = self.simplify_js(website_data['js'])
        
        # 合并简化后的代码
        simplified = html_simplified + '\n' + css_simplified + '\n' + js_simplified
        
        return {
            'url': website_data['url'],
            'original': original,
            'simplified': simplified,
            'original_len': len(original),
            'simplified_len': len(simplified),
            'compression_ratio': len(simplified) / len(original) if len(original) > 0 else 1.0
        }


def create_simplified_dataset(input_file, output_file):
    """
    创建简化版本数据集
    
    Args:
        input_file: 原始网站数据 (website_complete.jsonl)
        output_file: 输出的配对数据 (website_paired.jsonl)
    """
    logger.info(f"Loading websites from {input_file}")
    
    websites = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 数据结构：{url, original: {html, css, js}, metadata}
            if 'original' in data:
                websites.append({
                    'url': data['url'],
                    'html': data['original'].get('html', ''),
                    'css': data['original'].get('css', ''),
                    'js': data['original'].get('js', '')
                })
    
    logger.info(f"Loaded {len(websites)} websites")
    
    simplifier = WebsiteSimplifier()
    paired_data = []
    
    total_original_size = 0
    total_simplified_size = 0
    
    for i, website in enumerate(websites):
        try:
            result = simplifier.simplify_website(website)
            paired_data.append(result)
            
            total_original_size += result['original_len']
            total_simplified_size += result['simplified_len']
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(websites)} websites")
        
        except Exception as e:
            logger.error(f"Failed to simplify {website.get('url', 'unknown')}: {e}")
    
    # 保存配对数据
    logger.info(f"Saving paired data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in paired_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 统计信息
    avg_compression = total_simplified_size / total_original_size if total_original_size > 0 else 1.0
    
    logger.info(f"\n✅ 简化数据集创建完成:")
    logger.info(f"  - 网站数量: {len(paired_data)}")
    logger.info(f"  - 原始代码总量: {total_original_size / 1024:.1f} KB")
    logger.info(f"  - 简化代码总量: {total_simplified_size / 1024:.1f} KB")
    logger.info(f"  - 平均压缩率: {avg_compression:.2%}")
    logger.info(f"  - 数据文件: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='创建简化版本数据集')
    parser.add_argument('--input', type=str, 
                       default='../data/website_complete.jsonl',
                       help='输入的完整网站数据')
    parser.add_argument('--output', type=str,
                       default='../data/website_paired.jsonl',
                       help='输出的配对数据')
    
    args = parser.parse_args()
    
    create_simplified_dataset(args.input, args.output)
