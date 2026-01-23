#!/usr/bin/env python3
"""
自动数据标注器
从URL推断框架类别，重新标注所有数据
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AutoAnnotator:
    """自动标注器 - 基于URL和HTML特征"""
    
    # URL关键字映射（更全面）
    URL_PATTERNS = {
        'React': ['react', 'next', 'gatsby', 'create-react-app', 'cra-'],
        'Vue': ['vue', 'nuxt', 'vuepress', 'vuetify', 'quasar'],
        'Angular': ['angular', 'ng-', 'ngrx', 'material.angular'],
        'jQuery': ['jquery', 'plugins.jquery'],
        'Svelte': ['svelte', 'sveltekit'],
        'Express': ['express', 'expressjs', 'koa', 'fastify'],
    }
    
    # HTML特征检测（关键字密度）
    HTML_FEATURES = {
        'React': ['react', 'reactdom', 'jsx', 'usestate', 'useeffect'],
        'Vue': ['vue', 'v-bind', 'v-model', 'v-if', 'v-for'],
        'Angular': ['angular', '@angular', 'ng-app', 'ng-controller'],
        'jQuery': ['jquery', '$(', '$.ajax', '.on('],
        'Svelte': ['svelte', 'bind:', 'on:', 'sveltekit'],
        'Express': ['express', 'app.get', 'app.post', 'req.', 'res.'],
    }
    
    def infer_category_from_html(self, html: str) -> Optional[str]:
        """从HTML特征推断类别"""
        html_lower = html.lower()[:50000]  # 只检查前50KB
        
        scores = {}
        for framework, keywords in self.HTML_FEATURES.items():
            score = sum(html_lower.count(kw.lower()) for kw in keywords)
            if score > 5:  # 至少出现5次
                scores[framework] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return None
    
    def infer_category_from_url(self, url: str) -> Optional[str]:
        """从URL推断类别"""
        url_lower = url.lower()
        
        for framework, patterns in self.URL_PATTERNS.items():
            if any(pattern in url_lower for pattern in patterns):
                return framework
        
        return None
    
    def annotate_file(self, input_path: Path, output_path: Path):
        """标注单个文件"""
        annotated = []
        total = 0
        labeled_url = 0
        labeled_html = 0
        
        with open(input_path) as f:
            for line in f:
                data = json.loads(line)
                total += 1
                
                # 如果没有category，尝试推断
                if 'category' not in data or not data['category'] or data['category'] == 'Unknown':
                    # 1. 先从URL推断
                    inferred = self.infer_category_from_url(data['url'])
                    if inferred:
                        data['category'] = inferred
                        labeled_url += 1
                    else:
                        # 2. 再从HTML推断
                        html_cat = self.infer_category_from_html(data.get('html', ''))
                        if html_cat:
                            data['category'] = html_cat
                            labeled_html += 1
                        else:
                            data['category'] = 'Unknown'
                
                annotated.append(data)
        
        # 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for item in annotated:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"  {input_path.name}: {total}条, URL标注{labeled_url}条, HTML标注{labeled_html}条")
        return labeled_url + labeled_html


def main():
    """标注所有数据集"""
    logger.info("🏷️  开始自动标注...")
    
    annotator = AutoAnnotator()
    
    # 要处理的文件
    files = [
        ("training/real_data/websites/websites_data.jsonl", "training/real_data/annotated/websites_annotated.jsonl"),
        ("training/real_data/expanded/expanded_websites.jsonl", "training/real_data/annotated/expanded_annotated.jsonl"),
        ("training/real_data/final/complete_websites.jsonl", "training/real_data/annotated/final_annotated.jsonl"),
    ]
    
    total_labeled = 0
    for input_file, output_file in files:
        input_path = Path(input_file)
        if input_path.exists():
            labeled = annotator.annotate_file(input_path, Path(output_file))
            total_labeled += labeled
    
    logger.info(f"\n✅ 标注完成！共新增{total_labeled}条标签")
    logger.info(f"📂 标注数据保存在: training/real_data/annotated/")


if __name__ == "__main__":
    main()
