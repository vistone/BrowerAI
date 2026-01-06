#!/usr/bin/env python3
"""
网站生成与推理脚本

使用训练好的模型进行:
1. 网站分类预测
2. 框架识别
3. 风格分析
4. 相似网站推荐
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
import sys
from typing import Dict, List
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.tokenizers import CodeTokenizer
from core.models.website_learner import HolisticWebsiteLearner
from core.data.website_dataset import WebsiteDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebsiteInference:
    """网站推理引擎"""
    
    CATEGORIES = WebsiteDataset.CATEGORIES
    FRAMEWORKS = ["React", "Vue", "Angular", "jQuery", "Svelte", "Tailwind", "Bootstrap", "Unknown"]
    
    def __init__(self, model_path: Path, vocab_size: int = 10000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  推理设备: {self.device}")
        
        # 加载分词器
        self.tokenizer = CodeTokenizer(vocab_size=vocab_size)
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        logger.info("✅ 模型加载完成")
    
    def load_model(self, model_path: Path):
        """加载训练好的模型"""
        logger.info(f"📂 加载模型: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # 创建模型
        model = HolisticWebsiteLearner(
            vocab_size=config.get('vocab_size', 10000),
            d_model=config.get('d_model', 512),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6),
            num_categories=len(self.CATEGORIES),
            url_feature_dim=128
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def extract_url_features(self, url: str) -> torch.Tensor:
        """提取URL特征"""
        features = [0.0] * 128
        url_hash = hash(url) % 128
        features[url_hash] = 1.0
        
        if ".com" in url:
            features[0] = 1.0
        if ".org" in url:
            features[1] = 1.0
        if ".net" in url:
            features[2] = 1.0
        if ".edu" in url:
            features[3] = 1.0
        if ".gov" in url:
            features[4] = 1.0
        
        return torch.tensor(features, dtype=torch.float32)
    
    def preprocess_website(self, html: str, css: str, js: str, url: str) -> Dict[str, torch.Tensor]:
        """预处理网站数据"""
        # 分词
        html_ids = self.tokenizer.encode(html, max_length=2048)
        css_ids = self.tokenizer.encode(css, max_length=1024)
        js_ids = self.tokenizer.encode(js, max_length=2048)
        
        # 填充/截断
        html_ids = html_ids + [0] * (2048 - len(html_ids))
        css_ids = css_ids + [0] * (1024 - len(css_ids))
        js_ids = js_ids + [0] * (2048 - len(js_ids))
        
        html_ids = html_ids[:2048]
        css_ids = css_ids[:1024]
        js_ids = js_ids[:2048]
        
        # URL特征
        url_features = self.extract_url_features(url)
        
        return {
            'html_ids': torch.tensor([html_ids], dtype=torch.long),
            'css_ids': torch.tensor([css_ids], dtype=torch.long),
            'js_ids': torch.tensor([js_ids], dtype=torch.long),
            'url_features': url_features.unsqueeze(0)
        }
    
    @torch.no_grad()
    def predict(self, html: str, css: str, js: str, url: str) -> Dict:
        """预测网站属性"""
        # 预处理
        inputs = self.preprocess_website(html, css, js, url)
        
        # 移动到设备
        html_ids = inputs['html_ids'].to(self.device)
        css_ids = inputs['css_ids'].to(self.device)
        js_ids = inputs['js_ids'].to(self.device)
        url_features = inputs['url_features'].to(self.device)
        
        # 前向传播
        outputs = self.model(html_ids, css_ids, js_ids, url_features)
        
        # 类别预测
        category_probs = F.softmax(outputs['category_logits'], dim=1)[0]
        category_idx = category_probs.argmax().item()
        category_conf = category_probs[category_idx].item()
        
        # Top-3类别
        top3_categories = []
        top3_probs, top3_indices = category_probs.topk(3)
        for prob, idx in zip(top3_probs, top3_indices):
            top3_categories.append({
                'category': self.CATEGORIES[idx.item()],
                'confidence': prob.item()
            })
        
        # 框架预测
        framework_probs = F.softmax(outputs['framework_logits'], dim=1)[0]
        framework_idx = framework_probs.argmax().item()
        framework_conf = framework_probs[framework_idx].item()
        
        # 风格嵌入
        style_embedding = outputs['style_embedding'][0].cpu().numpy()
        
        return {
            'category': self.CATEGORIES[category_idx],
            'category_confidence': category_conf,
            'top3_categories': top3_categories,
            'framework': self.FRAMEWORKS[min(framework_idx, len(self.FRAMEWORKS)-1)],
            'framework_confidence': framework_conf,
            'style_embedding': style_embedding,
            'url': url
        }
    
    def predict_from_file(self, website_file: Path) -> Dict:
        """从JSONL文件预测"""
        with open(website_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
        
        # 提取数据
        if 'pages' in data:
            # 新格式（多页面）
            main_page = data['pages']['main']
            html = main_page['html']
            css = '\n'.join([f['content'] for f in main_page.get('css_files', [])])
            js = '\n'.join([f['content'] for f in main_page.get('js_files', [])])
        else:
            # 旧格式（单页面）
            html = data.get('html', '')
            css = '\n'.join([f['content'] for f in data.get('css_files', [])])
            js = '\n'.join([f['content'] for f in data.get('js_files', [])])
        
        url = data['url']
        
        return self.predict(html, css, js, url)
    
    def find_similar_websites(self, target_embedding, embeddings_db: List[Dict], top_k: int = 5) -> List[Dict]:
        """找到相似的网站"""
        similarities = []
        target_tensor = torch.tensor(target_embedding)
        
        for item in embeddings_db:
            emb_tensor = torch.tensor(item['embedding'])
            similarity = F.cosine_similarity(target_tensor, emb_tensor, dim=0).item()
            
            similarities.append({
                'url': item['url'],
                'category': item['category'],
                'similarity': similarity
            })
        
        # 排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def batch_inference(self, data_file: Path, output_file: Path, max_samples: int = None):
        """批量推理"""
        logger.info(f"📊 批量推理: {data_file}")
        
        results = []
        embeddings_db = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # 提取数据
                    if 'pages' in data:
                        main_page = data['pages']['main']
                        html = main_page['html']
                        css = '\n'.join([f['content'] for f in main_page.get('css_files', [])])
                        js = '\n'.join([f['content'] for f in main_page.get('js_files', [])])
                    else:
                        html = data.get('html', '')
                        css = '\n'.join([f['content'] for f in data.get('css_files', [])])
                        js = '\n'.join([f['content'] for f in data.get('js_files', [])])
                    
                    url = data['url']
                    
                    # 预测
                    result = self.predict(html, css, js, url)
                    results.append(result)
                    
                    # 保存嵌入
                    embeddings_db.append({
                        'url': url,
                        'category': result['category'],
                        'embedding': result['style_embedding'].tolist()
                    })
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"✅ 已处理 {i+1} 个网站")
                
                except Exception as e:
                    logger.error(f"❌ 错误 (line {i+1}): {e}")
                    continue
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'embeddings_db': embeddings_db,
                'total': len(results)
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 推理完成，结果保存到: {output_file}")
        
        # 统计
        category_counts = {}
        framework_counts = {}
        
        for r in results:
            cat = r['category']
            fw = r['framework']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            framework_counts[fw] = framework_counts.get(fw, 0) + 1
        
        logger.info("\n📊 分类统计:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat}: {count}")
        
        logger.info("\n🎨 框架统计:")
        for fw, count in sorted(framework_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {fw}: {count}")
        
        return results, embeddings_db


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="网站推理与生成")
    parser.add_argument("--model", type=Path, required=True, help="模型检查点路径")
    parser.add_argument("--mode", choices=['single', 'batch'], default='single', help="推理模式")
    parser.add_argument("--input", type=Path, help="输入文件")
    parser.add_argument("--output", type=Path, help="输出文件")
    parser.add_argument("--url", type=str, help="网站URL（单个推理）")
    parser.add_argument("--html", type=str, help="HTML内容")
    parser.add_argument("--css", type=str, help="CSS内容")
    parser.add_argument("--js", type=str, help="JS内容")
    parser.add_argument("--max-samples", type=int, help="最大样本数")
    
    args = parser.parse_args()
    
    # 创建推理引擎
    engine = WebsiteInference(args.model)
    
    if args.mode == 'single':
        if args.input:
            # 从文件推理
            result = engine.predict_from_file(args.input)
        else:
            # 从参数推理
            result = engine.predict(
                html=args.html or "",
                css=args.css or "",
                js=args.js or "",
                url=args.url or "http://example.com"
            )
        
        print("\n" + "="*60)
        print("🎯 推理结果:")
        print("="*60)
        print(f"\n📍 URL: {result['url']}")
        print(f"\n🏷️  分类: {result['category']} ({result['category_confidence']*100:.1f}%)")
        print(f"\n🎨 框架: {result['framework']} ({result['framework_confidence']*100:.1f}%)")
        print(f"\n📊 Top-3 分类:")
        for item in result['top3_categories']:
            print(f"  - {item['category']}: {item['confidence']*100:.1f}%")
        print("\n" + "="*60)
        
    elif args.mode == 'batch':
        if not args.input or not args.output:
            parser.error("批量模式需要 --input 和 --output 参数")
        
        results, embeddings_db = engine.batch_inference(
            args.input,
            args.output,
            max_samples=args.max_samples
        )
        
        print(f"\n✅ 批量推理完成: {len(results)} 个网站")


if __name__ == "__main__":
    main()
