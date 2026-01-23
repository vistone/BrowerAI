#!/usr/bin/env python3
"""
快速从 1000+ URL 库生成训练数据 (无需真实爬虫)
基于 URL 分类和特征生成多样化网站数据
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 数据模板库 ====================

# 按分类的网站模板
CATEGORY_TEMPLATES = {
    "documentation": {
        "html_pattern": """<!DOCTYPE html>
        <html>
        <head><title>{title}</title><meta name="viewport" content="width=device-width"></head>
        <body>
            <header><h1>{title}</h1><nav><a href="#docs">文档</a><a href="#api">API</a></nav></header>
            <main><section><h2>入门指南</h2><p>欢迎来到 {title}。</p></section>
            <section><h2>API 参考</h2><p>详细的接口文档</p></section></main>
            <footer><p>&copy; 2026</p></footer>
        </body>
        </html>""",
        "css_pattern": """
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; }}
        header {{ background: #2c3e50; color: white; padding: 20px; }}
        nav a {{ margin-right: 20px; color: white; text-decoration: none; }}
        main {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        section {{ margin: 20px 0; padding: 20px; background: #ecf0f1; border-radius: 5px; }}
        footer {{ background: #34495e; color: white; padding: 20px; text-align: center; margin-top: 40px; }}
        """,
        "js_pattern": """
        document.querySelectorAll('nav a').forEach(link => {{
            link.addEventListener('click', (e) => {{
                e.preventDefault();
                console.log('导航到: ' + link.textContent);
            }});
        }});
        """
    },
    
    "ecommerce": {
        "html_pattern": """<!DOCTYPE html>
        <html>
        <head><title>{title} - 购物</title><meta name="viewport" content="width=device-width"></head>
        <body>
            <header><h1>{title}</h1><input type="search" placeholder="搜索..."><span class="cart">购物车</span></header>
            <nav><a href="#electronics">电子</a><a href="#fashion">服装</a><a href="#home">家居</a></nav>
            <main><div class="products">
                <div class="product"><img src="product.jpg"><h3>产品 1</h3><p>¥99</p><button>加入购物车</button></div>
                <div class="product"><img src="product.jpg"><h3>产品 2</h3><p>¥199</p><button>加入购物车</button></div>
            </div></main>
            <footer><p>&copy; 2026</p></footer>
        </body>
        </html>""",
        "css_pattern": """
        body {{ font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }}
        header {{ background: #ff6b6b; color: white; padding: 20px; display: flex; justify-content: space-between; }}
        nav {{ display: flex; gap: 20px; padding: 10px 20px; background: white; }}
        .products {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; padding: 20px; }}
        .product {{ background: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .product button {{ background: #ff6b6b; color: white; border: none; padding: 10px 20px; cursor: pointer; }}
        """,
        "js_pattern": """
        document.querySelectorAll('.product button').forEach(btn => {{
            btn.addEventListener('click', () => {{
                console.log('产品已添加到购物车');
                btn.textContent = '已添加 ✓';
                setTimeout(() => btn.textContent = '加入购物车', 2000);
            }});
        }});
        """
    },
    
    "blog": {
        "html_pattern": """<!DOCTYPE html>
        <html>
        <head><title>{title} - 博客</title><meta name="viewport" content="width=device-width"></head>
        <body>
            <header><h1>{title}</h1><p>分享技术和生活见解</p></header>
            <nav><a href="#latest">最新</a><a href="#tech">技术</a><a href="#about">关于</a></nav>
            <main><article><h2>文章标题 1</h2><p class="date">2026-01-23</p><p>文章摘要...</p></article>
            <article><h2>文章标题 2</h2><p class="date">2026-01-22</p><p>文章摘要...</p></article></main>
            <footer><p>&copy; 2026 {title}</p></footer>
        </body>
        </html>""",
        "css_pattern": """
        body {{ font-family: Georgia, serif; line-height: 1.6; margin: 0; }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 20px; }}
        nav {{ background: white; padding: 10px 20px; }}
        nav a {{ margin-right: 20px; text-decoration: none; color: #333; }}
        main {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
        article {{ margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; }}
        .date {{ color: #666; font-size: 0.9em; }}
        footer {{ background: #333; color: white; text-align: center; padding: 20px; margin-top: 40px; }}
        """,
        "js_pattern": """
        document.querySelectorAll('article').forEach(article => {{
            article.addEventListener('click', () => {{
                console.log('阅读文章: ' + article.querySelector('h2').textContent);
            }});
        }});
        """
    },
    
    "saas": {
        "html_pattern": """<!DOCTYPE html>
        <html>
        <head><title>{title} - 云服务</title><meta name="viewport" content="width=device-width"></head>
        <body>
            <header><h1>{title}</h1><button class="cta">开始免费试用</button></header>
            <section class="hero"><h2>现代云解决方案</h2><p>简化您的工作流程</p></section>
            <section class="features">
                <div class="feature"><h3>🚀 高速</h3><p>超快的性能</p></div>
                <div class="feature"><h3>🔒 安全</h3><p>企业级安全</p></div>
                <div class="feature"><h3>💰 经济</h3><p>灵活的定价</p></div>
            </section>
            <footer><p>&copy; 2026</p></footer>
        </body>
        </html>""",
        "css_pattern": """
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; }}
        header {{ background: #0066cc; color: white; padding: 20px; display: flex; justify-content: space-between; align-items: center; }}
        .cta {{ background: #ff9900; color: white; border: none; padding: 12px 30px; font-size: 16px; cursor: pointer; border-radius: 5px; }}
        .hero {{ background: linear-gradient(135deg, #0066cc 0%, #00ccff 100%); color: white; padding: 60px 20px; text-align: center; }}
        .features {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; padding: 40px 20px; }}
        .feature {{ text-align: center; padding: 20px; }}
        footer {{ background: #333; color: white; text-align: center; padding: 20px; }}
        """,
        "js_pattern": """
        document.querySelector('.cta').addEventListener('click', () => {{
            console.log('用户点击了 CTA 按钮');
            alert('欢迎开始免费试用!');
        }});
        """
    },
}

# ==================== 数据生成器 ====================

class TrainingDataGenerator:
    """从 URL 列表生成训练数据"""
    
    def __init__(self):
        self.templates = CATEGORY_TEMPLATES
    
    def read_urls(self, urls_file: Path) -> List[Tuple[str, str]]:
        """读取 URL 列表"""
        urls = []
        with open(urls_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                url = parts[0].strip()
                category = parts[1].strip() if len(parts) > 1 else "saas"
                
                # 简化类别名称
                if category not in self.templates:
                    if 'doc' in category.lower():
                        category = 'documentation'
                    elif 'ecom' in category.lower():
                        category = 'ecommerce'
                    elif 'blog' in category.lower():
                        category = 'blog'
                    else:
                        category = 'saas'
                
                urls.append((url, category))
        
        return urls
    
    def generate_website(self, url: str, category: str) -> Dict:
        """生成一个网站的训练数据"""
        
        template = self.templates.get(category, self.templates['saas'])
        
        # 从 URL 提取标题
        title = url.split('//')[1].split('/')[0].replace('www.', '').replace('.com', '').title()
        
        html = template['html_pattern'].format(title=title)
        css = template['css_pattern']
        js = template['js_pattern']
        
        return {
            "url": url,
            "category": category,
            "input": html[:200],
            "output": html,
            "css": css,
            "js": js,
            "intent": {
                "website_type": category,
                "source": "1000_url_library",
                "has_responsive": True,
            },
            "metadata": {
                "source_url": url,
                "generation_method": "template_based"
            }
        }
    
    def generate_all(self, urls_file: Path, limit: int = 200) -> List[Dict]:
        """生成所有网站数据"""
        
        urls = self.read_urls(urls_file)
        logger.info(f"从 {len(urls)} 个 URL 生成训练数据 (限制: {limit})")
        
        websites = []
        for i, (url, category) in enumerate(urls[:limit]):
            if (i + 1) % 20 == 0:
                logger.info(f"已生成 {i + 1}/{min(limit, len(urls))} 个")
            
            website = self.generate_website(url, category)
            websites.append(website)
        
        return websites
    
    def save(self, websites: List[Dict], output_file: Path):
        """保存为 JSONL 格式"""
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for website in websites:
                f.write(json.dumps(website, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ 训练数据已保存: {output_file}")
        logger.info(f"   样本总数: {len(websites)}")
        
        # 统计
        categories = {}
        for w in websites:
            cat = w.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        logger.info(f"📊 类别分布:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            logger.info(f"   - {cat}: {count}")

# ==================== 主程序 ====================

if __name__ == "__main__":
    logger.info("🚀 从 1000+ URL 库生成训练数据")
    
    generator = TrainingDataGenerator()
    websites = generator.generate_all(
        Path("training/data/large_urls.txt"),
        limit=200
    )
    
    output_file = Path("data/website_training_1000_generated.jsonl")
    generator.save(websites, output_file)
    
    logger.info(f"""
════════════════════════════════════════════════════════════════
✅ 生成完成! 获得 {len(websites)} 个训练样本

下一步: 用生成的数据训练模型

python3 training/large_scale_website_trainer.py \\
    --data-file data/website_training_1000_generated.jsonl \\
    --epochs 40 \\
    --batch-size 8 \\
    --output-dir checkpoints/website_generator_1000_library_v1
════════════════════════════════════════════════════════════════
    """)
