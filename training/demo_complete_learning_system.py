#!/usr/bin/env python3
"""
BrowerAI Complete Learning System Integration Demo
完整的端到端学习管道演示

演示内容:
1. 特征提取 (HTML/CSS/JS → 48D向量)
2. 潜在编码 (48D → 256D)
3. 代码生成 (256D → HTML/CSS/JS)
4. 质量验证 (质量评分)
5. 学习追踪 (指标收集)
6. 批量处理 (多网站管道)
7. 模型持久化 (保存/加载)
"""

import sys
sys.path.insert(0, '/home/stone/BrowerAI/training')

from model_library import ModelLibrary, ModelLibraryConfig
import json
from datetime import datetime


def print_section(title: str, level: int = 1):
    """打印格式化的章节标题"""
    if level == 1:
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    else:
        print(f"\n► {title}")
        print("-"*70)


def demo_single_website():
    """演示1: 处理单个网站"""
    print_section("演示1: 单个网站处理", 2)
    
    # 初始化
    library = ModelLibrary()
    print("✓ 模型库已初始化")
    
    # 创建网站数据
    website = {
        'html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Tech Blog</title>
            </head>
            <body>
                <header>
                    <nav>
                        <a href="/">Home</a>
                        <a href="/blog">Blog</a>
                        <a href="/about">About</a>
                    </nav>
                </header>
                <main>
                    <article>
                        <h1>Introduction to AI</h1>
                        <p>Artificial Intelligence is transforming the world...</p>
                        <img src="ai.jpg" alt="AI illustration">
                        <button onclick="shareArticle()">Share</button>
                    </article>
                </main>
                <footer>
                    <p>&copy; 2026 Tech Blog. All rights reserved.</p>
                </footer>
            </body>
            </html>
        ''',
        'css': '''
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            header { background: rgba(0,0,0,0.8); padding: 20px; color: white; }
            nav { display: flex; gap: 20px; }
            nav a { color: white; text-decoration: none; }
            main { padding: 40px; background: white; margin: 20px; }
            article { max-width: 800px; }
            h1 { font-size: 2.5em; margin-bottom: 20px; }
            p { line-height: 1.6; margin-bottom: 15px; }
            img { max-width: 100%; height: auto; }
            button { 
                background: #667eea; 
                color: white; 
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            footer { background: #333; color: white; padding: 20px; text-align: center; }
        ''',
        'scripts': '''
            function shareArticle() {
                const title = document.querySelector('h1').innerText;
                if (navigator.share) {
                    navigator.share({
                        title: title,
                        text: 'Check out this article!',
                        url: window.location.href
                    });
                } else {
                    console.log('Share API not supported');
                }
            }
            
            document.addEventListener('DOMContentLoaded', () => {
                console.log('Blog loaded successfully');
                const links = document.querySelectorAll('nav a');
                links.forEach(link => {
                    link.addEventListener('click', (e) => {
                        if (link.href === window.location.href) {
                            e.preventDefault();
                        }
                    });
                });
            });
        ''',
        'framework': 'vanilla',
        'intent': 'blog',
        'style': 'modern',
    }
    
    print(f"网站数据: HTML({len(website['html'])}字符), CSS({len(website['css'])}字符), JS({len(website['scripts'])}字符)")
    
    # 处理
    result = library.process_website(website)
    
    # 显示结果
    print(f"\n处理结果:")
    print(f"  特征向量: {result['features'].shape}")
    print(f"  潜在向量: {result['latent'].shape}")
    print(f"  HTML质量: {result['quality_scores']['html_quality']:.2%}")
    print(f"  CSS质量:  {result['quality_scores']['css_quality']:.2%}")
    print(f"  JS质量:   {result['quality_scores']['js_quality']:.2%}")
    print(f"  总体质量: {result['quality_scores']['overall_quality']:.2%}")
    print(f"  处理时间: {result['processing_time_ms']:.2f}ms")
    
    # 显示生成的代码片段
    print(f"\n生成的代码:")
    print(f"  HTML长度: {len(result['generated_code']['html'])} 字符")
    print(f"  CSS长度:  {len(result['generated_code']['css'])} 字符")
    print(f"  JS长度:   {len(result['generated_code']['javascript'])} 字符")
    
    return result


def demo_batch_processing():
    """演示2: 批量处理多个网站"""
    print_section("演示2: 批量处理9个网站", 2)
    
    library = ModelLibrary()
    
    # 创建不同类型的网站
    websites = []
    
    # Blog网站
    for i in range(3):
        websites.append({
            'html': f'<html><head><title>Blog {i}</title></head><body><article><h1>Article {i}</h1><p>Content...</p></article></body></html>',
            'css': f'article {{ background: #f5f5f5; padding: 20px; }} h1 {{ color: #333; }}',
            'scripts': 'document.addEventListener("DOMContentLoaded", () => {});',
            'framework': 'react',
            'intent': 'blog',
            'style': 'modern',
        })
    
    # Ecommerce网站
    for i in range(3):
        websites.append({
            'html': f'<html><body><div class="shop"><h1>Shop {i}</h1><div class="products"><button>Buy</button></div></div></body></html>',
            'css': f'.shop {{ display: grid; }} .products {{ display: flex; gap: 20px; }}',
            'scripts': 'function checkout() { console.log("Checkout"); }',
            'framework': 'vue',
            'intent': 'ecommerce',
            'style': 'minimal',
        })
    
    # Portfolio网站
    for i in range(3):
        websites.append({
            'html': f'<html><body><section><h1>Portfolio {i}</h1><img src="work{i}.jpg"><p>Project {i}</p></section></body></html>',
            'css': f'section {{ text-align: center; }} img {{ max-width: 100%; }}',
            'scripts': 'gsap.to("img", { opacity: 1 });',
            'framework': 'angular',
            'intent': 'portfolio',
            'style': 'classic',
        })
    
    print(f"准备处理 {len(websites)} 个网站:")
    print(f"  - 3个博客网站 (React)")
    print(f"  - 3个电商网站 (Vue)")
    print(f"  - 3个作品集网站 (Angular)")
    
    # 批量处理
    batch_result = library.batch_process(websites)
    
    # 显示统计
    print(f"\n批量处理结果:")
    print(f"  总数:     {batch_result['total_processed']}")
    print(f"  成功:     {batch_result['successful']}")
    print(f"  失败:     {batch_result['failed']}")
    print(f"  成功率:   {(batch_result['successful']/batch_result['total_processed']*100):.1f}%")
    
    summary = batch_result['summary']
    print(f"\n性能指标:")
    print(f"  平均质量:     {summary['average_quality']:.2%}")
    print(f"  平均损失:     {summary['average_loss']:.4f}")
    print(f"  平均处理时间: {summary['average_processing_time_ms']:.2f}ms")
    print(f"  学习迭代数:   {summary['learning_iterations']}")
    
    print(f"\n框架分布:")
    for framework, count in summary['framework_distribution'].items():
        print(f"  {framework}: {count}")
    
    # 处理时间详情
    print(f"\n处理时间详情 (前3个):")
    for i, result in enumerate(batch_result['results'][:3]):
        print(f"  网站 {i}: {result['processing_time_ms']:.2f}ms")
    
    return batch_result


def demo_learning_process():
    """演示3: 模拟学习过程"""
    print_section("演示3: 模拟10轮学习迭代", 2)
    
    library = ModelLibrary()
    
    print("处理10个不同的网站...")
    for round_num in range(10):
        website = {
            'html': f'<html><body><h{(round_num % 6) + 1}>Iteration {round_num}</h{(round_num % 6) + 1}></body></html>',
            'css': f'body {{ background: hsl({round_num * 36}deg, 100%, 50%); }}',
            'scripts': f'console.log("Round {round_num}");',
            'framework': ['react', 'vue', 'angular'][round_num % 3],
            'intent': ['blog', 'ecommerce', 'portfolio'][round_num % 3],
            'style': ['modern', 'minimal'][round_num % 2],
        }
        
        result = library.process_website(website)
        
        if (round_num + 1) % 2 == 0:
            status = library.get_model_status()
            print(f"  轮次 {round_num + 1}: 样本={status['learning_tracker']['total_samples']}, "
                  f"质量={status['learning_tracker']['average_quality']:.3f}")
    
    # 最终统计
    final_status = library.get_model_status()
    summary = final_status['learning_tracker']
    
    print(f"\n最终学习摘要:")
    print(f"  处理样本:    {summary['total_samples']}")
    print(f"  平均质量:    {summary['average_quality']:.3f}")
    print(f"  平均损失:    {summary['average_loss']:.4f}")
    print(f"  平均处理时间: {summary['average_processing_time_ms']:.2f}ms")
    print(f"  框架分布:    {summary['framework_distribution']}")
    print(f"  总用时:      {summary['elapsed_seconds']:.2f}秒")
    
    return final_status


def demo_model_persistence():
    """演示4: 模型保存和加载"""
    print_section("演示4: 模型持久化", 2)
    
    import tempfile
    import os
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, 'browerai_demo_model.pkl')
    
    # 第一个库: 训练
    print("第一步: 创建并训练模型")
    library1 = ModelLibrary()
    
    for i in range(5):
        website = {
            'html': f'<h{i+1}>Site {i}</h{i+1}>',
            'css': f'h{i+1} {{ color: #333; }}',
            'scripts': 'console.log("Site");',
            'framework': 'react',
        }
        library1.process_website(website)
    
    status1 = library1.get_model_status()
    print(f"  样本数: {status1['learning_tracker']['total_samples']}")
    print(f"  质量:   {status1['learning_tracker']['average_quality']:.3f}")
    
    # 保存
    print(f"\n第二步: 保存模型到 {model_path}")
    library1.save_model(model_path)
    print("  ✓ 模型已保存")
    print(f"  文件大小: {os.path.getsize(model_path)} 字节")
    
    # 第二个库: 加载和继续
    print(f"\n第三步: 创建新库并加载模型")
    library2 = ModelLibrary()
    library2.load_model(model_path)
    print("  ✓ 模型已加载")
    
    # 继续处理
    print(f"\n第四步: 使用加载的模型继续处理")
    for i in range(5, 8):
        website = {
            'html': f'<h{i+1}>Site {i}</h{i+1}>',
            'css': f'h{i+1} {{ color: #666; }}',
            'scripts': 'console.log("New Site");',
            'framework': 'vue',
        }
        library2.process_website(website)
    
    status2 = library2.get_model_status()
    print(f"  新样本数: {status2['learning_tracker']['total_samples']}")
    print(f"  新质量:   {status2['learning_tracker']['average_quality']:.3f}")
    
    # 清理
    os.remove(model_path)
    os.rmdir(temp_dir)
    print(f"\n  ✓ 临时文件已清理")


def demo_status_monitoring():
    """演示5: 状态监控和指标"""
    print_section("演示5: 实时状态监控", 2)
    
    library = ModelLibrary()
    
    print("处理5个网站并监控实时状态...")
    
    for i in range(5):
        website = {
            'html': f'<html><h1>Monitor Demo {i}</h1></html>',
            'css': f'h1 {{ font-size: {16 + i * 4}px; }}',
            'scripts': f'const site = {i};',
            'framework': ['react', 'vue', 'angular', 'svelte', 'nextjs'][i],
        }
        
        library.process_website(website)
        
        # 每处理一个网站就显示状态
        status = library.get_model_status()
        
        print(f"\n  网站 {i+1}:")
        print(f"    特征提取数:   {status['feature_extractor']['extractions']}")
        print(f"    编码数:       {status['latent_encoder']['encodings']}")
        print(f"    生成数:       {status['code_generator']['generations']}")
        print(f"    验证数:       {status['quality_validator']['validations']}")
        
        tracker = status['learning_tracker']
        print(f"    样本总数:     {tracker['total_samples']}")
        print(f"    平均质量:     {tracker['average_quality']:.3f}")
        print(f"    平均处理时间: {tracker['average_processing_time_ms']:.2f}ms")


def main():
    """主展示函数"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  🧠 BrowerAI 完整学习系统演示                                      ║")
    print("║     Complete Learning System Integration Demo                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\n演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 执行所有演示
    print_section("开始演示", 1)
    
    try:
        # 演示1: 单个网站
        demo_single_website()
        
        # 演示2: 批量处理
        demo_batch_processing()
        
        # 演示3: 学习过程
        demo_learning_process()
        
        # 演示4: 模型持久化
        demo_model_persistence()
        
        # 演示5: 状态监控
        demo_status_monitoring()
        
        # 完成
        print_section("演示完成", 1)
        print("✅ 所有演示已成功完成！")
        print("\n关键成就:")
        print("  ✓ 48维特征提取 - 网站转换为标准特征向量")
        print("  ✓ 256维潜在编码 - 高维压缩表示")
        print("  ✓ 代码生成 - 从潜在向量生成HTML/CSS/JS")
        print("  ✓ 质量验证 - 代码质量评估")
        print("  ✓ 批量处理 - 大规模网站处理")
        print("  ✓ 学习追踪 - 完整的指标收集")
        print("  ✓ 模型持久化 - 保存/加载功能")
        print("  ✓ 实时监控 - 状态检查和指标")
        
        print("\n下一步:")
        print("  → 查看 MODEL_LIBRARY_GUIDE.md 获取详细文档")
        print("  → 查看 model_library.py 了解实现细节")
        print("  → 查看 test_model_library.py 查看8个完整测试")
        
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
