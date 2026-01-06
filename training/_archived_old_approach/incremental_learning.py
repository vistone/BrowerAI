#!/usr/bin/env python3
"""
增量学习脚本 - 边爬边学
Incremental Learning: Crawl one website → Learn immediately → Update model

优势:
1. 无需保存大量中间数据 (节省3-5GB存储)
2. 实时更新模型 (随时可用)
3. 内存友好 (只处理当前网站)
4. 中断安全 (模型已保存)
"""

import asyncio
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from prepare_website_data import WebsiteCrawler, load_urls_from_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplifiedWebsiteLearner(nn.Module):
    """简化版网站学习模型"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # 多任务头
        self.framework_classifier = nn.Linear(hidden_dim * 2, 20)  # 20种框架
        self.category_classifier = nn.Linear(hidden_dim * 2, 10)   # 10种分类
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, _) = self.lstm(embedded)
        # 使用最后一个时间步
        features = lstm_out[:, -1, :]  # (batch, hidden_dim*2)
        features = self.dropout(features)
        
        framework_logits = self.framework_classifier(features)
        category_logits = self.category_classifier(features)
        
        return framework_logits, category_logits


class WebsiteTokenizer:
    """简单的网站内容分词器"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = {}
    
    def build_vocab(self, text: str):
        """构建词汇表"""
        words = text.lower().split()
        for word in words:
            self.word_freq[word] = self.word_freq.get(word, 0) + 1
    
    def finalize_vocab(self):
        """固定词汇表（取最常见的词）"""
        sorted_words = sorted(self.word_freq.items(), key=lambda x: -x[1])
        for idx, (word, _) in enumerate(sorted_words[:self.vocab_size-2], start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, text: str, max_len: int = 512) -> list:
        """编码文本为token IDs"""
        words = text.lower().split()[:max_len]
        tokens = [self.word2idx.get(w, 1) for w in words]  # 1 = <UNK>
        
        # Padding
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        
        return tokens


class IncrementalLearner:
    """增量学习器 - 边爬边学"""
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints/incremental',
        learning_rate: float = 1e-4,
        device: str = 'auto'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"🖥️  使用设备: {self.device}")
        
        # 初始化模型
        self.model = SimplifiedWebsiteLearner().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.tokenizer = WebsiteTokenizer()
        
        # 标签映射
        self.framework_map = {}
        self.category_map = {}
        
        # 统计
        self.total_trained = 0
        self.training_history = []
        
        # 尝试加载已有模型
        self.load_checkpoint()
    
    def extract_text(self, website_data: Dict[str, Any]) -> str:
        """提取网站文本内容"""
        texts = []
        
        # 主页面HTML内容
        if 'pages' in website_data and 'main' in website_data['pages']:
            main_page = website_data['pages']['main']
            if 'html' in main_page:
                texts.append(main_page['html'][:5000])  # 限制长度
        
        # 子页面内容
        if 'pages' in website_data and 'sub_pages' in website_data['pages']:
            for page in website_data['pages']['sub_pages'][:5]:
                if 'html' in page:
                    texts.append(page['html'][:3000])
        
        # CSS内容
        if 'pages' in website_data and 'main' in website_data['pages']:
            main_page = website_data['pages']['main']
            if 'css_files' in main_page:
                for css in main_page['css_files'][:5]:
                    if 'content' in css:
                        texts.append(css['content'][:1000])
        
        # JS内容
        if 'pages' in website_data and 'main' in website_data['pages']:
            main_page = website_data['pages']['main']
            if 'js_files' in main_page:
                for js in main_page['js_files'][:5]:
                    if 'content' in js:
                        texts.append(js['content'][:1000])
        
        return ' '.join(texts)
    
    def prepare_labels(self, website_data: Dict[str, Any]) -> tuple:
        """准备标签"""
        # Framework标签
        framework = website_data.get('metadata', {}).get('framework', 'Unknown')
        if framework not in self.framework_map:
            self.framework_map[framework] = len(self.framework_map)
        framework_idx = self.framework_map[framework]
        
        # Category标签
        category = website_data.get('category', 'unknown')
        if category not in self.category_map:
            self.category_map[category] = len(self.category_map)
        category_idx = self.category_map[category]
        
        return framework_idx, category_idx
    
    def train_on_website(self, website_data: Dict[str, Any]) -> Dict[str, float]:
        """在单个网站上训练"""
        # 提取文本
        text = self.extract_text(website_data)
        if not text:
            logger.warning(f"网站 {website_data.get('url', 'unknown')} 无有效内容")
            return {
                'loss': 0.0,
                'loss_framework': 0.0,
                'loss_category': 0.0
            }
        
        # 更新词汇表
        self.tokenizer.build_vocab(text)
        
        # 编码
        tokens = self.tokenizer.encode(text)
        x = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # 准备标签
        framework_idx, category_idx = self.prepare_labels(website_data)
        y_framework = torch.tensor([framework_idx], dtype=torch.long).to(self.device)
        y_category = torch.tensor([category_idx], dtype=torch.long).to(self.device)
        
        # 训练
        self.model.train()
        self.optimizer.zero_grad()
        
        framework_logits, category_logits = self.model(x)
        
        # 多任务损失
        loss_framework = nn.CrossEntropyLoss()(framework_logits, y_framework)
        loss_category = nn.CrossEntropyLoss()(category_logits, y_category)
        loss = loss_framework + loss_category
        
        loss.backward()
        self.optimizer.step()
        
        self.total_trained += 1
        
        return {
            'loss': loss.item(),
            'loss_framework': loss_framework.item(),
            'loss_category': loss_category.item()
        }
    
    def save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer_word2idx': self.tokenizer.word2idx,
            'tokenizer_word_freq': self.tokenizer.word_freq,
            'framework_map': self.framework_map,
            'category_map': self.category_map,
            'total_trained': self.total_trained,
            'training_history': self.training_history
        }, checkpoint_path)
        
        logger.info(f"💾 检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self):
        """加载检查点"""
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        
        if not checkpoint_path.exists():
            logger.info("📝 从头开始训练")
            return
        
        logger.info(f"📂 加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.tokenizer.word2idx = checkpoint['tokenizer_word2idx']
        self.tokenizer.word_freq = checkpoint['tokenizer_word_freq']
        self.framework_map = checkpoint['framework_map']
        self.category_map = checkpoint['category_map']
        self.total_trained = checkpoint['total_trained']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"✅ 已恢复训练状态: {self.total_trained} 个网站")


async def incremental_learning_pipeline(
    urls_file: str,
    checkpoint_dir: str,
    max_depth: int = 2,
    max_pages: int = 5,
    learning_rate: float = 1e-4,
    save_frequency: int = 10
):
    """增量学习流水线：爬取 → 立即学习 → 保存"""
    
    # 加载URL列表
    logger.info(f"📋 加载URL列表: {urls_file}")
    urls = load_urls_from_file(urls_file)
    total_urls = len(urls)
    logger.info(f"📊 总共 {total_urls} 个网站")
    
    # 初始化学习器
    learner = IncrementalLearner(
        checkpoint_dir=checkpoint_dir,
        learning_rate=learning_rate
    )
    
    # 初始化爬虫
    async with WebsiteCrawler(max_files=50, max_depth=max_depth, max_pages=max_pages) as crawler:
        
        start_idx = learner.total_trained
        logger.info(f"🚀 从第 {start_idx + 1} 个网站开始")
        
        for idx, (url, category) in enumerate(urls[start_idx:], start=start_idx + 1):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"[{idx}/{total_urls}] 🌐 爬取: {url}")
                
                # 1️⃣ 爬取网站
                website_data = await crawler.crawl_website(url, category)
                
                if not website_data:
                    logger.warning(f"❌ 爬取失败: {url}")
                    continue
                
                logger.info(f"✅ 爬取完成: {website_data.get('depth', 0)} 个页面")
                
                # 2️⃣ 立即学习
                logger.info(f"🧠 开始学习...")
                losses = learner.train_on_website(website_data)
                
                if losses['loss'] > 0:
                    logger.info(f"📈 损失: {losses['loss']:.4f} "
                              f"(框架:{losses['loss_framework']:.4f}, "
                              f"分类:{losses['loss_category']:.4f})")
                else:
                    logger.info(f"⏭️  跳过（无有效内容）")
                
                # 记录历史
                learner.training_history.append({
                    'url': url,
                    'category': category,
                    'loss': losses['loss'],
                    'timestamp': datetime.now().isoformat()
                })
                
                # 3️⃣ 定期保存
                if idx % save_frequency == 0:
                    learner.save_checkpoint()
                    logger.info(f"💾 已保存检查点 ({idx}/{total_urls})")
                
            except KeyboardInterrupt:
                logger.info("\n⚠️  用户中断，保存当前进度...")
                learner.save_checkpoint()
                raise
            
            except Exception as e:
                logger.error(f"❌ 处理失败 {url}: {e}")
                continue
        
        # 最终保存
        logger.info(f"\n{'='*60}")
        logger.info("🎉 全部完成！保存最终模型...")
        learner.save_checkpoint()
        
        # 固化词汇表
        learner.tokenizer.finalize_vocab()
        
        # 保存最终模型
        final_path = Path(checkpoint_dir) / 'final_model.pt'
        torch.save({
            'model_state_dict': learner.model.state_dict(),
            'tokenizer': learner.tokenizer,
            'framework_map': learner.framework_map,
            'category_map': learner.category_map
        }, final_path)
        
        logger.info(f"✅ 最终模型已保存: {final_path}")
        logger.info(f"📊 总共训练: {learner.total_trained} 个网站")


def main():
    parser = argparse.ArgumentParser(description='增量学习 - 边爬边学')
    parser.add_argument('--urls-file', type=str, required=True,
                       help='URL列表文件')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/incremental',
                       help='检查点保存目录')
    parser.add_argument('--depth', type=int, default=2,
                       help='最大爬取深度')
    parser.add_argument('--max-pages', type=int, default=5,
                       help='每个网站最大页面数')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--save-frequency', type=int, default=10,
                       help='每N个网站保存一次检查点')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🚀 增量学习模式启动")
    logger.info("="*60)
    logger.info(f"URL文件: {args.urls_file}")
    logger.info(f"检查点目录: {args.checkpoint_dir}")
    logger.info(f"深度: {args.depth}, 页面数: {args.max_pages}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"保存频率: 每 {args.save_frequency} 个网站")
    logger.info("="*60)
    
    asyncio.run(incremental_learning_pipeline(
        urls_file=args.urls_file,
        checkpoint_dir=args.checkpoint_dir,
        max_depth=args.depth,
        max_pages=args.max_pages,
        learning_rate=args.learning_rate,
        save_frequency=args.save_frequency
    ))


if __name__ == '__main__':
    main()
