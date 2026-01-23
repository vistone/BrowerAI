#!/usr/bin/env python3
"""
训练端到端网站生成器（原始→简化）

使用新的 WebsiteGenerator 模型：
- CodeEncoder: 理解原始网站 (HTML+CSS+JS 融合)
- CodeDecoder: 生成简化版本
- WebsiteIntentClassifier: 分类网站意图
- 多任务学习: 代码重建 + 意图分类

这符合项目设计要求：整体网站学习，而非独立的技术层面学习
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import logging
from pathlib import Path
import sys

# 添加模型路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.website_generator import WebsiteGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairedWebsiteDataset(Dataset):
    """
    配对网站数据集（原始→简化）
    
    数据格式:
    {
        "original": "原始网站 HTML+CSS+JS 代码",
        "simplified": "简化网站 HTML+CSS+JS 代码",
        "intent": {
            "layout": "flex|grid|float|table",
            "interaction": "click_nav|form_submit|search|scroll|hover",
            "components": "header|footer|sidebar|main|pagination",
            "theme": "dark|light|colorful|minimal"
        }
    }
    """
    
    def __init__(self, data_file, max_len=1024, token_mode='word'):
        self.max_len = max_len
        self.token_mode = token_mode  # 'word' or 'char'
        
        # 加载配对数据
        logger.info(f"Loading paired websites from {data_file}")
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} website pairs")
        
        # 构建字符词汇表（简化版：使用字符级token）
        all_chars = set()
        for item in self.data:
            all_chars.update(item['original'])
            all_chars.update(item['simplified'])
        
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        for i, char in enumerate(sorted(all_chars), 4):
            self.char2idx[char] = i
        
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        
        # 意图标签映射
        self.layout_to_idx = {
            'flex': 0, 'grid': 1, 'float': 2, 'table': 3
        }
        self.interaction_to_idx = {
            'click_nav': 0, 'form_submit': 1, 'search': 2, 'scroll': 3, 'hover': 4
        }
        self.components_to_idx = {
            'header': 0, 'footer': 1, 'sidebar': 2, 'main': 3, 'pagination': 4
        }
        self.theme_to_idx = {
            'dark': 0, 'light': 1, 'colorful': 2, 'minimal': 3
        }
        
        logger.info(f"Vocab size: {self.vocab_size}")
    
    def encode(self, text):
        """文本编码为token IDs"""
        return [self.char2idx.get(c, self.char2idx['<UNK>']) for c in text[:self.max_len-2]]
    
    def decode(self, tokens):
        """token IDs解码为文本"""
        chars = [self.idx2char.get(t, '') for t in tokens if t not in [0, 1, 2]]
        return ''.join(chars)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            original: 原始网站代码序列
            simplified: 简化网站代码序列
            intent_labels: 意图标签字典
        """
        item = self.data[idx]
        
        # 编码原始代码
        original_encoded = [1] + self.encode(item['original']) + [2]  # SOS + content + EOS
        original_tensor = torch.tensor(original_encoded, dtype=torch.long)
        
        # 编码简化代码
        simplified_encoded = [1] + self.encode(item['simplified']) + [2]
        simplified_tensor = torch.tensor(simplified_encoded, dtype=torch.long)
        
        # 意图标签
        intent = item.get('intent', {})
        intent_labels = {
            'layout': self.layout_to_idx.get(intent.get('layout', 'flex'), 0),
            'interaction': self.interaction_to_idx.get(intent.get('interaction', 'click_nav'), 0),
            'components': self.components_to_idx.get(intent.get('components', 'main'), 3),
            'theme': self.theme_to_idx.get(intent.get('theme', 'light'), 1)
        }
        
        return original_tensor, simplified_tensor, intent_labels



def collate_fn(batch):
    """批处理，padding到相同长度"""
    originals, simplified, intent_labels_list = zip(*batch)
    
    # 找到最大长度
    max_orig_len = max(len(s) for s in originals)
    max_simp_len = max(len(t) for t in simplified)
    
    # Padding
    orig_padded = torch.zeros(len(originals), max_orig_len, dtype=torch.long)
    simp_padded = torch.zeros(len(simplified), max_simp_len, dtype=torch.long)
    
    for i, (orig, simp) in enumerate(zip(originals, simplified)):
        orig_padded[i, :len(orig)] = orig
        simp_padded[i, :len(simp)] = simp
    
    # 堆叠意图标签
    intent_batch = {
        'layout': torch.tensor([labels['layout'] for labels in intent_labels_list], dtype=torch.long),
        'interaction': torch.tensor([labels['interaction'] for labels in intent_labels_list], dtype=torch.long),
        'components': torch.tensor([labels['components'] for labels in intent_labels_list], dtype=torch.long),
        'theme': torch.tensor([labels['theme'] for labels in intent_labels_list], dtype=torch.long)
    }
    
    return orig_padded, simp_padded, intent_batch


class WebsiteGeneratorTrainer:
    """网站生成器训练器"""
    
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD
    
    def train_epoch(self, dataloader, dataset, lambda_weights=None):
        """训练一个epoch"""
        if lambda_weights is None:
            lambda_weights = {
                'html': 1.0,
                'css': 1.0,
                'js': 1.0,
                'reconstruction': 0.7,
                'intent': 0.3
            }
        
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_intent_loss = 0
        
        for batch_idx, (original, simplified, intent_labels) in enumerate(dataloader):
            original = original.to(self.device)
            simplified = simplified.to(self.device)
            intent_labels = {k: v.to(self.device) for k, v in intent_labels.items()}
            
            # Forward pass
            batch_size, tgt_len = simplified.shape
            
            # 准备目标代码（去掉EOS token）
            tgt_html = simplified[:, :-1]  # 简化版本作为HTML目标
            tgt_css = simplified[:, :-1]   # 简化版本作为CSS目标
            tgt_js = simplified[:, :-1]    # 简化版本作为JS目标
            
            output = self.model(original, original, original)  # HTML, CSS, JS tokens都是原始代码
            
            # 计算损失
            loss_result = self.model.compute_loss(
                output,
                tgt_html,
                tgt_css,
                tgt_js,
                intent_labels,
                lambda_weights
            )
            
            loss = loss_result['total_loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 提取标量值用于日志
            loss_val = loss.item()
            recon_loss_val = loss_result['reconstruction_loss'].item()
            intent_loss_val = loss_result['intent_loss'].item()
            
            total_loss += loss_val
            total_recon_loss += recon_loss_val
            total_intent_loss += intent_loss_val
            
            # 日志
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Batch {batch_idx+1} - "
                    f"Loss: {loss_val:.4f}, "
                    f"Recon: {recon_loss_val:.4f}, "
                    f"Intent: {intent_loss_val:.4f}"
                )
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_intent = total_intent_loss / len(dataloader)
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon,
            'intent_loss': avg_intent
        }
    
    def save_checkpoint(self, checkpoint_path, epoch, metrics):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")


def train():
    """训练端到端网站生成器"""
    # 超参数
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
    d_model = 256
    vocab_size = None
    
    # 数据
    data_file = 'data/website_paired.jsonl'
    if not Path(data_file).exists():
        logger.warning(f"Data file {data_file} not found, using dummy data for testing")
        # 创建测试数据
        Path('data').mkdir(parents=True, exist_ok=True)
        with open(data_file, 'w') as f:
            f.write('{"original": "<html><body>Test</body></html>", "simplified": "<html><body>T</body></html>", "intent": {"layout": "flex", "interaction": "click_nav", "components": "main", "theme": "light"}}\n')
    
    dataset = PairedWebsiteDataset(data_file, max_len=512)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    vocab_size = dataset.vocab_size
    
    logger.info(f"Dataset: {len(dataset)} pairs, {len(dataloader)} batches per epoch")
    logger.info(f"Vocab size: {vocab_size}")
    
    # 模型 - 创建配置字典
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3
    }
    
    model = WebsiteGenerator(model_config).to(device)
    
    logger.info(f"Model initialized on {device}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练器
    trainer = WebsiteGeneratorTrainer(model, optimizer, device)
    
    # 训练
    checkpoint_dir = Path('checkpoints/website_generator')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Starting Training: End-to-End Website Generation")
    logger.info("Architecture: CodeEncoder -> IntentClassifier + CodeDecoder")
    logger.info("Learning: Holistic website generation (HTML+CSS+JS)")
    logger.info("=" * 60)
    
    lambda_weights = {
        'html': 1.0,
        'css': 1.0,
        'js': 1.0,
        'reconstruction': 0.7,
        'intent': 0.3
    }
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        metrics = trainer.train_epoch(dataloader, dataset, lambda_weights)
        
        logger.info(
            f"  Total Loss: {metrics['total_loss']:.4f}, "
            f"  Recon Loss: {metrics['recon_loss']:.4f}, "
            f"  Intent Loss: {metrics['intent_loss']:.4f}"
        )
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f'epoch_{epoch+1}.pt'
            trainer.save_checkpoint(checkpoint_path, epoch + 1, metrics)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training completed!")
    logger.info("=" * 60)
    
    # 保存最终模型
    final_model_path = checkpoint_dir / 'final_model.pt'
    trainer.save_checkpoint(final_model_path, num_epochs, metrics)


if __name__ == '__main__':
    train()
