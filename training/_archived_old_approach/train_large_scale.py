#!/usr/bin/env python3
"""
大规模网站学习训练脚本

训练1000+网站的完整模型，支持:
- 大数据集训练
- 检查点保存
- 训练恢复
- 模型导出ONNX
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.website_dataset import WebsiteDataset
from core.data.tokenizers import CodeTokenizer
from core.models.website_learner import HolisticWebsiteLearner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LargeScaleTrainer:
    """大规模训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🖥️  训练设备: {self.device}")
        logger.info(f"📁 检查点目录: {self.checkpoint_dir}")
    
    def prepare_data(self):
        """准备数据集"""
        logger.info("📊 加载数据集...")
        
        # 创建分词器
        tokenizer = CodeTokenizer(vocab_size=config['vocab_size'])
        
        # 加载数据集
        dataset = WebsiteDataset(
            data_file=Path(config['data_file']),
            tokenizer=tokenizer,
            max_html_len=config['max_html_len'],
            max_css_len=config['max_css_len'],
            max_js_len=config['max_js_len']
        )
        
        # 划分训练/验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        logger.info(f"✅ 数据集准备完成:")
        logger.info(f"  - 训练集: {train_size} 个网站")
        logger.info(f"  - 验证集: {val_size} 个网站")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 0)
        )
        
        return train_loader, val_loader, tokenizer
    
    def create_model(self, tokenizer):
        """创建模型"""
        logger.info("🤖 创建模型...")
        
        model = HolisticWebsiteLearner(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            num_categories=len(WebsiteDataset.CATEGORIES),
            url_feature_dim=128
        )
        
        model = model.to(self.device)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ 模型创建完成:")
        logger.info(f"  - 总参数: {total_params:,}")
        logger.info(f"  - 可训练参数: {trainable_params:,}")
        logger.info(f"  - 模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
        
        return model
    
    def load_checkpoint(self, model, optimizer, scheduler):
        """加载检查点"""
        checkpoint_file = self.checkpoint_dir / "latest_checkpoint.pt"
        
        if checkpoint_file.exists():
            logger.info(f"📂 加载检查点: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            logger.info(f"✅ 检查点加载完成 (从epoch {start_epoch}继续)")
            return start_epoch, best_val_loss
        
        return 0, float('inf')
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存最新检查点
        latest_file = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_file)
        
        # 保存最佳模型
        if is_best:
            best_file = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_file)
            logger.info(f"💎 保存最佳模型: {best_file}")
        
        # 定期保存epoch检查点
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            epoch_file = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, epoch_file)
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            html_ids = batch['html_ids'].to(self.device)
            css_ids = batch['css_ids'].to(self.device)
            js_ids = batch['js_ids'].to(self.device)
            url_features = batch['url_features'].to(self.device)
            category = batch['category'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(html_ids, css_ids, js_ids, url_features)
            
            # 计算损失
            loss = criterion(outputs['category_logits'], category)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs['category_logits'].max(1)
            total += category.size(0)
            correct += predicted.eq(category).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
        
        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total
        
        return avg_loss, acc
    
    def validate(self, model, val_loader, criterion):
        """验证"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证"):
                html_ids = batch['html_ids'].to(self.device)
                css_ids = batch['css_ids'].to(self.device)
                js_ids = batch['js_ids'].to(self.device)
                url_features = batch['url_features'].to(self.device)
                category = batch['category'].to(self.device)
                
                outputs = model(html_ids, css_ids, js_ids, url_features)
                loss = criterion(outputs['category_logits'], category)
                
                total_loss += loss.item()
                _, predicted = outputs['category_logits'].max(1)
                total += category.size(0)
                correct += predicted.eq(category).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        acc = 100. * correct / total
        
        return avg_loss, acc
    
    def export_onnx(self, model, tokenizer):
        """导出ONNX模型"""
        logger.info("📦 导出ONNX模型...")
        
        model.eval()
        model.to('cpu')
        
        # 创建示例输入
        batch_size = 1
        html_ids = torch.randint(0, config['vocab_size'], (batch_size, config['max_html_len']))
        css_ids = torch.randint(0, config['vocab_size'], (batch_size, config['max_css_len']))
        js_ids = torch.randint(0, config['vocab_size'], (batch_size, config['max_js_len']))
        url_features = torch.randn(batch_size, 128)
        
        # 导出
        onnx_file = self.checkpoint_dir / "website_learner.onnx"
        torch.onnx.export(
            model,
            (html_ids, css_ids, js_ids, url_features),
            onnx_file,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['html_ids', 'css_ids', 'js_ids', 'url_features'],
            output_names=['category_logits', 'framework_logits', 'style_embedding'],
            dynamic_axes={
                'html_ids': {0: 'batch_size'},
                'css_ids': {0: 'batch_size'},
                'js_ids': {0: 'batch_size'},
                'url_features': {0: 'batch_size'}
            }
        )
        
        logger.info(f"✅ ONNX模型已导出: {onnx_file}")
    
    def train(self):
        """完整训练流程"""
        logger.info("\n" + "="*60)
        logger.info("🚀 开始大规模网站学习训练")
        logger.info("="*60 + "\n")
        
        # 准备数据
        train_loader, val_loader, tokenizer = self.prepare_data()
        
        # 创建模型
        model = self.create_model(tokenizer)
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs']
        )
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 加载检查点
        start_epoch, best_val_loss = self.load_checkpoint(model, optimizer, scheduler)
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info(f"\n📈 开始训练 (epochs: {start_epoch} -> {config['epochs']})\n")
        
        # 训练循环
        for epoch in range(start_epoch, config['epochs']):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{config['epochs']}")
            logger.info(f"{'='*60}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # 验证
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # 打印结果
            logger.info(f"\n训练: loss={train_loss:.4f}, acc={train_acc:.2f}%")
            logger.info(f"验证: loss={val_loss:.4f}, acc={val_acc:.2f}%")
            logger.info(f"学习率: {scheduler.get_last_lr()[0]:.6f}")
            
            # 保存检查点
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(model, optimizer, scheduler, epoch, val_loss, is_best)
            
            # 保存历史
            history_file = self.checkpoint_dir / "training_history.json"
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("🎉 训练完成!")
        logger.info(f"最佳验证损失: {best_val_loss:.4f}")
        logger.info("="*60 + "\n")
        
        # 导出ONNX
        self.export_onnx(model, tokenizer)
        
        return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="大规模网站学习训练")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--data-file", type=Path, default=Path("data/websites/large_train.jsonl"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/large_scale"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true", help="从检查点恢复")
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'data_file': str(args.data_file),
        'checkpoint_dir': str(args.checkpoint_dir),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'vocab_size': 10000,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'max_html_len': 2048,
        'max_css_len': 1024,
        'max_js_len': 2048,
        'num_workers': 4,
        'weight_decay': 0.01,
        'save_interval': 5
    }
    
    # 创建训练器
    trainer = LargeScaleTrainer(config)
    
    # 开始训练
    model, history = trainer.train()
    
    logger.info("\n✨ 所有任务完成!")
