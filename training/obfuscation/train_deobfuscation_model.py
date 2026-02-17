#!/usr/bin/env python3
"""
🎓 反混淆模型训练脚本

训练深度学习模型来识别和反混淆JS代码
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import sys
from pathlib import Path
import json
import re
from typing import List, Tuple, Dict
import random

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

# 导入我们的系统
from training.obfuscation.global_js_obfuscation_deobfuscation_system import (
    DeobfuscationModel,
    GlobalObfuscationKnowledgeBase,
    ObfuscationType,
    PracticalDeobfuscator
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 数据集生成
# ============================================================================

class SyntheticObfuscationDataset(Dataset):
    """合成混淆数据集"""
    
    def __init__(self, num_samples=1000, max_length=512):
        self.num_samples = num_samples
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        # 生成训练样本
        self.samples = self._generate_samples()
        
        logger.info(f"✓ 数据集生成完成")
        logger.info(f"  样本数: {len(self.samples)}")
        logger.info(f"  词汇表大小: {self.vocab_size}")
    
    def _build_vocab(self) -> Dict[str, int]:
        """构建词汇表"""
        
        # 基础tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        # JS关键字
        keywords = ['var', 'let', 'const', 'function', 'if', 'else', 'for', 'while', 
                    'return', 'console', 'log', 'String', 'fromCharCode']
        
        # 常见符号
        symbols = ['(', ')', '{', '}', '[', ']', ';', ',', '=', '+', '-', '*', '/', '.', '_']
        
        # 混淆标识符
        obf_identifiers = [f'_0x{i:04x}' for i in range(256)]
        
        # 常见单词
        words = ['hello', 'world', 'test', 'data', 'value', 'result', 'message']
        
        all_tokens = special_tokens + keywords + symbols + obf_identifiers + words
        
        return {token: idx for idx, token in enumerate(all_tokens)}
    
    def _generate_samples(self) -> List[Tuple[str, str, List[ObfuscationType]]]:
        """生成训练样本 (原始代码, 混淆代码, 混淆类型)"""
        
        samples = []
        
        for _ in range(self.num_samples):
            # 随机选择混淆类型
            obf_type = random.choice([
                ObfuscationType.STRING_ENCODING,
                ObfuscationType.IDENTIFIER_MANGLING,
                ObfuscationType.CONTROL_FLOW,
            ])
            
            # 生成原始代码
            original = self._generate_clean_code()
            
            # 应用混淆
            obfuscated = self._apply_obfuscation(original, obf_type)
            
            samples.append((original, obfuscated, [obf_type]))
        
        return samples
    
    def _generate_clean_code(self) -> str:
        """生成简单的JS代码"""
        
        templates = [
            "var message = 'hello';\nconsole.log(message);",
            "function test() {\n  return 'world';\n}",
            "var value = 42;\nvar result = value + 1;",
            "console.log('test');",
        ]
        
        return random.choice(templates)
    
    def _apply_obfuscation(self, code: str, obf_type: ObfuscationType) -> str:
        """应用混淆"""
        
        if obf_type == ObfuscationType.STRING_ENCODING:
            # 字符串编码为十六进制
            def encode_string(match):
                s = match.group(1)
                return "'" + ''.join(f'\\x{ord(c):02x}' for c in s) + "'"
            
            return re.sub(r"'([^']+)'", encode_string, code)
        
        elif obf_type == ObfuscationType.IDENTIFIER_MANGLING:
            # 标识符混淆
            identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)
            identifier_map = {name: f'_0x{random.randint(0, 65535):04x}' 
                             for name in set(identifiers) 
                             if name not in ['var', 'let', 'const', 'function', 'console', 'log', 'return']}
            
            for original, mangled in identifier_map.items():
                code = re.sub(r'\b' + original + r'\b', mangled, code)
            
            return code
        
        elif obf_type == ObfuscationType.CONTROL_FLOW:
            # 简单的控制流混淆
            return code.replace('\n', ';').replace('  ', '')
        
        return code
    
    def _tokenize(self, code: str) -> List[int]:
        """将代码转换为token序列"""
        
        tokens = []
        
        # 简单分词
        for token in re.findall(r'\w+|[^\w\s]', code):
            if token in self.vocab:
                tokens.append(self.vocab[token])
            else:
                tokens.append(self.vocab['<UNK>'])
        
        # 填充/截断
        if len(tokens) < self.max_length:
            tokens += [self.vocab['<PAD>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        return tokens
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        original, obfuscated, obf_types = self.samples[idx]
        
        # 转换为tensor
        obf_tokens = torch.tensor(self._tokenize(obfuscated), dtype=torch.long)
        orig_tokens = torch.tensor(self._tokenize(original), dtype=torch.long)
        
        # 混淆类型标签 (multi-hot encoding)
        obf_labels = torch.zeros(len(ObfuscationType), dtype=torch.float)
        for obf_type in obf_types:
            obf_labels[list(ObfuscationType).index(obf_type)] = 1.0
        
        return obf_tokens, orig_tokens, obf_labels


# ============================================================================
# 训练器
# ============================================================================

class DeobfuscationTrainer:
    """反混淆模型训练器"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # 损失函数
        self.deobf_criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略<PAD>
        self.obf_criterion = nn.BCELoss()
        
        logger.info(f"✓ 训练器初始化")
        logger.info(f"  设备: {device}")
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        
        self.model.train()
        total_loss = 0
        total_deobf_loss = 0
        total_obf_loss = 0
        
        for batch_idx, (obf_tokens, orig_tokens, obf_labels) in enumerate(dataloader):
            obf_tokens = obf_tokens.to(self.device)
            orig_tokens = orig_tokens.to(self.device)
            obf_labels = obf_labels.to(self.device)
            
            # 前向传播
            deobf_logits, obf_preds = self.model(obf_tokens)
            
            # 计算损失
            deobf_loss = self.deobf_criterion(
                deobf_logits.view(-1, deobf_logits.size(-1)),
                orig_tokens[:, 0]  # 简化: 只预测第一个token
            )
            
            obf_loss = self.obf_criterion(obf_preds, obf_labels)
            
            # 组合损失
            loss = deobf_loss + obf_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_deobf_loss += deobf_loss.item()
            total_obf_loss += obf_loss.item()
        
        return {
            'loss': total_loss / len(dataloader),
            'deobf_loss': total_deobf_loss / len(dataloader),
            'obf_loss': total_obf_loss / len(dataloader),
        }
    
    def evaluate(self, dataloader):
        """评估模型"""
        
        self.model.eval()
        total_loss = 0
        correct_obf_preds = 0
        total_samples = 0
        
        with torch.no_grad():
            for obf_tokens, orig_tokens, obf_labels in dataloader:
                obf_tokens = obf_tokens.to(self.device)
                orig_tokens = orig_tokens.to(self.device)
                obf_labels = obf_labels.to(self.device)
                
                # 前向传播
                deobf_logits, obf_preds = self.model(obf_tokens)
                
                # 计算损失
                deobf_loss = self.deobf_criterion(
                    deobf_logits.view(-1, deobf_logits.size(-1)),
                    orig_tokens[:, 0]
                )
                obf_loss = self.obf_criterion(obf_preds, obf_labels)
                loss = deobf_loss + obf_loss
                
                total_loss += loss.item()
                
                # 计算混淆类型预测准确率
                obf_preds_binary = (obf_preds > 0.5).float()
                correct_obf_preds += (obf_preds_binary == obf_labels).sum().item()
                total_samples += obf_labels.numel()
        
        return {
            'loss': total_loss / len(dataloader),
            'obf_accuracy': correct_obf_preds / total_samples,
        }
    
    def train(self, train_loader, val_loader, epochs=10):
        """完整训练流程"""
        
        logger.info(f"\n🎓 开始训练 ({epochs} epochs)")
        logger.info("="*80 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_metrics['loss'])
            
            # 日志
            logger.info(f"Epoch {epoch}/{epochs}:")
            logger.info(f"  训练损失: {train_metrics['loss']:.4f} "
                       f"(反混淆={train_metrics['deobf_loss']:.4f}, "
                       f"分类={train_metrics['obf_loss']:.4f})")
            logger.info(f"  验证损失: {val_metrics['loss']:.4f}")
            logger.info(f"  混淆类型准确率: {val_metrics['obf_accuracy']:.2%}")
            logger.info("")
            
            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                
                save_path = Path('models/deobfuscation_model_best.pth')
                save_path.parent.mkdir(exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, save_path)
                
                logger.info(f"💾 保存最佳模型 (验证损失: {best_val_loss:.4f})\n")
        
        logger.info("="*80)
        logger.info(f"✅ 训练完成! 最佳验证损失: {best_val_loss:.4f}")
        logger.info("="*80 + "\n")


# ============================================================================
# 主程序
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("🎓 反混淆模型训练系统")
    logger.info("="*80 + "\n")
    
    # 1. 生成数据集
    logger.info("【步骤1】生成训练数据集")
    logger.info("-"*80)
    
    train_dataset = SyntheticObfuscationDataset(num_samples=1000)
    val_dataset = SyntheticObfuscationDataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(val_dataset)} 样本")
    logger.info(f"  批次大小: 32\n")
    
    # 2. 创建模型
    logger.info("【步骤2】创建反混淆模型")
    logger.info("-"*80)
    
    model = DeobfuscationModel(
        vocab_size=train_dataset.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  模型参数: {total_params:,}\n")
    
    # 3. 训练模型
    logger.info("【步骤3】训练模型")
    logger.info("-"*80 + "\n")
    
    trainer = DeobfuscationTrainer(model)
    trainer.train(train_loader, val_loader, epochs=10)
    
    # 4. 保存最终模型
    logger.info("【步骤4】保存最终模型")
    logger.info("-"*80)
    
    final_save_path = Path('models/deobfuscation_model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': train_dataset.vocab_size,
        'vocab': train_dataset.vocab,
    }, final_save_path)
    
    logger.info(f"  保存至: {final_save_path}")
    logger.info(f"  模型大小: {final_save_path.stat().st_size / 1024 / 1024:.2f} MB\n")
    
    # 5. 测试反混淆
    logger.info("【步骤5】测试反混淆")
    logger.info("-"*80)
    
    test_code = "var _0x1234 = '\\x48\\x65\\x6c\\x6c\\x6f';"
    logger.info(f"  测试代码: {test_code}")
    
    deobfuscator = PracticalDeobfuscator()
    result = deobfuscator.deobfuscate(test_code)
    
    logger.info(f"  检测到: {[name for name, _ in result['improvement']['detected_obfuscators']]}")
    logger.info(f"  应用规则: {result['improvement']['applied_rules']}")
    
    # 最终总结
    logger.info("\n" + "="*80)
    logger.info("✅ 反混淆系统构建完成!")
    logger.info("="*80)
    
    logger.info(f"\n📊 系统统计:")
    logger.info(f"  训练样本: {len(train_dataset):,}")
    logger.info(f"  模型参数: {total_params:,}")
    logger.info(f"  词汇表: {train_dataset.vocab_size}")
    logger.info(f"  混淆类型: {len(ObfuscationType)}")
    
    logger.info(f"\n📦 输出文件:")
    logger.info(f"  最佳模型: models/deobfuscation_model_best.pth")
    logger.info(f"  最终模型: models/deobfuscation_model_final.pth")
    
    logger.info(f"\n🎯 下一步:")
    logger.info(f"  1. 在真实混淆JS上测试")
    logger.info(f"  2. 收集更多混淆样本")
    logger.info(f"  3. 微调模型参数")
    logger.info(f"  4. 部署为在线服务")


if __name__ == '__main__':
    main()
