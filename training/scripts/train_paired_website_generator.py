#!/usr/bin/env python3
"""
训练配对的网站生成器（原始→简化）
不再是自编码器，而是真正的生成模型
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairedWebsiteDataset(Dataset):
    """配对网站数据集（原始→简化）"""
    
    def __init__(self, data_file, max_len=2048):
        self.max_len = max_len
        
        # 加载配对数据
        logger.info(f"Loading paired websites from {data_file}")
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    'original': item['original'],
                    'simplified': item['simplified']
                })
        
        logger.info(f"Loaded {len(self.data)} website pairs")
        
        # 构建字符词汇表（合并原始和简化的所有字符）
        all_chars = set()
        for item in self.data:
            all_chars.update(item['original'])
            all_chars.update(item['simplified'])
        
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        for i, char in enumerate(sorted(all_chars), 3):
            self.char2idx[char] = i
        
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        
        logger.info(f"Vocab size: {self.vocab_size}")
    
    def encode(self, text):
        """文本编码为token IDs"""
        return [self.char2idx.get(c, 0) for c in text[:self.max_len-2]]
    
    def decode(self, tokens):
        """token IDs解码为文本"""
        chars = [self.idx2char.get(t, '') for t in tokens if t not in [0, 1, 2]]
        return ''.join(chars)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            src: 原始网站代码序列
            tgt: 简化网站代码序列
        """
        item = self.data[idx]
        
        # 编码原始代码
        src_encoded = [1] + self.encode(item['original']) + [2]  # SOS + content + EOS
        src_tensor = torch.tensor(src_encoded, dtype=torch.long)
        
        # 编码简化代码
        tgt_encoded = [1] + self.encode(item['simplified']) + [2]
        tgt_tensor = torch.tensor(tgt_encoded, dtype=torch.long)
        
        return src_tensor, tgt_tensor


def collate_fn(batch):
    """批处理，padding到相同长度"""
    srcs, tgts = zip(*batch)
    
    # 找到最大长度
    max_src_len = max(len(s) for s in srcs)
    max_tgt_len = max(len(t) for t in tgts)
    
    # Padding
    src_padded = torch.zeros(len(srcs), max_src_len, dtype=torch.long)
    tgt_padded = torch.zeros(len(tgts), max_tgt_len, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(srcs, tgts)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
    
    return src_padded, tgt_padded


class WebsiteGenerator(nn.Module):
    """网站生成器（原始→简化）"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_len=2048):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        batch_size, src_len = src.shape
        tgt_len = tgt.shape[1]
        
        # 创建mask
        src_mask = (src == 0)
        tgt_mask = (tgt == 0)
        tgt_attn_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(src.device)
        
        # Encode原始网站
        src_emb = self.embedding(src) + self.pos_encoding[:, :src_len, :]
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        
        # Decode生成简化版本
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt_len, :]
        output = self.decoder(
            tgt_emb, 
            memory,
            tgt_mask=tgt_attn_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        
        logits = self.fc_out(output)
        return logits


def train():
    """训练配对生成模型"""
    # 超参数
    batch_size = 2
    num_epochs = 30
    learning_rate = 1e-4
    
    # 数据
    dataset = PairedWebsiteDataset('data/website_paired.jsonl', max_len=1024)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    logger.info(f"Dataset: {len(dataset)} pairs, {len(dataloader)} batches per epoch")
    
    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WebsiteGenerator(
        vocab_size=dataset.vocab_size,
        d_model=256,
        nhead=8,
        num_layers=3,
        max_len=2048
    ).to(device)
    
    logger.info(f"Model: vocab={dataset.vocab_size}, d_model=256, layers=3, device={device}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD
    
    # 训练
    checkpoint_dir = Path('checkpoints/paired_generator')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training (原始→简化)...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Forward
            tgt_input = tgt[:, :-1]  # 去掉最后一个token
            tgt_output = tgt[:, 1:]  # 去掉第一个token (SOS)
            
            logits = model(src, tgt_input)
            
            # Loss
            loss = criterion(
                logits.reshape(-1, dataset.vocab_size),
                tgt_output.reshape(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 日志
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        
        # 保存检查点
        checkpoint_path = checkpoint_dir / f'epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'vocab_size': dataset.vocab_size,
            'char2idx': dataset.char2idx,
            'idx2char': dataset.idx2char
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("✅ Training completed!")


if __name__ == '__main__':
    train()
