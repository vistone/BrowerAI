#!/usr/bin/env python3
"""
端到端网站生成模型 - 简单直接版本
输入: 完整网站 (HTML+CSS+JS)
输出: AI重建的网站 (不同代码，相同功能)
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebsiteDataset(Dataset):
    """完整网站数据集"""
    
    def __init__(self, data_file, max_len=2048):
        self.data = []
        self.max_len = max_len
        
        with open(data_file, 'r') as f:
            for line in f:
                site = json.loads(line)
                # 拼接完整网站代码
                full_code = (
                    site['original']['html'] + '\n' +
                    site['original']['css'] + '\n' +
                    site['original']['js']
                )
                if len(full_code) > 100:  # 至少100字符
                    self.data.append(full_code)
        
        logger.info(f"Loaded {len(self.data)} websites")
        
        # 构建词汇表
        self.build_vocab()
    
    def build_vocab(self):
        """构建字符级词汇表"""
        chars = set()
        for code in self.data:
            chars.update(code[:self.max_len])
        
        self.char_to_idx = {char: idx + 4 for idx, char in enumerate(sorted(chars))}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<SOS>'] = 1
        self.char_to_idx['<EOS>'] = 2
        self.char_to_idx['<UNK>'] = 3
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        logger.info(f"Vocab size: {self.vocab_size}")
    
    def encode(self, text):
        """文本转ID"""
        return [self.char_to_idx.get(c, 3) for c in text[:self.max_len]]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        code = self.data[idx]
        
        # 输入和目标相同（自编码器）
        # 实际应用中，目标应该是简化/重构的版本
        encoded = [1] + self.encode(code) + [2]  # SOS + content + EOS
        
        # Pad到固定长度
        if len(encoded) < self.max_len:
            encoded += [0] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        
        return torch.tensor(encoded, dtype=torch.long)


class WebsiteGenerator(nn.Module):
    """
    完整网站生成模型
    Transformer Encoder-Decoder架构
    """
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        logger.info(f"Model: vocab={vocab_size}, d_model={d_model}, layers={num_layers}")
    
    def forward(self, src, tgt=None):
        """
        src: [batch, seq_len] - 输入网站代码
        tgt: [batch, seq_len] - 目标网站代码（训练时）
        """
        batch_size, seq_len = src.shape
        
        # Encode source
        src_emb = self.embedding(src) + self.pos_encoding[:, :seq_len, :]
        src_mask = (src == 0)
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        
        if tgt is not None:
            # Training: teacher forcing
            tgt_len = tgt.size(1)
            tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt_len, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(src.device)
            tgt_padding_mask = (tgt == 0)
            
            output = self.decoder(
                tgt_emb, 
                memory, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_mask
            )
        else:
            # Inference: autoregressive
            output = self.generate(memory, src_mask, max_len=seq_len)
        
        logits = self.fc_out(output)
        return logits
    
    def generate(self, memory, memory_mask, max_len=512):
        """生成新网站代码"""
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with SOS token
        generated = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            tgt_emb = self.embedding(generated) + self.pos_encoding[:, :generated.size(1), :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1)).to(device)
            
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
            next_token_logits = self.fc_out(output[:, -1, :])
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences generated EOS
            if (next_token == 2).all():
                break
        
        return output


def train(model, dataloader, epochs=10, lr=1e-4):
    """训练模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # 输入: 完整序列, 目标: 同样序列（自编码）
            src = batch
            tgt_input = batch[:, :-1]
            tgt_output = batch[:, 1:]
            
            optimizer.zero_grad()
            
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, model.vocab_size), tgt_output.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = Path(__file__).parent.parent / 'checkpoints' / 'website_generator'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': avg_loss
        }, checkpoint_dir / f'epoch_{epoch+1}.pt')
        
        logger.info(f"Saved checkpoint: epoch_{epoch+1}.pt")


def main():
    """主函数"""
    # 加载数据
    data_file = Path(__file__).parent.parent / 'data' / 'website_complete.jsonl'
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Run: python scripts/extract_website_complete.py first")
        return
    
    dataset = WebsiteDataset(data_file, max_len=1024)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 创建模型
    model = WebsiteGenerator(
        vocab_size=dataset.vocab_size,
        d_model=256,
        nhead=8,
        num_layers=3
    )
    
    # 训练
    logger.info("Starting training...")
    train(model, dataloader, epochs=20, lr=1e-4)
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
