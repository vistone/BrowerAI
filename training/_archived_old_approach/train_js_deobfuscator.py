#!/usr/bin/env python3
"""
JS反混淆模型训练脚本
Seq2Seq模型: 混淆JS → 清晰JS

与BrowerAI Rust集成:
- 输入格式: [1, 60] i64 张量 (token IDs)
- 输出格式: [1, 60] i64 张量 (token IDs) 
- 词汇表大小: 160
- 特殊token: PAD=0, SOS=1, EOS=2, UNK=3

集成路径: src/ai/integration.rs::JsDeobfuscatorIntegration::deobfuscate()
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 特殊token (与Rust代码一致)
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

MAX_LENGTH = 60  # 与ONNX模型匹配
VOCAB_SIZE = 160  # 与Rust JsTokenizer匹配


class JsTokenizer:
    """
    JS Tokenizer (与 src/ai/integration.rs 的 JsTokenizer 一致)
    词汇表: 160个token
    """
    
    def __init__(self):
        self.vocab = self._build_vocab()
        self.token_to_id = {token: idx + 4 for idx, token in enumerate(self.vocab)}  # 4个特殊token
        self.token_to_id[PAD_TOKEN] = PAD_ID
        self.token_to_id[SOS_TOKEN] = SOS_ID
        self.token_to_id[EOS_TOKEN] = EOS_ID
        self.token_to_id[UNK_TOKEN] = UNK_ID
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        logger.info(f"Tokenizer initialized: {len(self.token_to_id)} tokens")
    
    def _build_vocab(self) -> List[str]:
        """构建词汇表 (与Rust代码一致)"""
        vocab = [
            # JS关键字
            'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'return',
            'class', 'new', 'this', 'super', 'import', 'export', 'async', 'await',
            'try', 'catch', 'throw', 'typeof', 'instanceof', 'extends', 'static',
            'default', 'switch', 'case', 'break', 'continue', 'do', 'in', 'of',
            'delete', 'void', 'yield', 'with', 'debugger', 'finally',
            
            # 操作符
            '=', '+', '-', '*', '/', '%', '++', '--', '==', '!=', '===', '!==',
            '<', '>', '<=', '>=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>',
            '>>>', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '>>>=',
            '?', ':', '=>', '...', '.', ',',
            
            # 符号
            '{', '}', '[', ']', '(', ')', ';',
            
            # 变量名模式
            'var0', 'var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9',
            'tmp0', 'tmp1', 'tmp2', 'tmp3', 'tmp4', 'tmp5', 'tmp6', 'tmp7', 'tmp8', 'tmp9',
            'val0', 'val1', 'val2', 'val3', 'val4', 'val5', 'val6', 'val7', 'val8', 'val9',
            'data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9',
            'result0', 'result1', 'result2', 'result3', 'result4', 'result5', 'result6', 'result7', 'result8', 'result9',
            'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9',
            
            # 单字母
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        return vocab[:156]  # 限制到156个 (160 - 4个特殊token)
    
    def tokenize(self, code: str) -> List[str]:
        """
        Tokenize JS代码 (简单空格+操作符分割)
        """
        tokens = []
        current = ''
        
        operators = ['===', '!==', '>>>', '>>=', '<<=', '>>>=', '+=', '-=', '*=', '/=',
                     '%=', '&=', '|=', '^=', '==', '!=', '<=', '>=', '&&', '||', 
                     '++', '--', '=>', '...', '<<', '>>', '>>>', 
                     '=', '+', '-', '*', '/', '%', '<', '>', '!', '&', '|', '^', '~',
                     '?', ':', '.', ',', '{', '}', '[', ']', '(', ')', ';']
        
        i = 0
        while i < len(code):
            # 尝试匹配最长操作符
            matched = False
            for op_len in [3, 2, 1]:
                if i + op_len <= len(code):
                    substr = code[i:i+op_len]
                    if substr in operators:
                        if current:
                            tokens.append(current)
                            current = ''
                        tokens.append(substr)
                        i += op_len
                        matched = True
                        break
            
            if not matched:
                ch = code[i]
                if ch.isspace():
                    if current:
                        tokens.append(current)
                        current = ''
                else:
                    current += ch
                i += 1
        
        if current:
            tokens.append(current)
        
        return tokens
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Token转ID"""
        return [self.token_to_id.get(token, UNK_ID) for token in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """ID转Token"""
        return [self.id_to_token.get(id, UNK_TOKEN) for id in ids]


class Seq2SeqDataset(Dataset):
    """JS反混淆数据集"""
    
    def __init__(self, data_pairs: List[Dict[str, str]], tokenizer: JsTokenizer, max_length: int = MAX_LENGTH):
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        pair = self.data_pairs[idx]
        
        # Tokenize
        obf_tokens = self.tokenizer.tokenize(pair['obfuscated'])
        clean_tokens = self.tokenizer.tokenize(pair['clean'])
        
        # Encode
        obf_ids = self.tokenizer.encode(obf_tokens)
        clean_ids = self.tokenizer.encode(clean_tokens)
        
        # 添加SOS/EOS并截断
        obf_ids = [SOS_ID] + obf_ids[:self.max_length-2] + [EOS_ID]
        clean_ids = [SOS_ID] + clean_ids[:self.max_length-2] + [EOS_ID]
        
        # Pad到固定长度
        obf_ids += [PAD_ID] * (self.max_length - len(obf_ids))
        clean_ids += [PAD_ID] * (self.max_length - len(clean_ids))
        
        return {
            'input': torch.tensor(obf_ids, dtype=torch.long),
            'target': torch.tensor(clean_ids, dtype=torch.long)
        }


class Seq2SeqModel(nn.Module):
    """
    Seq2Seq LSTM模型用于JS反混淆
    输入: [batch, max_len] - token IDs
    输出: [batch, max_len] - token IDs
    """
    
    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Decoder
        self.decoder = nn.LSTM(embed_dim, hidden_dim * 2, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
        logger.info(f"Model initialized: vocab_size={vocab_size}, embed_dim={embed_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, input_seq, target_seq=None):
        """
        input_seq: [batch, max_len]
        target_seq: [batch, max_len] (训练时使用teacher forcing)
        """
        batch_size = input_seq.size(0)
        max_len = input_seq.size(1)
        
        # Encode
        embedded = self.embedding(input_seq)
        encoder_output, (hidden, cell) = self.encoder(embedded)
        
        # Decoder初始状态 (合并bidirectional hidden states)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)  # [num_layers, batch, hidden*2]
        
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        
        # Decode
        if target_seq is not None:
            # Training: teacher forcing
            decoder_input = self.embedding(target_seq)
            decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(decoder_output)  # [batch, max_len, vocab_size]
        else:
            # Inference: autoregressive
            outputs = []
            decoder_input = torch.full((batch_size, 1), SOS_ID, dtype=torch.long, device=input_seq.device)
            
            for t in range(max_len):
                decoder_embedded = self.embedding(decoder_input)
                decoder_output, (hidden, cell) = self.decoder(decoder_embedded, (hidden, cell))
                output = self.fc(decoder_output)  # [batch, 1, vocab_size]
                outputs.append(output)
                
                # Next input
                decoder_input = output.argmax(dim=2)
            
            output = torch.cat(outputs, dim=1)  # [batch, max_len, vocab_size]
        
        return output


def load_data(data_file: Path) -> List[Dict[str, str]]:
    """加载混淆/反混淆数据对"""
    logger.info(f"Loading data from {data_file}")
    data_pairs = []
    
    with open(data_file, 'r') as f:
        for line in f:
            pair = json.loads(line.strip())
            if 'obfuscated' in pair and 'clean' in pair:
                # 简单清理
                obf = pair['obfuscated'].strip()
                clean = pair['clean'].strip()
                
                if obf and clean:
                    data_pairs.append({
                        'obfuscated': obf,
                        'clean': clean
                    })
    
    logger.info(f"Loaded {len(data_pairs)} data pairs")
    return data_pairs


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 50, lr: float = 1e-3, device: str = 'cpu'):
    """训练模型"""
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_seq = batch['input'].to(device)
            target_seq = batch['target'].to(device)
            
            optimizer.zero_grad()
            output = model(input_seq, target_seq[:, :-1])  # 不包括最后一个token
            
            # 计算loss
            output = output.reshape(-1, model.vocab_size)
            target = target_seq[:, 1:].reshape(-1)  # 不包括SOS
            
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_seq = batch['input'].to(device)
                target_seq = batch['target'].to(device)
                
                output = model(input_seq, target_seq[:, :-1])
                output = output.reshape(-1, model.vocab_size)
                target = target_seq[:, 1:].reshape(-1)
                
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(__file__).parent.parent / 'checkpoints' / 'js_deobfuscator'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path / 'best_model.pt')
            logger.info(f"Saved best model (val_loss: {avg_val_loss:.4f})")


def main():
    """主函数"""
    # 数据路径
    data_dir = Path(__file__).parent.parent / 'data'
    data_file = data_dir / 'obfuscation_pairs.jsonl'
    
    # 加载数据
    data_pairs = load_data(data_file)
    
    if len(data_pairs) == 0:
        logger.error("No data found! Please run data preparation first.")
        return
    
    # 划分训练/验证集
    split_idx = int(len(data_pairs) * 0.8)
    train_pairs = data_pairs[:split_idx]
    val_pairs = data_pairs[split_idx:]
    
    logger.info(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # 创建tokenizer和datasets
    tokenizer = JsTokenizer()
    train_dataset = Seq2SeqDataset(train_pairs, tokenizer)
    val_dataset = Seq2SeqDataset(val_pairs, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 创建模型
    model = Seq2SeqModel(vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256, num_layers=2)
    
    # 训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training on {device}")
    
    train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device=device)
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
