#!/usr/bin/env python3
"""
增强型 Transformer 代码生成模型
支持 HTML/CSS/JS 的智能生成

特性:
1. 使用 Transformer 架构替代传统 RNN
2. 支持多任务学习（HTML/CSS/JS）
3. 注意力机制增强代码理解
4. 预训练 + 微调策略
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerCodeGenerator(nn.Module):
    """Transformer 代码生成模型"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None):
        """前向传播"""
        # 嵌入 + 位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        # Transformer 编码
        output = self.transformer_encoder(src, src_mask)
        
        # 输出投影
        output = self.output(output)
        
        return output
    
    def generate(self, prompt_ids: torch.Tensor, max_len: int = 100, temperature: float = 1.0):
        """生成代码序列"""
        self.eval()
        device = next(self.parameters()).device
        
        generated = prompt_ids.clone().to(device)
        
        with torch.no_grad():
            for _ in range(max_len):
                # 前向传播
                output = self.forward(generated)
                
                # 获取最后一个 token 的 logits
                logits = output[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # 采样下一个 token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)
                
                # 检查是否生成结束符（假设 EOS = 2）
                if next_token.item() == 2:
                    break
        
        return generated


class CodeTokenizer:
    """代码词法分析器（增强版）"""
    
    def __init__(self):
        self.PAD = '<PAD>'
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.UNK = '<UNK>'
        
        # 构建词汇表
        self.vocab = self._build_vocab()
        self.vocab2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2vocab = {idx: token for token, idx in self.vocab2idx.items()}
        self.vocab_size = len(self.vocab)
    
    def _build_vocab(self) -> List[str]:
        """构建词汇表"""
        vocab = [self.PAD, self.SOS, self.EOS, self.UNK]
        
        # HTML 标签
        html_tags = [
            'html', 'head', 'body', 'title', 'div', 'span', 'p', 'a', 'img',
            'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'form', 'input',
            'button', 'textarea', 'select', 'option', 'label', 'h1', 'h2',
            'h3', 'h4', 'h5', 'h6', 'header', 'footer', 'nav', 'section',
            'article', 'aside', 'main', 'script', 'style', 'link', 'meta'
        ]
        
        # CSS 属性
        css_props = [
            'color', 'background', 'font', 'margin', 'padding', 'border',
            'width', 'height', 'display', 'position', 'flex', 'grid',
            'align', 'justify', 'text', 'transform', 'transition', 'animation'
        ]
        
        # JS 关键字
        js_keywords = [
            'const', 'let', 'var', 'function', 'return', 'if', 'else',
            'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
            'class', 'extends', 'constructor', 'this', 'super', 'new',
            'async', 'await', 'import', 'export', 'from', 'default'
        ]
        
        # 操作符和标点
        operators = ['=', '==', '===', '<', '>', '{', '}', '(', ')', '[', ']', 
                     '.', ',', ';', ':', '+', '-', '*', '/', '%']
        
        vocab.extend(html_tags)
        vocab.extend([f'<{tag}>' for tag in html_tags])
        vocab.extend([f'</{tag}>' for tag in html_tags])
        vocab.extend(css_props)
        vocab.extend(js_keywords)
        vocab.extend(operators)
        
        return vocab
    
    def encode(self, code: str, max_len: int = 200) -> torch.Tensor:
        """编码代码为 token IDs"""
        tokens = [self.SOS] + self.tokenize(code)[:max_len-2] + [self.EOS]
        indices = [self.vocab2idx.get(token, self.vocab2idx[self.UNK]) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """解码 token IDs 为代码"""
        tokens = []
        for idx in indices:
            if idx == self.vocab2idx[self.EOS]:
                break
            if idx == self.vocab2idx[self.PAD]:
                continue
            token = self.idx2vocab.get(idx.item(), self.UNK)
            if token not in [self.SOS, self.PAD]:
                tokens.append(token)
        return ' '.join(tokens)
    
    def tokenize(self, code: str) -> List[str]:
        """简单分词"""
        import re
        pattern = r'<[^>]+>|[a-zA-Z_]\w*|[^\w\s]'
        tokens = re.findall(pattern, code)
        return [t for t in tokens if t.strip()]


class CodeGenerationDataset(Dataset):
    """代码生成数据集"""
    
    def __init__(self, data_path: Path, tokenizer: CodeTokenizer, max_len: int = 200):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        # 加载数据
        if data_path.exists():
            with open(data_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        code = item.get('code', '')
                        if code:
                            self.samples.append(code)
                    except:
                        continue
        
        print(f"Loaded {len(self.samples)} code samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        code = self.samples[idx]
        encoded = self.tokenizer.encode(code, max_len=self.max_len)
        
        # 输入是除了最后一个 token，目标是除了第一个 token
        src = encoded[:-1]
        tgt = encoded[1:]
        
        return src, tgt


def collate_fn(batch):
    """批处理函数"""
    src_batch, tgt_batch = zip(*batch)
    
    # 找到最大长度
    max_len = max(len(s) for s in src_batch)
    
    # 填充
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_batch, tgt_batch):
        src_pad = torch.cat([src, torch.zeros(max_len - len(src), dtype=torch.long)])
        tgt_pad = torch.cat([tgt, torch.zeros(max_len - len(tgt), dtype=torch.long)])
        src_padded.append(src_pad)
        tgt_padded.append(tgt_pad)
    
    return torch.stack(src_padded), torch.stack(tgt_padded)


def main():
    print("=" * 60)
    print("🚀 Transformer 代码生成模型训练")
    print("=" * 60)
    
    # 初始化 tokenizer
    tokenizer = CodeTokenizer()
    print(f"📖 词汇表大小: {tokenizer.vocab_size}")
    
    # 加载数据
    data_path = Path(__file__).parent.parent / 'data' / 'code_samples.jsonl'
    
    # 如果没有真实数据，创建一些示例
    if not data_path.exists():
        print("⚠️  未找到训练数据，创建示例数据...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        examples = [
            {'code': '<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>'},
            {'code': 'body { margin: 0; padding: 20px; background: #f5f5f5; }'},
            {'code': 'function calculate(a, b) { return a + b; }'},
            {'code': 'const result = async () => { const data = await fetch("/api"); return data; }'},
        ]
        
        with open(data_path, 'w') as f:
            for ex in examples * 50:  # 重复以有足够数据
                f.write(json.dumps(ex) + '\n')
    
    dataset = CodeGenerationDataset(data_path, tokenizer)
    
    if len(dataset) == 0:
        print("❌ 没有训练数据")
        return 1
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # 创建模型
    model = TransformerCodeGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数: {total_params:,}")
    
    # 训练设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 设备: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            
            output = model(src)
            
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt.reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
    
    # 测试生成
    print("\n🧪 测试代码生成:")
    model.eval()
    
    prompts = [
        "<html>",
        "function",
        "body {",
    ]
    
    for prompt in prompts:
        print(f"\n提示: {prompt}")
        prompt_ids = tokenizer.encode(prompt, max_len=10)
        generated_ids = model.generate(prompt_ids.unsqueeze(0), max_len=50, temperature=0.8)
        generated_code = tokenizer.decode(generated_ids[0])
        print(f"生成: {generated_code[:100]}...")
    
    # 导出 ONNX
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    onnx_path = models_dir / 'transformer_code_generator_v1.onnx'
    
    model.eval()
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 50)).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    print(f"\n💾 ONNX 模型已导出: {onnx_path}")
    
    # 保存 tokenizer 配置
    config_path = models_dir / 'transformer_tokenizer_config.json'
    config = {
        'vocab': tokenizer.vocab,
        'vocab_size': tokenizer.vocab_size,
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Tokenizer 配置已保存: {config_path}")
    
    print("\n✅ 训练完成!")
    print("\n建议:")
    print("  1. 收集更多真实代码样本用于训练")
    print("  2. 调整模型超参数以提升性能")
    print("  3. 使用预训练模型进行迁移学习")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
