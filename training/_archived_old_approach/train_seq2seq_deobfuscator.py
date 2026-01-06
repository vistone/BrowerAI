#!/usr/bin/env python3
"""
Seq2Seq去混淆模型 - JavaScript代码转换
输入: 混淆的JS代码
输出: 去混淆的JS代码
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
from typing import List, Tuple


class JSTokenizer:
    """JavaScript词法分析器"""
    
    def __init__(self):
        # 特殊标记
        self.PAD = '<PAD>'
        self.SOS = '<SOS>'  # Start of sequence
        self.EOS = '<EOS>'  # End of sequence
        self.UNK = '<UNK>'  # Unknown token
        
        # JS关键字
        self.keywords = {
            'const', 'let', 'var', 'function', 'return', 'if', 'else',
            'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
            'class', 'extends', 'constructor', 'this', 'super', 'new',
            'async', 'await', 'promise', 'then', 'catch', 'try', 'finally',
            'import', 'export', 'from', 'default', 'as',
        }
        
        # 操作符
        self.operators = {
            '=', '==', '===', '!=', '!==', '<', '>', '<=', '>=',
            '+', '-', '*', '/', '%', '**', '++', '--',
            '&&', '||', '!', '&', '|', '^', '~', '<<', '>>',
            '?', ':', '=>', '.', ',', ';',
        }
        
        # 构建词汇表
        self.vocab = [self.PAD, self.SOS, self.EOS, self.UNK]
        self.vocab.extend(sorted(self.keywords))
        self.vocab.extend(sorted(self.operators))
        
        # 添加括号
        self.vocab.extend(['(', ')', '{', '}', '[', ']'])
        
        # 添加常见变量名模式
        for prefix in ['var', 'tmp', 'val', 'data', 'result', 'item']:
            for i in range(10):
                self.vocab.append(f'{prefix}{i}')
        
        # 添加单字母变量 (混淆代码常见)
        for c in 'abcdefghijklmnopqrstuvwxyz':
            self.vocab.append(c)
        
        self.vocab2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2vocab = {idx: token for token, idx in self.vocab2idx.items()}
        self.vocab_size = len(self.vocab)
    
    def tokenize(self, code: str) -> list:
        """将代码分词"""
        tokens = []
        
        # 简单的词法分析
        pattern = r'\w+|[^\w\s]'
        matches = re.findall(pattern, code)
        
        for match in matches:
            if match in self.vocab2idx:
                tokens.append(match)
            elif match in self.keywords:
                tokens.append(match)
            elif len(match) == 1 and match.isalpha():
                tokens.append(match)
            else:
                tokens.append(self.UNK)
        
        return tokens
    
    def encode(self, code: str, max_len=100) -> torch.Tensor:
        """编码为索引序列"""
        tokens = [self.SOS] + self.tokenize(code)[:max_len-2] + [self.EOS]
        indices = [self.vocab2idx.get(token, self.vocab2idx[self.UNK]) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """解码为代码"""
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


class Seq2SeqDeobfuscator(nn.Module):
    """Seq2Seq去混淆模型"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 编码器
        self.encoder_embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # 解码器
        self.decoder_embed = nn.Embedding(vocab_size, embed_dim)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim * 2, batch_first=True)
        
        # 输出层
        self.output = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, src, tgt):
        """前向传播"""
        # 编码
        src_embed = self.encoder_embed(src)
        encoder_output, (hidden, cell) = self.encoder_lstm(src_embed)
        
        # 将双向编码器的隐状态拼接为解码器初始状态
        # hidden, cell: (2, batch, hidden) -> (1, batch, hidden*2)
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1).unsqueeze(0)
        cell_cat = torch.cat([cell[0], cell[1]], dim=1).unsqueeze(0)

        # 解码
        tgt_embed = self.decoder_embed(tgt)
        decoder_output, _ = self.decoder_lstm(tgt_embed, (hidden_cat, cell_cat))
        
        # 输出
        logits = self.output(decoder_output)
        
        return logits
    
    def generate(self, src, tokenizer, max_len=100):
        """生成去混淆代码"""
        device = next(self.parameters()).device
        
        # 编码
        src_embed = self.encoder_embed(src)
        encoder_output, (hidden, cell) = self.encoder_lstm(src_embed)
        
        # 初始化解码器状态
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1).unsqueeze(0)
        cell_cat = torch.cat([cell[0], cell[1]], dim=1).unsqueeze(0)
        
        # 从SOS开始逐token生成
        batch_size = src.size(0)
        current = torch.tensor([[tokenizer.vocab2idx[tokenizer.SOS]]] * batch_size).to(device)
        generated = []
        
        for _ in range(max_len):
            # 解码一步
            tgt_embed = self.decoder_embed(current)
            decoder_output, (hidden_cat, cell_cat) = self.decoder_lstm(tgt_embed, (hidden_cat, cell_cat))
            logits = self.output(decoder_output[:, -1, :])
            
            # 贪婪采样
            next_token = logits.argmax(dim=-1).unsqueeze(1)
            generated.append(next_token.item())
            
            # 判断结束
            if next_token.item() == tokenizer.vocab2idx[tokenizer.EOS]:
                break
            
            current = next_token
        
        return torch.tensor(generated)
    
    def inference(self, src, max_len=60):
        """ONNX推理模式：编码器+解码器完整前向，返回token ids"""
        device = next(self.parameters()).device
        batch_size = src.size(0)
        
        # 编码
        src_embed = self.encoder_embed(src)
        encoder_output, (hidden, cell) = self.encoder_lstm(src_embed)
        
        # 初始化解码器状态
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1).unsqueeze(0)
        cell_cat = torch.cat([cell[0], cell[1]], dim=1).unsqueeze(0)
        
        # 生成输出序列（固定长度，用于ONNX导出）
        outputs = []
        current = torch.full((batch_size, 1), 1, dtype=torch.long).to(device)  # SOS token = 1
        
        for _ in range(max_len):
            tgt_embed = self.decoder_embed(current)
            decoder_output, (hidden_cat, cell_cat) = self.decoder_lstm(tgt_embed, (hidden_cat, cell_cat))
            logits = self.output(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            outputs.append(next_token)
            current = next_token
        
        # 拼接所有输出 [batch, max_len]
        return torch.cat(outputs, dim=1)


class InferenceWrapper(nn.Module):
    """ONNX导出包装器"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, src):
        return self.model.inference(src, max_len=60)


def create_synthetic_dataset(tokenizer, num_samples=1000):
    """创建合成训练数据
    由于没有真实的混淆-原始代码对，我们生成简单的合成数据
    实际使用中需要真实的配对数据
    """
    samples = []
    
    patterns = [
        # 模式1: 变量重命名
        ('const a = 5;', 'const value = 5;'),
        ('let b = "hello";', 'let message = "hello";'),
        ('var c = true;', 'var isActive = true;'),
        
        # 模式2: 函数简化
        ('function a(b){return b*2}', 'function double(value){return value*2}'),
        ('const b=c=>c+1', 'const increment = value => value+1'),
        
        # 模式3: 逻辑简化
        ('if(a){b}else{c}', 'if(condition){doA}else{doB}'),
        ('a?b:c', 'condition?trueValue:falseValue'),
    ]
    
    # 生成变体
    for obfuscated, clean in patterns:
        for i in range(num_samples // len(patterns)):
            # 添加一些随机变化
            samples.append((obfuscated, clean))
    
    return samples


class PairedCodeDataset(Dataset):
    """混淆-原始成对数据集"""

    def __init__(self, pairs: List[Tuple[str, str]], tokenizer: JSTokenizer, max_len: int = 120):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        obf, clean = self.pairs[idx]
        src = self.tokenizer.encode(obf, max_len=self.max_len)
        tgt = self.tokenizer.encode(clean, max_len=self.max_len)
        return src, tgt


def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded


def main():
    print("=" * 60)
    print("🚀 Seq2Seq去混淆模型训练")
    print("=" * 60)
    
    # 1. 创建tokenizer
    tokenizer = JSTokenizer()
    print(f"📖 词汇表大小: {tokenizer.vocab_size}")
    
    # 2. 读取真实配对数据（如果存在），否则使用合成数据
    pairs_path = Path(__file__).parent.parent / 'data' / 'obfuscation_pairs.jsonl'
    pairs: List[Tuple[str, str]] = []
    if pairs_path.exists():
        print(f"📂 发现真实配对数据: {pairs_path}")
        with open(pairs_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    obf = item.get('obfuscated') or item.get('obf')
                    clean = item.get('clean') or item.get('original')
                    if obf and clean:
                        pairs.append((obf, clean))
                except Exception:
                    continue
    if not pairs:
        print("⚠️  未找到真实配对数据，使用合成样本演示")
        pairs = create_synthetic_dataset(tokenizer, num_samples=200)

    print(f"✅ 训练样本: {len(pairs)}")
    dataset = PairedCodeDataset(pairs, tokenizer)
    
    # 3. 创建模型
    model = Seq2SeqDeobfuscator(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        hidden_dim=256
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数: {total_params:,}")

    # 4. DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)

    # 5. 训练（轻量级示例）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 设备: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()
            # 教师强制：输入是 tgt[:-1]，预测 tgt[1:]
            logits = model(src, tgt[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    # 6. 简单推理演示
    model.eval()
    sample_src, sample_tgt = dataset[0]
    gen_indices = model.generate(sample_src.unsqueeze(0).to(device), tokenizer, max_len=60)
    generated = tokenizer.decode(gen_indices.cpu())
    print("\n🧪 生成示例：")
    print("混淆:", tokenizer.decode(sample_src))
    print("期望:", tokenizer.decode(sample_tgt))
    print("生成:", generated)

    # 7. 保存tokenizer配置
    config_path = Path(__file__).parent.parent / 'models' / 'tokenizer_config.json'
    config = {
        'vocab': tokenizer.vocab,
        'vocab_size': tokenizer.vocab_size,
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Tokenizer配置已保存: {config_path}")

    # 8. 导出ONNX（使用inference包装器）
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    onnx_path = models_dir / 'js_deobfuscator_v1.onnx'

    model.eval()
    wrapper = InferenceWrapper(model)
    dummy_src = torch.randint(0, tokenizer.vocab_size, (1, 60)).to(device)

    # 导出包装模型
    torch.onnx.export(
        wrapper,
        dummy_src,
        str(onnx_path),
        input_names=['src'],
        output_names=['output'],
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes={
            'src': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
    )

    print(f"💾 ONNX 已导出: {onnx_path} (使用inference模式)")

    print("\n⚠️  注意: 要取得真实效果，请准备1万+真实混淆↔原始配对样本，并可改用Transformer架构")
    print("   本脚本支持：将真实配对数据放到 training/data/obfuscation_pairs.jsonl (每行 {obfuscated, clean})")
    print("\n📚 建议:")
    print("   1. 使用 uglify/terser/obfuscator-js 对开源JS做混淆，保留原始代码作标签")
    print("   2. 增加样本覆盖事件循环/异步/模块化等多模式")
    print("   3. 换用 Transformer (小型) 替代 LSTM 提升表现")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
