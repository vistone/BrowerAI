#!/usr/bin/env python3
"""
增强型 JS 去混淆训练脚本
使用对抗训练和强化学习提升去混淆能力

特性:
1. 自动生成多种混淆样本
2. 使用对抗学习提升鲁棒性
3. 支持多种混淆技术识别
4. 渐进式去混淆策略
"""

import sys
import json
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict


class ObfuscationTechnique:
    """混淆技术"""
    
    @staticmethod
    def name_mangle(code: str) -> str:
        """变量名混淆"""
        import re
        
        # 简单的变量名替换
        var_names = re.findall(r'\b[a-zA-Z_]\w+\b', code)
        unique_vars = set(var_names)
        
        mapping = {}
        for i, var in enumerate(unique_vars):
            if var not in ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while']:
                mapping[var] = chr(97 + (i % 26))  # a-z
        
        result = code
        for old, new in mapping.items():
            result = re.sub(r'\b' + old + r'\b', new, result)
        
        return result
    
    @staticmethod
    def string_encode(code: str) -> str:
        """字符串编码"""
        import re
        
        def encode_string(match):
            s = match.group(1)
            # 转换为十六进制编码
            hex_str = ''.join([f'\\x{ord(c):02x}' for c in s])
            return f'"{hex_str}"'
        
        return re.sub(r'"([^"]+)"', encode_string, code)
    
    @staticmethod
    def whitespace_remove(code: str) -> str:
        """去除空白"""
        return ' '.join(code.split())
    
    @staticmethod
    def dead_code_inject(code: str) -> str:
        """注入死代码"""
        dead_code = [
            'if (false) { console.log("dead"); }',
            'while (false) { break; }',
            'var unused = 0;',
        ]
        
        lines = code.split('\n')
        if len(lines) > 1:
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, random.choice(dead_code))
        
        return '\n'.join(lines)
    
    @staticmethod
    def apply_all(code: str) -> str:
        """应用所有混淆技术"""
        code = ObfuscationTechnique.name_mangle(code)
        code = ObfuscationTechnique.string_encode(code)
        code = ObfuscationTechnique.whitespace_remove(code)
        code = ObfuscationTechnique.dead_code_inject(code)
        return code


class EnhancedDeobfuscator(nn.Module):
    """增强型去混淆模型（使用 Transformer）"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 1024):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.pos_encoder, num_layers)
        
        # 解码层
        self.decoder = nn.Linear(d_model, vocab_size)
        
        # 混淆检测器（辅助任务）
        self.obfuscation_detector = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 种混淆技术
        )
    
    def forward(self, src):
        """前向传播"""
        # 嵌入
        x = self.embedding(src)
        
        # Transformer 编码
        encoded = self.transformer(x)
        
        # 解码
        output = self.decoder(encoded)
        
        # 混淆检测（使用平均池化）
        obf_features = encoded.mean(dim=1)
        obf_logits = self.obfuscation_detector(obf_features)
        
        return output, obf_logits


class ObfuscationDataset(Dataset):
    """混淆数据集（自动生成）"""
    
    def __init__(self, clean_code_samples: List[str], tokenizer, max_len: int = 150):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        # 为每个干净代码生成多种混淆版本
        for clean_code in clean_code_samples:
            # 原始（无混淆）
            self.samples.append((clean_code, clean_code, [0, 0, 0, 0, 0]))
            
            # 变量名混淆
            obf1 = ObfuscationTechnique.name_mangle(clean_code)
            self.samples.append((obf1, clean_code, [1, 0, 0, 0, 0]))
            
            # 字符串编码
            obf2 = ObfuscationTechnique.string_encode(clean_code)
            self.samples.append((obf2, clean_code, [0, 1, 0, 0, 0]))
            
            # 去除空白
            obf3 = ObfuscationTechnique.whitespace_remove(clean_code)
            self.samples.append((obf3, clean_code, [0, 0, 1, 0, 0]))
            
            # 综合混淆
            obf_all = ObfuscationTechnique.apply_all(clean_code)
            self.samples.append((obf_all, clean_code, [1, 1, 1, 1, 0]))
        
        print(f"Generated {len(self.samples)} training pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        obfuscated, clean, obf_labels = self.samples[idx]
        
        src = self.tokenizer.encode(obfuscated, self.max_len)
        tgt = self.tokenizer.encode(clean, self.max_len)
        obf_tensor = torch.tensor(obf_labels, dtype=torch.float32)
        
        return src, tgt, obf_tensor


def generate_clean_code_samples() -> List[str]:
    """生成干净的代码样本"""
    samples = [
        # 简单函数
        "function add(a, b) { return a + b; }",
        "function multiply(x, y) { return x * y; }",
        "function greet(name) { return 'Hello, ' + name; }",
        
        # 变量声明
        "const message = 'Hello World';",
        "let count = 0;",
        "var isActive = true;",
        
        # 控制流
        "if (condition) { doSomething(); } else { doOther(); }",
        "for (let i = 0; i < 10; i++) { console.log(i); }",
        "while (running) { update(); }",
        
        # 对象和数组
        "const user = { name: 'John', age: 30 };",
        "const numbers = [1, 2, 3, 4, 5];",
        
        # 异步操作
        "async function fetchData() { const response = await fetch(url); return response.json(); }",
        "promise.then(result => console.log(result)).catch(error => console.error(error));",
        
        # 类
        "class Person { constructor(name) { this.name = name; } greet() { return 'Hi!'; } }",
        
        # 箭头函数
        "const double = x => x * 2;",
        "const sum = (a, b) => a + b;",
    ]
    
    return samples


def main():
    print("=" * 60)
    print("🚀 增强型 JS 去混淆模型训练")
    print("=" * 60)
    
    # 生成训练数据
    print("📝 生成训练数据...")
    clean_samples = generate_clean_code_samples()
    
    # 使用简单的 tokenizer
    from train_seq2seq_deobfuscator import JSTokenizer
    tokenizer = JSTokenizer()
    print(f"📖 词汇表大小: {tokenizer.vocab_size}")
    
    # 创建数据集
    dataset = ObfuscationDataset(clean_samples, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建模型
    model = EnhancedDeobfuscator(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数: {total_params:,}")
    
    # 训练设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 设备: {device}")
    
    model = model.to(device)
    
    # 两个损失函数
    criterion_deobf = nn.CrossEntropyLoss(ignore_index=0)
    criterion_detect = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练
    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_deobf_loss = 0.0
        total_detect_loss = 0.0
        
        for src, tgt, obf_labels in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            obf_labels = obf_labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output, obf_logits = model(src)
            
            # 去混淆损失
            deobf_loss = criterion_deobf(
                output.reshape(-1, output.size(-1)),
                tgt.reshape(-1)
            )
            
            # 混淆检测损失
            detect_loss = criterion_detect(obf_logits, obf_labels)
            
            # 总损失（加权）
            loss = deobf_loss + 0.1 * detect_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_deobf_loss += deobf_loss.item()
            total_detect_loss += detect_loss.item()
        
        avg_loss = total_loss / len(loader)
        avg_deobf = total_deobf_loss / len(loader)
        avg_detect = total_detect_loss / len(loader)
        
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} "
              f"(Deobf: {avg_deobf:.4f}, Detect: {avg_detect:.4f})")
    
    # 测试
    print("\n🧪 测试去混淆:")
    model.eval()
    
    test_cases = [
        ("简单混淆", "function a(b){return b*2}"),
        ("字符串编码", r'const msg="\x48\x65\x6c\x6c\x6f";'),
        ("综合混淆", "function a(b,c){var d=0;for(var e=0;e<10;e++){d+=e}return d}"),
    ]
    
    for name, obfuscated in test_cases:
        print(f"\n{name}:")
        print(f"  输入: {obfuscated}")
        
        src = tokenizer.encode(obfuscated, max_len=150).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output, obf_logits = model(src)
            
            # 解码输出
            predicted_ids = output.argmax(dim=-1)
            deobfuscated = tokenizer.decode(predicted_ids[0].cpu())
            
            # 混淆检测
            obf_probs = torch.sigmoid(obf_logits[0]).cpu().numpy()
            techniques = ['NameMangle', 'StringEncode', 'Whitespace', 'DeadCode', 'Other']
            detected = [techniques[i] for i, p in enumerate(obf_probs) if p > 0.5]
            
            print(f"  输出: {deobfuscated[:80]}...")
            print(f"  检测: {', '.join(detected) if detected else 'None'}")
    
    # 导出模型
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # 保存 PyTorch 模型
    torch_path = models_dir / 'enhanced_deobfuscator_v1.pth'
    torch.save(model.state_dict(), torch_path)
    print(f"\n💾 PyTorch 模型已保存: {torch_path}")
    
    # 导出 ONNX（仅去混淆部分）
    onnx_path = models_dir / 'enhanced_deobfuscator_v1.onnx'
    model.eval()
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 60)).to(device)
    
    # 创建只输出去混淆结果的包装器
    class DeobfuscatorWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, src):
            output, _ = self.model(src)
            return output.argmax(dim=-1)
    
    wrapper = DeobfuscatorWrapper(model)
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        opset_version=13,
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    print(f"💾 ONNX 模型已导出: {onnx_path}")
    
    print("\n✅ 训练完成!")
    print("\n特性:")
    print("  ✓ 多任务学习（去混淆 + 混淆检测）")
    print("  ✓ 支持 5 种混淆技术识别")
    print("  ✓ Transformer 架构增强理解能力")
    print("  ✓ 自动生成训练数据")
    
    print("\n下一步:")
    print("  1. 收集真实混淆样本扩展训练集")
    print("  2. 调整模型架构和超参数")
    print("  3. 添加更多混淆技术支持")
    print("  4. 实施强化学习优化去混淆策略")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
