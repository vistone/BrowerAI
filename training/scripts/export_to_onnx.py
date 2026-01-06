#!/usr/bin/env python3
"""
导出训练好的网站生成器模型到ONNX格式
用于Rust集成
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebsiteGenerator(nn.Module):
    """网站生成器模型（与train_website_generator.py相同）"""
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
        """
        Args:
            src: [batch_size, src_len] - 输入网站代码
            tgt: [batch_size, tgt_len] - 目标网站代码（训练时）
        Returns:
            logits: [batch_size, tgt_len, vocab_size]
        """
        batch_size, src_len = src.shape
        tgt_len = tgt.shape[1]
        
        # 创建mask
        src_mask = (src == 0)
        tgt_mask = (tgt == 0)
        tgt_attn_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        
        # Encode
        src_emb = self.embedding(src) + self.pos_encoding[:, :src_len, :]
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        
        # Decode
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


def export_onnx(checkpoint_path, output_path, vocab_size=110, seq_len=1024):
    """
    导出模型到ONNX格式
    
    Args:
        checkpoint_path: 训练好的模型检查点路径
        output_path: ONNX输出路径
        vocab_size: 词汇表大小
        seq_len: 序列长度
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # 加载模型
    model = WebsiteGenerator(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=3,
        max_len=2048
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded, vocab_size={vocab_size}, d_model=256, layers=3")
    
    # 创建dummy输入
    batch_size = 1
    dummy_src = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    dummy_tgt = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    
    logger.info(f"Exporting to ONNX with input shape: src={dummy_src.shape}, tgt={dummy_tgt.shape}")
    
    # 导出ONNX
    torch.onnx.export(
        model,
        (dummy_src, dummy_tgt),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['src', 'tgt'],
        output_names=['logits'],
        dynamic_axes={
            'src': {0: 'batch_size', 1: 'src_len'},
            'tgt': {0: 'batch_size', 1: 'tgt_len'},
            'logits': {0: 'batch_size', 1: 'tgt_len'}
        }
    )
    
    logger.info(f"✅ ONNX模型已导出到: {output_path}")
    
    # 保存vocab和配置信息
    config = {
        'vocab_size': vocab_size,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 3,
        'max_len': 2048,
        'input_names': ['src', 'tgt'],
        'output_names': ['logits'],
        'checkpoint_epoch': checkpoint.get('epoch', 'unknown'),
        'checkpoint_loss': checkpoint.get('loss', 'unknown')
    }
    
    config_path = output_path.replace('.onnx', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ 配置文件已保存到: {config_path}")
    logger.info(f"\n模型信息:")
    logger.info(f"  - 输入1: src (网站源代码序列) - shape: [batch, src_len]")
    logger.info(f"  - 输入2: tgt (目标代码序列起始) - shape: [batch, tgt_len]")
    logger.info(f"  - 输出: logits (字符概率分布) - shape: [batch, tgt_len, {vocab_size}]")
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='导出网站生成器模型到ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output', type=str, default='../../models/local/website_generator_v1.onnx', 
                       help='ONNX输出路径')
    parser.add_argument('--vocab-size', type=int, default=110, help='词汇表大小')
    parser.add_argument('--seq-len', type=int, default=1024, help='序列长度')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    export_onnx(args.checkpoint, args.output, args.vocab_size, args.seq_len)
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_onnx(args.checkpoint, args.output, args.vocab_size, args.seq_len)

