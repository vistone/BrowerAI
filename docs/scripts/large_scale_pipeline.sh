#!/bin/bash
# 大规模数据训练管道 - 10,000+样本 → 增强模型

echo "🚀 大规模学习管道 - 目标10,000+样本"
echo "=================================================="
echo ""

# 等待数据生成完成
echo "⏳ 步骤1: 等待数据生成 (最多30分钟)..."

timeout=1800
start_time=$(date +%s)

while [ $(($(date +%s) - start_time)) -lt $timeout ]; do
    if [ -f real_data/obfuscated_code/training_pairs.jsonl ]; then
        pairs=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
        if [ "$pairs" -gt 5000 ]; then
            echo "✅ 数据生成完成: $pairs 个训练对"
            break
        fi
    fi
    sleep 5
    echo -n "."
done

echo ""

if [ ! -f real_data/obfuscated_code/training_pairs.jsonl ]; then
    echo "❌ 数据生成失败"
    exit 1
fi

pairs=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
size=$(du -h real_data/obfuscated_code/training_pairs.jsonl | awk '{print $1}')

echo ""
echo "📊 步骤2: 数据统计"
echo "   训练对数: $pairs"
echo "   文件大小: $size"
echo ""

# 运行大规模训练
echo "🤖 步骤3: 启动大规模GPU训练 (30个epoch)"
echo ""

cat > /tmp/large_scale_trainer.py << 'TRAINER_EOF'
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class LargeDataset(Dataset):
    def __init__(self, data_file, vocab_size=10000, max_length=512):
        self.data = []
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        with open(data_file) as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except:
                    pass
        
        logger.info(f"📂 加载了 {len(self.data)} 个数据样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        code = record.get('obfuscated', '')
        
        # 简单分词
        tokens = [hash(word) % self.vocab_size for word in code.split()[:self.max_length]]
        
        # 填充
        while len(tokens) < self.max_length:
            tokens.append(0)
        tokens = tokens[:self.max_length]
        
        return torch.tensor(tokens[:self.max_length]), 0

class SimpleModel(nn.Module):
    def __init__(self, vocab_size=10000, hidden_size=512, num_classes=23):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        emb = self.embedding(x)
        _, (h, c) = self.lstm(emb)
        h_combined = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h_combined)

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  设备: {device}\n")
    
    # 加载数据
    logger.info("📊 加载数据...")
    dataset = LargeDataset('real_data/obfuscated_code/training_pairs.jsonl')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    logger.info(f"✅ 数据加载完成\n")
    
    # 创建模型
    logger.info("🤖 创建模型...")
    model = SimpleModel().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()
    logger.info("✅ 模型创建完成\n")
    
    # 训练
    logger.info("🚀 开始训练 (30个epoch)...\n")
    
    best_loss = float('inf')
    
    for epoch in range(30):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = torch.zeros(x.size(0), dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1:2d}/30 - Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'models/local/large_scale_best.pt')
        
        scheduler.step()
    
    # 保存最终模型
    torch.save(model.state_dict(), 'models/local/large_scale_final.pt')
    logger.info("\n✅ 训练完成!")
    logger.info("   best_model: models/local/large_scale_best.pt")
    logger.info("   final_model: models/local/large_scale_final.pt")

if __name__ == '__main__':
    train()
TRAINER_EOF

python3 /tmp/large_scale_trainer.py

echo ""
echo "="*50
echo "✅ 大规模学习完成!"
echo "="*50
echo ""
echo "模型文件:"
ls -lh models/local/large_scale*.pt
