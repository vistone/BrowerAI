#!/usr/bin/env python3
"""
真实数据处理和模型训练
使用真实网站数据训练框架检测模型
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class RealWebsiteDataset(Dataset):
    """真实网站数据集"""
    
    def __init__(self, data_file: Path, max_samples: int = None):
        self.samples = []
        self.frameworks = set()
        
        logger.info(f"📖 加载数据集: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    if data.get('success') and data.get('detected_frameworks'):
                        # 使用最高概率的框架作为标签
                        primary_framework = max(
                            data['detected_frameworks'].items(),
                            key=lambda x: x[1]
                        )[0]
                        
                        # 提取代码特征
                        code_text = data.get('html', '')[:10000]
                        scripts = data.get('scripts', [])
                        for script in scripts[:3]:  # 前3个脚本
                            if 'content' in script:
                                code_text += "\n" + script['content'][:5000]
                        
                        self.samples.append({
                            'code': code_text,
                            'framework': primary_framework,
                            'confidence': max(data['detected_frameworks'].values()),
                            'all_frameworks': data['detected_frameworks'],
                            'url': data['url'],
                        })
                        self.frameworks.add(primary_framework)
                except Exception as e:
                    logger.warning(f"⚠️ 解析行 {i} 失败: {e}")
        
        logger.info(f"✅ 加载 {len(self.samples)} 个样本，{len(self.frameworks)} 个框架")
        logger.info(f"   框架分布: {self._get_distribution()}")
    
    def _get_distribution(self) -> Dict[str, int]:
        """获取数据分布"""
        dist = defaultdict(int)
        for sample in self.samples:
            dist[sample['framework']] += 1
        return dict(dist)
    
    def extract_features(self, code: str, vocab_size: int = 10000) -> np.ndarray:
        """提取代码特征向量"""
        # 简单的词频特征
        tokens = re.findall(r'\b[a-zA-Z_]\w*\b', code.lower())
        
        # 框架特定关键字
        framework_keywords = {
            'React': ['react', 'jsx', 'components', 'hooks', 'useState', 'useEffect'],
            'Vue': ['vue', 'template', 'v-bind', 'v-model', 'data', 'computed'],
            'Angular': ['angular', 'decorator', 'component', 'service', 'module'],
            'Express': ['express', 'app.get', 'app.post', 'middleware', 'router'],
            'jQuery': ['jquery', 'plugin', 'selector', 'ajax', 'event'],
        }
        
        features = np.zeros(100)
        
        # 关键字频率
        for i, (fw, keywords) in enumerate(framework_keywords.items()):
            count = sum(tokens.count(kw) for kw in keywords)
            features[i] = count
        
        # 代码特性
        features[5] = len(tokens)
        features[6] = code.count('function')
        features[7] = code.count('class')
        features[8] = code.count('import')
        features[9] = code.count('async')
        
        # 字符特性
        features[10] = code.count('{')
        features[11] = code.count('(')
        features[12] = code.count('[')
        
        return features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = self.extract_features(sample['code'])
        return {
            'features': torch.FloatTensor(features),
            'framework': sample['framework'],
            'code': sample['code'],
            'url': sample['url'],
        }


class FrameworkDetectionModel(nn.Module):
    """框架检测深度学习模型"""
    
    def __init__(self, input_size: int = 100, num_frameworks: int = 8, hidden_size: int = 256):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, num_frameworks),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


class RealDataTrainer:
    """真实数据训练器"""
    
    def __init__(self, data_file: Path = Path("real_data/websites/websites_data.jsonl"),
                 output_dir: Path = Path("models/real_trained")):
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🚀 使用设备: {self.device}")
    
    def prepare_data(self, batch_size: int = 32, train_ratio: float = 0.8):
        """准备数据集"""
        logger.info("📊 准备数据...")
        
        dataset = RealWebsiteDataset(self.data_file)
        
        if len(dataset) == 0:
            logger.error("❌ 数据集为空，请先运行爬取器")
            return None, None
        
        # 分割数据
        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 框架编码
        frameworks = list(dataset.frameworks)
        label_encoder = {fw: i for i, fw in enumerate(sorted(frameworks))}
        
        logger.info(f"✅ 数据准备完成")
        logger.info(f"   总样本: {len(dataset)}")
        logger.info(f"   训练: {train_size}, 测试: {test_size}")
        logger.info(f"   框架: {frameworks}")
        
        return (train_loader, test_loader, label_encoder, dataset), dataset
    
    def train_model(self, epochs: int = 50, batch_size: int = 32):
        """训练模型"""
        loaders_info, dataset = self.prepare_data(batch_size)
        
        if loaders_info is None:
            return
        
        train_loader, test_loader, label_encoder, _ = loaders_info
        
        # 创建模型
        num_frameworks = len(label_encoder)
        model = FrameworkDetectionModel(
            input_size=100,
            num_frameworks=num_frameworks,
            hidden_size=256
        ).to(self.device)
        
        # 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎓 开始训练 ({epochs} epochs)")
        logger.info(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        # 建立逆向标签映射
        inverse_encoder = {v: k for k, v in label_encoder.items()}
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                frameworks = batch['framework']
                labels = torch.tensor([label_encoder.get(fw, 0) for fw in frameworks]).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # 验证
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    features = batch['features'].to(self.device)
                    frameworks = batch['framework']
                    labels = torch.tensor([label_encoder.get(fw, 0) for fw in frameworks]).to(self.device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100
            
            logger.info(f"Epoch {epoch+1:3d} | "
                       f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
                       f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), self.output_dir / "best_model.pt")
            
            scheduler.step(avg_val_loss)
        
        logger.info(f"\n✅ 训练完成！最佳epoch: {best_epoch+1}, Loss: {best_val_loss:.4f}")
        
        # 加载最佳模型并评估
        model.load_state_dict(torch.load(self.output_dir / "best_model.pt"))
        self.evaluate_model(model, test_loader, label_encoder, inverse_encoder)
        
        return model, label_encoder
    
    def evaluate_model(self, model, test_loader, label_encoder, inverse_encoder):
        """评估模型"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 模型评估")
        logger.info(f"{'='*60}\n")
        
        model.eval()
        all_predicted = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                frameworks = batch['framework']
                labels = torch.tensor([label_encoder.get(fw, 0) for fw in frameworks]).to(self.device)
                
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算准确率
        accuracy = accuracy_score(all_labels, all_predicted)
        logger.info(f"总体准确率: {accuracy*100:.2f}%")
        
        # 分类报告
        logger.info("\n分类报告:")
        target_names = [inverse_encoder[i] for i in sorted(inverse_encoder.keys())]
        print(classification_report(all_labels, all_predicted, target_names=target_names))


async def main():
    """主函数"""
    trainer = RealDataTrainer()
    trainer.train_model(epochs=50, batch_size=32)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
