#!/usr/bin/env python3
"""
完整的生产级训练系统
合并所有爬取数据 + 训练 + 优化
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class CombinedWebsiteDataset(Dataset):
    """合并所有网站数据"""
    
    def __init__(self, data_files: List[Path], min_confidence: float = 0.6, min_html_len: int = 800):
        self.samples = []
        self.frameworks = set()
        self.min_confidence = min_confidence
        self.min_html_len = min_html_len
        
        logger.info("📖 加载合并数据集...")
        
        for data_file in data_files:
            if not data_file.exists():
                logger.warning(f"⚠️ 文件不存在: {data_file}")
                continue
            
            logger.info(f"  加载: {data_file}")
            
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)

                        code = data.get('html', '') if data.get('html') else ''
                        if len(code) < self.min_html_len:
                            continue

                        # 新格式：直接category + confidence
                        category = data.get('category')
                        confidence = data.get('confidence', 0)

                        if category and category != 'Unknown' and confidence >= self.min_confidence:
                            self.samples.append({
                                'code': code[:5000],
                                'framework': category,
                                'confidence': confidence,
                                'all_indicators': None,
                                'url': data.get('url', 'unknown'),
                            })
                            self.frameworks.add(category)
                            continue

                        # 兼容旧格式：indicators/detected_frameworks
                        indicators = data.get('indicators') or data.get('detected_frameworks', {})
                        if indicators:
                            primary_framework, score = max(indicators.items(), key=lambda x: x[1])
                            if score >= self.min_confidence:
                                self.samples.append({
                                    'code': code[:5000],
                                    'framework': primary_framework,
                                    'confidence': score,
                                    'all_indicators': indicators,
                                    'url': data.get('url', 'unknown'),
                                })
                                self.frameworks.add(primary_framework)
                    except Exception as e:
                        logger.debug(f"⚠️ 解析错误: {e}")
        
        logger.info(f"✅ 加载完成: {len(self.samples)} 样本, {len(self.frameworks)} 框架")
        logger.info(f"   框架: {sorted(self.frameworks)}")
        self._print_distribution()
    
    def _print_distribution(self):
        """打印数据分布"""
        dist = defaultdict(int)
        for sample in self.samples:
            dist[sample['framework']] += 1
        
        logger.info("   分布:")
        for fw, count in sorted(dist.items(), key=lambda x: -x[1]):
            logger.info(f"     {fw}: {count} ({count*100//len(self.samples)}%)")
    
    def extract_features(self, code: str) -> np.ndarray:
        """提取代码特征"""
        features = np.zeros(50)
        
        code_lower = code.lower()
        
        # 框架特定关键字计数
        react_keywords = ['react', 'jsx', 'usestate', 'useeffect', 'hooks', 'component']
        vue_keywords = ['vue', 'v-', 'template', 'computed', 'watch', 'component']
        angular_keywords = ['angular', 'component', 'service', 'module', 'decorator', 'injectable']
        jquery_keywords = ['jquery', '$.', 'plugin', 'selector', 'ajax']
        express_keywords = ['express', 'app.', 'router', 'middleware', 'request', 'response']
        
        features[0] = sum(code_lower.count(kw) for kw in react_keywords)
        features[1] = sum(code_lower.count(kw) for kw in vue_keywords)
        features[2] = sum(code_lower.count(kw) for kw in angular_keywords)
        features[3] = sum(code_lower.count(kw) for kw in jquery_keywords)
        features[4] = sum(code_lower.count(kw) for kw in express_keywords)
        
        # 代码结构特征
        features[5] = code.count('function')
        features[6] = code.count('class')
        features[7] = code.count('const')
        features[8] = code.count('let')
        features[9] = code.count('var')
        
        # 导入/模块特征
        features[10] = code.count('import')
        features[11] = code.count('require')
        features[12] = code.count('export')
        features[13] = code.count('module.exports')
        
        # 异步特征
        features[14] = code.count('async')
        features[15] = code.count('await')
        features[16] = code.count('promise')
        features[17] = code.count('.then(')
        
        # 代码大小特征
        features[18] = len(code) / 1000  # KB
        features[19] = code.count('\n') / 100  # 行数
        features[20] = len(code.split()) / 100  # 单词数
        
        # HTML/DOM特征
        features[21] = code.count('<')
        features[22] = code.count('>')
        features[23] = code.count('getElementById')
        features[24] = code.count('querySelector')
        
        # 数据库/API特征
        features[25] = code.count('fetch(')
        features[26] = code.count('axios')
        features[27] = code.count('database')
        features[28] = code.count('mongodb')
        features[29] = code.count('sql')
        
        # 测试特征
        features[30] = code.count('jest')
        features[31] = code.count('test')
        features[32] = code.count('describe')
        features[33] = code.count('it(')
        
        # 配置特征
        features[34] = code.count('package.json')
        features[35] = code.count('webpack')
        features[36] = code.count('babel')
        features[37] = code.count('tsconfig')
        
        # 规范化
        features = np.clip(features, 0, 100)
        
        return features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = self.extract_features(sample['code'])
        return {
            'features': torch.FloatTensor(features),
            'framework': sample['framework'],
        }


class ProductionFrameworkDetector(nn.Module):
    """生产级框架检测模型"""
    
    def __init__(self, num_frameworks: int):
        super().__init__()
        
        # 特征提取层
        self.feature_layers = nn.Sequential(
            nn.Linear(50, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_frameworks),
        )
    
    def forward(self, x):
        features = self.feature_layers(x)
        return self.classifier(features)


class ProductionTrainer:
    """生产级训练器"""
    
    def __init__(self, output_dir: Path = Path("models/production")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🚀 设备: {self.device}")
    
    async def run_complete_training(self):
        """完整训练流程"""
        logger.info(f"\n{'='*70}")
        logger.info("🎓 完整训练流程")
        logger.info(f"{'='*70}\n")
        
        # 加载数据
        data_files = [
            Path("training/real_data/scaleable/scaleable_websites.jsonl"),
            Path("training/real_data/annotated/final_annotated.jsonl"),
            Path("training/real_data/annotated/expanded_annotated.jsonl"),
            Path("training/real_data/annotated/websites_annotated.jsonl"),
        ]
        
        dataset = CombinedWebsiteDataset(data_files, min_confidence=0.6, min_html_len=800)
        
        if len(dataset) == 0:
            logger.error("❌ 数据集为空")
            return
        
        # 分割数据
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 框架编码
        frameworks = sorted(list(dataset.frameworks))
        label_encoder = {fw: i for i, fw in enumerate(frameworks)}
        
        logger.info(f"📊 数据准备:")
        logger.info(f"  总样本: {len(dataset)}")
        logger.info(f"  训练: {train_size}, 测试: {test_size}")
        logger.info(f"  框架数: {len(frameworks)}\n")
        
        # 创建模型
        model = ProductionFrameworkDetector(num_frameworks=len(frameworks)).to(self.device)
        
        # 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # 训练循环
        best_acc = 0
        best_epoch = 0
        
        logger.info("🎯 开始训练...\n")
        
        for epoch in range(100):
            # 训练
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                features = batch['features'].to(self.device)
                frameworks_batch = batch['framework']
                labels = torch.tensor([label_encoder[fw] for fw in frameworks_batch]).to(self.device)
                
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
                    frameworks_batch = batch['framework']
                    labels = torch.tensor([label_encoder[fw] for fw in frameworks_batch]).to(self.device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d} | "
                          f"Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}% | "
                          f"Val: Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), self.output_dir / "best_model.pt")
                torch.save(label_encoder, self.output_dir / "label_encoder.pkl")
            
            scheduler.step()
        
        logger.info(f"\n✅ 训练完成! 最佳准确率: {best_acc:.2f}% (epoch {best_epoch+1})")
        
        # 生成最终报告
        self.generate_report(best_acc, len(frameworks), len(dataset))
    
    def generate_report(self, accuracy: float, num_frameworks: int, num_samples: int):
        """生成最终报告"""
        report = {
            "status": "PRODUCTION_READY" if accuracy >= 70 else "NEEDS_IMPROVEMENT",
            "accuracy": round(accuracy, 2),
            "num_frameworks": num_frameworks,
            "num_samples": num_samples,
            "model_path": str(self.output_dir / "best_model.pt"),
            "device": str(self.device),
        }
        
        with open(self.output_dir / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n{'='*70}")
        logger.info("📋 最终报告:")
        logger.info(f"  状态: {report['status']}")
        logger.info(f"  准确率: {report['accuracy']}%")
        logger.info(f"  框架数: {num_frameworks}")
        logger.info(f"  样本数: {num_samples}")
        logger.info(f"  模型: {report['model_path']}")
        logger.info(f"{'='*70}\n")


async def main():
    trainer = ProductionTrainer()
    await trainer.run_complete_training()


if __name__ == "__main__":
    asyncio.run(main())
