#!/usr/bin/env python3
"""
Week 6 Phase 2 Step 6 - GPU 加速混合模型训练
使用 TensorFlow GPU 训练神经网络
针对 600+ 样本和 48 维特征优化
"""
import json
import os
import warnings
import logging
from pathlib import Path
import time

import numpy as np

# GPU 加速库
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# 启用 GPU 内存增长
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class GPUTrainer:
    """GPU 加速训练器"""
    
    def __init__(self, batch_size=64, epochs=30):
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
        logger.info(f"🖥️  GPU 可用: {self.gpu_available}")
        if self.gpu_available:
            logger.info(f"📊 GPU 设备: {[gpu.name for gpu in gpus]}")
    
    def generate_training_data(self, n_samples=600, n_features=48):
        """生成训练数据"""
        logger.info(f"📊 生成 {n_samples} 个样本，{n_features} 维特征...")
        
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)
        
        # 框架样本 (正样本, 200 个)
        X[:200, :] = np.random.randn(200, n_features) * 0.4 + 1.2
        y[:200] = 1
        
        # 混淆样本 (负样本, 200 个)
        X[200:400, :] = np.random.randn(200, n_features) * 0.5 + 0.4
        y[200:400] = 0
        
        # 其他样本 (200 个)
        X[400:, :] = np.random.randn(200, n_features) * 0.45 + 0.8
        y[400:] = np.random.randint(0, 2, 200)
        
        return X, y
    
    def build_nn_model(self, input_dim):
        """构建神经网络"""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # 第一层
            layers.Dense(256, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # 第二层
            layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # 第三层
            layers.Dense(64, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # 第四层
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            # 输出层
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_nn_gpu(self, X_train, y_train, X_val, y_val):
        """GPU 训练神经网络"""
        logger.info("🎯 使用 GPU 训练神经网络...")
        
        model = self.build_nn_model(X_train.shape[1])
        
        # 数据集优化
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # 早停
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        start_time = time.time()
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=[early_stopping],
            verbose=0
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ NN 训练完成 (GPU, 耗时: {training_time:.2f}s)")
        
        # 预测
        y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_val, y_pred)
        
        return {
            'model': model,
            'accuracy': acc,
            'predictions': y_pred,
            'training_time': training_time
        }
    
    def train_rf(self, X_train, y_train, X_val, y_val):
        """CPU 随机森林 (作为基准)"""
        logger.info("🎯 训练随机森林 (CPU 基准)...")
        
        start_time = time.time()
        
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        logger.info(f"✅ RF 训练完成 (CPU, 耗时: {training_time:.2f}s)")
        
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        return {
            'model': model,
            'accuracy': acc,
            'predictions': y_pred,
            'training_time': training_time
        }
    
    def train_kfold(self, X, y, n_splits=5):
        """K-fold 交叉验证"""
        logger.info(f"🔄 执行 {n_splits}-fold 交叉验证...")
        logger.info(f"📊 数据: {X.shape[0]} 样本, {X.shape[1]} 维特征")
        logger.info("")
        
        X_scaled = self.scaler.fit_transform(X)
        
        results = {
            'nn_accs': [],
            'rf_accs': [],
            'ensemble_accs': [],
            'fold_details': []
        }
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            logger.info(f"{'='*70}")
            logger.info(f"Fold {fold_idx + 1}/{n_splits}")
            logger.info(f"{'='*70}")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # GPU 神经网络
            nn_result = self.train_nn_gpu(X_train, y_train, X_val, y_val)
            nn_acc = nn_result['accuracy']
            results['nn_accs'].append(nn_acc)
            
            # RF 基准
            rf_result = self.train_rf(X_train, y_train, X_val, y_val)
            rf_acc = rf_result['accuracy']
            results['rf_accs'].append(rf_acc)
            
            # 集成
            ensemble_pred = (
                0.6 * nn_result['predictions'] +
                0.4 * rf_result['predictions']
            ).round().astype(int)
            
            ensemble_acc = accuracy_score(y_val, ensemble_pred)
            results['ensemble_accs'].append(ensemble_acc)
            
            fold_data = {
                'fold': fold_idx + 1,
                'nn_accuracy': float(nn_acc),
                'rf_accuracy': float(rf_acc),
                'ensemble_accuracy': float(ensemble_acc),
                'nn_training_time': nn_result['training_time'],
                'rf_training_time': rf_result['training_time']
            }
            results['fold_details'].append(fold_data)
            
            logger.info(f"\n📊 Fold {fold_idx + 1} 结果:")
            logger.info(f"   🧠 神经网络 (GPU): {nn_acc:.4f}")
            logger.info(f"   🌲 随机森林 (CPU): {rf_acc:.4f}")
            logger.info(f"   🎯 集成模型: {ensemble_acc:.4f}")
            logger.info("")
        
        return results
    
    def save_results(self, results, output_dir="data/week6_training_results"):
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'gpu_accelerated': self.gpu_available,
            'neural_network_gpu': {
                'mean_accuracy': float(np.mean(results['nn_accs'])),
                'std_accuracy': float(np.std(results['nn_accs'])),
                'accuracies': [float(x) for x in results['nn_accs']]
            },
            'random_forest_cpu': {
                'mean_accuracy': float(np.mean(results['rf_accs'])),
                'std_accuracy': float(np.std(results['rf_accs'])),
                'accuracies': [float(x) for x in results['rf_accs']]
            },
            'ensemble': {
                'mean_accuracy': float(np.mean(results['ensemble_accs'])),
                'std_accuracy': float(np.std(results['ensemble_accs'])),
                'accuracies': [float(x) for x in results['ensemble_accs']]
            }
        }
        
        results_file = output_path / "gpu_training_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'fold_details': results['fold_details']
            }, f, indent=2)
        
        logger.info(f"✅ 结果已保存到: {results_file}\n")
        
        # 显示摘要
        logger.info("="*70)
        logger.info("🎉 GPU 加速训练完成!")
        logger.info("="*70)
        logger.info(f"\n📊 模型性能对比:")
        logger.info(f"   🧠 神经网络 (GPU):")
        logger.info(f"      准确度: {summary['neural_network_gpu']['mean_accuracy']:.4f} ± {summary['neural_network_gpu']['std_accuracy']:.4f}")
        logger.info(f"\n   🌲 随机森林 (CPU):")
        logger.info(f"      准确度: {summary['random_forest_cpu']['mean_accuracy']:.4f} ± {summary['random_forest_cpu']['std_accuracy']:.4f}")
        logger.info(f"\n   🎯 集成模型 (GPU+CPU):")
        logger.info(f"      准确度: {summary['ensemble']['mean_accuracy']:.4f} ± {summary['ensemble']['std_accuracy']:.4f}")
        logger.info(f"\n✨ GPU 加速: {self.gpu_available}")
        logger.info("="*70)

def main():
    """主函数"""
    logger.info("\n🚀 Week 6 Phase 2 Step 6 - GPU 加速模型训练")
    logger.info("="*70)
    logger.info("")
    
    trainer = GPUTrainer(batch_size=64, epochs=30)
    
    # 生成训练数据
    X, y = trainer.generate_training_data(n_samples=600, n_features=48)
    logger.info("")
    
    # K-fold 训练
    results = trainer.train_kfold(X, y, n_splits=5)
    
    # 保存结果
    trainer.save_results(results)

if __name__ == "__main__":
    main()
