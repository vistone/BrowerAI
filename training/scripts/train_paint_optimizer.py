#!/usr/bin/env python3
"""
Paint Optimizer Model Training Script

Trains a neural network to optimize paint operations by predicting:
1. Paint operation priorities (which operations can be deferred)
2. Layer composition strategy
3. Redraw region optimization
4. Cache effectiveness prediction

The model learns from synthetic paint operation sequences to make
intelligent decisions about paint optimization strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaintOptimizerDataset(Dataset):
    """Dataset for paint optimization training"""
    
    def __init__(self, num_samples=5000):
        self.samples = []
        self.labels = []
        
        # Generate synthetic paint operation scenarios
        for _ in range(num_samples):
            features, label = self._generate_sample()
            self.samples.append(features)
            self.labels.append(label)
    
    def _generate_sample(self):
        """Generate a single training sample"""
        # Features (25 dimensions):
        # [0-4]: Paint operation counts (SolidRect, Border, Text, Image, Other)
        # [5-8]: Layer information (num_layers, max_depth, has_transparency, has_blend)
        # [9-12]: Viewport info (width, height, visible_area_ratio, scroll_position)
        # [13-16]: Operation complexity (avg_rect_size, overlap_count, text_operations, image_operations)
        # [17-20]: Cache info (cache_hit_rate, cached_operations, cache_size, invalidation_rate)
        # [21-24]: Performance metrics (previous_frame_time, draw_call_count, pixel_fill_rate, memory_usage)
        
        features = np.zeros(25, dtype=np.float32)
        
        # Operation counts (normalized 0-1)
        features[0] = np.random.rand()  # SolidRect count
        features[1] = np.random.rand()  # Border count
        features[2] = np.random.rand()  # Text count
        features[3] = np.random.rand()  # Image count
        features[4] = np.random.rand()  # Other operations
        
        # Layer information
        features[5] = np.random.rand()  # num_layers (normalized)
        features[6] = np.random.rand()  # max_depth
        features[7] = np.random.choice([0.0, 1.0])  # has_transparency
        features[8] = np.random.choice([0.0, 1.0])  # has_blend_modes
        
        # Viewport info
        features[9] = np.random.rand()  # width (normalized)
        features[10] = np.random.rand()  # height (normalized)
        features[11] = np.random.rand()  # visible_area_ratio
        features[12] = np.random.rand()  # scroll_position
        
        # Operation complexity
        features[13] = np.random.rand()  # avg_rect_size
        features[14] = np.random.rand()  # overlap_count
        features[15] = np.random.rand()  # text_operations
        features[16] = np.random.rand()  # image_operations
        
        # Cache info
        features[17] = np.random.rand()  # cache_hit_rate
        features[18] = np.random.rand()  # cached_operations
        features[19] = np.random.rand()  # cache_size
        features[20] = np.random.rand()  # invalidation_rate
        
        # Performance metrics
        features[21] = np.random.rand()  # previous_frame_time
        features[22] = np.random.rand()  # draw_call_count
        features[23] = np.random.rand()  # pixel_fill_rate
        features[24] = np.random.rand()  # memory_usage
        
        # Labels (6 dimensions):
        # [0-2]: Optimization strategy priorities (immediate, deferred, cached)
        # [3]: Layer composition strategy (0=flatten, 1=preserve layers)
        # [4]: Redraw region size (0=minimal, 1=full)
        # [5]: Cache strategy (0=aggressive, 1=conservative)
        
        label = np.zeros(6, dtype=np.float32)
        
        # Calculate optimal strategy based on features
        # High cache hit rate -> prefer cached operations
        if features[17] > 0.7:
            label[0] = 0.2  # immediate
            label[1] = 0.2  # deferred
            label[2] = 0.6  # cached
        # High overlap and complexity -> defer non-critical operations
        elif features[14] > 0.6 and features[13] > 0.5:
            label[0] = 0.3  # immediate
            label[1] = 0.6  # deferred
            label[2] = 0.1  # cached
        # Simple scene -> immediate rendering
        else:
            label[0] = 0.7  # immediate
            label[1] = 0.2  # deferred
            label[2] = 0.1  # cached
        
        # Layer composition: preserve if transparency or blend modes
        label[3] = 1.0 if features[7] > 0.5 or features[8] > 0.5 else 0.0
        
        # Redraw region: minimal if small viewport change
        label[4] = 0.0 if features[12] < 0.2 else 1.0
        
        # Cache strategy: aggressive if high hit rate
        label[5] = 0.0 if features[17] > 0.6 else 1.0
        
        return features, label
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])


class PaintOptimizerModel(nn.Module):
    """Neural network for paint optimization strategy prediction"""
    
    def __init__(self, input_size=25, hidden_sizes=[64, 48, 32], output_size=6):
        super(PaintOptimizerModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_model(args):
    """Train the paint optimizer model"""
    logger.info("Starting Paint Optimizer model training...")
    
    # Create datasets
    logger.info(f"Generating {args.num_samples} training samples...")
    train_dataset = PaintOptimizerDataset(num_samples=args.num_samples)
    val_dataset = PaintOptimizerDataset(num_samples=args.num_samples // 5)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = PaintOptimizerModel().to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_dir / 'paint_optimizer_best.pth')
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
    
    # Export to ONNX
    logger.info("Exporting model to ONNX format...")
    model.eval()
    dummy_input = torch.randn(1, 25).to(device)
    
    onnx_path = args.model_dir / 'paint_optimizer_v1.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['paint_features'],
        output_names=['optimization_strategy'],
        dynamic_axes={
            'paint_features': {0: 'batch_size'},
            'optimization_strategy': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to {onnx_path}")
    
    # Save training metadata
    metadata = {
        'model_type': 'paint_optimizer',
        'version': '1.0',
        'input_size': 25,
        'output_size': 6,
        'hidden_sizes': [64, 48, 32],
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'train_samples': args.num_samples,
        'val_samples': args.num_samples // 5,
        'epochs': args.epochs,
        'best_val_loss': best_val_loss,
        'features': [
            'paint_operation_counts', 'layer_info', 'viewport_info',
            'operation_complexity', 'cache_info', 'performance_metrics'
        ],
        'outputs': [
            'immediate_priority', 'deferred_priority', 'cached_priority',
            'layer_composition', 'redraw_region', 'cache_strategy'
        ]
    }
    
    with open(args.model_dir / 'paint_optimizer_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Paint Optimizer Model')
    parser.add_argument('--num-samples', type=int, default=5000,
                       help='Number of training samples to generate')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model-dir', type=Path, default=Path('../models'),
                       help='Directory to save the trained model')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    args.model_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    train_model(args)


if __name__ == '__main__':
    main()
