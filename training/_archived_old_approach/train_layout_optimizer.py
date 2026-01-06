#!/usr/bin/env python3
"""
Layout Optimizer Model Training for BrowerAI Phase 3.3.

This script trains an AI model to optimize layout calculations by predicting
optimal layout strategies based on DOM structure and CSS properties.

Usage:
    python train_layout_optimizer.py [--data-dir DATA_DIR] [--epochs N]
"""

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class LayoutDataset(Dataset):
    """Dataset for layout optimization training."""
    
    def __init__(self, data_path: Path, max_nodes: int = 100):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.max_nodes = max_nodes
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Encode DOM structure features
        features = self.encode_features(sample)
        
        # Target: optimal layout strategy
        # 0: block, 1: inline, 2: flex, 3: grid
        strategy = sample.get('optimal_strategy', 0)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(strategy, dtype=torch.long)
    
    def encode_features(self, sample):
        """Encode DOM structure into feature vector."""
        features = []
        
        # Number of children
        num_children = sample.get('num_children', 0)
        features.append(min(num_children / 10.0, 1.0))  # Normalize
        
        # Depth of tree
        depth = sample.get('depth', 0)
        features.append(min(depth / 10.0, 1.0))
        
        # Element types distribution
        element_types = sample.get('element_types', {})
        for elem_type in ['div', 'span', 'p', 'section', 'article']:
            features.append(element_types.get(elem_type, 0) / max(sum(element_types.values()), 1))
        
        # Viewport size category (small, medium, large)
        viewport_width = sample.get('viewport_width', 800)
        features.append(1.0 if viewport_width < 600 else 0.0)  # Mobile
        features.append(1.0 if 600 <= viewport_width < 1200 else 0.0)  # Tablet
        features.append(1.0 if viewport_width >= 1200 else 0.0)  # Desktop
        
        # Content density
        content_density = sample.get('content_density', 0.5)
        features.append(content_density)
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]


class LayoutOptimizerModel(nn.Module):
    """Neural network for layout strategy prediction."""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_strategies: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_strategies)
        )
    
    def forward(self, x):
        return self.network(x)


def generate_training_data(output_path: Path, num_samples: int = 5000):
    """Generate synthetic training data for layout optimization."""
    data = []
    
    for _ in range(num_samples):
        num_children = random.randint(0, 20)
        depth = random.randint(1, 8)
        viewport_width = random.choice([375, 768, 1024, 1920])
        
        # Generate element type distribution
        element_types = {}
        for elem_type in ['div', 'span', 'p', 'section', 'article', 'header', 'footer']:
            element_types[elem_type] = random.randint(0, num_children)
        
        # Determine optimal strategy based on heuristics
        if num_children <= 2:
            optimal_strategy = 0  # Block
        elif depth > 5 and num_children > 5:
            optimal_strategy = 3  # Grid
        elif num_children > 3 and viewport_width >= 768:
            optimal_strategy = 2  # Flex
        else:
            optimal_strategy = 0  # Block
        
        data.append({
            'num_children': num_children,
            'depth': depth,
            'element_types': element_types,
            'viewport_width': viewport_width,
            'content_density': random.uniform(0.1, 0.9),
            'optimal_strategy': optimal_strategy
        })
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {len(data)} training samples")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in tqdm(dataloader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validation"):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def export_to_onnx(model, output_path: Path):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 20)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=15,
        input_names=['features'],
        output_names=['strategy'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'strategy': {0: 'batch_size'}
        }
    )
    print(f"\nModel exported to ONNX: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Layout Optimizer Model')
    parser.add_argument('--data-dir', type=str, default='../data/layout',
                       help='Directory for training data')
    parser.add_argument('--output-dir', type=str, default='../models',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate training data')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data if requested
    if args.generate_data:
        print("Generating training data...")
        generate_training_data(data_dir / 'train.json', 5000)
        generate_training_data(data_dir / 'val.json', 1000)
        generate_training_data(data_dir / 'test.json', 500)
    
    # Load datasets
    train_path = data_dir / 'train.json'
    val_path = data_dir / 'val.json'
    
    if not train_path.exists():
        print("Training data not found. Run with --generate-data flag.")
        return
    
    train_dataset = LayoutDataset(train_path)
    val_dataset = LayoutDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = LayoutOptimizerModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = output_dir / 'layout_optimizer_best.pth'
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved best model (acc: {best_val_acc:.4f})")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    
    # Export to ONNX
    output_dir = Path(args.output_dir)
    onnx_path = output_dir / 'layout_optimizer_v1.onnx'
    export_to_onnx(model, onnx_path)
    
    print(f"\n📊 Layout Strategy Distribution:")
    print(f"  0: Block layout")
    print(f"  1: Inline layout")
    print(f"  2: Flex layout")
    print(f"  3: Grid layout")
    print(f"\nTo use this model with BrowerAI:")
    print(f"  1. Copy {onnx_path} to ../models/local/")
    print(f"  2. Update ../models/model_config.toml")
    print(f"  3. Integrate with rendering engine")


if __name__ == '__main__':
    main()
