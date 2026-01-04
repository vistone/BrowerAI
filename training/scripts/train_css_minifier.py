#!/usr/bin/env python3
"""
CSS Minification Model Training Script

This script trains a neural network to analyze CSS code and predict safe
minification strategies. The model determines what can be safely minified,
shortened, or removed without affecting functionality.

Model outputs:
- Whitespace removal safety (0-1)
- Comment removal safety (0-1)
- Property shorthand potential (0-1)
- Value optimization potential (0-1)
- Overall minification score (0-1)

Uses PyTorch for training and exports to ONNX format for deployment.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class CSSMinificationDataset(Dataset):
    """Dataset for CSS minification training"""
    
    def __init__(self, num_samples=5500):
        self.features = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate synthetic CSS file features
            features = self._generate_minification_features()
            labels = self._compute_minification_targets(features)
            
            self.features.append(features)
            self.labels.append(labels)
    
    def _generate_minification_features(self):
        """Generate synthetic CSS minification features (17 dimensions)"""
        return np.array([
            np.random.randint(100, 10000),  # file_size_bytes
            np.random.uniform(0, 0.5),      # whitespace_ratio
            np.random.uniform(0, 0.3),      # comment_ratio
            np.random.randint(5, 500),      # num_rules
            np.random.randint(10, 1000),    # num_properties
            np.random.randint(0, 100),      # num_comments
            np.random.uniform(0, 1),        # shorthand_usage_ratio
            np.random.uniform(0, 1),        # color_format_optimization
            np.random.uniform(0, 1),        # zero_value_optimization
            np.random.uniform(0, 1),        # unit_optimization
            np.random.uniform(0, 1),        # quote_usage_ratio
            np.random.randint(0, 50),       # calc_expressions
            np.random.randint(0, 30),       # var_declarations
            np.random.uniform(0, 1),        # has_important_rules
            np.random.uniform(0, 1),        # has_media_queries
            np.random.uniform(0, 1),        # has_keyframes
            np.random.uniform(0, 1)         # vendor_prefix_ratio
        ], dtype=np.float32)
    
    def _compute_minification_targets(self, features):
        """Compute minification targets based on features"""
        whitespace_ratio = features[1]
        comment_ratio = features[2]
        shorthand_usage = features[6]
        color_opt = features[7]
        zero_opt = features[8]
        unit_opt = features[9]
        has_important = features[13]
        has_media = features[14]
        has_keyframes = features[15]
        
        # Whitespace removal safety (always safe unless special cases)
        whitespace_safety = max(0.7, 1.0 - has_keyframes * 0.2 - has_media * 0.1)
        
        # Comment removal safety (usually safe, but preserve special comments)
        comment_safety = max(0.8, 1.0 - comment_ratio * 0.2)  # Some comments might be important
        
        # Shorthand potential (higher when not already using shorthand)
        shorthand_potential = max(0.0, 1.0 - shorthand_usage)
        
        # Value optimization (color, zero, units)
        value_opt_potential = min(1.0, (1 - color_opt) * 0.35 + 
                                  (1 - zero_opt) * 0.35 + 
                                  (1 - unit_opt) * 0.3)
        
        # Overall minification score
        overall_score = min(1.0, 
            whitespace_ratio * 0.3 +
            comment_ratio * 0.2 +
            shorthand_potential * 0.25 +
            value_opt_potential * 0.25
        )
        
        return np.array([
            whitespace_safety, 
            comment_safety, 
            shorthand_potential, 
            value_opt_potential,
            overall_score
        ], dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])


class CSSMinificationModel(nn.Module):
    """Neural network for CSS minification prediction"""
    
    def __init__(self, input_size=17, output_size=5):
        super(CSSMinificationModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 40),
            nn.ReLU(),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.25),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
            nn.Dropout(0.2),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Sigmoid()  # Output scores [0, 1]
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(num_samples=5500, epochs=20, batch_size=32, lr=0.001):
    """Train the CSS minification model"""
    
    print(f"Generating {num_samples} CSS minification samples...")
    dataset = CSSMinificationDataset(num_samples)
    
    # Split into train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CSSMinificationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nTraining CSS Minification Model...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training
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
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"  → New best model (val_loss: {val_loss:.6f})")
    
    # Load best model for testing
    model.load_state_dict(best_model_state)
    model.eval()
    
    # Test
    test_loss = 0.0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    print("=" * 60)
    print(f"Final Test Loss: {test_loss:.6f}")
    
    return model


def export_to_onnx(model, output_path="css_minifier_v1.onnx"):
    """Export trained model to ONNX format"""
    
    model.eval()
    model.cpu()  # Move to CPU for ONNX export
    dummy_input = torch.randn(1, 17)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['css_features'],
        output_names=['minification_scores'],
        dynamic_axes={
            'css_features': {0: 'batch_size'},
            'minification_scores': {0: 'batch_size'}
        },
        opset_version=11,
        export_params=True
    )
    
    print(f"\nModel exported to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train CSS Minification Model')
    parser.add_argument('--num-samples', type=int, default=5500, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../../models/css_minifier_v1.onnx',
                        help='Output ONNX file path')
    
    args = parser.parse_args()
    
    print("CSS Minification Model Training")
    print("=" * 60)
    print(f"Samples: {args.num_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)
    
    # Train model
    model = train_model(args.num_samples, args.epochs, args.batch_size, args.lr)
    
    # Export to ONNX
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    export_to_onnx(model, args.output)
    
    print("\n✓ CSS Minification model training complete!")
    print(f"✓ Model saved to {args.output}")
    print("\nModel predicts 5 scores:")
    print("  1. Whitespace Removal Safety - Safe to remove whitespace (0-1)")
    print("  2. Comment Removal Safety - Safe to remove comments (0-1)")
    print("  3. Shorthand Potential - Can use shorthand properties (0-1)")
    print("  4. Value Optimization Potential - Can optimize values (0-1)")
    print("  5. Overall Minification Score - Total minification potential (0-1)")
    print("\nUsage in BrowerAI:")
    print("  - Analyze CSS files for minification")
    print("  - Safe minification strategies")
    print("  - Reduce file size")
    print("  - Maintain functionality")


if __name__ == "__main__":
    main()
