#!/usr/bin/env python3
"""
CSS Optimizer Model Training Script

Trains a neural network to optimize CSS code by predicting:
- Rule deduplication opportunities
- Selector simplification potential
- Minification safety scores
- Rule merge opportunities

Uses PyTorch for training and exports to ONNX format for deployment.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class CSSOptimizerDataset(Dataset):
    """Dataset for CSS optimization training"""
    
    def __init__(self, num_samples=4000):
        self.features = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate synthetic CSS features
            features = self._generate_css_features()
            labels = self._compute_optimization_targets(features)
            
            self.features.append(features)
            self.labels.append(labels)
    
    def _generate_css_features(self):
        """Generate synthetic CSS file features (18 dimensions)"""
        return np.array([
            np.random.randint(5, 500),      # num_rules
            np.random.randint(1, 100),      # num_selectors
            np.random.randint(10, 200),     # num_properties
            np.random.uniform(0, 1),        # avg_specificity
            np.random.uniform(0, 1),        # redundancy_score
            np.random.randint(0, 20),       # duplicate_rules
            np.random.randint(0, 50),       # unused_rules
            np.random.uniform(0, 1),        # complexity_score
            np.random.randint(0, 10),       # media_queries
            np.random.randint(0, 5),        # keyframes
            np.random.randint(0, 10),       # animations
            np.random.randint(0, 30),       # important_count
            np.random.uniform(0, 1),        # vendor_prefix_ratio
            np.random.randint(100, 50000),  # file_size_bytes
            np.random.uniform(0, 1),        # selector_nesting_depth
            np.random.uniform(0, 1),        # property_shorthand_usage
            np.random.randint(0, 100),      # color_variations
            np.random.uniform(0, 1)         # whitespace_ratio
        ], dtype=np.float32)
    
    def _compute_optimization_targets(self, features):
        """Compute optimization targets based on features"""
        num_rules, num_selectors, num_properties = features[0], features[1], features[2]
        redundancy = features[4]
        duplicate_rules = features[5]
        unused_rules = features[6]
        complexity = features[7]
        
        # Deduplication score (higher = more duplication to remove)
        dedup_score = min(1.0, (duplicate_rules / max(num_rules, 1)) + redundancy * 0.3)
        
        # Selector simplification (higher = more complex selectors to simplify)
        selector_simp = min(1.0, complexity * 0.5 + (num_selectors / max(num_rules, 1)) * 0.5)
        
        # Minification safety (higher = safer to minify)
        minify_safety = max(0.0, 1.0 - redundancy * 0.3 - unused_rules / max(num_rules, 1) * 0.2)
        
        # Merge opportunity (higher = more rules can be merged)
        merge_opp = min(1.0, (num_properties / max(num_rules * 3, 1)) * 0.6 + redundancy * 0.4)
        
        return np.array([dedup_score, selector_simp, minify_safety, merge_opp], dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])


class CSSOptimizerModel(nn.Module):
    """Neural network for CSS optimization prediction"""
    
    def __init__(self, input_size=18, output_size=4):
        super(CSSOptimizerModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Sigmoid()  # Output probabilities [0, 1]
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(num_samples=4000, epochs=15, batch_size=32, lr=0.001):
    """Train the CSS optimizer model"""
    
    print(f"Generating {num_samples} CSS samples...")
    dataset = CSSOptimizerDataset(num_samples)
    
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
    model = CSSOptimizerModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nTraining CSS Optimizer Model...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
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
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    print("=" * 60)
    print(f"Final Test Loss: {test_loss:.6f}")
    
    return model


def export_to_onnx(model, output_path="css_optimizer_v1.onnx"):
    """Export trained model to ONNX format"""
    
    model.eval()
    dummy_input = torch.randn(1, 18)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['css_features'],
        output_names=['optimization_scores'],
        dynamic_axes={
            'css_features': {0: 'batch_size'},
            'optimization_scores': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print(f"\nModel exported to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train CSS Optimizer Model')
    parser.add_argument('--num-samples', type=int, default=4000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../../models/css_optimizer_v1.onnx',
                        help='Output ONNX file path')
    
    args = parser.parse_args()
    
    print("CSS Optimizer Model Training")
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
    
    print("\n✓ CSS Optimizer model training complete!")
    print(f"✓ Model saved to {args.output}")
    print("\nModel predicts 4 optimization scores:")
    print("  1. Deduplication Score - Remove duplicate rules")
    print("  2. Selector Simplification - Simplify complex selectors")
    print("  3. Minification Safety - Safe to minify aggressively")
    print("  4. Merge Opportunity - Rules can be merged")


if __name__ == "__main__":
    main()
