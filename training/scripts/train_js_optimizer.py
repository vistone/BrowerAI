#!/usr/bin/env python3
"""
JavaScript Optimizer Model Training Script

Trains a neural network to optimize JavaScript code by predicting:
- Minification safety scores
- Dead code detection
- Optimization potential
- Bundle optimization opportunities
- Async/await conversion potential

Uses PyTorch for training and exports to ONNX format for deployment.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class JSOptimizerDataset(Dataset):
    """Dataset for JavaScript optimization training"""
    
    def __init__(self, num_samples=4500):
        self.features = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate synthetic JS features
            features = self._generate_js_features()
            labels = self._compute_optimization_targets(features)
            
            self.features.append(features)
            self.labels.append(labels)
    
    def _generate_js_features(self):
        """Generate synthetic JavaScript file features (20 dimensions)"""
        return np.array([
            np.random.randint(50, 5000),    # num_tokens
            np.random.randint(10, 500),     # num_statements
            np.random.randint(1, 50),       # num_functions
            np.random.randint(0, 100),      # num_variables
            np.random.randint(0, 50),       # num_classes
            np.random.uniform(1, 20),       # cyclomatic_complexity
            np.random.uniform(0, 1),        # code_duplication_score
            np.random.randint(0, 30),       # unused_variables
            np.random.randint(0, 20),       # unused_functions
            np.random.uniform(0, 1),        # dead_code_ratio
            np.random.uniform(0, 1),        # modern_syntax_ratio
            np.random.randint(0, 50),       # callback_count
            np.random.randint(0, 30),       # promise_count
            np.random.randint(0, 20),       # async_await_count
            np.random.uniform(0, 1),        # es6_feature_usage
            np.random.randint(0, 100),      # dependency_count
            np.random.randint(500, 100000), # file_size_bytes
            np.random.uniform(0, 1),        # comment_ratio
            np.random.uniform(0, 1),        # whitespace_ratio
            np.random.uniform(0, 1)         # scope_complexity
        ], dtype=np.float32)
    
    def _compute_optimization_targets(self, features):
        """Compute optimization targets based on features"""
        num_statements = features[1]
        complexity = features[5]
        duplication = features[6]
        unused_vars = features[7]
        unused_funcs = features[8]
        dead_code = features[9]
        modern_syntax = features[10]
        callbacks = features[11]
        promises = features[12]
        async_await = features[13]
        
        # Minification safety (higher = safer to minify)
        minify_safety = max(0.0, 1.0 - dead_code * 0.3 - duplication * 0.2)
        
        # Dead code score (higher = more dead code to remove)
        dead_code_score = min(1.0, dead_code + (unused_vars + unused_funcs) / max(num_statements, 1))
        
        # Optimization potential (higher = more optimizations possible)
        opt_potential = min(1.0, complexity / 10.0 * 0.4 + duplication * 0.3 + (1 - modern_syntax) * 0.3)
        
        # Bundle opportunity (higher = better bundling potential)
        bundle_score = min(1.0, modern_syntax * 0.5 + (1 - duplication) * 0.5)
        
        # Async conversion potential (higher = more callbacks to convert)
        async_conv = min(1.0, (callbacks / max(num_statements / 10, 1)) * 0.6 + 
                        (1 - (async_await / max(callbacks + promises + 1, 1))) * 0.4)
        
        return np.array([minify_safety, dead_code_score, opt_potential, bundle_score, async_conv], 
                       dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])


class JSOptimizerModel(nn.Module):
    """Neural network for JavaScript optimization prediction"""
    
    def __init__(self, input_size=20, output_size=5):
        super(JSOptimizerModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 40),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, output_size),
            nn.Sigmoid()  # Output probabilities [0, 1]
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(num_samples=4500, epochs=15, batch_size=32, lr=0.001):
    """Train the JavaScript optimizer model"""
    
    print(f"Generating {num_samples} JavaScript samples...")
    dataset = JSOptimizerDataset(num_samples)
    
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
    model = JSOptimizerModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nTraining JavaScript Optimizer Model...")
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


def export_to_onnx(model, output_path="js_optimizer_v1.onnx"):
    """Export trained model to ONNX format"""
    
    model.eval()
    dummy_input = torch.randn(1, 20)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['js_features'],
        output_names=['optimization_scores'],
        dynamic_axes={
            'js_features': {0: 'batch_size'},
            'optimization_scores': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print(f"\nModel exported to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train JavaScript Optimizer Model')
    parser.add_argument('--num-samples', type=int, default=4500, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../../models/js_optimizer_v1.onnx',
                        help='Output ONNX file path')
    
    args = parser.parse_args()
    
    print("JavaScript Optimizer Model Training")
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
    
    print("\n✓ JavaScript Optimizer model training complete!")
    print(f"✓ Model saved to {args.output}")
    print("\nModel predicts 5 optimization scores:")
    print("  1. Minification Safety - Safe to minify code")
    print("  2. Dead Code Score - Amount of dead code to remove")
    print("  3. Optimization Potential - Overall optimization opportunity")
    print("  4. Bundle Score - Code bundling potential")
    print("  5. Async Conversion - Callback to async/await conversion")


if __name__ == "__main__":
    main()
