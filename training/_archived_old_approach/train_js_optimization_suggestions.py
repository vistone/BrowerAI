#!/usr/bin/env python3
"""
JavaScript Optimization Suggestions Model Training Script

This script trains a neural network to analyze JavaScript code and suggest
specific optimizations such as code refactoring opportunities, performance
improvements, and modern syntax upgrades.

Model outputs:
- Loop optimization potential (0-1)
- Function optimization score (0-1)
- Memory optimization potential (0-1)
- Modern syntax upgrade score (0-1)
- Async/await conversion potential (0-1)
- Bundle size reduction score (0-1)

Uses PyTorch for training and exports to ONNX format for deployment.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class JSOptimizationDataset(Dataset):
    """Dataset for JavaScript optimization suggestions training"""
    
    def __init__(self, num_samples=7500):
        self.features = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate synthetic JS code features
            features = self._generate_optimization_features()
            labels = self._compute_optimization_targets(features)
            
            self.features.append(features)
            self.labels.append(labels)
    
    def _generate_optimization_features(self):
        """Generate synthetic JavaScript optimization features (22 dimensions)"""
        return np.array([
            np.random.randint(10, 1000),    # num_lines
            np.random.randint(1, 50),       # num_functions
            np.random.randint(0, 30),       # num_loops
            np.random.randint(0, 20),       # nested_loop_count
            np.random.randint(0, 100),      # variable_count
            np.random.randint(0, 50),       # closure_count
            np.random.randint(0, 40),       # callback_count
            np.random.randint(0, 25),       # promise_count
            np.random.randint(0, 15),       # async_function_count
            np.random.uniform(0, 1),        # es6_usage_ratio
            np.random.uniform(0, 1),        # es5_pattern_ratio
            np.random.uniform(0, 1),        # immutability_score
            np.random.randint(0, 50),       # array_operations
            np.random.randint(0, 30),       # object_operations
            np.random.randint(0, 20),       # dom_operations
            np.random.uniform(0, 1),        # functional_programming_ratio
            np.random.uniform(0, 1),        # code_duplication_score
            np.random.randint(0, 15),       # inline_function_count
            np.random.uniform(1, 20),       # avg_function_complexity
            np.random.randint(0, 10),       # eval_usage_count
            np.random.uniform(0, 1),        # memory_leak_indicators
            np.random.randint(100, 50000)   # estimated_file_size
        ], dtype=np.float32)
    
    def _compute_optimization_targets(self, features):
        """Compute optimization suggestion targets based on features"""
        num_loops = features[2]
        nested_loops = features[3]
        num_functions = features[1]
        closures = features[5]
        callbacks = features[6]
        promises = features[7]
        async_funcs = features[8]
        es6_usage = features[9]
        es5_patterns = features[10]
        duplication = features[16]
        complexity = features[18]
        memory_indicators = features[20]
        file_size = features[21]
        
        # Loop optimization potential (high with many/nested loops)
        loop_opt = min(1.0,
            (num_loops / 30.0) * 0.5 +
            (nested_loops / 20.0) * 0.5
        )
        
        # Function optimization score (high complexity or many small functions)
        func_opt = min(1.0,
            (complexity / 20.0) * 0.4 +
            (closures / 50.0) * 0.3 +
            duplication * 0.3
        )
        
        # Memory optimization potential
        memory_opt = min(1.0,
            memory_indicators * 0.5 +
            (closures / 50.0) * 0.3 +
            (num_functions / 50.0) * 0.2
        )
        
        # Modern syntax upgrade score (high with ES5 patterns, low ES6 usage)
        modern_upgrade = min(1.0,
            es5_patterns * 0.6 +
            (1 - es6_usage) * 0.4
        )
        
        # Async/await conversion potential (high callbacks, low async usage)
        total_async_patterns = callbacks + promises + async_funcs
        async_ratio = async_funcs / max(total_async_patterns, 1)
        async_conversion = min(1.0,
            (callbacks / 40.0) * 0.6 +
            (1 - async_ratio) * 0.4
        )
        
        # Bundle size reduction score
        bundle_reduction = min(1.0,
            (file_size / 50000.0) * 0.3 +
            duplication * 0.4 +
            (1 - es6_usage) * 0.3
        )
        
        return np.array([
            loop_opt,
            func_opt,
            memory_opt,
            modern_upgrade,
            async_conversion,
            bundle_reduction
        ], dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])


class JSOptimizationModel(nn.Module):
    """Neural network for JavaScript optimization suggestions"""
    
    def __init__(self, input_size=22, output_size=6):
        super(JSOptimizationModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Dropout(0.25),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 20),
            nn.ReLU(),
            nn.Linear(20, output_size),
            nn.Sigmoid()  # Output scores [0, 1]
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(num_samples=7500, epochs=20, batch_size=32, lr=0.001):
    """Train the JavaScript optimization suggestions model"""
    
    print(f"Generating {num_samples} JavaScript code samples...")
    dataset = JSOptimizationDataset(num_samples)
    
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
    
    model = JSOptimizationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nTraining JavaScript Optimization Suggestions Model...")
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


def export_to_onnx(model, output_path="js_optimization_suggestions_v1.onnx"):
    """Export trained model to ONNX format"""
    
    model.eval()
    model.cpu()  # Move to CPU for ONNX export
    dummy_input = torch.randn(1, 22)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['code_features'],
        output_names=['optimization_scores'],
        dynamic_axes={
            'code_features': {0: 'batch_size'},
            'optimization_scores': {0: 'batch_size'}
        },
        opset_version=11,
        export_params=True
    )
    
    print(f"\nModel exported to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train JavaScript Optimization Suggestions Model')
    parser.add_argument('--num-samples', type=int, default=7500, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../../models/js_optimization_suggestions_v1.onnx',
                        help='Output ONNX file path')
    
    args = parser.parse_args()
    
    print("JavaScript Optimization Suggestions Model Training")
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
    
    print("\n✓ JavaScript Optimization Suggestions model training complete!")
    print(f"✓ Model saved to {args.output}")
    print("\nModel predicts 6 optimization scores:")
    print("  1. Loop Optimization Potential - Loop performance improvements (0-1)")
    print("  2. Function Optimization Score - Function refactoring opportunities (0-1)")
    print("  3. Memory Optimization Potential - Memory usage improvements (0-1)")
    print("  4. Modern Syntax Upgrade Score - ES6+ conversion opportunities (0-1)")
    print("  5. Async/Await Conversion Potential - Callback to async/await (0-1)")
    print("  6. Bundle Size Reduction Score - Code size optimization (0-1)")
    print("\nUsage in BrowerAI:")
    print("  - Analyze JavaScript code patterns")
    print("  - Suggest specific optimizations")
    print("  - Improve code performance")
    print("  - Modernize codebase")
    print("  - Reduce bundle sizes")


if __name__ == "__main__":
    main()
