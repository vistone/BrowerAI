#!/usr/bin/env python3
"""
CSS Selector Optimization Model Training Script

This script trains a neural network to analyze CSS selectors and suggest
optimizations. The model identifies overly complex selectors, suggests
simplifications, and predicts performance improvements.

Model outputs:
- Complexity score (0-1, higher = more complex)
- Simplification potential (0-1)
- Performance impact (0-1, higher = better performance gain)
- Specificity balance (0-1, higher = well-balanced)

Uses PyTorch for training and exports to ONNX format for deployment.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class CSSSelectorDataset(Dataset):
    """Dataset for CSS selector optimization training"""
    
    def __init__(self, num_samples=6000):
        self.features = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate synthetic selector features
            features = self._generate_selector_features()
            labels = self._compute_optimization_targets(features)
            
            self.features.append(features)
            self.labels.append(labels)
    
    def _generate_selector_features(self):
        """Generate synthetic CSS selector features (16 dimensions)"""
        return np.array([
            np.random.randint(1, 10),       # selector_length (num parts)
            np.random.randint(0, 5),        # id_count
            np.random.randint(0, 8),        # class_count
            np.random.randint(0, 5),        # tag_count
            np.random.randint(0, 4),        # pseudo_class_count
            np.random.randint(0, 3),        # pseudo_element_count
            np.random.randint(0, 5),        # attribute_selector_count
            np.random.randint(0, 4),        # descendant_combinator_count
            np.random.randint(0, 3),        # child_combinator_count
            np.random.randint(0, 2),        # adjacent_sibling_count
            np.random.randint(0, 2),        # general_sibling_count
            np.random.uniform(0, 1),        # specificity_score
            np.random.randint(0, 100),      # estimated_match_count
            np.random.uniform(0, 1),        # universal_selector_usage
            np.random.uniform(0, 1),        # negation_usage
            np.random.uniform(0, 1)         # has_inefficient_patterns
        ], dtype=np.float32)
    
    def _compute_optimization_targets(self, features):
        """Compute optimization targets based on features"""
        sel_length = features[0]
        id_count = features[1]
        class_count = features[2]
        desc_count = features[7]
        child_count = features[8]
        specificity = features[11]
        match_count = features[12]
        universal = features[13]
        inefficient = features[15]
        
        # Complexity score (higher = more complex, needs optimization)
        complexity = min(1.0, (sel_length / 10.0) * 0.3 + 
                       (desc_count / 4.0) * 0.3 +
                       universal * 0.2 + 
                       inefficient * 0.2)
        
        # Simplification potential (high when overly complex)
        simplification = min(1.0, max(0.0,
            (sel_length > 4) * 0.3 +
            (desc_count > 2) * 0.3 +
            (class_count > 4) * 0.2 +
            inefficient * 0.2
        ))
        
        # Performance impact of optimization (high when complex + many matches)
        perf_impact = min(1.0, complexity * 0.5 + (match_count / 100.0) * 0.5)
        
        # Specificity balance (well-balanced is good)
        # Too high or too low specificity is bad
        ideal_specificity = 0.5
        spec_balance = 1.0 - abs(specificity - ideal_specificity) * 2
        
        return np.array([complexity, simplification, perf_impact, spec_balance], dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])


class CSSSelectorOptimizerModel(nn.Module):
    """Neural network for CSS selector optimization prediction"""
    
    def __init__(self, input_size=16, output_size=4):
        super(CSSSelectorOptimizerModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 40),
            nn.ReLU(),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),
            nn.Linear(40, 32),
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


def train_model(num_samples=6000, epochs=20, batch_size=32, lr=0.001):
    """Train the CSS selector optimizer model"""
    
    print(f"Generating {num_samples} CSS selector samples...")
    dataset = CSSSelectorDataset(num_samples)
    
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
    
    model = CSSSelectorOptimizerModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nTraining CSS Selector Optimizer Model...")
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


def export_to_onnx(model, output_path="css_selector_optimizer_v1.onnx"):
    """Export trained model to ONNX format"""
    
    model.eval()
    model.cpu()  # Move to CPU for ONNX export
    dummy_input = torch.randn(1, 16)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['selector_features'],
        output_names=['optimization_scores'],
        dynamic_axes={
            'selector_features': {0: 'batch_size'},
            'optimization_scores': {0: 'batch_size'}
        },
        opset_version=11,
        export_params=True
    )
    
    print(f"\nModel exported to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train CSS Selector Optimizer Model')
    parser.add_argument('--num-samples', type=int, default=6000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../../models/css_selector_optimizer_v1.onnx',
                        help='Output ONNX file path')
    
    args = parser.parse_args()
    
    print("CSS Selector Optimizer Model Training")
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
    
    print("\n✓ CSS Selector Optimizer model training complete!")
    print(f"✓ Model saved to {args.output}")
    print("\nModel predicts 4 scores:")
    print("  1. Complexity Score - Selector complexity level (0-1)")
    print("  2. Simplification Potential - How much can be simplified (0-1)")
    print("  3. Performance Impact - Performance gain from optimization (0-1)")
    print("  4. Specificity Balance - How well-balanced the specificity is (0-1)")
    print("\nUsage in BrowerAI:")
    print("  - Analyze CSS selectors")
    print("  - Suggest simplifications")
    print("  - Improve rendering performance")
    print("  - Optimize selector specificity")


if __name__ == "__main__":
    main()
