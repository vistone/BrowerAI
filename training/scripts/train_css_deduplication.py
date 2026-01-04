#!/usr/bin/env python3
"""
CSS Rule Deduplication Model Training Script

This script trains a neural network to detect and predict duplicate CSS rules
that can be safely merged or removed. The model analyzes CSS rule patterns,
selector specificity, and property overlap to identify deduplication opportunities.

Model outputs:
- Duplicate rule probability (0-1)
- Merge opportunity score (0-1)
- Safety confidence (0-1)

Uses PyTorch for training and exports to ONNX format for deployment.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class CSSDeduplicationDataset(Dataset):
    """Dataset for CSS deduplication training"""
    
    def __init__(self, num_samples=5000):
        self.features = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate synthetic CSS rule pair features
            features = self._generate_rule_pair_features()
            labels = self._compute_deduplication_targets(features)
            
            self.features.append(features)
            self.labels.append(labels)
    
    def _generate_rule_pair_features(self):
        """Generate synthetic CSS rule pair features (15 dimensions)"""
        return np.array([
            np.random.uniform(0, 1),        # selector_similarity
            np.random.uniform(0, 1),        # property_overlap_ratio
            np.random.randint(0, 20),       # num_shared_properties
            np.random.randint(1, 50),       # total_properties_rule1
            np.random.randint(1, 50),       # total_properties_rule2
            np.random.uniform(0, 1),        # specificity_rule1
            np.random.uniform(0, 1),        # specificity_rule2
            np.random.uniform(0, 1),        # value_similarity
            np.random.randint(0, 10),       # cascade_distance
            np.random.uniform(0, 1),        # media_query_match
            np.random.uniform(0, 1),        # importance_conflict
            np.random.uniform(0, 1),        # vendor_prefix_match
            np.random.randint(0, 5),        # pseudo_class_count
            np.random.uniform(0, 1),        # order_dependency_risk
            np.random.uniform(0, 1)         # shorthand_property_ratio
        ], dtype=np.float32)
    
    def _compute_deduplication_targets(self, features):
        """Compute deduplication targets based on features"""
        selector_sim = features[0]
        prop_overlap = features[1]
        shared_props = features[2]
        spec1, spec2 = features[5], features[6]
        value_sim = features[7]
        cascade_dist = features[8]
        importance = features[10]
        order_risk = features[13]
        
        # Duplicate probability (high when very similar)
        duplicate_prob = min(1.0, selector_sim * 0.3 + prop_overlap * 0.4 + value_sim * 0.3)
        
        # Merge opportunity (high overlap, similar specificity)
        specificity_diff = abs(spec1 - spec2)
        merge_score = max(0.0, min(1.0, prop_overlap * 0.5 + 
                                   (1 - specificity_diff) * 0.3 +
                                   selector_sim * 0.2))
        
        # Safety confidence (low risk factors)
        safety = max(0.0, 1.0 - importance * 0.3 - 
                     order_risk * 0.3 - 
                     (cascade_dist / 10.0) * 0.4)
        
        return np.array([duplicate_prob, merge_score, safety], dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])


class CSSDeduplicationModel(nn.Module):
    """Neural network for CSS rule deduplication prediction"""
    
    def __init__(self, input_size=15, output_size=3):
        super(CSSDeduplicationModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
            nn.Dropout(0.2),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Sigmoid()  # Output probabilities [0, 1]
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(num_samples=5000, epochs=20, batch_size=32, lr=0.001):
    """Train the CSS deduplication model"""
    
    print(f"Generating {num_samples} CSS rule pair samples...")
    dataset = CSSDeduplicationDataset(num_samples)
    
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
    
    model = CSSDeduplicationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nTraining CSS Deduplication Model...")
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


def export_to_onnx(model, output_path="css_deduplication_v1.onnx"):
    """Export trained model to ONNX format"""
    
    model.eval()
    model.cpu()  # Move to CPU for ONNX export
    dummy_input = torch.randn(1, 15)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['rule_pair_features'],
        output_names=['deduplication_scores'],
        dynamic_axes={
            'rule_pair_features': {0: 'batch_size'},
            'deduplication_scores': {0: 'batch_size'}
        },
        opset_version=11,
        export_params=True
    )
    
    print(f"\nModel exported to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train CSS Deduplication Model')
    parser.add_argument('--num-samples', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../../models/css_deduplication_v1.onnx',
                        help='Output ONNX file path')
    
    args = parser.parse_args()
    
    print("CSS Rule Deduplication Model Training")
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
    
    print("\n✓ CSS Deduplication model training complete!")
    print(f"✓ Model saved to {args.output}")
    print("\nModel predicts 3 scores:")
    print("  1. Duplicate Probability - How likely rules are duplicates (0-1)")
    print("  2. Merge Opportunity Score - How safely rules can be merged (0-1)")
    print("  3. Safety Confidence - Confidence in safe deduplication (0-1)")
    print("\nUsage in BrowerAI:")
    print("  - Analyze CSS rule pairs")
    print("  - Identify redundant rules")
    print("  - Suggest safe merge operations")
    print("  - Reduce CSS file size")


if __name__ == "__main__":
    main()
