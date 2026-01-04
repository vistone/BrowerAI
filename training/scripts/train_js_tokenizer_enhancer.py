#!/usr/bin/env python3
"""
JavaScript Tokenization Enhancer Model Training Script

This script trains a neural network to enhance JavaScript tokenization by
predicting token types, identifying problematic patterns, and suggesting
corrections for ambiguous or malformed tokens.

Model outputs:
- Token validity score (0-1)
- Token type confidence (0-1)
- Correction needed probability (0-1)
- Syntax complexity score (0-1)

Uses PyTorch for training and exports to ONNX format for deployment.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class JSTokenizationDataset(Dataset):
    """Dataset for JavaScript tokenization enhancement training"""
    
    def __init__(self, num_samples=7000):
        self.features = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate synthetic token features
            features = self._generate_token_features()
            labels = self._compute_tokenization_targets(features)
            
            self.features.append(features)
            self.labels.append(labels)
    
    def _generate_token_features(self):
        """Generate synthetic JavaScript token features (18 dimensions)"""
        return np.array([
            np.random.randint(1, 100),      # token_length
            np.random.uniform(0, 1),        # alphanumeric_ratio
            np.random.uniform(0, 1),        # special_char_ratio
            np.random.randint(0, 10),       # nested_bracket_depth
            np.random.uniform(0, 1),        # whitespace_before
            np.random.uniform(0, 1),        # whitespace_after
            np.random.randint(0, 5),        # line_breaks_in_token
            np.random.uniform(0, 1),        # starts_with_number
            np.random.uniform(0, 1),        # contains_unicode
            np.random.uniform(0, 1),        # is_keyword_like
            np.random.uniform(0, 1),        # is_operator_like
            np.random.uniform(0, 1),        # is_identifier_like
            np.random.uniform(0, 1),        # is_literal_like
            np.random.uniform(0, 1),        # has_escape_sequences
            np.random.randint(0, 3),        # quote_type (0=none, 1=single, 2=double, 3=backtick)
            np.random.uniform(0, 1),        # ambiguous_context
            np.random.uniform(0, 1),        # potential_asi_issue
            np.random.uniform(0, 1)         # unicode_normalization_needed
        ], dtype=np.float32)
    
    def _compute_tokenization_targets(self, features):
        """Compute tokenization targets based on features"""
        token_len = features[0]
        alpha_ratio = features[1]
        special_ratio = features[2]
        nested_depth = features[3]
        line_breaks = features[6]
        is_keyword = features[9]
        is_operator = features[10]
        is_identifier = features[11]
        is_literal = features[12]
        has_escape = features[13]
        ambiguous = features[15]
        asi_issue = features[16]
        unicode_norm = features[17]
        
        # Token validity score (high when well-formed)
        validity = max(0.0, min(1.0, 
            (alpha_ratio + (1 - special_ratio)) * 0.4 +
            (1 - ambiguous) * 0.3 +
            (1 - asi_issue) * 0.3
        ))
        
        # Token type confidence (high when clearly matches a type)
        type_matches = is_keyword + is_operator + is_identifier + is_literal
        type_confidence = min(1.0, type_matches) if type_matches > 0 else 0.3
        
        # Correction needed probability
        correction_needed = min(1.0, 
            ambiguous * 0.4 +
            asi_issue * 0.3 +
            unicode_norm * 0.2 +
            (has_escape * special_ratio) * 0.1
        )
        
        # Syntax complexity score
        complexity = min(1.0,
            (nested_depth / 10.0) * 0.3 +
            (line_breaks / 5.0) * 0.2 +
            special_ratio * 0.3 +
            (token_len / 100.0) * 0.2
        )
        
        return np.array([validity, type_confidence, correction_needed, complexity], dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])


class JSTokenizationModel(nn.Module):
    """Neural network for JavaScript tokenization enhancement"""
    
    def __init__(self, input_size=18, output_size=4):
        super(JSTokenizationModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 48),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Dropout(0.3),
            nn.Linear(48, 36),
            nn.ReLU(),
            nn.BatchNorm1d(36),
            nn.Dropout(0.25),
            nn.Linear(36, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
            nn.Dropout(0.2),
            nn.Linear(24, output_size),
            nn.Sigmoid()  # Output scores [0, 1]
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(num_samples=7000, epochs=20, batch_size=32, lr=0.001):
    """Train the JavaScript tokenization enhancer model"""
    
    print(f"Generating {num_samples} JavaScript token samples...")
    dataset = JSTokenizationDataset(num_samples)
    
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
    
    model = JSTokenizationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nTraining JavaScript Tokenization Enhancer Model...")
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


def export_to_onnx(model, output_path="js_tokenizer_enhancer_v1.onnx"):
    """Export trained model to ONNX format"""
    
    model.eval()
    model.cpu()  # Move to CPU for ONNX export
    dummy_input = torch.randn(1, 18)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['token_features'],
        output_names=['tokenization_scores'],
        dynamic_axes={
            'token_features': {0: 'batch_size'},
            'tokenization_scores': {0: 'batch_size'}
        },
        opset_version=11,
        export_params=True
    )
    
    print(f"\nModel exported to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train JavaScript Tokenization Enhancer Model')
    parser.add_argument('--num-samples', type=int, default=7000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../../models/js_tokenizer_enhancer_v1.onnx',
                        help='Output ONNX file path')
    
    args = parser.parse_args()
    
    print("JavaScript Tokenization Enhancer Model Training")
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
    
    print("\n✓ JavaScript Tokenization Enhancer model training complete!")
    print(f"✓ Model saved to {args.output}")
    print("\nModel predicts 4 scores:")
    print("  1. Token Validity Score - How valid the token is (0-1)")
    print("  2. Token Type Confidence - Confidence in token type (0-1)")
    print("  3. Correction Needed - Token needs correction (0-1)")
    print("  4. Syntax Complexity - Token complexity level (0-1)")
    print("\nUsage in BrowerAI:")
    print("  - Enhanced tokenization accuracy")
    print("  - Identify malformed tokens")
    print("  - Suggest token corrections")
    print("  - Handle edge cases in JS parsing")


if __name__ == "__main__":
    main()
