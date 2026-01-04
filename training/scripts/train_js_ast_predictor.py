#!/usr/bin/env python3
"""
JavaScript AST Predictor Model Training Script

This script trains a neural network to predict Abstract Syntax Tree (AST) node
types and structures from JavaScript code features. The model helps with faster
parsing by predicting likely AST patterns before full parsing.

Model outputs:
- Statement type probability (0-1)
- Expression complexity (0-1)
- Nesting depth prediction (0-1)
- AST confidence score (0-1)
- Declaration pattern score (0-1)

Uses PyTorch for training and exports to ONNX format for deployment.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class JSASTDataset(Dataset):
    """Dataset for JavaScript AST prediction training"""
    
    def __init__(self, num_samples=8000):
        self.features = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate synthetic AST features
            features = self._generate_ast_features()
            labels = self._compute_ast_targets(features)
            
            self.features.append(features)
            self.labels.append(labels)
    
    def _generate_ast_features(self):
        """Generate synthetic JavaScript AST features (20 dimensions)"""
        return np.array([
            np.random.uniform(0, 1),        # starts_with_keyword
            np.random.uniform(0, 1),        # has_function_keyword
            np.random.uniform(0, 1),        # has_class_keyword
            np.random.uniform(0, 1),        # has_const_let_var
            np.random.uniform(0, 1),        # has_arrow_function
            np.random.uniform(0, 1),        # has_async_await
            np.random.uniform(0, 1),        # has_import_export
            np.random.uniform(0, 1),        # has_destructuring
            np.random.randint(0, 10),       # bracket_pairs_count
            np.random.randint(0, 8),        # paren_pairs_count
            np.random.randint(0, 6),        # brace_pairs_count
            np.random.randint(0, 20),       # operator_count
            np.random.randint(0, 30),       # identifier_count
            np.random.randint(0, 15),       # literal_count
            np.random.uniform(0, 1),        # has_ternary
            np.random.uniform(0, 1),        # has_spread_operator
            np.random.uniform(0, 1),        # has_template_literal
            np.random.randint(0, 5),        # semicolon_count
            np.random.randint(1, 100),      # code_length
            np.random.uniform(0, 1)         # es6_feature_density
        ], dtype=np.float32)
    
    def _compute_ast_targets(self, features):
        """Compute AST prediction targets based on features"""
        has_function = features[1]
        has_class = features[2]
        has_declaration = features[3]
        has_arrow = features[4]
        has_async = features[5]
        has_import = features[6]
        bracket_pairs = features[8]
        paren_pairs = features[9]
        brace_pairs = features[10]
        operator_count = features[11]
        identifier_count = features[12]
        has_ternary = features[14]
        has_spread = features[15]
        
        # Statement type probability (declaration, expression, control flow)
        # Higher when has declaration keywords
        statement_prob = min(1.0, 
            has_declaration * 0.3 + 
            has_function * 0.25 + 
            has_class * 0.25 +
            has_import * 0.2
        )
        
        # Expression complexity (higher with more operators and nesting)
        expr_complexity = min(1.0,
            (operator_count / 20.0) * 0.4 +
            (bracket_pairs / 10.0) * 0.3 +
            has_ternary * 0.2 +
            has_spread * 0.1
        )
        
        # Nesting depth prediction
        nesting_depth = min(1.0,
            (brace_pairs / 6.0) * 0.4 +
            (paren_pairs / 8.0) * 0.3 +
            (bracket_pairs / 10.0) * 0.3
        )
        
        # AST confidence (clear patterns = high confidence)
        pattern_clarity = (has_function + has_class + has_declaration + has_import)
        ast_confidence = min(1.0, 0.5 + pattern_clarity * 0.125)
        
        # Declaration pattern score (variable, function, class declarations)
        decl_score = min(1.0,
            has_declaration * 0.35 +
            has_function * 0.3 +
            has_class * 0.25 +
            has_async * 0.1
        )
        
        return np.array([
            statement_prob, 
            expr_complexity, 
            nesting_depth, 
            ast_confidence,
            decl_score
        ], dtype=np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])


class JSASTModel(nn.Module):
    """Neural network for JavaScript AST prediction"""
    
    def __init__(self, input_size=20, output_size=5):
        super(JSASTModel, self).__init__()
        
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


def train_model(num_samples=8000, epochs=20, batch_size=32, lr=0.001):
    """Train the JavaScript AST predictor model"""
    
    print(f"Generating {num_samples} JavaScript AST samples...")
    dataset = JSASTDataset(num_samples)
    
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
    
    model = JSASTModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nTraining JavaScript AST Predictor Model...")
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


def export_to_onnx(model, output_path="js_ast_predictor_v1.onnx"):
    """Export trained model to ONNX format"""
    
    model.eval()
    model.cpu()  # Move to CPU for ONNX export
    dummy_input = torch.randn(1, 20)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['code_features'],
        output_names=['ast_predictions'],
        dynamic_axes={
            'code_features': {0: 'batch_size'},
            'ast_predictions': {0: 'batch_size'}
        },
        opset_version=11,
        export_params=True
    )
    
    print(f"\nModel exported to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train JavaScript AST Predictor Model')
    parser.add_argument('--num-samples', type=int, default=8000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../../models/js_ast_predictor_v1.onnx',
                        help='Output ONNX file path')
    
    args = parser.parse_args()
    
    print("JavaScript AST Predictor Model Training")
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
    
    print("\n✓ JavaScript AST Predictor model training complete!")
    print(f"✓ Model saved to {args.output}")
    print("\nModel predicts 5 scores:")
    print("  1. Statement Type Probability - Likelihood of statement type (0-1)")
    print("  2. Expression Complexity - Expression nesting/complexity (0-1)")
    print("  3. Nesting Depth Prediction - Predicted AST depth (0-1)")
    print("  4. AST Confidence Score - Prediction confidence (0-1)")
    print("  5. Declaration Pattern Score - Declaration detection (0-1)")
    print("\nUsage in BrowerAI:")
    print("  - Predict AST structure before full parse")
    print("  - Speed up parsing with informed decisions")
    print("  - Better error recovery")
    print("  - Optimized parsing strategies")


if __name__ == "__main__":
    main()
