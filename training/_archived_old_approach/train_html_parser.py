#!/usr/bin/env python3
"""
HTML Parser Model Training Script for BrowerAI.

This script trains a simple neural network model to understand HTML structure
and predict parsing strategies. The model is then exported to ONNX format for
use with BrowerAI's ONNX Runtime inference engine.

Usage:
    python train_html_parser.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class HTMLDataset(Dataset):
    """Dataset for HTML parsing training."""
    
    def __init__(self, data_path: Path, max_length: int = 512):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.max_length = max_length
        
        # Simple character-level vocabulary
        all_text = ' '.join([sample['input'] for sample in self.data])
        self.vocab = sorted(set(all_text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Convert text to indices
        text = sample['input'][:self.max_length]
        indices = [self.char_to_idx.get(ch, 0) for ch in text]
        
        # Pad to max_length
        if len(indices) < self.max_length:
            indices.extend([0] * (self.max_length - len(indices)))
        
        # Label: 1 for valid, 0 for malformed
        label = 1 if sample['label'] == 'valid' else 0
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


class HTMLParserModel(nn.Module):
    """Simple LSTM-based model for HTML parsing."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim * 2)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        output = self.fc(last_output)  # (batch_size, 1)
        output = self.sigmoid(output)
        return output.squeeze()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def export_to_onnx(model, vocab_size, output_path: Path, max_length: int = 512):
    """Export trained model to ONNX format."""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, (1, max_length))
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=15,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"\nModel exported to ONNX: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train HTML Parser Model')
    parser.add_argument('--data-dir', type=str, default='../data/html',
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='../models',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    data_dir = Path(args.data_dir)
    train_dataset = HTMLDataset(data_dir / 'train.json', args.max_length)
    val_dataset = HTMLDataset(data_dir / 'val.json', args.max_length)
    
    vocab_size = len(train_dataset.vocab) + 1  # +1 for padding
    print(f"Vocabulary size: {vocab_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = HTMLParserModel(vocab_size).to(device)
    criterion = nn.BCELoss()
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save PyTorch model
            model_path = output_dir / 'html_parser_best.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    
    # Export to ONNX
    output_dir = Path(args.output_dir)
    onnx_path = output_dir / 'html_parser_v1.onnx'
    export_to_onnx(model, vocab_size, onnx_path, args.max_length)
    
    print(f"\nTo use this model with BrowerAI:")
    print(f"  1. Copy {onnx_path} to ../models/local/")
    print(f"  2. Update ../models/model_config.toml with model information")
    print(f"  3. Build BrowerAI with AI features: cargo build --features ai")


if __name__ == '__main__':
    main()
