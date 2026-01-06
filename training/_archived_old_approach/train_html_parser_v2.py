#!/usr/bin/env python3
"""
Enhanced HTML Parser Model Training for BrowerAI Phase 2.

This script implements advanced model architectures including:
- Transformer-based models for better context understanding
- Attention mechanisms for structure prediction
- Multi-task learning for validation and correction
- Optimized for ONNX deployment

Usage:
    python train_html_parser_v2.py [--data-dir DATA_DIR] [--model-type TYPE]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class HTMLDatasetV2(Dataset):
    """Enhanced dataset with better preprocessing."""
    
    def __init__(self, data_path: Path, max_length: int = 512, use_enhanced: bool = True):
        # Load both original and enhanced data if available
        self.max_length = max_length
        self.data = []
        
        # Try to load enhanced data first
        if use_enhanced:
            enhanced_path = data_path.parent / data_path.name.replace('.json', '_enhanced.json')
            if enhanced_path.exists():
                with open(enhanced_path, 'r', encoding='utf-8') as f:
                    self.data.extend(json.load(f))
                print(f"Loaded {len(self.data)} enhanced samples")
        
        # Load original data
        if data_path.exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
                self.data.extend(original_data)
            print(f"Total samples: {len(self.data)}")
        
        # Build vocabulary from all text
        all_text = ' '.join([sample['input'] for sample in self.data])
        self.vocab = sorted(set(all_text))
        self.char_to_idx = {ch: idx + 1 for idx, ch in enumerate(self.vocab)}  # 0 reserved for padding
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.vocab) + 1
        
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
        
        # Multi-label: valid/malformed and complexity
        label = 1.0 if sample['label'] == 'valid' else 0.0
        complexity = 0.5  # Default
        if 'complexity' in sample:
            complexity = {'low': 0.0, 'medium': 0.5, 'high': 1.0}.get(sample['complexity'], 0.5)
        
        return (torch.tensor(indices, dtype=torch.long), 
                torch.tensor([label, complexity], dtype=torch.float32))


class TransformerHTMLParser(nn.Module):
    """Transformer-based model for HTML parsing with attention."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 num_heads: int = 4, num_layers: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = nn.Embedding(512, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.fc_validity = nn.Linear(embedding_dim, 1)
        self.fc_complexity = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch_size, seq_length)
        batch_size, seq_length = x.shape
        
        # Create position indices
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed and add positional encoding
        embedded = self.embedding(x) + self.pos_encoder(positions)
        
        # Create attention mask for padding
        padding_mask = (x == 0)
        
        # Transform
        transformed = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Pool: take mean of non-padded tokens
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (transformed * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Multi-task outputs
        validity = self.sigmoid(self.fc_validity(pooled)).squeeze()
        complexity = self.sigmoid(self.fc_complexity(pooled)).squeeze()
        
        return torch.stack([validity, complexity], dim=1)


class ImprovedLSTMParser(nn.Module):
    """Improved LSTM with attention mechanism."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 96, hidden_dim: int = 192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                           bidirectional=True, num_layers=2, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output heads
        self.dropout = nn.Dropout(0.3)
        self.fc_validity = nn.Linear(hidden_dim * 2, 1)
        self.fc_complexity = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Embed
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = (lstm_out * attention_weights).sum(dim=1)
        
        # Outputs
        attended = self.dropout(attended)
        validity = self.sigmoid(self.fc_validity(attended)).squeeze()
        complexity = self.sigmoid(self.fc_complexity(attended)).squeeze()
        
        return torch.stack([validity, complexity], dim=1)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with multi-task loss."""
    model.train()
    total_loss = 0
    correct_validity = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Multi-task loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy for validity prediction
        predictions = (outputs[:, 0] > 0.5).float()
        correct_validity += (predictions == labels[:, 0]).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct_validity / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct_validity = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            predictions = (outputs[:, 0] > 0.5).float()
            correct_validity += (predictions == labels[:, 0]).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct_validity / total


def export_to_onnx(model, vocab_size, output_path: Path, max_length: int = 512):
    """Export trained model to ONNX format."""
    model.eval()
    
    dummy_input = torch.randint(0, vocab_size, (1, max_length))
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=15,
        input_names=['input'],
        output_names=['validity', 'complexity'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'validity': {0: 'batch_size'},
            'complexity': {0: 'batch_size'}
        }
    )
    print(f"\nModel exported to ONNX: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced HTML Parser Model (Phase 2)')
    parser.add_argument('--data-dir', type=str, default='../data/html',
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='../models',
                       help='Directory to save trained model')
    parser.add_argument('--model-type', choices=['transformer', 'lstm'], default='transformer',
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    
    # Load datasets
    data_dir = Path(args.data_dir)
    train_dataset = HTMLDatasetV2(data_dir / 'train.json', args.max_length, use_enhanced=True)
    val_dataset = HTMLDatasetV2(data_dir / 'val.json', args.max_length, use_enhanced=False)
    
    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    if args.model_type == 'transformer':
        model = TransformerHTMLParser(vocab_size).to(device)
        print("Using Transformer architecture")
    else:
        model = ImprovedLSTMParser(vocab_size).to(device)
        print("Using Improved LSTM architecture")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=3)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = output_dir / f'html_parser_{args.model_type}_best.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'model_type': args.model_type,
                'accuracy': best_val_acc,
                'loss': best_val_loss
            }, model_path)
            print(f"✓ Saved best model (acc: {best_val_acc:.4f})")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    
    # Export to ONNX
    output_dir = Path(args.output_dir)
    onnx_path = output_dir / f'html_parser_{args.model_type}_v2.onnx'
    export_to_onnx(model, vocab_size, onnx_path, args.max_length)
    
    print(f"\n📊 Training Summary:")
    print(f"  Model: {args.model_type.capitalize()}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Best Accuracy: {best_val_acc:.4f}")
    print(f"  Best Loss: {best_val_loss:.4f}")
    print(f"\nTo use this model with BrowerAI:")
    print(f"  1. Copy {onnx_path} to ../models/local/")
    print(f"  2. Update ../models/model_config.toml")
    print(f"  3. Build with: cargo build --features ai --release")


if __name__ == '__main__':
    main()
