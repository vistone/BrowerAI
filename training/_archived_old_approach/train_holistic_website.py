"""
Holistic Website Learning Training Script

This script trains the HolisticWebsiteLearner model to understand complete websites
as integrated systems rather than isolated components.

Learning objectives:
1. Website intent classification (ecommerce, news, education, etc.)
2. Code style and company fingerprint recognition
3. File dependency and loading order understanding
4. Device adaptation strategy recognition
5. Framework and build tool detection

Usage:
    python scripts/train_holistic_website.py --config configs/website_learner.yaml
    python scripts/train_holistic_website.py --config configs/website_learner.yaml --resume checkpoint.pt
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import HolisticWebsiteLearner
from core.data import WebsiteDataset, collate_website_batch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HolisticWebsiteTrainer:
    """Trainer for holistic website learning"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = HolisticWebsiteLearner(config['model']).to(self.device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
        
        # Initialize datasets
        from core.data.tokenizers import CodeTokenizer
        
        tokenizer = CodeTokenizer(vocab_size=config['model']['vocab_size'])
        
        data_cfg = config['data']
        data_dir = Path(data_cfg['data_dir'])
        
        self.train_dataset = WebsiteDataset(
            data_file=data_dir / data_cfg['train_file'],
            tokenizer=tokenizer,
            max_html_len=data_cfg['max_html_len'],
            max_css_len=data_cfg['max_css_len'],
            max_js_len=data_cfg['max_js_len']
        )
        
        self.val_dataset = WebsiteDataset(
            data_file=data_dir / data_cfg['val_file'],
            tokenizer=tokenizer,
            max_html_len=data_cfg['max_html_len'],
            max_css_len=data_cfg['max_css_len'],
            max_js_len=data_cfg['max_js_len']
        )
        
        # Data loaders
        train_cfg = config['training']
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=collate_website_batch
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=collate_website_batch
        )
        
        # Optimizer
        opt_cfg = config['optimizer']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            betas=opt_cfg['betas'],
            eps=opt_cfg['eps']
        )
        
        # Learning rate scheduler
        sched_cfg = config['scheduler']
        total_steps = len(self.train_loader) * train_cfg['epochs']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=sched_cfg['min_lr']
        )
        
        # Mixed precision training
        self.use_amp = config['advanced']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Task weights for multi-task learning
        self.task_weights = config['training']['task_weights']
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def compute_multi_task_loss(
        self,
        outputs: dict,
        batch: dict
    ) -> tuple[torch.Tensor, dict]:
        """Compute weighted multi-task loss"""
        
        losses = {}
        
        # 1. Category classification loss
        if 'category_logits' in outputs and 'category' in batch:
            category_loss = nn.CrossEntropyLoss()(
                outputs['category_logits'],
                batch['category']
            )
            losses['category'] = category_loss * self.task_weights['category']
        
        # 2. Framework detection loss
        if 'framework_logits' in outputs and 'framework' in batch:
            framework_loss = nn.CrossEntropyLoss()(
                outputs['framework_logits'],
                batch['framework']
            )
            losses['framework'] = framework_loss * self.task_weights['framework']
        
        # 3. Build tool detection loss
        if 'build_tool_logits' in outputs and 'build_tool' in batch:
            build_tool_loss = nn.CrossEntropyLoss()(
                outputs['build_tool_logits'],
                batch['build_tool']
            )
            losses['build_tool'] = build_tool_loss * self.task_weights['build_tool']
        
        # 4. Company style classification loss
        if 'company_logits' in outputs and 'company' in batch:
            company_loss = nn.CrossEntropyLoss()(
                outputs['company_logits'],
                batch['company']
            )
            losses['company_style'] = company_loss * self.task_weights['company_style']
        
        # 5. Dependency prediction loss (optional, depends on model output)
        if 'dependency_pred' in outputs and 'adjacency_matrix' in batch:
            # Binary cross-entropy for adjacency matrix prediction
            dependency_loss = nn.BCEWithLogitsLoss()(
                outputs['dependency_pred'],
                batch['adjacency_matrix'].float()
            )
            losses['dependency'] = dependency_loss * self.task_weights['dependency']
        
        # 6. Loading order ranking loss (if available)
        if 'loading_order_pred' in outputs and 'loading_order' in batch:
            # Use margin ranking loss or similar
            order_loss = nn.MSELoss()(
                outputs['loading_order_pred'],
                batch['loading_order'].float()
            )
            losses['loading_order'] = order_loss * self.task_weights['loading_order']
        
        # Total loss
        total_loss = sum(losses.values())
        
        return total_loss, losses
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        task_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        batch['html_ids'],
                        batch['css_ids'],
                        batch['js_ids'],
                        batch.get('adjacency_matrix'),
                        batch.get('url_features')
                    )
                    loss, losses = self.compute_multi_task_loss(outputs, batch)
            else:
                outputs = self.model(
                    batch['html_ids'],
                    batch['css_ids'],
                    batch['js_ids'],
                    batch.get('adjacency_matrix'),
                    batch.get('url_features')
                )
                loss, losses = self.compute_multi_task_loss(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for task, task_loss in losses.items():
                if task not in task_losses:
                    task_losses[task] = 0
                task_losses[task] += task_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            self.global_step += 1
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_task_losses = {k: v / len(self.train_loader) for k, v in task_losses.items()}
        
        return {
            'total_loss': avg_loss,
            **avg_task_losses
        }
    
    def validate(self) -> dict:
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0
        task_losses = {}
        
        # Metrics
        correct = {
            'category': 0,
            'framework': 0,
            'build_tool': 0,
            'company': 0
        }
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    batch['html_ids'],
                    batch['css_ids'],
                    batch['js_ids'],
                    batch.get('adjacency_matrix'),
                    batch.get('url_features')
                )
                
                loss, losses = self.compute_multi_task_loss(outputs, batch)
                
                # Accumulate losses
                total_loss += loss.item()
                for task, task_loss in losses.items():
                    if task not in task_losses:
                        task_losses[task] = 0
                    task_losses[task] += task_loss.item()
                
                # Compute accuracies
                if 'category_logits' in outputs:
                    preds = outputs['category_logits'].argmax(dim=1)
                    correct['category'] += (preds == batch['category']).sum().item()
                
                if 'framework_logits' in outputs:
                    preds = outputs['framework_logits'].argmax(dim=1)
                    correct['framework'] += (preds == batch['framework']).sum().item()
                
                if 'build_tool_logits' in outputs:
                    preds = outputs['build_tool_logits'].argmax(dim=1)
                    correct['build_tool'] += (preds == batch['build_tool']).sum().item()
                
                if 'company_logits' in outputs:
                    preds = outputs['company_logits'].argmax(dim=1)
                    correct['company'] += (preds == batch['company']).sum().item()
                
                total += batch['html'].size(0)
        
        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        avg_task_losses = {k: v / len(self.val_loader) for k, v in task_losses.items()}
        
        # Compute accuracies
        accuracies = {f'{k}_acc': v / total for k, v in correct.items()}
        
        return {
            'total_loss': avg_loss,
            **avg_task_losses,
            **accuracies
        }
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def train(self, num_epochs: int, checkpoint_dir: str = "checkpoints"):
        """Main training loop"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch+1} - Train: {train_metrics}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch+1} - Val: {val_metrics}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            self.save_checkpoint(checkpoint_path)
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                self.save_checkpoint(best_path)
                logger.info(f"New best model saved with val_loss={self.best_val_loss:.4f}")
        
        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Holistic Website Learner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/website_learner",
        help="Directory to save checkpoints"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Initialize trainer
    trainer = HolisticWebsiteTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    num_epochs = config['training']['epochs']
    trainer.train(num_epochs, checkpoint_dir=args.checkpoint_dir)
    
    # Export to ONNX
    logger.info("Exporting model to ONNX...")
    export_path = config['export']['onnx_path']
    
    # Create dummy inputs for ONNX export
    dummy_html = torch.randint(0, config['model']['vocab_size'], (1, 512)).to(trainer.device)
    dummy_css = torch.randint(0, config['model']['vocab_size'], (1, 512)).to(trainer.device)
    dummy_js = torch.randint(0, config['model']['vocab_size'], (1, 512)).to(trainer.device)
    
    trainer.model.export_to_onnx(
        export_path,
        dummy_html,
        dummy_css,
        dummy_js
    )
    
    logger.info(f"Model exported to {export_path}")


if __name__ == "__main__":
    main()
