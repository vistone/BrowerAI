"""
Unified Trainer

Modern training loop with best practices from:
- PyTorch Lightning
- HuggingFace Transformers Trainer
- Fast.ai
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from typing import Optional, Dict, List, Callable
from tqdm import tqdm
import time


class Trainer:
    """
    Unified trainer for all BrowerAI models.
    
    Features:
        - Automatic mixed precision (AMP)
        - Gradient accumulation
        - Gradient clipping
        - Learning rate warmup
        - Callbacks system
        - Metrics tracking
        - Automatic checkpointing
        - TensorBoard logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        # Training config
        max_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 0,
        # AMP config
        use_amp: bool = False,
        # Checkpointing
        checkpoint_dir: Optional[Path] = None,
        save_every_n_epochs: int = 1,
        # Callbacks
        callbacks: Optional[List[Callable]] = None,
        # Logging
        log_every_n_steps: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
        # Device handling
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Training config
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        
        # AMP
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs
        
        # Callbacks
        self.callbacks = callbacks or []
        
        # Logging
        self.log_every_n_steps = log_every_n_steps
        
        # State tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics history
        self.train_history = {
            "loss": [],
            "lr": [],
            "step_time": []
        }
        self.val_history = {
            "loss": [],
            "metrics": []
        }
        
        print(f"✅ Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   AMP: {self.use_amp}")
        print(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
    
    def train(self):
        """Main training loop."""
        print(f"\n🚀 Starting training for {self.max_epochs} epochs")
        print(f"   Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"   Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Validation epoch
            val_metrics = {}
            if self.val_loader and (epoch + 1) % 1 == 0:  # Validate every epoch
                val_metrics = self._validate_epoch()
            
            # Print epoch summary
            self._print_epoch_summary(train_metrics, val_metrics)
            
            # Callbacks
            self._run_callbacks('on_epoch_end', {
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
            
            # Save checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_metrics.get('loss', train_metrics['loss']))
        
        print(f"\n✅ Training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        
        return self.train_history, self.val_history
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self._compute_loss(batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Scheduler step (if per-step scheduler)
                if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Warmup
                if self.global_step < self.warmup_steps:
                    lr_scale = min(1.0, self.global_step / self.warmup_steps)
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = pg['initial_lr'] * lr_scale
            
            # Track metrics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.log_every_n_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        # Epoch metrics
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time
        
        metrics = {
            'loss': avg_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
            'time': epoch_time
        }
        
        # Update history
        self.train_history['loss'].append(avg_loss)
        self.train_history['lr'].append(metrics['lr'])
        self.train_history['step_time'].append(epoch_time / num_batches)
        
        return metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = self._move_to_device(batch)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = self._compute_loss(batch)
                
                epoch_loss += loss.item()
                num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        metrics = {'loss': avg_loss}
        
        # Update history
        self.val_history['loss'].append(avg_loss)
        
        # Update best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint(self.current_epoch, avg_loss, is_best=True)
        
        # Scheduler step (if validation-based)
        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        
        return metrics
    
    def _compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for a batch."""
        if self.criterion:
            # Standard supervised learning
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[-1]
            else:
                inputs = batch['input_ids']
                targets = batch['labels']
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        else:
            # Model computes its own loss
            loss = self.model(**batch)
            if isinstance(loss, dict):
                loss = loss['loss']
        
        return loss
    
    def _move_to_device(self, batch):
        """Move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (tuple, list)):
            return [self._move_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        else:
            return batch
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model: {best_path}")
    
    def _print_epoch_summary(self, train_metrics: Dict, val_metrics: Dict):
        """Print epoch summary."""
        print(f"\nEpoch {self.current_epoch + 1}/{self.max_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  LR: {train_metrics['lr']:.2e}")
        print(f"  Time: {train_metrics['time']:.2f}s")
        
        if val_metrics:
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
    
    def _run_callbacks(self, event: str, data: Dict):
        """Run callbacks for an event."""
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(data)


class DistributedTrainer(Trainer):
    """
    Distributed training with DDP (DistributedDataParallel).
    
    For multi-GPU training following modern practices.
    """
    
    def __init__(self, *args, local_rank: int = 0, world_size: int = 1, **kwargs):
        self.local_rank = local_rank
        self.world_size = world_size
        
        super().__init__(*args, **kwargs)
        
        # Wrap model in DDP
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank
            )
        
        print(f"✅ Distributed trainer initialized (rank {local_rank}/{world_size})")
