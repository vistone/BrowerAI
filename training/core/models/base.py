"""
Base Model Class

Provides common interface for all BrowerAI models following modern ML practices.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import json


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all BrowerAI models.
    
    Features:
        - Automatic parameter counting
        - Model checkpointing
        - ONNX export support
        - Configuration management
        - Device handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._device = torch.device('cpu')
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses."""
        pass
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "frozen": total - trainable,
            "total": total
        }
    
    def save_checkpoint(self, path: Path, epoch: int, optimizer_state: Optional[Dict] = None, 
                       metrics: Optional[Dict] = None):
        """Save model checkpoint with training state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "parameter_count": self.count_parameters(),
        }
        
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if metrics:
            checkpoint["metrics"] = metrics
            
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")
    
    @classmethod
    def load_checkpoint(cls, path: Path, device: Optional[torch.device] = None):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device or torch.device('cpu'))
        
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if device:
            model = model.to(device)
            
        print(f"✅ Checkpoint loaded: {path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Parameters: {checkpoint.get('parameter_count', {}).get('total', 'N/A'):,}")
        
        return model, checkpoint
    
    def export_onnx(self, path: Path, input_shape: tuple, input_names: list, 
                    output_names: list, opset_version: int = 14):
        """
        Export model to ONNX format.
        
        Args:
            path: Output file path
            input_shape: Example input shape (batch_size, seq_length, ...)
            input_names: Names for input tensors
            output_names: Names for output tensors
            opset_version: ONNX opset version (14+ recommended)
        """
        self.eval()
        
        dummy_input = torch.randint(0, 100, input_shape).to(self._device)
        
        torch.onnx.export(
            self,
            dummy_input,
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes={
                input_names[0]: {0: "batch_size", 1: "sequence_length"},
                output_names[0]: {0: "batch_size"}
            },
            export_params=True,
            do_constant_folding=True
        )
        
        print(f"✅ ONNX model exported: {path}")
        print(f"   Opset version: {opset_version}")
        
    def save_config(self, path: Path):
        """Save model configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✅ Config saved: {path}")
    
    def to_device(self, device: torch.device):
        """Move model to device and update internal state."""
        self._device = device
        return self.to(device)
    
    def freeze_layers(self, layer_names: list):
        """Freeze specific layers for transfer learning."""
        for name, param in self.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = False
                print(f"🔒 Froze layer: {name}")
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("🔓 All layers unfrozen")
    
    def get_layer_lr(self, base_lr: float, layer_decay: float = 0.95) -> Dict:
        """
        Generate layer-wise learning rates (used in modern LLMs).
        
        Args:
            base_lr: Base learning rate
            layer_decay: Multiplicative decay per layer
            
        Returns:
            Dict mapping parameter names to learning rates
        """
        layer_lrs = {}
        num_layers = len([n for n, _ in self.named_parameters()])
        
        for i, (name, _) in enumerate(self.named_parameters()):
            decay_factor = layer_decay ** (num_layers - i - 1)
            layer_lrs[name] = base_lr * decay_factor
            
        return layer_lrs
