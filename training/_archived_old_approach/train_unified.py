"""
Unified Training Script for BrowerAI

Modern training pipeline consolidating all scattered scripts.

Usage:
    # Train HTML parser
    python train_unified.py --task html_parser --config configs/html_parser.yaml
    
    # Train deobfuscator
    python train_unified.py --task deobfuscator --config configs/deobfuscator.yaml
    
    # Train with custom settings
    python train_unified.py --task html_parser --epochs 20 --batch-size 64 --lr 0.001
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

from core.models import HTMLParser, CSSParser, JSParser, CodeDeobfuscator
from core.trainers import Trainer
from core.data.tokenizers import UnifiedWebTokenizer


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model(task: str, config: dict) -> nn.Module:
    """Get model based on task type."""
    models = {
        "html_parser": HTMLParser,
        "css_parser": CSSParser,
        "js_parser": JSParser,
        "deobfuscator": CodeDeobfuscator,
    }
    
    model_class = models.get(task)
    if not model_class:
        raise ValueError(f"Unknown task: {task}. Available: {list(models.keys())}")
    
    model = model_class(config["model"])
    
    # Print model info
    params = model.count_parameters()
    print(f"\n📊 Model Statistics:")
    print(f"   Task: {task}")
    print(f"   Architecture: {model_class.__name__}")
    print(f"   Parameters: {params['total']:,} ({params['trainable']:,} trainable)")
    
    return model


def get_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Get optimizer with modern best practices."""
    opt_config = config["optimizer"]
    opt_type = opt_config["type"].lower()
    lr = opt_config["lr"]
    weight_decay = opt_config.get("weight_decay", 0.01)
    
    # AdamW is the modern standard (used in GPT, BERT, etc.)
    if opt_type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=opt_config.get("betas", (0.9, 0.999)),
            eps=opt_config.get("eps", 1e-8)
        )
    elif opt_type == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif opt_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_config.get("momentum", 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    print(f"\n⚙️  Optimizer: {opt_type.upper()}")
    print(f"   Learning rate: {lr}")
    print(f"   Weight decay: {weight_decay}")
    
    return optimizer


def get_scheduler(optimizer: optim.Optimizer, config: dict, 
                 num_training_steps: int) -> optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler."""
    sched_config = config.get("scheduler", {})
    if not sched_config:
        return None
    
    sched_type = sched_config["type"].lower()
    
    # Cosine annealing with warmup (modern standard)
    if sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=sched_config.get("min_lr", 1e-6)
        )
    elif sched_type == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=sched_config.get("end_factor", 0.1),
            total_iters=num_training_steps
        )
    elif sched_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_config.get("factor", 0.5),
            patience=sched_config.get("patience", 5)
        )
    else:
        return None
    
    print(f"\n📈 Scheduler: {sched_type}")
    
    return scheduler


def prepare_data(task: str, config: dict):
    """Prepare datasets and dataloaders."""
    data_config = config["data"]
    data_dir = Path(data_config["data_dir"])
    
    # Build/load tokenizer
    tokenizer_path = Path("tokenizers") / f"{task}_tokenizer.json"
    
    if tokenizer_path.exists():
        tokenizer = UnifiedWebTokenizer.load(tokenizer_path)
    else:
        print(f"⚠️  Tokenizer not found. Please run data preparation first.")
        print(f"   Expected: {tokenizer_path}")
        sys.exit(1)
    
    # For now, return dummy loaders (will be implemented with actual datasets)
    print(f"\n📂 Data:")
    print(f"   Directory: {data_dir}")
    print(f"   Tokenizer: {tokenizer_path}")
    print(f"   Vocab size: {len(tokenizer.vocab)}")
    
    # TODO: Implement actual dataset loading
    # For now, return None to indicate data needs to be prepared
    return None, None, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Unified BrowerAI Training")
    
    # Task selection
    parser.add_argument("--task", type=str, required=True,
                       choices=["html_parser", "css_parser", "js_parser", "deobfuscator"],
                       help="Training task")
    
    # Configuration
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config YAML file")
    
    # Training hyperparameters (override config)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    
    # Model architecture (override config)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    
    # Training features
    parser.add_argument("--use-amp", action="store_true",
                       help="Use automatic mixed precision")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BrowerAI Unified Training Pipeline v2.0")
    print("=" * 80)
    
    # Load configuration
    if args.config:
        config = load_config(Path(args.config))
    else:
        # Use default config for task
        config_path = Path("configs") / f"{args.task}.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            print(f"⚠️  No config found for task: {args.task}")
            print(f"   Please create: {config_path}")
            print(f"   Or specify with --config")
            sys.exit(1)
    
    # Override config with command-line arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["optimizer"]["lr"] = args.lr
    if args.weight_decay:
        config["optimizer"]["weight_decay"] = args.weight_decay
    if args.d_model:
        config["model"]["d_model"] = args.d_model
    if args.num_heads:
        config["model"]["num_heads"] = args.num_heads
    if args.num_layers:
        config["model"]["num_layers"] = args.num_layers
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Prepare data
    train_loader, val_loader, tokenizer = prepare_data(args.task, config)
    
    if train_loader is None:
        print("\n❌ Data preparation required!")
        print("   Please run data preparation scripts first:")
        print(f"   python scripts/prepare_{args.task}_data.py")
        return
    
    # Get model
    config["model"]["vocab_size"] = len(tokenizer.vocab)
    model = get_model(args.task, config)
    
    # Get optimizer
    optimizer = get_optimizer(model, config)
    
    # Get scheduler
    num_training_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = get_scheduler(optimizer, config, num_training_steps)
    
    # Get loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        max_epochs=config["training"]["epochs"],
        gradient_accumulation_steps=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp,
        checkpoint_dir=Path(args.checkpoint_dir) / args.task,
        log_every_n_steps=config["training"].get("log_every_n_steps", 10),
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"✅ Resumed from checkpoint: {args.resume}")
    
    # Train
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    train_history, val_history = trainer.train()
    
    # Export to ONNX
    print("\n" + "=" * 80)
    print("Exporting to ONNX")
    print("=" * 80)
    
    onnx_path = Path("../models/local") / f"{args.task}_v2.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.export_onnx(
        path=onnx_path,
        input_shape=(1, config["model"].get("max_len", 512)),
        input_names=["input_ids"],
        output_names=["logits"]
    )
    
    # Save config
    config_save_path = onnx_path.with_suffix(".json")
    model.save_config(config_save_path)
    
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Best model: {trainer.checkpoint_dir / 'best_model.pt'}")
    print(f"ONNX model: {onnx_path}")
    print(f"Config: {config_save_path}")


if __name__ == "__main__":
    main()
