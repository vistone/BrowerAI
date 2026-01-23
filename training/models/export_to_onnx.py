#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export trained PyTorch models to ONNX format for Rust integration.

Usage:
  python training/models/export_to_onnx.py --output models/onnx_exports/
"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_selector_embedding import SelectorEmbeddingModel
from train_all_models import PropertyPredictorModel, ColorLearningModel, CompletePageModel, FinetunedModel

CHECKPOINTS = {
    "selector_embedding": "checkpoints/phase2/selector_embedding_v2/model.pt",
    "property_predictor": "checkpoints/phase2/property_predictor_v2/model.pt",
    "color_model": "checkpoints/phase2/color_model_v2/model.pt",
    "complete_model": "checkpoints/phase2/complete_model_v2/model.pt",
    "finetuned_model": "checkpoints/phase2/finetuned_models/model_lora.pt",
}

CONFIGS = {
    "selector_embedding": {"vocab_size": 64, "embed_dim": 128, "hidden_dim": 256, "num_layers": 2},
    "property_predictor": {"input_dim": 128, "hidden_dim": 256, "num_properties": 50},
    "color_model": {},
    "complete_model": {"input_dim": 256, "hidden_dim": 512, "num_heads": 8, "num_layers": 3},
    "finetuned_model": {"base_dim": 512, "lora_rank": 8},
}

INPUTS = {
    "selector_embedding": lambda: torch.randint(0, 64, (1, 50), dtype=torch.long),
    "property_predictor": lambda: torch.randn(1, 10, 128),
    "color_model": lambda: torch.randn(1, 3, 32, 32),
    "complete_model": lambda: torch.randn(1, 50, 256),
    "finetuned_model": lambda: torch.randn(1, 512),
}


def export_model(name: str, ckpt_path: str, output_dir: str) -> None:
    device = torch.device("cpu")
    if name == "selector_embedding":
        model = SelectorEmbeddingModel(**CONFIGS[name]).to(device)
    elif name == "property_predictor":
        model = PropertyPredictorModel(**CONFIGS[name]).to(device)
    elif name == "color_model":
        model = ColorLearningModel().to(device)
    elif name == "complete_model":
        model = CompletePageModel(**CONFIGS[name]).to(device)
    elif name == "finetuned_model":
        model = FinetunedModel(**CONFIGS[name]).to(device)
    else:
        raise ValueError(f"Unknown model: {name}")

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()

    dummy = INPUTS[name]()
    out_path = os.path.join(output_dir, f"{name}.onnx")
    os.makedirs(output_dir, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"✅ Exported {name} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="models/onnx_exports", help="Output directory for ONNX models")
    args = ap.parse_args()

    for name, ckpt in CHECKPOINTS.items():
        try:
            export_model(name, ckpt, args.output)
        except Exception as e:
            print(f"⚠️  Failed to export {name}: {e}")

    print(f"\n🎉 ONNX export complete. Files in: {args.output}")


if __name__ == "__main__":
    main()
