"""
Count and breakdown parameters in HolisticWebsiteLearner model
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.website_learner import HolisticWebsiteLearner

def count_parameters_detailed(model):
    """Count parameters with detailed breakdown"""
    
    print("=" * 70)
    print("HolisticWebsiteLearner Parameter Breakdown")
    print("=" * 70)
    
    total = 0
    
    # Group parameters by component
    components = {}
    
    for name, param in model.named_parameters():
        # Get component name (first part before dot)
        component = name.split('.')[0]
        
        if component not in components:
            components[component] = {'params': 0, 'details': []}
        
        param_count = param.numel()
        components[component]['params'] += param_count
        components[component]['details'].append((name, param_count, list(param.shape)))
        total += param_count
    
    # Print summary by component
    print("\nComponent Summary:")
    print("-" * 70)
    for component, info in sorted(components.items(), key=lambda x: x[1]['params'], reverse=True):
        percentage = (info['params'] / total) * 100
        print(f"{component:30s} {info['params']:>12,} ({percentage:>5.2f}%)")
    
    print("-" * 70)
    print(f"{'TOTAL':30s} {total:>12,} (100.00%)")
    print()
    
    # Detailed breakdown for each component
    print("\nDetailed Parameter List:")
    print("=" * 70)
    for component in sorted(components.keys()):
        info = components[component]
        print(f"\n{component.upper()} - Total: {info['params']:,}")
        print("-" * 70)
        for name, count, shape in info['details'][:10]:  # Show first 10
            print(f"  {name:50s} {count:>10,}  {shape}")
        if len(info['details']) > 10:
            print(f"  ... and {len(info['details']) - 10} more parameters")
    
    return total


if __name__ == "__main__":
    # Create model with default config
    config = {
        "vocab_size": 10000,
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 4,
        "d_ff": 1024,
        "dropout": 0.1,
        "max_len": 512
    }
    
    print("\nConfiguration:")
    print("-" * 70)
    for key, value in config.items():
        print(f"  {key:20s} {value}")
    print()
    
    model = HolisticWebsiteLearner(config)
    
    total_params = count_parameters_detailed(model)
    
    print("\n" + "=" * 70)
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size (FP32): ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"Model Size (FP16): ~{total_params * 2 / 1024 / 1024:.1f} MB")
    print("=" * 70)
