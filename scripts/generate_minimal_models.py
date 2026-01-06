#!/usr/bin/env python3
"""
Minimal ONNX Model Generator for BrowerAI
Creates lightweight demo models for testing when full training is not available
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto
import argparse


def create_html_structure_analyzer():
    """Create a minimal HTML structure analyzer model"""
    # Input: HTML tokens (sequence of 128 tokens, each embedded to 32 dims)
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 128, 32]
    )
    
    # Output: Structure complexity score (single float)
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 1]
    )
    
    # Simple linear layer: 128*32 -> 1
    # Flatten 128*32 = 4096 -> 1
    weight = np.random.randn(1, 4096).astype(np.float32) * 0.01
    bias = np.zeros(1).astype(np.float32)
    
    # Create constant nodes
    weight_tensor = helper.make_tensor(
        'weight', TensorProto.FLOAT, [1, 4096], weight.flatten()
    )
    bias_tensor = helper.make_tensor(
        'bias', TensorProto.FLOAT, [1], bias
    )
    
    weight_node = helper.make_node(
        'Constant', [], ['weight_const'], value=weight_tensor
    )
    bias_node = helper.make_node(
        'Constant', [], ['bias_const'], value=bias_tensor
    )
    
    # Flatten input
    flatten_node = helper.make_node(
        'Flatten', ['input'], ['flattened'], axis=1
    )
    
    # Matrix multiply
    matmul_node = helper.make_node(
        'MatMul', ['flattened', 'weight_const'], ['matmul_out']
    )
    
    # Add bias
    add_node = helper.make_node(
        'Add', ['matmul_out', 'bias_const'], ['output']
    )
    
    # Create graph
    graph = helper.make_graph(
        [weight_node, bias_node, flatten_node, matmul_node, add_node],
        'html_structure_analyzer',
        [input_tensor],
        [output_tensor]
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='BrowerAI')
    model.opset_import[0].version = 13
    
    return model


def create_css_selector_optimizer():
    """Create a minimal CSS selector optimizer model"""
    # Input: CSS selector tokens (64 tokens, 24 dims each)
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 64, 24]
    )
    
    # Output: Optimization confidence (single float)
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 1]
    )
    
    # Simple averaging pooling + linear
    weight = np.random.randn(1, 64 * 24).astype(np.float32) * 0.01
    bias = np.zeros(1).astype(np.float32)
    
    weight_tensor = helper.make_tensor(
        'weight', TensorProto.FLOAT, [1, 64 * 24], weight.flatten()
    )
    bias_tensor = helper.make_tensor(
        'bias', TensorProto.FLOAT, [1], bias
    )
    
    weight_node = helper.make_node(
        'Constant', [], ['weight_const'], value=weight_tensor
    )
    bias_node = helper.make_node(
        'Constant', [], ['bias_const'], value=bias_tensor
    )
    
    flatten_node = helper.make_node(
        'Flatten', ['input'], ['flattened'], axis=1
    )
    
    matmul_node = helper.make_node(
        'MatMul', ['flattened', 'weight_const'], ['matmul_out']
    )
    
    add_node = helper.make_node(
        'Add', ['matmul_out', 'bias_const'], ['output']
    )
    
    graph = helper.make_graph(
        [weight_node, bias_node, flatten_node, matmul_node, add_node],
        'css_selector_optimizer',
        [input_tensor],
        [output_tensor]
    )
    
    model = helper.make_model(graph, producer_name='BrowerAI')
    model.opset_import[0].version = 13
    
    return model


def create_js_syntax_analyzer():
    """Create a minimal JS syntax analyzer model"""
    # Input: JS tokens (256 tokens, 48 dims each)
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 256, 48]
    )
    
    # Output: Syntax complexity score
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 1]
    )
    
    weight = np.random.randn(1, 256 * 48).astype(np.float32) * 0.01
    bias = np.zeros(1).astype(np.float32)
    
    weight_tensor = helper.make_tensor(
        'weight', TensorProto.FLOAT, [1, 256 * 48], weight.flatten()
    )
    bias_tensor = helper.make_tensor(
        'bias', TensorProto.FLOAT, [1], bias
    )
    
    weight_node = helper.make_node(
        'Constant', [], ['weight_const'], value=weight_tensor
    )
    bias_node = helper.make_node(
        'Constant', [], ['bias_const'], value=bias_tensor
    )
    
    flatten_node = helper.make_node(
        'Flatten', ['input'], ['flattened'], axis=1
    )
    
    matmul_node = helper.make_node(
        'MatMul', ['flattened', 'weight_const'], ['matmul_out']
    )
    
    add_node = helper.make_node(
        'Add', ['matmul_out', 'bias_const'], ['output']
    )
    
    graph = helper.make_graph(
        [weight_node, bias_node, flatten_node, matmul_node, add_node],
        'js_syntax_analyzer',
        [input_tensor],
        [output_tensor]
    )
    
    model = helper.make_model(graph, producer_name='BrowerAI')
    model.opset_import[0].version = 13
    
    return model


def save_model(model, filename):
    """Save ONNX model and verify it"""
    # Check and create directory
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Check model validity
    try:
        onnx.checker.check_model(model)
        print(f"✓ Model validation passed for {filename}")
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False
    
    # Save model
    onnx.save(model, filename)
    print(f"✓ Model saved to {filename}")
    
    # Verify file size
    size_kb = os.path.getsize(filename) / 1024
    print(f"  Size: {size_kb:.2f} KB")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate minimal ONNX models for BrowerAI')
    parser.add_argument('--output-dir', default='models/local', 
                       help='Output directory for models')
    parser.add_argument('--model', choices=['html', 'css', 'js', 'all'], default='all',
                       help='Which model to generate')
    args = parser.parse_args()
    
    print("BrowerAI Minimal Model Generator")
    print("=" * 50)
    
    models_to_create = {
        'html': ('html_structure_analyzer_v1.onnx', create_html_structure_analyzer),
        'css': ('css_selector_optimizer_v1.onnx', create_css_selector_optimizer),
        'js': ('js_syntax_analyzer_v1.onnx', create_js_syntax_analyzer),
    }
    
    if args.model == 'all':
        selected_models = models_to_create.items()
    else:
        selected_models = [(args.model, models_to_create[args.model])]
    
    success_count = 0
    for name, (filename, creator) in selected_models:
        print(f"\nGenerating {name} model...")
        model = creator()
        output_path = os.path.join(args.output_dir, filename)
        
        if save_model(model, output_path):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Generated {success_count} model(s) successfully")
    print(f"Output directory: {args.output_dir}")
    print("\nTo use these models:")
    print(f"  1. Verify models exist in {args.output_dir}")
    print("  2. Update models/model_config.toml if needed")
    print("  3. Run: cargo build --features ai")
    print("  4. Run: cargo run")


if __name__ == '__main__':
    main()
