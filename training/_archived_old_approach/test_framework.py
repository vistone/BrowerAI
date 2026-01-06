#!/usr/bin/env python3
"""
Test script for BrowerAI Training Framework v2.0

Tests all core components without requiring actual training.
"""

import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / "core"))


def test_structure():
    """Test directory structure."""
    print("=" * 70)
    print("Testing Directory Structure")
    print("=" * 70)
    
    required_dirs = [
        "core",
        "core/models",
        "core/data",
        "core/trainers",
        "configs",
    ]
    
    required_files = [
        "core/__init__.py",
        "core/models/__init__.py",
        "core/models/base.py",
        "core/models/transformer.py",
        "core/models/attention.py",
        "core/models/parsers.py",
        "core/models/deobfuscator.py",
        "core/data/__init__.py",
        "core/data/tokenizers.py",
        "core/trainers/__init__.py",
        "core/trainers/trainer.py",
        "configs/html_parser.yaml",
        "configs/deobfuscator.yaml",
        "train_unified.py",
        "README_V2.md",
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            all_ok = False
    
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - MISSING")
            all_ok = False
    
    return all_ok


def test_imports():
    """Test that modules can be imported."""
    print("\n" + "=" * 70)
    print("Testing Module Imports")
    print("=" * 70)
    
    try:
        import torch
        torch_available = True
        print(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        torch_available = False
        print("⚠️  PyTorch not installed (run: pip install torch)")
    
    if torch_available:
        try:
            from core.models import BaseModel, TransformerEncoder
            print("✅ core.models imports successful")
        except Exception as e:
            print(f"❌ core.models import failed: {e}")
            return False
        
        try:
            from core.data.tokenizers import CodeTokenizer
            print("✅ core.data imports successful")
        except Exception as e:
            print(f"❌ core.data import failed: {e}")
            return False
        
        try:
            from core.trainers import Trainer
            print("✅ core.trainers imports successful")
        except Exception as e:
            print(f"❌ core.trainers import failed: {e}")
            return False
    
    return torch_available


def test_configs():
    """Test configuration files."""
    print("\n" + "=" * 70)
    print("Testing Configuration Files")
    print("=" * 70)
    
    try:
        import yaml
        yaml_available = True
    except ImportError:
        print("⚠️  PyYAML not installed (run: pip install pyyaml)")
        return False
    
    configs = [
        "configs/html_parser.yaml",
        "configs/deobfuscator.yaml",
    ]
    
    all_ok = True
    for config_file in configs:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required keys
            required_keys = ["model", "training", "optimizer", "data"]
            missing_keys = [k for k in required_keys if k not in config]
            
            if missing_keys:
                print(f"❌ {config_file} - Missing keys: {missing_keys}")
                all_ok = False
            else:
                print(f"✅ {config_file} - Valid")
                print(f"   Model: d_model={config['model'].get('d_model')}, "
                      f"layers={config['model'].get('num_layers', 'N/A')}")
                print(f"   Training: epochs={config['training']['epochs']}, "
                      f"batch_size={config['training']['batch_size']}")
        except Exception as e:
            print(f"❌ {config_file} - Error: {e}")
            all_ok = False
    
    return all_ok


def test_code_quality():
    """Test code syntax and quality."""
    print("\n" + "=" * 70)
    print("Testing Code Quality")
    print("=" * 70)
    
    python_files = list(Path("core").rglob("*.py"))
    python_files.extend([
        Path("train_unified.py"),
    ])
    
    all_ok = True
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                code = f.read()
            
            # Compile to check syntax
            compile(code, str(py_file), 'exec')
            
            lines = len(code.split('\n'))
            print(f"✅ {py_file} ({lines} lines)")
        except SyntaxError as e:
            print(f"❌ {py_file} - Syntax error: {e}")
            all_ok = False
        except Exception as e:
            print(f"⚠️  {py_file} - Warning: {e}")
    
    return all_ok


def print_summary():
    """Print framework summary."""
    print("\n" + "=" * 70)
    print("Framework Summary")
    print("=" * 70)
    
    # Count files and lines
    core_files = list(Path("core").rglob("*.py"))
    total_lines = 0
    for f in core_files:
        try:
            with open(f) as file:
                total_lines += len(file.readlines())
        except:
            pass
    
    print(f"\n📊 Statistics:")
    print(f"   Core modules: {len(core_files)} files")
    print(f"   Total lines: {total_lines:,}")
    print(f"   Config files: 2")
    
    print(f"\n🎯 Key Features:")
    print("   ✅ Unified training interface")
    print("   ✅ Modern transformer architectures")
    print("   ✅ Multi-task learning support")
    print("   ✅ Advanced training techniques (AMP, gradient accumulation)")
    print("   ✅ Automatic ONNX export")
    print("   ✅ Comprehensive configuration system")
    
    print(f"\n📚 Models Available:")
    print("   • HTML Parser (transformer encoder)")
    print("   • CSS Parser (lightweight transformer)")
    print("   • JS Parser (deep transformer with contrastive learning)")
    print("   • Code Deobfuscator (seq2seq with copy mechanism)")
    
    print(f"\n🚀 Quick Start:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Prepare data: python scripts/prepare_html_data.py")
    print("   3. Train model: python train_unified.py --task html_parser")
    print("   4. Export to ONNX: Automatic after training")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BrowerAI Training Framework v2.0 - Test Suite")
    print("=" * 70)
    
    results = {
        "Structure": test_structure(),
        "Imports": test_imports(),
        "Configs": test_configs(),
        "Code Quality": test_code_quality(),
    }
    
    print_summary()
    
    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Framework is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review above output.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
