#!/usr/bin/env python3
"""
Dataset Manager for BrowerAI Training Data Repository

This tool provides functionality to manage, download, upload, and validate
training datasets for BrowerAI's AI models.

Usage:
    python dataset_manager.py list
    python dataset_manager.py download --dataset html/structure_prediction
    python dataset_manager.py upload --path /path/to/data --name html/custom
    python dataset_manager.py validate --dataset html/structure_prediction
    python dataset_manager.py stats
"""

import argparse
import json
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil


class DatasetManager:
    """Manages training datasets for BrowerAI"""
    
    def __init__(self, data_root: str = None):
        """
        Initialize dataset manager
        
        Args:
            data_root: Root directory for datasets (default: ./data)
        """
        if data_root is None:
            # Use script directory's parent as root
            script_dir = Path(__file__).parent
            data_root = script_dir.parent / "data"
        
        self.data_root = Path(data_root)
        self.metadata_dir = self.data_root / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category directories
        for category in ["html", "css", "js", "combined", "benchmarks"]:
            (self.data_root / category).mkdir(parents=True, exist_ok=True)
            for subdir in ["raw", "processed", "synthetic"]:
                (self.data_root / category / subdir).mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available datasets
        
        Args:
            category: Filter by category (html, css, js, combined, benchmarks)
        
        Returns:
            List of dataset information dictionaries
        """
        datasets = []
        
        # Search for manifest files
        if category:
            search_paths = [self.data_root / category]
        else:
            search_paths = [
                self.data_root / cat 
                for cat in ["html", "css", "js", "combined", "benchmarks"]
            ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            for manifest_file in search_path.rglob("manifest.json"):
                try:
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                    
                    # Add path information
                    rel_path = manifest_file.parent.relative_to(self.data_root)
                    manifest["path"] = str(rel_path)
                    datasets.append(manifest)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Failed to load manifest {manifest_file}: {e}", 
                          file=sys.stderr)
        
        return datasets
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific dataset
        
        Args:
            dataset_name: Name/path of the dataset (e.g., "html/structure_prediction")
        
        Returns:
            Dataset manifest or None if not found
        """
        dataset_path = self.data_root / dataset_name
        manifest_file = dataset_path / "manifest.json"
        
        if not manifest_file.exists():
            return None
        
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            manifest["path"] = dataset_name
            return manifest
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading manifest: {e}", file=sys.stderr)
            return None
    
    def create_manifest(
        self,
        name: str,
        description: str,
        size_samples: int,
        size_bytes: int,
        format_type: str = "json",
        schema: Dict[str, str] = None,
        source: str = "synthetic",
        tags: List[str] = None,
        license_type: str = "Apache-2.0"
    ) -> Dict[str, Any]:
        """
        Create a dataset manifest
        
        Args:
            name: Dataset name
            description: Dataset description
            size_samples: Number of samples
            size_bytes: Size in bytes
            format_type: Data format (json, csv, parquet)
            schema: Input/output schema
            source: Data source (synthetic, crawled, manual)
            tags: List of tags
            license_type: License type
        
        Returns:
            Manifest dictionary
        """
        now = datetime.utcnow().isoformat() + "Z"
        
        if schema is None:
            schema = {
                "input": "Data input description",
                "output": "Data output description"
            }
        
        if tags is None:
            tags = []
        
        manifest = {
            "name": name,
            "version": "1.0.0",
            "created": now,
            "updated": now,
            "description": description,
            "size": {
                "samples": size_samples,
                "bytes": size_bytes
            },
            "format": format_type,
            "schema": schema,
            "splits": {
                "train": 0.8,
                "validation": 0.1,
                "test": 0.1
            },
            "license": license_type,
            "source": source,
            "tags": tags,
            "checksum": ""
        }
        
        return manifest
    
    def save_manifest(self, dataset_name: str, manifest: Dict[str, Any]) -> bool:
        """
        Save manifest to dataset directory
        
        Args:
            dataset_name: Name/path of the dataset
            manifest: Manifest dictionary
        
        Returns:
            True if successful, False otherwise
        """
        dataset_path = self.data_root / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        manifest_file = dataset_path / "manifest.json"
        
        try:
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"Manifest saved to {manifest_file}")
            return True
        except IOError as e:
            print(f"Error saving manifest: {e}", file=sys.stderr)
            return False
    
    def validate_dataset(self, dataset_name: str) -> bool:
        """
        Validate dataset structure and manifest
        
        Args:
            dataset_name: Name/path of the dataset
        
        Returns:
            True if valid, False otherwise
        """
        print(f"Validating dataset: {dataset_name}")
        
        # Check if dataset exists
        dataset_path = self.data_root / dataset_name
        if not dataset_path.exists():
            print(f"  ✗ Dataset directory not found: {dataset_path}", file=sys.stderr)
            return False
        print(f"  ✓ Dataset directory exists")
        
        # Check manifest
        manifest = self.get_dataset_info(dataset_name)
        if manifest is None:
            print(f"  ✗ Manifest not found or invalid", file=sys.stderr)
            return False
        print(f"  ✓ Manifest valid")
        
        # Check required fields
        required_fields = ["name", "version", "description", "size", "format"]
        for field in required_fields:
            if field not in manifest:
                print(f"  ✗ Missing required field: {field}", file=sys.stderr)
                return False
        print(f"  ✓ All required fields present")
        
        # Check if data files exist
        data_files = list(dataset_path.glob("*.json")) + \
                     list(dataset_path.glob("*.csv")) + \
                     list(dataset_path.glob("*.parquet"))
        
        if not data_files:
            print(f"  ! Warning: No data files found in dataset directory", file=sys.stderr)
        else:
            print(f"  ✓ Found {len(data_files)} data file(s)")
        
        print(f"  ✓ Dataset validation passed")
        return True
    
    def calculate_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA256 checksum of a file
        
        Args:
            file_path: Path to file
        
        Returns:
            Hex digest of SHA256 checksum
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all datasets
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_datasets": 0,
            "total_samples": 0,
            "total_bytes": 0,
            "by_category": {}
        }
        
        datasets = self.list_datasets()
        
        for dataset in datasets:
            stats["total_datasets"] += 1
            
            if "size" in dataset:
                stats["total_samples"] += dataset["size"].get("samples", 0)
                stats["total_bytes"] += dataset["size"].get("bytes", 0)
            
            # Get category from path
            path_parts = dataset.get("path", "").split("/")
            if path_parts:
                category = path_parts[0]
                if category not in stats["by_category"]:
                    stats["by_category"][category] = {
                        "datasets": 0,
                        "samples": 0,
                        "bytes": 0
                    }
                
                stats["by_category"][category]["datasets"] += 1
                if "size" in dataset:
                    stats["by_category"][category]["samples"] += dataset["size"].get("samples", 0)
                    stats["by_category"][category]["bytes"] += dataset["size"].get("bytes", 0)
        
        return stats


def format_bytes(bytes_count: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} TB"


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="BrowerAI Dataset Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all datasets
  python dataset_manager.py list

  # List HTML datasets only
  python dataset_manager.py list --category html

  # Get info about specific dataset
  python dataset_manager.py info --dataset html/structure_prediction

  # Validate dataset
  python dataset_manager.py validate --dataset html/structure_prediction

  # Create new dataset manifest
  python dataset_manager.py create --name html/custom_dataset \\
      --description "Custom HTML dataset" \\
      --samples 1000 --bytes 5000000

  # Show statistics
  python dataset_manager.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    list_parser.add_argument("--category", help="Filter by category")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.add_argument("--dataset", required=True, help="Dataset name")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--dataset", required=True, help="Dataset name")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create dataset manifest")
    create_parser.add_argument("--name", required=True, help="Dataset name")
    create_parser.add_argument("--description", required=True, help="Dataset description")
    create_parser.add_argument("--samples", type=int, required=True, help="Number of samples")
    create_parser.add_argument("--bytes", type=int, required=True, help="Size in bytes")
    create_parser.add_argument("--format", default="json", help="Data format")
    create_parser.add_argument("--source", default="synthetic", help="Data source")
    create_parser.add_argument("--license", default="Apache-2.0", help="License type")
    create_parser.add_argument("--tags", nargs="+", help="Dataset tags")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    manager = DatasetManager()
    
    if args.command == "list":
        datasets = manager.list_datasets(args.category)
        
        if not datasets:
            print("No datasets found")
            return
        
        print(f"Found {len(datasets)} dataset(s):\n")
        for dataset in datasets:
            print(f"📦 {dataset.get('name', 'Unknown')}")
            print(f"   Path: {dataset.get('path', 'Unknown')}")
            print(f"   Version: {dataset.get('version', 'Unknown')}")
            print(f"   Description: {dataset.get('description', 'No description')}")
            if "size" in dataset:
                samples = dataset["size"].get("samples", 0)
                bytes_size = dataset["size"].get("bytes", 0)
                print(f"   Size: {samples:,} samples, {format_bytes(bytes_size)}")
            print()
    
    elif args.command == "info":
        info = manager.get_dataset_info(args.dataset)
        
        if info is None:
            print(f"Dataset not found: {args.dataset}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Dataset Information:")
        print(f"  Name: {info.get('name')}")
        print(f"  Version: {info.get('version')}")
        print(f"  Description: {info.get('description')}")
        print(f"  Created: {info.get('created')}")
        print(f"  Updated: {info.get('updated')}")
        print(f"  Format: {info.get('format')}")
        print(f"  Source: {info.get('source')}")
        print(f"  License: {info.get('license')}")
        
        if "size" in info:
            print(f"  Samples: {info['size'].get('samples', 0):,}")
            print(f"  Size: {format_bytes(info['size'].get('bytes', 0))}")
        
        if "splits" in info:
            print(f"  Splits:")
            for split_name, split_ratio in info["splits"].items():
                print(f"    {split_name}: {split_ratio * 100:.0f}%")
        
        if "tags" in info:
            print(f"  Tags: {', '.join(info['tags'])}")
    
    elif args.command == "validate":
        is_valid = manager.validate_dataset(args.dataset)
        sys.exit(0 if is_valid else 1)
    
    elif args.command == "create":
        manifest = manager.create_manifest(
            name=args.name,
            description=args.description,
            size_samples=args.samples,
            size_bytes=args.bytes,
            format_type=args.format,
            source=args.source,
            tags=args.tags or [],
            license_type=args.license
        )
        
        success = manager.save_manifest(args.name, manifest)
        sys.exit(0 if success else 1)
    
    elif args.command == "stats":
        stats = manager.get_statistics()
        
        print("Dataset Repository Statistics\n")
        print(f"Total Datasets: {stats['total_datasets']}")
        print(f"Total Samples: {stats['total_samples']:,}")
        print(f"Total Size: {format_bytes(stats['total_bytes'])}")
        print()
        
        if stats["by_category"]:
            print("By Category:")
            for category, cat_stats in sorted(stats["by_category"].items()):
                print(f"  {category}:")
                print(f"    Datasets: {cat_stats['datasets']}")
                print(f"    Samples: {cat_stats['samples']:,}")
                print(f"    Size: {format_bytes(cat_stats['bytes'])}")


if __name__ == "__main__":
    main()
