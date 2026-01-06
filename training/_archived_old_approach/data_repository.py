"""
BrowerAI Data Repository Module

This module provides utilities for loading and managing training datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any, Tuple
import random


class Dataset:
    """Represents a training dataset"""
    
    def __init__(self, path: Path, manifest: Dict[str, Any], split: str = "train"):
        """
        Initialize dataset
        
        Args:
            path: Path to dataset directory
            manifest: Dataset manifest
            split: Data split (train, validation, test)
        """
        self.path = path
        self.manifest = manifest
        self.split = split
        self._data = None
        self._index = 0
    
    def __len__(self) -> int:
        """Get number of samples in current split"""
        if self._data is None:
            self._load_data()
        return len(self._data)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples"""
        if self._data is None:
            self._load_data()
        
        self._index = 0
        return self
    
    def __next__(self) -> Dict[str, Any]:
        """Get next sample"""
        if self._index >= len(self._data):
            raise StopIteration
        
        sample = self._data[self._index]
        self._index += 1
        return sample
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index"""
        if self._data is None:
            self._load_data()
        return self._data[idx]
    
    def _load_data(self):
        """Load data from disk"""
        # Look for data files
        data_files = list(self.path.glob(f"{self.split}*.json"))
        
        if not data_files:
            # Try without split prefix
            data_files = list(self.path.glob("*.json"))
            if data_files and data_files[0].name != "manifest.json":
                data_files = [f for f in data_files if f.name != "manifest.json"]
        
        if not data_files:
            raise FileNotFoundError(f"No data files found in {self.path}")
        
        # Load first data file
        with open(data_files[0], 'r') as f:
            all_data = json.load(f)
        
        # If data is a list, split it
        if isinstance(all_data, list):
            self._data = self._split_data(all_data)
        else:
            self._data = [all_data]
    
    def _split_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split data according to manifest splits"""
        splits = self.manifest.get("splits", {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1
        })
        
        total = len(data)
        train_size = int(total * splits.get("train", 0.8))
        val_size = int(total * splits.get("validation", 0.1))
        
        if self.split == "train":
            return data[:train_size]
        elif self.split == "validation":
            return data[train_size:train_size + val_size]
        elif self.split == "test":
            return data[train_size + val_size:]
        else:
            return data
    
    def shuffle(self):
        """Shuffle dataset"""
        if self._data is None:
            self._load_data()
        random.shuffle(self._data)
    
    def batch(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate over batches
        
        Args:
            batch_size: Size of each batch
        
        Yields:
            Batches of samples
        """
        if self._data is None:
            self._load_data()
        
        for i in range(0, len(self._data), batch_size):
            yield self._data[i:i + batch_size]


class DatasetManager:
    """Manages access to training datasets"""
    
    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize dataset manager
        
        Args:
            data_root: Root directory for datasets
        """
        if data_root is None:
            # Use default location
            data_root = Path(__file__).parent.parent / "data"
        
        self.data_root = Path(data_root)
    
    def list_datasets(self, category: Optional[str] = None) -> List[str]:
        """
        List available datasets
        
        Args:
            category: Filter by category
        
        Returns:
            List of dataset names
        """
        datasets = []
        
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
                rel_path = manifest_file.parent.relative_to(self.data_root)
                datasets.append(str(rel_path))
        
        return sorted(datasets)
    
    def load(self, dataset_name: str, split: str = "train") -> Dataset:
        """
        Load a dataset
        
        Args:
            dataset_name: Name/path of dataset (e.g., "html/structure_prediction")
            split: Data split to load (train, validation, test)
        
        Returns:
            Dataset object
        
        Raises:
            FileNotFoundError: If dataset not found
            ValueError: If invalid split
        """
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be train, validation, or test")
        
        dataset_path = self.data_root / dataset_name
        manifest_path = dataset_path / "manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        return Dataset(dataset_path, manifest, split)
    
    def get_manifest(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get dataset manifest
        
        Args:
            dataset_name: Name/path of dataset
        
        Returns:
            Manifest dictionary
        
        Raises:
            FileNotFoundError: If dataset not found
        """
        dataset_path = self.data_root / dataset_name
        manifest_path = dataset_path / "manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        
        with open(manifest_path, 'r') as f:
            return json.load(f)


def load_dataset(dataset_name: str, split: str = "train", data_root: Optional[str] = None) -> Dataset:
    """
    Convenience function to load a dataset
    
    Args:
        dataset_name: Name/path of dataset
        split: Data split to load
        data_root: Root directory for datasets
    
    Returns:
        Dataset object
    """
    manager = DatasetManager(data_root)
    return manager.load(dataset_name, split)


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = DatasetManager()
    
    # List datasets
    print("Available datasets:")
    for dataset in manager.list_datasets():
        print(f"  - {dataset}")
    
    # Example: Load and iterate
    # dataset = manager.load("html/structure_prediction", split="train")
    # for sample in dataset:
    #     print(sample)
