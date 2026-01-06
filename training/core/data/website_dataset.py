"""
Holistic Website Dataset

Dataset for learning complete websites as integrated systems.
Captures loading order, dependencies, and multi-modal relationships.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import networkx as nx
from collections import defaultdict


class WebsiteDataset(Dataset):
    """
    Dataset for complete website understanding.
    
    Each sample contains:
        - HTML content
        - CSS stylesheets (in loading order)
        - JavaScript files (in loading order)
        - Resource dependencies
        - Website metadata (category, URL, etc.)
        - Dependency graph
    
    Data format (JSONL):
    {
        "url": "https://example.com",
        "category": "ecommerce",
        "html": "<html>...</html>",
        "css_files": [
            {"path": "main.css", "content": "...", "order": 0},
            {"path": "theme.css", "content": "...", "order": 1}
        ],
        "js_files": [
            {"path": "vendor.js", "content": "...", "order": 0, "type": "blocking"},
            {"path": "app.js", "content": "...", "order": 1, "type": "defer"}
        ],
        "dependencies": [
            {"from": "app.js", "to": "vendor.js", "type": "import"},
            {"from": "index.html", "to": "main.css", "type": "link"}
        ],
        "metadata": {
            "viewport": "width=device-width",
            "responsive": true,
            "framework": "React",
            "build_tool": "Webpack",
            "company": "Amazon"
        }
    }
    """
    
    CATEGORIES = [
        "ecommerce", "news", "education", "entertainment", "social",
        "business", "government", "personal", "documentation", "tools"
    ]
    
    def __init__(self, data_file: Path, tokenizer, max_html_len: int = 2048,
                 max_css_len: int = 1024, max_js_len: int = 2048):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_html_len = max_html_len
        self.max_css_len = max_css_len
        self.max_js_len = max_js_len
        
        # Load dataset
        self.samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        print(f"✅ Loaded {len(self.samples)} websites from {data_file}")
        
        # Build category mapping
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.CATEGORIES)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Handle multi-page structure (new format with depth crawling)
        if "pages" in sample:
            # New format: {"pages": {"main": {...}, "sub_pages": [...]}}
            main_page = sample["pages"]["main"]
            sub_pages = sample["pages"].get("sub_pages", [])
            
            # Combine main page with sub-pages for holistic learning
            html_content = main_page["html"]
            css_files = main_page.get("css_files", [])
            js_files = main_page.get("js_files", [])
            
            # Add features from sub-pages (sample first N sub-pages to avoid overflow)
            max_sub_pages = 3
            for sub_page in sub_pages[:max_sub_pages]:
                # Extract key elements from sub-pages (simplified)
                html_content += f"\n<!-- SUB-PAGE: {sub_page['url']} -->\n"
                # Note: For now we're just adding metadata; in future can do full multi-page modeling
        else:
            # Old format: {"html": "...", "css_files": [...], ...}
            html_content = sample.get("html", "")
            css_files = sample.get("css_files", [])
            js_files = sample.get("js_files", [])
        
        # Tokenize HTML
        html_ids = self.tokenizer.encode(
            html_content,
            max_length=self.max_html_len
        )
        html_ids = self._pad_or_truncate(html_ids, self.max_html_len)
        
        # Combine CSS files (in loading order)
        css_combined = self._combine_files(css_files)
        css_ids = self.tokenizer.encode(css_combined, max_length=self.max_css_len)
        css_ids = self._pad_or_truncate(css_ids, self.max_css_len)
        
        # Combine JS files (in loading order)
        js_combined = self._combine_files(js_files)
        js_ids = self.tokenizer.encode(js_combined, max_length=self.max_js_len)
        js_ids = self._pad_or_truncate(js_ids, self.max_js_len)
        
        # Extract URL features (simple hash-based for now)
        url_features = self._extract_url_features(sample["url"])
        
        # Build dependency graph
        adjacency_matrix = self._build_dependency_graph(sample)
        
        # Get category
        category = sample.get("category", "unknown")
        category_idx = self.category_to_idx.get(category, 0)
        
        # Extract metadata features
        metadata = sample.get("metadata", {})
        
        return {
            "html_ids": torch.tensor(html_ids, dtype=torch.long),
            "css_ids": torch.tensor(css_ids, dtype=torch.long),
            "js_ids": torch.tensor(js_ids, dtype=torch.long),
            "url_features": torch.tensor(url_features, dtype=torch.float),
            "adjacency_matrix": adjacency_matrix,
            "category": torch.tensor(category_idx, dtype=torch.long),
            
            # Metadata labels for multi-task learning
            "framework": self._encode_framework(metadata.get("framework")),
            "build_tool": self._encode_build_tool(metadata.get("build_tool")),
            "company_style": self._encode_company(metadata.get("company")),
            "responsive": torch.tensor(metadata.get("responsive", False), dtype=torch.float),
            
            # Original data for debugging
            "url": sample["url"],
        }
    
    def _combine_files(self, files: List[Dict]) -> str:
        """Combine multiple files in loading order."""
        if not files:
            return ""
        
        # Sort by loading order
        sorted_files = sorted(files, key=lambda x: x.get("order", 0))
        
        # Concatenate with separators
        combined = []
        for file_info in sorted_files:
            content = file_info.get("content", "")
            file_type = file_info.get("type", "")
            
            # Add metadata comment
            combined.append(f"/* File: {file_info.get('path', 'unknown')} */")
            if file_type:
                combined.append(f"/* Type: {file_type} */")
            combined.append(content)
            combined.append("\n\n")
        
        return "\n".join(combined)
    
    def _build_dependency_graph(self, sample: Dict) -> torch.Tensor:
        """
        Build adjacency matrix from dependencies.
        
        Nodes: HTML (0), CSS files, JS files
        Edges: Dependencies between them
        """
        # Collect all files
        files = ["index.html"]  # HTML is always node 0
        files.extend([f["path"] for f in sample.get("css_files", [])])
        files.extend([f["path"] for f in sample.get("js_files", [])])
        
        num_files = len(files)
        file_to_idx = {f: i for i, f in enumerate(files)}
        
        # Initialize adjacency matrix
        adj = torch.zeros(num_files, num_files, dtype=torch.float)
        
        # Add edges from dependencies
        for dep in sample.get("dependencies", []):
            from_file = dep.get("from", "")
            to_file = dep.get("to", "")
            
            if from_file in file_to_idx and to_file in file_to_idx:
                from_idx = file_to_idx[from_file]
                to_idx = file_to_idx[to_file]
                adj[from_idx, to_idx] = 1.0
        
        return adj
    
    def _extract_url_features(self, url: str) -> List[float]:
        """Extract features from URL."""
        features = [0.0] * 128
        
        # Simple hashing for demo (in production, use learned embeddings)
        url_hash = hash(url) % 128
        features[url_hash] = 1.0
        
        # Add domain-based features
        if ".com" in url:
            features[0] = 1.0
        if ".org" in url:
            features[1] = 1.0
        if ".edu" in url:
            features[2] = 1.0
        if ".gov" in url:
            features[3] = 1.0
        
        # Protocol
        if url.startswith("https"):
            features[4] = 1.0
        
        return features
    
    def _encode_framework(self, framework: Optional[str]) -> torch.Tensor:
        """Encode framework as one-hot."""
        frameworks = [
            "React", "Vue", "Angular", "Svelte", "jQuery", "Backbone",
            "Ember", "Preact", "Alpine", "Lit", "Unknown"
        ]
        idx = frameworks.index(framework) if framework in frameworks else len(frameworks) - 1
        return torch.tensor(idx, dtype=torch.long)
    
    def _encode_build_tool(self, build_tool: Optional[str]) -> torch.Tensor:
        """Encode build tool as one-hot."""
        tools = [
            "Webpack", "Rollup", "Vite", "Parcel", "Gulp", "Grunt",
            "Browserify", "esbuild", "Snowpack", "Unknown"
        ]
        idx = tools.index(build_tool) if build_tool in tools else len(tools) - 1
        return torch.tensor(idx, dtype=torch.long)
    
    def _encode_company(self, company: Optional[str]) -> torch.Tensor:
        """Encode company style."""
        companies = [
            "Google", "Facebook", "Amazon", "Microsoft", "Apple",
            "Alibaba", "Tencent", "Baidu", "Netflix", "Airbnb",
            "Unknown"
        ]
        idx = companies.index(company) if company in companies else len(companies) - 1
        return torch.tensor(idx, dtype=torch.long)
    
    def _pad_or_truncate(self, ids: List[int], max_length: int) -> List[int]:
        """Pad or truncate to max_length."""
        if len(ids) > max_length:
            return ids[:max_length]
        else:
            return ids + [0] * (max_length - len(ids))


class WebsiteCrawlDataset(Dataset):
    """
    Dataset from real website crawls.
    
    Automatically extracts dependencies, loading order, and metadata
    from crawled websites.
    """
    
    def __init__(self, crawl_dir: Path, tokenizer, **kwargs):
        self.crawl_dir = Path(crawl_dir)
        self.tokenizer = tokenizer
        
        # Find all website directories
        self.website_dirs = [d for d in self.crawl_dir.iterdir() if d.is_dir()]
        
        print(f"✅ Found {len(self.website_dirs)} crawled websites")
    
    def __len__(self):
        return len(self.website_dirs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        website_dir = self.website_dirs[idx]
        
        # Load HTML
        html_file = website_dir / "index.html"
        html_content = html_file.read_text(encoding='utf-8') if html_file.exists() else ""
        
        # Load CSS files
        css_files = list((website_dir / "css").glob("*.css")) if (website_dir / "css").exists() else []
        css_combined = "\n".join([f.read_text(encoding='utf-8') for f in css_files])
        
        # Load JS files
        js_files = list((website_dir / "js").glob("*.js")) if (website_dir / "js").exists() else []
        js_combined = "\n".join([f.read_text(encoding='utf-8') for f in js_files])
        
        # Load metadata
        metadata_file = website_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
        
        # Tokenize
        html_ids = self.tokenizer.encode(html_content, max_length=2048)
        css_ids = self.tokenizer.encode(css_combined, max_length=1024)
        js_ids = self.tokenizer.encode(js_combined, max_length=2048)
        
        # Extract URL features
        url = metadata.get("url", str(website_dir.name))
        url_features = self._extract_url_features(url)
        
        # Build simple dependency graph (HTML -> CSS, HTML -> JS)
        num_files = 1 + len(css_files) + len(js_files)
        adj = torch.zeros(num_files, num_files)
        for i in range(1, num_files):
            adj[0, i] = 1.0  # HTML depends on everything
        
        # Get category from metadata or directory structure
        category_name = metadata.get("category", "unknown")
        category_idx = WebsiteDataset.CATEGORIES.index(category_name) if category_name in WebsiteDataset.CATEGORIES else 0
        
        return {
            "html_ids": torch.tensor(self._pad(html_ids, 2048), dtype=torch.long),
            "css_ids": torch.tensor(self._pad(css_ids, 1024), dtype=torch.long),
            "js_ids": torch.tensor(self._pad(js_ids, 2048), dtype=torch.long),
            "url_features": torch.tensor(url_features, dtype=torch.float),
            "adjacency_matrix": adj,
            "category": torch.tensor(category_idx, dtype=torch.long),
            "url": url
        }
    
    def _extract_url_features(self, url: str) -> List[float]:
        """Extract URL features."""
        features = [0.0] * 128
        url_hash = hash(url) % 128
        features[url_hash] = 1.0
        return features
    
    def _pad(self, ids: List[int], max_len: int) -> List[int]:
        """Pad to max length."""
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [0] * (max_len - len(ids))


def collate_website_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for website batches.
    
    Handles variable-sized dependency graphs by padding.
    """
    # Stack simple tensors
    html_ids = torch.stack([item["html_ids"] for item in batch])
    css_ids = torch.stack([item["css_ids"] for item in batch])
    js_ids = torch.stack([item["js_ids"] for item in batch])
    url_features = torch.stack([item["url_features"] for item in batch])
    categories = torch.stack([item["category"] for item in batch])
    
    # Handle variable-sized adjacency matrices
    max_nodes = max(item["adjacency_matrix"].size(0) for item in batch)
    batch_size = len(batch)
    
    adjacency_batch = torch.zeros(batch_size, max_nodes, max_nodes)
    for i, item in enumerate(batch):
        size = item["adjacency_matrix"].size(0)
        adjacency_batch[i, :size, :size] = item["adjacency_matrix"]
    
    return {
        "html_ids": html_ids,
        "css_ids": css_ids,
        "js_ids": js_ids,
        "url_features": url_features,
        "adjacency_matrix": adjacency_batch,
        "category": categories,
        
        # Pass through metadata if present
        "framework": torch.stack([item["framework"] for item in batch]) if "framework" in batch[0] else None,
        "build_tool": torch.stack([item["build_tool"] for item in batch]) if "build_tool" in batch[0] else None,
        "company_style": torch.stack([item["company_style"] for item in batch]) if "company_style" in batch[0] else None,
    }
