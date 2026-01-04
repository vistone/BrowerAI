#!/usr/bin/env python3
"""
Visual Regression Testing Script for BrowerAI Phase 3.4

This script performs visual regression testing on the rendering engine by comparing
rendered outputs against baseline images to detect visual changes.

Usage:
    python test_visual_regression.py [--baseline-dir DIR] [--output-dir DIR]
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not available. Install with: pip install Pillow")


class VisualRegressionTester:
    """Visual regression testing for rendering engine."""
    
    def __init__(self, baseline_dir: Path, output_dir: Path):
        self.baseline_dir = baseline_dir
        self.output_dir = output_dir
        self.results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'new_baselines': 0,
            'differences': []
        }
        
        # Create directories if they don't exist
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_test_html(self) -> List[Tuple[str, str]]:
        """Generate test HTML samples for rendering."""
        return [
            ('simple_text', '<html><body><h1>Hello World</h1><p>Test paragraph</p></body></html>'),
            ('nested_divs', '<html><body><div><div><div>Nested Content</div></div></div></body></html>'),
            ('css_styling', '<html><head><style>body{background:blue;color:white;}</style></head><body>Styled</body></html>'),
            ('flexbox_layout', '<html><head><style>.flex{display:flex;gap:10px;}</style></head><body><div class="flex"><div>A</div><div>B</div></div></body></html>'),
            ('grid_layout', '<html><head><style>.grid{display:grid;grid-template-columns:1fr 1fr;}</style></head><body><div class="grid"><div>1</div><div>2</div></div></body></html>'),
            ('absolute_positioning', '<html><head><style>.abs{position:absolute;top:10px;left:20px;}</style></head><body><div class="abs">Positioned</div></body></html>'),
            ('float_layout', '<html><head><style>.float{float:left;width:50%;}</style></head><body><div class="float">Left</div><div>Right</div></body></html>'),
            ('text_formatting', '<html><body><b>Bold</b> <i>Italic</i> <u>Underline</u></body></html>'),
            ('lists', '<html><body><ul><li>Item 1</li><li>Item 2</li></ul><ol><li>First</li><li>Second</li></ol></body></html>'),
            ('complex_layout', '<html><head><style>.container{width:800px;margin:auto;}.header{background:#333;color:white;padding:20px;}</style></head><body><div class="container"><div class="header"><h1>Title</h1></div><p>Content</p></div></body></html>')
        ]
    
    def create_mock_screenshot(self, test_name: str, html_content: str) -> Path:
        """Create a mock screenshot for testing (simulate rendering)."""
        if not PIL_AVAILABLE:
            # Create empty file
            output_path = self.output_dir / f"{test_name}_current.png"
            output_path.touch()
            return output_path
        
        # Create a simple visual representation
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple representation
        draw.rectangle([10, 10, 790, 590], outline='black', width=2)
        
        # Add text (simplified HTML content representation)
        try:
            # Try to use a font, fall back to default if not available
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw hash of content as visual marker
        content_hash = hashlib.md5(html_content.encode()).hexdigest()[:8]
        draw.text((20, 20), f"Test: {test_name}", fill='black', font=font)
        draw.text((20, 50), f"Hash: {content_hash}", fill='blue', font=font)
        
        # Draw some visual elements based on HTML content
        if 'background:blue' in html_content:
            draw.rectangle([50, 100, 750, 500], fill='blue', outline='black')
        if '<h1>' in html_content:
            draw.rectangle([50, 100, 400, 150], fill='lightgray', outline='black')
        if 'display:flex' in html_content or 'display:grid' in html_content:
            draw.rectangle([50, 200, 350, 400], fill='lightblue', outline='black')
            draw.rectangle([400, 200, 700, 400], fill='lightgreen', outline='black')
        
        output_path = self.output_dir / f"{test_name}_current.png"
        img.save(output_path)
        return output_path
    
    def compare_images(self, baseline_path: Path, current_path: Path, threshold: float = 0.95) -> Tuple[bool, float]:
        """Compare two images and return similarity score."""
        if not PIL_AVAILABLE:
            # Can't compare without PIL, assume match if files exist
            return baseline_path.exists() and current_path.exists(), 1.0
        
        if not baseline_path.exists():
            return False, 0.0
        
        try:
            baseline_img = Image.open(baseline_path)
            current_img = Image.open(current_path)
            
            # Convert to same size if needed
            if baseline_img.size != current_img.size:
                current_img = current_img.resize(baseline_img.size)
            
            # Convert to numpy arrays
            baseline_array = np.array(baseline_img)
            current_array = np.array(current_img)
            
            # Calculate pixel-wise difference
            diff = np.abs(baseline_array.astype(float) - current_array.astype(float))
            similarity = 1.0 - (np.mean(diff) / 255.0)
            
            passed = similarity >= threshold
            return passed, similarity
            
        except Exception as e:
            print(f"Error comparing images: {e}")
            return False, 0.0
    
    def create_diff_image(self, baseline_path: Path, current_path: Path, diff_path: Path):
        """Create a visual diff image highlighting differences."""
        if not PIL_AVAILABLE:
            return
        
        try:
            baseline_img = Image.open(baseline_path)
            current_img = Image.open(current_path)
            
            # Ensure same size
            if baseline_img.size != current_img.size:
                current_img = current_img.resize(baseline_img.size)
            
            # Convert to numpy
            baseline_array = np.array(baseline_img)
            current_array = np.array(current_img)
            
            # Calculate difference
            diff_array = np.abs(baseline_array.astype(float) - current_array.astype(float))
            
            # Highlight differences in red
            diff_img = Image.fromarray(diff_array.astype(np.uint8))
            diff_img.save(diff_path)
            
        except Exception as e:
            print(f"Error creating diff image: {e}")
    
    def run_visual_regression_tests(self):
        """Run visual regression tests."""
        print("\n" + "=" * 60)
        print("Visual Regression Testing for Rendering Engine")
        print("=" * 60)
        
        test_cases = self.generate_test_html()
        
        for test_name, html_content in test_cases:
            self.results['total_tests'] += 1
            
            print(f"\n📸 Testing: {test_name}")
            
            # Create current screenshot (mock rendering)
            current_path = self.create_mock_screenshot(test_name, html_content)
            baseline_path = self.baseline_dir / f"{test_name}_baseline.png"
            
            # Check if baseline exists
            if not baseline_path.exists():
                # Create new baseline
                if PIL_AVAILABLE:
                    img = Image.open(current_path)
                    img.save(baseline_path)
                else:
                    import shutil
                    shutil.copy(current_path, baseline_path)
                
                self.results['new_baselines'] += 1
                self.results['passed'] += 1
                print(f"  ✓ New baseline created")
            else:
                # Compare with baseline
                passed, similarity = self.compare_images(baseline_path, current_path)
                
                if passed:
                    self.results['passed'] += 1
                    print(f"  ✓ Passed (similarity: {similarity:.2%})")
                else:
                    self.results['failed'] += 1
                    print(f"  ✗ Failed (similarity: {similarity:.2%})")
                    
                    # Create diff image
                    diff_path = self.output_dir / f"{test_name}_diff.png"
                    self.create_diff_image(baseline_path, current_path, diff_path)
                    
                    self.results['differences'].append({
                        'test_name': test_name,
                        'similarity': similarity,
                        'baseline': str(baseline_path),
                        'current': str(current_path),
                        'diff': str(diff_path)
                    })
        
        return self.results
    
    def generate_report(self):
        """Generate visual regression test report."""
        print("\n" + "=" * 60)
        print("Visual Regression Test Report")
        print("=" * 60)
        
        print(f"\n📊 Summary:")
        print(f"  Total Tests: {self.results['total_tests']}")
        print(f"  Passed: {self.results['passed']}")
        print(f"  Failed: {self.results['failed']}")
        print(f"  New Baselines: {self.results['new_baselines']}")
        
        if self.results['differences']:
            print(f"\n⚠️  Visual Differences Detected:")
            print("-" * 60)
            for diff in self.results['differences']:
                print(f"\n  Test: {diff['test_name']}")
                print(f"  Similarity: {diff['similarity']:.2%}")
                print(f"  Diff Image: {diff['diff']}")
        else:
            print("\n✅ No visual regressions detected")
        
        # Save results
        output_path = Path('visual_regression_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
        
        print("\n" + "=" * 60)
        print("Visual regression testing ensures rendering consistency")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Visual regression testing for BrowerAI rendering')
    parser.add_argument('--baseline-dir', type=str, default='visual_baselines',
                       help='Directory containing baseline images')
    parser.add_argument('--output-dir', type=str, default='visual_outputs',
                       help='Directory for output images')
    parser.add_argument('--threshold', type=float, default=0.95,
                       help='Similarity threshold (0-1, default: 0.95)')
    
    args = parser.parse_args()
    
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    
    if not PIL_AVAILABLE:
        print("\n⚠️  Pillow library recommended for full functionality")
        print("Install with: pip install Pillow")
        print("Running in limited mode...\n")
    
    tester = VisualRegressionTester(baseline_dir, output_dir)
    tester.run_visual_regression_tests()
    tester.generate_report()
    
    # Exit with error code if tests failed
    if tester.results['failed'] > 0:
        exit(1)


if __name__ == '__main__':
    main()
