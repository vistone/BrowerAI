"""
Demo: Actually crawl a real website to show the system works

This demonstrates the ACTUAL data collection process.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the crawler from prepare_website_data
from scripts.prepare_website_data import WebsiteCrawler
import json


async def demo_crawl():
    """Actually crawl a real website"""
    
    # Use a simple, publicly accessible website
    test_url = "https://example.com"
    
    print("=" * 70)
    print("REAL Website Crawling Demo")
    print("=" * 70)
    print(f"\nCrawling: {test_url}")
    print("This will make REAL HTTP requests...\n")
    
    async with WebsiteCrawler(max_files=5) as crawler:
        website_data = await crawler.crawl_website(test_url, category="documentation")
        
        if website_data:
            print("\n" + "=" * 70)
            print("Crawl Results")
            print("=" * 70)
            
            print(f"\nURL: {website_data['url']}")
            print(f"Category: {website_data['category']}")
            print(f"HTML size: {len(website_data['html'])} characters")
            print(f"CSS files: {len(website_data['css_files'])}")
            print(f"JS files: {len(website_data['js_files'])}")
            print(f"Dependencies: {len(website_data['dependencies'])}")
            
            print(f"\nMetadata:")
            print(f"  Framework: {website_data['metadata']['framework']}")
            print(f"  Build Tool: {website_data['metadata']['build_tool']}")
            print(f"  Company: {website_data['metadata']['company']}")
            print(f"  Responsive: {website_data['metadata']['responsive']}")
            
            # Show first 500 chars of HTML
            print(f"\nHTML Preview (first 500 chars):")
            print("-" * 70)
            print(website_data['html'][:500])
            print("...")
            
            # Save to file
            output_file = "data/websites/demo_sample.jsonl"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(website_data, ensure_ascii=False) + '\n')
            
            print(f"\n✓ Saved to: {output_file}")
            print("\nThis is REAL DATA that can be used for training!")
            
        else:
            print("\n✗ Failed to crawl website")


if __name__ == "__main__":
    print("\n⚠️  This will make REAL network requests to example.com")
    print("Press Ctrl+C to cancel, or wait 3 seconds to proceed...")
    
    try:
        import time
        time.sleep(3)
        asyncio.run(demo_crawl())
    except KeyboardInterrupt:
        print("\nCancelled by user")
