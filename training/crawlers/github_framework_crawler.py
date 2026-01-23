#!/usr/bin/env python3
"""
真实JS框架GitHub爬虫 - 收集官方框架源代码
"""

import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class GitHubFrameworkCrawler:
    """从GitHub爬取真实框架代码"""
    
    # 全球主流JS框架
    FRAMEWORKS = [
        # 前端框架 (排名靠前)
        {'name': 'react', 'repo': 'facebook/react', 'category': 'frontend', 'priority': 1},
        {'name': 'vue', 'repo': 'vuejs/vue', 'category': 'frontend', 'priority': 2},
        {'name': 'angular', 'repo': 'angular/angular', 'category': 'frontend', 'priority': 3},
        {'name': 'svelte', 'repo': 'sveltejs/svelte', 'category': 'frontend', 'priority': 4},
        {'name': 'ember', 'repo': 'emberjs/ember.js', 'category': 'frontend', 'priority': 5},
        
        # 全栈框架
        {'name': 'next', 'repo': 'vercel/next.js', 'category': 'fullstack', 'priority': 6},
        {'name': 'nuxt', 'repo': 'nuxt/nuxt', 'category': 'fullstack', 'priority': 7},
        {'name': 'gatsby', 'repo': 'gatsbyjs/gatsby', 'category': 'fullstack', 'priority': 8},
        {'name': 'remix', 'repo': 'remix-run/remix', 'category': 'fullstack', 'priority': 9},
        
        # 后端框架
        {'name': 'express', 'repo': 'expressjs/express', 'category': 'backend', 'priority': 10},
        {'name': 'fastify', 'repo': 'fastify/fastify', 'category': 'backend', 'priority': 11},
        {'name': 'koa', 'repo': 'koajs/koa', 'category': 'backend', 'priority': 12},
        {'name': 'nest', 'repo': 'nestjs/nest', 'category': 'backend', 'priority': 13},
        
        # 构建工具
        {'name': 'webpack', 'repo': 'webpack/webpack', 'category': 'build', 'priority': 14},
        {'name': 'vite', 'repo': 'vitejs/vite', 'category': 'build', 'priority': 15},
        {'name': 'rollup', 'repo': 'rollup/rollup', 'category': 'build', 'priority': 16},
        {'name': 'esbuild', 'repo': 'evanw/esbuild', 'category': 'build', 'priority': 17},
        
        # 工具库
        {'name': 'lodash', 'repo': 'lodash/lodash', 'category': 'utility', 'priority': 18},
        {'name': 'axios', 'repo': 'axios/axios', 'category': 'utility', 'priority': 19},
        {'name': 'date-fns', 'repo': 'date-fns/date-fns', 'category': 'utility', 'priority': 20},
    ]
    
    def __init__(self, github_token: Optional[str] = None):
        """初始化爬虫"""
        token = github_token or os.getenv('GITHUB_TOKEN', '')
        
        # 清理token中的引号（处理smart quotes）
        if token:
            # 移除开头和结尾的任何引号
            token = token.strip('\'""\u201c\u201d')  # 包括ASCII和smart quotes
        
        self.token = token if token else None
        
        if not self.token:
            print("⚠️  警告: 未设置GITHUB_TOKEN，将使用更低的API速率限制")
            print("   建议: export GITHUB_TOKEN='your_token_here'")
        
        # 只使用ASCII字符的headers
        self.headers = {
            'Accept': 'application/json',
            'User-Agent': 'BrowserAI-Framework-Crawler/1.0'
        }
        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.data_dir = Path('real_data/github_frameworks')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total': len(self.FRAMEWORKS),
            'success': 0,
            'failed': 0,
            'files_downloaded': 0,
            'total_size': 0,
        }
    
    def get_repo_info(self, repo: str) -> Optional[Dict]:
        """获取仓库信息"""
        try:
            url = f"https://api.github.com/repos/{repo}"
            response = self.session.get(url, timeout=10)
            
            # 确保正确的编码
            response.encoding = 'utf-8'
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"  ❌ 无法获取仓库信息: {response.status_code}")
                return None
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            return None
    
    def get_repo_files(self, repo: str, path: str = '', max_files: int = 20) -> List[Dict]:
        """获取仓库中的文件列表(递归查找)"""
        all_files = []
        
        def explore_directory(dir_path: str, depth: int = 0):
            """递归探索目录"""
            if depth > 3 or len(all_files) >= max_files:  # 限制深度和文件数
                return
            
            try:
                url = f"https://api.github.com/repos/{repo}/contents/{dir_path}"
                response = self.session.get(url, timeout=10)
                # 确保正确的编码
                response.encoding = 'utf-8'
                
                if response.status_code == 200:
                    items = response.json()
                    if isinstance(items, list):
                        for item in items:
                            if len(all_files) >= max_files:
                                break
                            
                            # 如果是JS/TS文件，添加到列表
                            if item.get('type') == 'file':
                                name = item.get('name', '')
                                if (name.endswith('.js') or name.endswith('.ts') or 
                                    name.endswith('.jsx') or name.endswith('.tsx')):
                                    all_files.append(item)
                            
                            # 如果是目录，递归探索
                            elif item.get('type') == 'dir':
                                time.sleep(0.5)  # 避免API速率限制
                                explore_directory(item.get('path', ''), depth + 1)
            except:
                pass
        
        explore_directory(path)
        return all_files[:max_files]
    
    def download_file(self, file_url: str, save_path: Path) -> bool:
        """下载单个文件"""
        try:
            response = self.session.get(file_url, timeout=10)
            # 确保正确的编码
            response.encoding = 'utf-8'
            
            if response.status_code == 200:
                with open(save_path, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(response.text)
                
                self.stats['files_downloaded'] += 1
                self.stats['total_size'] += len(response.text)
                return True
            return False
        except:
            return False
    
    def crawl_framework(self, framework: Dict) -> bool:
        """爬取单个框架"""
        name = framework['name']
        repo = framework['repo']
        category = framework['category']
        
        print(f"\n{'='*70}")
        print(f"🔄 爬取框架: {name.upper()} ({repo})")
        print(f"类别: {category}")
        print('='*70)
        
        # 获取仓库信息
        repo_info = self.get_repo_info(repo)
        if not repo_info:
            print(f"❌ 失败: 无法获取仓库信息")
            return False
        
        # 创建框架目录
        framework_dir = self.data_dir / name
        framework_dir.mkdir(exist_ok=True)
        
        # 保存元数据
        metadata = {
            'name': name,
            'repo': repo,
            'category': category,
            'url': repo_info.get('html_url'),
            'description': repo_info.get('description'),
            'stars': repo_info.get('stargazers_count', 0),
            'forks': repo_info.get('forks_count', 0),
            'language': repo_info.get('language'),
            'updated_at': repo_info.get('updated_at'),
            'crawled_at': datetime.now().isoformat(),
        }
        
        with open(framework_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"⭐ Stars: {metadata['stars']}")
        print(f"🍴 Forks: {metadata['forks']}")
        print(f"📝 语言: {metadata['language']}")
        
        # 下载源文件
        src_dirs = ['src', 'lib', 'packages']
        total_files = 0
        
        for src_dir in src_dirs:
            print(f"\n📂 在 {src_dir}/ 中查找文件...")
            
            files = self.get_repo_files(repo, src_dir, max_files=15)
            
            if files:
                files_subdir = framework_dir / src_dir
                files_subdir.mkdir(exist_ok=True)
                
                for file in files:
                    file_name = file.get('name', '')
                    download_url = file.get('download_url')
                    
                    if download_url and self.download_file(download_url, files_subdir / file_name):
                        total_files += 1
                        print(f"  ✅ {file_name}")
        
        # 下载package.json
        package_json_files = self.get_repo_files(repo, '', max_files=50)
        for file in package_json_files:
            if file.get('name') == 'package.json':
                download_url = file.get('download_url')
                if download_url:
                    self.download_file(download_url, framework_dir / 'package.json')
                    print(f"  ✅ package.json")
        
        print(f"\n✅ {name} 完成: 下载 {total_files} 个源文件")
        return True
    
    def crawl_all(self) -> Dict:
        """爬取所有框架"""
        print("\n" + "="*70)
        print("🌍 全球JS框架真实源代码收集系统")
        print("="*70)
        print(f"要爬取的框架: {self.stats['total']} 个")
        print(f"数据保存位置: {self.data_dir}")
        print("="*70)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'frameworks': {},
            'statistics': self.stats,
        }
        
        for framework in self.FRAMEWORKS:
            try:
                if self.crawl_framework(framework):
                    self.stats['success'] += 1
                    results['frameworks'][framework['name']] = {
                        'status': 'success',
                        'category': framework['category'],
                    }
                else:
                    self.stats['failed'] += 1
                    results['frameworks'][framework['name']] = {
                        'status': 'failed',
                        'category': framework['category'],
                    }
            except Exception as e:
                print(f"❌ 异常错误: {str(e)}")
                self.stats['failed'] += 1
            
            # API速率限制
            time.sleep(2)
        
        # 保存报告
        report_file = self.data_dir / 'crawl_report.json'
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        print("\n" + "="*70)
        print("📊 爬取总结")
        print("="*70)
        print(f"✅ 成功: {self.stats['success']}/{self.stats['total']}")
        print(f"❌ 失败: {self.stats['failed']}/{self.stats['total']}")
        print(f"📄 下载文件: {self.stats['files_downloaded']} 个")
        print(f"💾 总大小: {self.stats['total_size'] / 1024 / 1024:.2f} MB")
        print(f"📝 报告: {report_file}")
        print("="*70)
        
        return results


def main():
    """主函数"""
    import sys
    
    token = sys.argv[1] if len(sys.argv) > 1 else None
    
    crawler = GitHubFrameworkCrawler(token)
    crawler.crawl_all()


if __name__ == '__main__':
    main()
