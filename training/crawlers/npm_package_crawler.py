#!/usr/bin/env python3
"""
NPM包爬虫 - 从npm registry下载真实的框架包
下载的是已发布的、可能被混淆的代码
"""

import os
import json
import requests
import tarfile
import time
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

class NPMPackageCrawler:
    """从NPM registry下载框架包"""
    
    PACKAGES = [
        # 核心框架
        'react', 'react-dom', 'vue', '@angular/core', 'svelte',
        'ember-source', 'backbone',
        
        # 全栈框架
        'next', 'nuxt', 'gatsby', 'remix-run',
        
        # 后端框架
        'express', 'fastify', 'koa', '@nestjs/core',
        
        # 构建工具
        'webpack', 'vite', 'rollup', 'esbuild', 'parcel',
        
        # 工具库
        'lodash', 'axios', 'date-fns', 'ramda', 'moment',
    ]
    
    def __init__(self):
        self.registry_url = 'https://registry.npmjs.org'
        self.data_dir = Path('real_data/npm_packages')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total': len(self.PACKAGES),
            'success': 0,
            'failed': 0,
            'total_size': 0,
        }
    
    def get_package_info(self, package_name: str) -> Optional[Dict]:
        """获取NPM包信息"""
        try:
            url = f"{self.registry_url}/{package_name}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"  ❌ 无法获取包信息: {response.status_code}")
                return None
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            return None
    
    def download_package(self, package_name: str) -> bool:
        """下载单个NPM包"""
        print(f"\n{'='*70}")
        print(f"📦 正在下载: {package_name}")
        print('='*70)
        
        try:
            # 获取包信息
            pkg_info = self.get_package_info(package_name)
            if not pkg_info:
                self.stats['failed'] += 1
                return False
            
            # 获取最新版本
            if 'dist-tags' not in pkg_info or 'latest' not in pkg_info['dist-tags']:
                print(f"  ⚠️  找不到最新版本")
                self.stats['failed'] += 1
                return False
            
            latest_version = pkg_info['dist-tags']['latest']
            
            if latest_version not in pkg_info.get('versions', {}):
                print(f"  ⚠️  版本信息不可用")
                self.stats['failed'] += 1
                return False
            
            version_info = pkg_info['versions'][latest_version]
            tarball_url = version_info['dist']['tarball']
            
            print(f"📌 版本: {latest_version}")
            print(f"🔗 下载: {tarball_url}")
            
            # 创建包目录
            safe_name = package_name.replace('/', '__')
            pkg_dir = self.data_dir / safe_name
            pkg_dir.mkdir(exist_ok=True)
            
            # 下载tar.gz
            response = requests.get(tarball_url, stream=True, timeout=30)
            response.raise_for_status()
            
            tar_path = pkg_dir / f"{safe_name}-{latest_version}.tar.gz"
            
            # 保存tar文件
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = tar_path.stat().st_size
            self.stats['total_size'] += file_size
            
            print(f"💾 大小: {file_size / 1024 / 1024:.2f} MB")
            
            # 解压
            print(f"📂 正在解压...")
            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(pkg_dir)
                
                # 删除tar文件以节省空间
                tar_path.unlink()
                
                print(f"✅ 完成: {safe_name}")
                self.stats['success'] += 1
                
                # 保存包元数据
                metadata = {
                    'name': package_name,
                    'version': latest_version,
                    'tarball': tarball_url,
                    'description': pkg_info.get('description'),
                    'repository': pkg_info.get('repository'),
                    'keywords': pkg_info.get('keywords', []),
                    'author': pkg_info.get('author'),
                    'downloads': version_info['dist'].get('downloads', 0),
                    'downloaded_at': datetime.now().isoformat(),
                }
                
                with open(pkg_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                return True
                
            except Exception as e:
                print(f"  ❌ 解压失败: {str(e)}")
                self.stats['failed'] += 1
                return False
        
        except Exception as e:
            print(f"  ❌ 下载失败: {str(e)}")
            self.stats['failed'] += 1
            return False
    
    def download_all(self) -> Dict:
        """下载所有包"""
        print("\n" + "="*70)
        print("🌍 NPM包爬虫 - 下载真实的框架包")
        print("="*70)
        print(f"要下载的包: {self.stats['total']} 个")
        print(f"数据保存位置: {self.data_dir}")
        print("="*70)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'packages': {},
            'statistics': self.stats,
        }
        
        for package in self.PACKAGES:
            try:
                if self.download_package(package):
                    results['packages'][package] = 'success'
                else:
                    results['packages'][package] = 'failed'
            except Exception as e:
                print(f"❌ 异常: {str(e)}")
                self.stats['failed'] += 1
            
            # API速率限制
            time.sleep(1)
        
        # 保存报告
        report_file = self.data_dir / 'download_report.json'
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        print("\n" + "="*70)
        print("📊 下载总结")
        print("="*70)
        print(f"✅ 成功: {self.stats['success']}/{self.stats['total']}")
        print(f"❌ 失败: {self.stats['failed']}/{self.stats['total']}")
        print(f"💾 总大小: {self.stats['total_size'] / 1024 / 1024:.2f} MB")
        print(f"📝 报告: {report_file}")
        print("="*70)
        
        return results


def main():
    crawler = NPMPackageCrawler()
    crawler.download_all()


if __name__ == '__main__':
    main()
