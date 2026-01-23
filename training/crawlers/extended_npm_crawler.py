#!/usr/bin/env python3
"""
扩展NPM包采集器 - 采集更多主流JavaScript框架
"""
import subprocess
import sys
import time

# 扩展的NPM包列表 - 覆盖更多框架生态
EXTENDED_PACKAGES = [
    # Vue生态
    "vue", "@vue/cli", "vuex", "vue-router", "pinia", "@vueuse/core",
    
    # Angular生态  
    "@angular/core", "@angular/cli", "@angular/router", "@angular/common",
    
    # 现代轻量框架
    "preact", "solid-js", "alpine", "lit", "petite-vue", "hyperapp",
    
    # 元框架
    "astro", "qwik", "sveltekit", "@remix-run/react",
    
    # 状态管理
    "redux", "mobx", "zustand", "recoil", "jotai", "valtio",
    
    # 构建工具
    "typescript", "@babel/core", "esbuild", "rollup", "tsup",
    
    # UI组件库
    "antd", "element-ui", "vuetify", "@mui/material", "chakra-ui",
    
    # 工具库（已有的补充更多版本）
    "underscore", "rambda", "immutable", "immer",
    
    # 测试框架
    "jest", "vitest", "@testing-library/react", "cypress", "playwright",
]

def download_package(package_name):
    """下载单个NPM包"""
    print(f"\n📦 下载 {package_name}...")
    try:
        result = subprocess.run(
            ["npm", "pack", package_name],
            cwd="real_data/npm_packages",
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"✅ {package_name} 下载成功")
            return True
        else:
            print(f"❌ {package_name} 下载失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {package_name} 下载异常: {e}")
        return False

def main():
    print("=" * 70)
    print("  📦 扩展NPM包采集器")
    print("  目标：采集 {} 个主流框架包".format(len(EXTENDED_PACKAGES)))
    print("=" * 70)
    
    success_count = 0
    fail_count = 0
    
    for i, package in enumerate(EXTENDED_PACKAGES, 1):
        print(f"\n[{i}/{len(EXTENDED_PACKAGES)}]", end=" ")
        if download_package(package):
            success_count += 1
        else:
            fail_count += 1
        time.sleep(1)  # 避免请求过快
    
    print("\n" + "=" * 70)
    print(f"✅ 下载完成: {success_count} 成功, {fail_count} 失败")
    print("=" * 70)

if __name__ == "__main__":
    main()
