#!/bin/bash
# 🚀 快速命令参考 - 下一步行动
# 使用: source quick_actions.sh

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🎯 下一步行动快速参考 (2026-01-23)                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# 监控函数
# ============================================================
monitor_training() {
    echo "📊 监控5倍增强训练进度..."
    tail -f /tmp/training_reduced_aug.log | grep -E "Epoch|Val Intent"
}

check_result() {
    echo "📈 查看最终结果..."
    if [ -f /tmp/training_reduced_aug.log ]; then
        echo "=== 训练完成摘要 ==="
        grep -E "Epoch [0-9]: Train Loss|最佳Intent|✅" /tmp/training_reduced_aug.log | tail -10
    else
        echo "❌ 日志文件不存在"
    fi
}

compare_results() {
    echo "📊 性能对比..."
    echo ""
    echo "数据集配置对比:"
    echo "  原始 Baseline:      53.12%  (325个样本)"
    echo "  25x增强 (失败):     50.88%  (373个样本, 过度相似)"
    echo "  5x增强 (进行中):    ??.??%  (333个样本, 低相似度)"
    echo ""
    echo "预期结果:"
    echo "  ✓ 如果 >= 52%  → 增强因子确实是问题，转向改进增强"
    echo "  ✗ 如果 < 52%  → 需要添加正则化或改变策略"
}

# ============================================================
# 改进增强脚本生成器
# ============================================================
create_advanced_augmentation() {
    echo "📝 创建改进增强脚本..."
    
    cat > training/advanced_augmentation.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""
改进的数据增强: 结构性变换而非简单复制
"""
import json
import random
import re
from pathlib import Path
from bs4 import BeautifulSoup

def shuffle_html_elements(html):
    """随机重排HTML元素"""
    soup = BeautifulSoup(html, 'html.parser')
    # 找所有product div，随机重排
    products = soup.find_all(class_=re.compile('product|item'))
    if len(products) > 1:
        random.shuffle(products)
    return str(soup)

def vary_css_classes(html):
    """变更CSS类名"""
    replacements = {
        'product': 'item',
        'price': 'cost',
        'cart': 'basket',
        'buy': 'purchase'
    }
    for old, new in replacements.items():
        html = html.replace(old, new + str(random.randint(100,999)))
    return html

def advanced_augment(sample, factor=5):
    """结构性增强"""
    augmented = []
    
    for i in range(factor):
        new_sample = sample.copy()
        html = sample.get('input', '')
        
        # 变换1: 重排HTML元素
        if i % 3 == 0:
            html = shuffle_html_elements(html)
        
        # 变换2: CSS类名变更
        html = vary_css_classes(html)
        
        # 变换3: 随机添加/移除元素
        if random.random() > 0.5:
            html += '<div class="recommendation">相关商品</div>'
        
        new_sample['input'] = html[:2000]  # 截断到合理长度
        augmented.append(new_sample)
    
    return augmented

def augment_dataset(input_file, output_file, target=15):
    """增强整个数据集"""
    with open(input_file, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    ecom = [s for s in samples if s.get('intent', {}).get('website_type', '').lower() == 'ecommerce']
    others = [s for s in samples if s.get('intent', {}).get('website_type', '').lower() != 'ecommerce']
    
    factor = (target - len(ecom)) // len(ecom) if ecom else 0
    
    augmented_ecom = []
    for sample in ecom:
        augmented_ecom.extend(advanced_augment(sample, factor=factor+1))
    
    all_samples = others + augmented_ecom[:target]
    
    with open(output_file, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✅ 增强完成: {len(all_samples)} 样本 ({len(augmented_ecom[:target])} ecommerce)")

if __name__ == '__main__':
    augment_dataset('data/website_training_fixed_input.jsonl', 
                   'data/website_training_augmented_advanced.jsonl',
                   target=15)
EOFPYTHON
    
    echo "✅ 脚本已创建: training/advanced_augmentation.py"
}

# ============================================================
# 主菜单
# ============================================================

while true; do
    echo ""
    echo "选择操作:"
    echo "  1) 监控5倍增强训练"
    echo "  2) 查看最终结果"
    echo "  3) 对比性能"
    echo "  4) 创建改进增强脚本"
    echo "  5) 显示所有命令"
    echo "  0) 退出"
    echo ""
    
    read -p "请选择 [0-5]: " choice
    
    case $choice in
        1) monitor_training ;;
        2) check_result ;;
        3) compare_results ;;
        4) create_advanced_augmentation ;;
        5) cat $0 ;;
        0) echo "再见!"; exit 0 ;;
        *) echo "❌ 无效选择" ;;
    esac
done
