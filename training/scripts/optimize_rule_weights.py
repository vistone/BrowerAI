#!/usr/bin/env python3
"""
Week 6 Phase 2 Step 7 - 规则权重学习 (轻量版)
基于已采集的框架样本学习规则权重 (多分类逻辑回归)
输出: data/week6_rule_weights.json
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

FRAMEWORKS = [
    "react", "vue", "angular", "jquery", "svelte", "express",
    "ember", "backbone", "alpine", "htmx", "nextjs", "nuxt"
]

INDICATORS = {
    "react": ["react", "react-dom", "__react", "jsx"],
    "vue": ["vue", "v-", "__vue"],
    "angular": ["angular", "ng-", "@angular"],
    "jquery": ["jquery", "$.fn", "$.ajax"],
    "svelte": ["svelte"],
    "express": ["express"],
    "ember": ["ember", "@ember"],
    "backbone": ["backbone", "marionette"],
    "alpine": ["alpine", "x-data"],
    "htmx": ["htmx", "hx-"] ,
    "nextjs": ["next", "__next", "_next"],
    "nuxt": ["nuxt", "__nuxt"],
}


def count_indicator(html: str, patterns: List[str]) -> int:
    html_lower = html.lower()
    count = 0
    for pat in patterns:
        count += html_lower.count(pat)
    return count


def load_samples(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    with open(path, "r") as f:
        for line in f:
            try:
                sample = json.loads(line)
            except Exception:
                continue
            html = sample.get("html", "")
            label = sample.get("framework", "unknown")
            if not html or label not in FRAMEWORKS:
                continue
            row = [count_indicator(html, INDICATORS[fw]) for fw in FRAMEWORKS]
            X.append(row)
            y.append(FRAMEWORKS.index(label))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    data_path = Path("data/week6_samples/framework_samples.jsonl")
    if not data_path.exists():
        print(f"⚠️  找不到样本: {data_path}")
        return

    print("📥 加载样本...")
    X, y = load_samples(data_path)
    if len(X) < 10:
        print("⚠️  样本过少，无法训练规则权重")
        return

    print(f"  样本数: {len(X)} | 特征维度: {X.shape[1]}")

    # 小样本/类别不平衡时，避免 stratify 报错
    class_counts = np.bincount(y, minlength=len(FRAMEWORKS))
    min_count = class_counts[class_counts > 0].min() if len(class_counts[class_counts > 0]) > 0 else 0

    if len(X) < 20 or min_count < 2:
        # 退化为全量训练 + 训练集评估
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    print("🧠 训练规则权重 (多分类逻辑回归)...")
    clf = LogisticRegression(
        max_iter=200,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ 规则权重学习完成 | 准确率: {acc:.4f}")

    print("\n分类报告:")
    print(classification_report(y_test, preds, target_names=[FRAMEWORKS[i] for i in sorted(set(y))]))

    # 保存权重
    weights = {
        "frameworks": FRAMEWORKS,
        "indicators": INDICATORS,
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "accuracy": float(acc),
    }
    output_path = Path("data/week6_rule_weights.json")
    with open(output_path, "w") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)

    print(f"\n💾 权重已保存: {output_path}")


if __name__ == "__main__":
    main()
