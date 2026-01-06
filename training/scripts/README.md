# BrowerAI 训练脚本（Python）

本目录提供网站简化模型的数据处理与训练脚本。依赖声明见 `training/requirements.txt`。

## 安装依赖

```bash
cd training
pip install -r requirements.txt
```

## 数据准备

- `extract_website_complete.py`：从抓取结果提取完整网站 JSONL
- `create_simplified_dataset.py`：生成原始→简化配对数据
- `prepare_website_data.py` / `prepare_data.py`：清洗、切分、校验
- `batch_crawl_websites.py`：批量抓取（可选）
- `dataset_manager.py`：数据集清单、校验、统计

## 训练与导出

- `train_paired_website_generator.py`：字符级 Transformer 训练（原始→简化）
- `count_parameters.py`：参数规模统计
- `export_to_onnx.py`：将检查点导出为 ONNX

典型流程：

```bash
# 生成配对数据
python scripts/create_simplified_dataset.py \
    --input data/website_complete.jsonl \
    --output data/website_paired.jsonl

# 训练
python scripts/train_paired_website_generator.py

# 导出 ONNX
python scripts/export_to_onnx.py \
    --checkpoint checkpoints/paired_generator/epoch_30.pt \
    --output ../models/local/website_generator_v1.onnx
```

## 数据格式

- `website_complete.jsonl`：每行完整网站代码
- `website_paired.jsonl`：每行 `{"original": "...", "simplified": "..."}`

## 部署

将导出的 `.onnx` 放入 `models/local/` 后即可在 Rust 端按配置加载。
