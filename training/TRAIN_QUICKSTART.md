# 🎓 BrowerAI 模型训练快速开始

从收集数据到部署模型的完整流程。

## 📋 前置要求

- Python 3.8+
- Rust 1.70+（已有）
- 至少 100+ 反馈样本

## 🚀 5 步开始训练

### 步骤 1: 安装 Python 依赖

```bash
cd training

# 方式 A: 自动设置（推荐）
./setup_env.sh

# 方式 B: 手动安装
pip install -r requirements.txt
```

### 步骤 2: 收集训练数据

回到项目根目录，访问真实网站收集反馈数据：

```bash
cd ..

# 访问多个网站
cargo run --bin browerai -- --learn \
    https://example.com \
    https://github.com \
    https://rust-lang.org \
    https://developer.mozilla.org \
    https://www.wikipedia.org
```

**目标**: 收集 100+ 反馈样本（访问 10-20 个网站）

### 步骤 3: 训练模型

```bash
cd training/scripts

# HTML 复杂度预测模型
python train_html_complexity.py \
    --data ../data/feedback_*.json \
    --epochs 100

# CSS 优化建议模型（如果有足够的 CSS 解析事件）
python train_css_optimizer.py \
    --data ../data/feedback_*.json \
    --epochs 100
```

**输出**: `training/models/*.onnx` 和 `*.pth`

### 步骤 4: 验证模型

```bash
# 验证 ONNX 格式
python validate_model.py ../models/html_complexity_v1.onnx

# 性能测试
python validate_model.py ../models/html_complexity_v1.onnx --benchmark
```

### 步骤 5: 部署模型

```bash
cd ../..

# 1. 复制模型到部署目录
cp training/models/html_complexity_v1.onnx models/local/

# 2. 更新模型配置
cat >> models/model_config.toml << EOF

[[models]]
name = "html_complexity_v1"
model_type = "HtmlParser"
path = "html_complexity_v1.onnx"
version = "1.0.0"
enabled = true
EOF

# 3. 重新编译启用 AI
cargo build --release --features ai

# 4. 测试效果
cargo run --release -- --ai-report
cargo run --release -- --learn https://example.com
```

## 🎯 完整示例

```bash
# 完整流程（复制粘贴运行）
cd /workspaces/BrowerAI

# 1. 安装依赖
cd training && ./setup_env.sh && cd ..

# 2. 收集数据（访问 10 个网站）
cargo run -- --learn \
    https://example.com \
    https://github.com \
    https://rust-lang.org \
    https://developer.mozilla.org \
    https://www.wikipedia.org \
    https://stackoverflow.com \
    https://news.ycombinator.com \
    https://reddit.com \
    https://www.python.org \
    https://nodejs.org

# 3. 检查数据量
cd training
python -c "
import json, glob
total = sum(len(json.load(open(f))) for f in glob.glob('data/feedback_*.json'))
print(f'总样本数: {total}')
print('HTML 解析样本:', sum(1 for f in glob.glob('data/feedback_*.json') for e in json.load(open(f)) if e.get('type')=='html_parsing'))
"

# 4. 训练模型
cd scripts
python train_html_complexity.py --epochs 100

# 5. 验证模型
python validate_model.py ../models/html_complexity_v1.onnx --benchmark

# 6. 部署
cd ../..
cp training/models/html_complexity_v1.onnx models/local/

# 7. 测试
cargo build --features ai && cargo run -- --ai-report
```

## 📊 推荐训练配置

### 数据量较少 (< 500 样本)

```bash
python train_html_complexity.py \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001
```

### 数据量中等 (500-5000 样本)

```bash
python train_html_complexity.py \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

### 数据量充足 (> 5000 样本)

```bash
python train_html_complexity.py \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.0005
```

## 🔍 检查训练效果

### 对比 AI 增强前后

```bash
# 没有 AI (stub mode)
cargo run -- --learn https://example.com
# 输出: complexity=0.500 (固定值)

# 启用真实模型
cargo build --features ai
cargo run -- --learn https://example.com
# 输出: complexity=0.732 (动态预测)
```

### 查看性能提升

```bash
# 运行 AI 报告
cargo run -- --ai-report

# 输出示例:
# 【性能监控】
# Model: html_complexity_v1
# Inference count: 50
# Average time: 0.234 ms  ⚡
```

## 🐛 常见问题

### Q1: ModuleNotFoundError: No module named 'torch'

**解决**: 安装依赖
```bash
cd training
pip install -r requirements.txt
```

### Q2: 训练数据不足（< 100 样本）

**解决**: 收集更多网站数据
```bash
# 创建 URL 列表
cat > websites.txt << EOF
https://example.com
https://github.com
https://rust-lang.org
... (更多网站)
EOF

# 批量访问
while read url; do
    cargo run -- --learn "$url"
    sleep 5
done < websites.txt
```

### Q3: CUDA out of memory

**解决**: 使用 CPU 或减小批次
```bash
# 强制使用 CPU
export CUDA_VISIBLE_DEVICES=""
python train_html_complexity.py --batch-size 16
```

### Q4: 训练损失不下降

**检查**:
1. 数据质量（是否有足够多样性）
2. 学习率（尝试 0.0001 或 0.01）
3. 模型容量（增加/减少层数）
4. 训练轮数（可能需要更多轮）

### Q5: Rust 端加载模型失败

**检查**:
1. 是否编译时启用了 `--features ai`
2. 模型路径是否正确（`models/local/*.onnx`）
3. `model_config.toml` 配置是否正确
4. ONNX 文件是否损坏（用 validate_model.py 验证）

## 📈 进阶技巧

### 自定义特征提取

修改 `train_html_complexity.py` 中的 `extract_html_features()`:

```python
def extract_html_features(event: dict) -> Tuple[List[float], float]:
    features = []
    
    # 添加自定义特征
    html_content = event.get('html_content', '')  # 需要在反馈中添加
    features.append(len(html_content) / 10000)  # 内容长度
    features.append(html_content.count('<table>'))  # 表格数量
    features.append(html_content.count('<form>'))  # 表单数量
    # ... 更多特征
    
    return features, label
```

### 超参数搜索

```bash
# 测试不同学习率
for lr in 0.0001 0.001 0.01; do
    python train_html_complexity.py \
        --lr $lr \
        --output ../models/html_lr_${lr}.onnx
done

# 对比效果
for model in ../models/html_lr_*.onnx; do
    echo "Testing $model"
    python validate_model.py $model --benchmark
done
```

### 模型融合

训练多个模型并集成：

```python
# 集成预测
predictions = []
for model_path in ['model_v1.onnx', 'model_v2.onnx', 'model_v3.onnx']:
    session = ort.InferenceSession(model_path)
    pred = session.run(None, {input_name: features})[0]
    predictions.append(pred)

# 平均融合
final_pred = np.mean(predictions, axis=0)
```

## 🎓 学习资源

- **PyTorch 教程**: https://pytorch.org/tutorials/
- **ONNX 文档**: https://onnx.ai/onnx/intro/
- **BrowerAI 文档**: 
  - [LEARNING_GUIDE.md](../../LEARNING_GUIDE.md) - 参数调优
  - [scripts/README.md](scripts/README.md) - 脚本详细文档
  - [QUICKSTART.md](QUICKSTART.md) - 原始快速开始

## ✅ 检查清单

训练前:
- [ ] Python 依赖已安装
- [ ] 收集了 100+ 反馈样本
- [ ] 数据文件存在于 `training/data/`

训练中:
- [ ] 训练损失逐渐下降
- [ ] 验证损失不上升（无过拟合）
- [ ] 没有错误或警告

训练后:
- [ ] ONNX 模型验证通过
- [ ] 推理速度 < 1ms
- [ ] 模型已复制到 `models/local/`
- [ ] 配置文件已更新
- [ ] Rust 编译启用了 `--features ai`

部署后:
- [ ] AI 报告显示模型已加载
- [ ] 真实网站测试显示动态复杂度
- [ ] 性能监控数据正常

---

🎉 祝训练顺利！有问题请查看 [scripts/README.md](scripts/README.md) 或提交 Issue。
