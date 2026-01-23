#!/bin/bash
# Flask API Server Systemd Service Setup
# 用于将Python API服务器部署为系统服务

set -e

echo "=== Flask API Server 部署脚本 ==="

# 1. 检查Python环境
if [ ! -d "/home/stone/BrowerAI/.venv" ]; then
    echo "❌ Python虚拟环境不存在"
    exit 1
fi

echo "✅ Python虚拟环境已找到"

# 2. 安装依赖
echo "📦 安装Flask依赖..."
source /home/stone/BrowerAI/.venv/bin/activate
pip install flask gunicorn -q
pip install torch -q  # 确保PyTorch已安装

# 3. 创建Systemd服务文件
UNIT_FILE="/tmp/framework-api.service"
cat > "$UNIT_FILE" << 'EOF'
[Unit]
Description=Framework Detection API Server
After=network.target

[Service]
Type=notify
User=stone
WorkingDirectory=/home/stone/BrowerAI
Environment="PATH=/home/stone/BrowerAI/.venv/bin"
ExecStart=/home/stone/BrowerAI/.venv/bin/gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 4 \
    --timeout 120 \
    --access-logfile /var/log/framework-api/access.log \
    --error-logfile /var/log/framework-api/error.log \
    training.api_server:app

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Systemd服务文件已创建: $UNIT_FILE"

# 4. 创建日志目录
echo "📁 创建日志目录..."
sudo mkdir -p /var/log/framework-api
sudo chown stone:stone /var/log/framework-api

# 5. 安装服务(需要sudo)
echo "📌 安装Systemd服务(需要sudo权限)..."
echo "请手动执行以下命令:"
echo ""
echo "  sudo cp $UNIT_FILE /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable framework-api.service"
echo "  sudo systemctl start framework-api.service"
echo "  sudo systemctl status framework-api.service"
echo ""
echo "或者手动启动(不使用systemd):"
echo ""
echo "  cd /home/stone/BrowerAI"
echo "  source .venv/bin/activate"
echo "  gunicorn --bind 0.0.0.0:5000 --workers 4 training.api_server:app"
echo ""

# 6. 测试API
echo "🧪 测试API健康检查..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ API服务已在运行"
else
    echo "⚠️  API服务未运行,请先启动"
fi

echo ""
echo "=== 部署脚本完成 ==="
