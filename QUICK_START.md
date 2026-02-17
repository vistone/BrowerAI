# 🚀 BrowerAI 快速开始指南

**版本**: 2026.02.17  
**语言**: 简体中文

---

## ⚡ 5分钟快速开始

### 前置要求

```bash
# 检查环境
rustc --version      # Rust 1.70+
node --version       # Node.js 18+
docker --version     # Docker 24+
git --version        # Git 2.30+
```

### 第1步: 克隆项目

```bash
git clone https://github.com/vistone/BrowerAI.git
cd BrowerAI
```

### 第2步: 本地开发环境

#### 后端 (Rust API)
```bash
# 构建后端
cargo build --release

# 运行API服务器 (http://localhost:3000)
cargo run --release -p browerai-api-server

# 测试API
curl http://localhost:3000/api/health
```

#### 前端 (React)
```bash
# 安装依赖
cd crates/browerai-webclient
npm install

# 开发模式 (http://localhost:5173)
npm run dev
```

### 第3步: 测试代码

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test parser::html

# 查看输出
cargo test -- --nocapture
```

### 第4步: Docker部署

```bash
# 构建镜像
docker build -t browerai:latest .

# 运行容器
docker run -p 3000:3000 browerai:latest

# 使用docker-compose
docker-compose up
```

---

## 📚 更多文档

| 文档 | 用途 |
|------|------|
| [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) | 完整的开发指南 |
| [docs/guides/DEPLOYMENT.md](docs/guides/DEPLOYMENT.md) | 生产部署说明 |
| [docs/api/](docs/api/) | API参考文档 |
| [docs/architecture/](docs/architecture/) | 架构设计文档 |
| [CHANGELOG.md](CHANGELOG.md) | 版本变更记录 |

---

## 🔗 常用链接

- 🏠 [主README](README.md)
- 📖 [完整文档](docs/README.md)
- 🤝 [贡献指南](CONTRIBUTING.md)
- 🐛 [问题追踪](https://github.com/vistone/BrowerAI/issues)
- 🌟 [GitHub](https://github.com/vistone/BrowerAI)

---

## 💡 遇到问题?

```
故障排查指南: docs/guides/TROUBLESHOOTING.md
常见问题: docs/references/FAQ.md
环境设置: docs/guides/SETUP.md
```

**准备好了? 开始[开发](DEVELOPMENT_GUIDE.md)吧!** 🎉

