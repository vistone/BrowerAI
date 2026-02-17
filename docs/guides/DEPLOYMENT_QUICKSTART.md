# 🔥 部署快速参考 - 5分钟速查

## 现在就做 (Right Now!)

### 1️⃣ 配置Secrets (GitHub网页 - 5分钟)

访问: https://github.com/vistone/BrowerAI/settings/secrets/actions

```
1. "New repository secret"
   Name: DOCKER_USERNAME
   Value: (你的Docker Hub用户名)
   
2. "New repository secret"
   Name: DOCKER_PASSWORD
   Value: (Docker Hub密码/PAT)
```

### 2️⃣ 创建Pull Request (GitHub网页 - 5分钟)

访问: https://github.com/vistone/BrowerAI/compare/main...week5-postgresql-persistence

```
创建PR，等待自动检查通过，然后合并
```

### 3️⃣ 推送版本标签 (CLI - 1分钟)

```bash
# 使用脚本 (最简单)
bash scripts/github_deploy_prepare.sh v1.0.0

# 或手动
git tag v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

---

## 监控部署 (自动执行)

访问: https://github.com/vistone/BrowerAI/actions

等待所有jobs完成 (约12分钟)

```
✓ Build-Backend
✓ Build-Frontend  
✓ Test-Backend
✓ Test-Frontend
✓ Build-Docker
✓ Scan-Docker
✓ Push-Docker
✓ Deploy-Kubernetes (可选)
✓ Verify-Deployment
✓ Create-Release
✓ Notify
```

---

## 验证部署 (完成后)

```bash
# 检查Docker镜像
docker pull your-username/browerai-api:v1.0.0

# 检查GitHub Release
open https://github.com/vistone/BrowerAI/releases
```

---

## 总耗时: ~28分钟

| 手动 | 自动 | 总计 |
|------|------|------|
| 11分钟 | 12分钟 | 28分钟 |

---

## 遇到问题?

📖 **详细指南**: [GITHUB_DEPLOYMENT_GUIDE.md](GITHUB_DEPLOYMENT_GUIDE.md)

📋 **完整清单**: [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)

💬 **故障排查**: 见清单中的"故障排查"章节

---

## 关键命令速查

```bash
# 查看标签
git tag -l

# 创建标签
git tag v1.0.0

# 推送标签
git push origin v1.0.0

# 查看标签详情
git show v1.0.0

# 学习更多
git help tag
```

---

**状态**: 🟢 准备就绪  
**版本**: v1.0.0  
**日期**: 2026-02-17  
**完成度**: 95% (等待Secrets配置)
