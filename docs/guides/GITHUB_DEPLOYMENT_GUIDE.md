# GitHub部署配置和发布指南

## 📋 第1步：配置GitHub Secrets

### 在GitHub网页界面配置Secrets

1. **打开GitHub仓库设置**
   - 访问: https://github.com/vistone/BrowerAI/settings/secrets/actions
   - 或在仓库页面: Settings → Secrets and variables → Actions

2. **添加DOCKER_USERNAME**
   - 点击 "New repository secret"
   - Name: `DOCKER_USERNAME`
   - Value: 你的Docker Hub用户名
   - 点击 "Add secret"

3. **添加DOCKER_PASSWORD**
   - 点击 "New repository secret"
   - Name: `DOCKER_PASSWORD`
   - Value: 你的Docker Hub密码或Access Token
   - 点击 "Add secret"

4. **可选：添加KUBE_CONFIG**
   - Name: `KUBE_CONFIG`
   - Value: base64编码的kubeconfig文件内容
   - 获取命令: `cat ~/.kube/config | base64 | tr -d '\n'`

### ✅ 验证Secrets已添加
- 访问上述链接后，应该能看到3个Secrets列出

---

## 📋 第2步：创建Pull Request到main分支

这一步会在GitHub网页界面进行：

1. **访问GitHub仓库**
   - 地址: https://github.com/vistone/BrowerAI

2. **创建Pull Request**
   - 点击 "Pull requests" 标签
   - 点击 "New pull request"
   - Base branch: `main`
   - Compare branch: `week5-postgresql-persistence`
   - 点击 "Create pull request"

3. **填写PR信息**
   - Title: "feat: Week 8 Phase E Complete - CI/CD Integration and Deployment Ready"
   - Description: 复制下面的内容

```markdown
## 🎉 Week 8 Phase E - 完整CI/CD集成

### ✨ 主要更新
- ✅ 完整的GitHub Actions CI/CD流程 (9阶段)
- ✅ React+TypeScript前端应用集成
- ✅ Docker容器化配置
- ✅ Kubernetes部署清单
- ✅ 自动化测试套件
- ✅ 回滚机制

### 📊 测试状态
- ✅ 28/28 测试通过 (100%)
- ✅ Rust编译成功
- ✅ 前端TypeScript编译成功
- ✅ Docker配置有效
- ✅ K8s清单验证通过

### 🚀 部署就绪
- API服务器: ✅ 运行正常 (平均延迟: 7ms)
- 前端应用: ✅ 编译成功
- 文档: ✅ 50+页完整

### 📝 关键文档
- [CI/CD使用指南](docs/CICD_USAGE_GUIDE.md)
- [项目最终状态](PROJECT_FINAL_STATUS.md)
- [测试报告](COMPREHENSIVE_TEST_AND_SUBMISSION_REPORT.md)

### ✅ 检查清单
- [x] 所有代码编译成功
- [x] 所有API端点测试通过
- [x] CI/CD工作流配置完整
- [x] Docker和K8s配置就绪
- [x] 文档完整
```

4. **审查和合并**
   - 等待CI/CD流程运行
   - 审查后合并到main分支

---

## 📋 第3步：推送版本标签触发自动部署

### 创建和推送版本标签

**命令行操作：**

```bash
# 1. 确保在main分支上 (合并PR后)
git checkout main
git pull origin main

# 2. 创建版本标签
git tag v1.0.0 -m "Release version 1.0.0 - Week 8 Phase E Complete"

# 3. 推送标签到GitHub
git push origin v1.0.0

# 4. 推送所有标签（可选）
git push origin --tags
```

### 标签命名规范

推荐使用语义化版本：
- `v1.0.0` - 主版本发布
- `v1.0.1` - 补丁版本
- `v1.1.0` - 次版本发布

### 触发CI/CD流程

推送标签后，GitHub Actions会自动触发：
1. ✅ 构建和测试
2. ✅ Docker镜像构建
3. ✅ 安全扫描
4. ✅ 镜像推送到Docker Hub
5. ✅ Kubernetes部署 (如果配置)
6. ✅ GitHub Release创建

---

## 🔍 验证部署状态

### 1. 检查GitHub Actions
- 访问: https://github.com/vistone/BrowerAI/actions
- 查看最新的workflow运行
- 验证所有jobs通过

### 2. 检查Docker Hub
- 访问: https://hub.docker.com
- 查看镜像: `your-username/browerai-api:latest` 和 `v1.0.0`

### 3. 检查GitHub Release
- 访问: https://github.com/vistone/BrowerAI/releases
- 应该能看到v1.0.0发布

---

## 🚀 完整部署流程总结

```
┌─────────────────────────────────────────────┐
│ 1. 配置GitHub Secrets                       │
│    - DOCKER_USERNAME                        │
│    - DOCKER_PASSWORD                        │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ 2. 创建Pull Request到main                   │
│    week5-postgresql-persistence → main      │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ 3. 合并PR (需要审查)                        │
│    代码进入main分支                         │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ 4. 推送版本标签                             │
│    git push origin v1.0.0                   │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ 5. 自动触发CI/CD流程                        │
│    - 构建 ✓ 测试 ✓ Docker ✓                │
│    - 扫描 ✓ 推送 ✓ 部署 ✓                  │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ 6. 创建GitHub Release                       │
│    v1.0.0 - 生产就绪                        │
└─────────────────────────────────────────────┘
```

---

## ⚙️ 配置示例

### 如果没有Docker Hub账户

1. 注册Docker Hub: https://hub.docker.com/signup
2. 创建Personal Access Token:
   - 访问: https://hub.docker.com/settings/security
   - 点击 "New Access Token"
   - 使用Token作为DOCKER_PASSWORD

### Kubernetes配置 (可选)

如果要启用K8s部署，需要额外配置：

1. 获取kubeconfig:
   ```bash
   cat ~/.kube/config | base64 | tr -d '\n'
   ```

2. 在GitHub Secrets中添加:
   - Name: `KUBE_CONFIG`
   - Value: 上面的base64编码内容

3. 添加KUBE_CONTEXT (可选):
   - Name: `KUBE_CONTEXT`
   - Value: kubernetes context名称

---

## 📞 故障排查

### 如果CI/CD失败

1. **检查Secrets**
   - 确保DOCKER_USERNAME和DOCKER_PASSWORD正确
   - 验证账户有权限推送镜像

2. **检查日志**
   - 访问Actions页面查看失败原因
   - 查看具体job的错误信息

3. **重新运行**
   - 在Actions页面点击"Re-run jobs"
   - 修复问题后重新推送

### 如果PR无法合并

- 检查代码冲突
- 解决冲突后重新推送
- 等待所有检查通过

---

## ✅ 检查清单

部署前验证：

- [ ] GitHub Secrets已配置 (DOCKER_USERNAME, DOCKER_PASSWORD)
- [ ] Pull Request已创建到main分支
- [ ] PR通过所有检查
- [ ] PR已合并到main分支
- [ ] 版本标签已创建和推送 (v1.0.0)
- [ ] GitHub Actions workflow已触发
- [ ] 所有jobs通过
- [ ] Docker镜像已推送
- [ ] GitHub Release已创建

---

## 📚 相关资源

- GitHub Actions文档: https://docs.github.com/actions
- Docker Hub: https://hub.docker.com
- Kubernetes文档: https://kubernetes.io/docs
- BrowerAI仓库: https://github.com/vistone/BrowerAI

---

**预计完成时间:** 10-15分钟  
**部署后验证时间:** 5-10分钟  
**总计:** ~20-25分钟

