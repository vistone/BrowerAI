# 故障排除指南

本指南提供常见问题的诊断和解决方案。

## 编译和构建问题

### ❌ Rust编译错误: "cannot find crate"

**症状**: 
```
error[E0463]: can't find crate for `browerai_xxx`
```

**原因**: 缺少crate依赖或工作区配置问题

**解决方案**:
```bash
# 1. 清理构建缓存
cargo clean

# 2. 更新依赖
cargo update

# 3. 从头编译
cargo build --release

# 4. 如果问题持续,检查Cargo.toml
cat Cargo.toml | grep -A 5 "\[workspace\]"
```

---

### ❌ Rust编译错误: "mismatched types"

**症状**:
```
error[E0308]: mismatched types
  expected `String`, found `&str`
```

**原因**: 类型不匹配,通常是String vs &str

**解决方案**:
```bash
# 运行类型检查
cargo check

# 修复提示的类型转换
cargo fix --allow-dirty

# 重新编译
cargo build
```

---

### ❌ TypeScript编译错误: "Cannot find module"

**症状**:
```
error TS2307: Cannot find module 'xxx'
```

**原因**: npm包未安装或导入路径错误

**解决方案**:
```bash
# 1. 清理node_modules
rm -rf node_modules package-lock.json

# 2. 重新安装依赖
npm install

# 3. 检查tsconfig.json路径设置
cat tsconfig.json | grep -A 5 "baseUrl\|paths"

# 4. 验证导入路径
grep -r "import.*from" src/ | head -5
```

---

### ❌ 编译超时错误

**症状**:
```
error: build script failed with: signal: killed
timeout after 600s
```

**原因**: 构建时间过长,通常由于大型Rust项目

**解决方案**:
```bash
# 1. 增加超时时间
export CARGO_BUILD_JOBS=4

# 2. 使用增量编译
cargo build -j 4

# 3. 使用release优化
cargo build --release -j 4

# 4. 清理并使用sccache (如果可用)
cargo install sccache
export RUSTC_WRAPPER=sccache
cargo build
```

---

## 运行时错误

### ❌ "failed to load model" 错误

**症状**:
```
Error: Failed to load ONNX model from '/path/to/model.onnx'
```

**原因**: 模型文件不存在或格式不正确

**解决方案**:
```bash
# 1. 检查模型文件是否存在
ls -lh models/local/

# 2. 验证模型格式
file models/local/*.onnx

# 3. 从model_config.toml验证配置
cat models/model_config.toml | grep -A 3 "name = \"html_parser"

# 4. 重新生成模型
cd training
python scripts/train_html_parser.py
cp models/*.onnx ../models/local/
```

---

### ❌ PostgreSQL连接错误

**症状**:
```
Error: could not connect to server: Connection refused
```

**原因**: PostgreSQL服务未运行或连接参数错误

**解决方案**:
```bash
# 1. 检查PostgreSQL状态
docker ps | grep postgres

# 2. 启动PostgreSQL (Docker Compose方式)
docker-compose up -d postgres

# 3. 检查连接参数
echo $DATABASE_URL
# 应该类似: postgresql://user:password@localhost:5432/browerai

# 4. 手动测试连接
psql "postgresql://user:password@localhost:5432/browerai" -c "SELECT 1;"

# 5. 检查迁移状态
sqlx database run

# 6. 如果数据损坏,重置数据库
docker-compose down -v
docker-compose up -d postgres
sqlx migrate run
```

---

### ❌ Redis连接错误

**症状**:
```
Error: connection refused, address: 127.0.0.1:6379
```

**原因**: Redis服务未运行

**解决方案**:
```bash
# 1. 检查Redis状态
docker ps | grep redis

# 2. 启动Redis
docker-compose up -d redis

# 3. 测试连接
redis-cli ping
# 应该返回: PONG

# 4. 检查Redis配置
docker exec $(docker ps -q -f "label=com.docker.compose.service=redis") \
  redis-cli CONFIG GET "maxmemory"
```

---

### ❌ API端点返回500错误

**症状**:
```json
{
  "error": "Internal Server Error",
  "status": 500
}
```

**原因**: 服务器内部错误

**解决方案**:
```bash
# 1. 检查API服务日志
docker logs $(docker ps -q -f "ancestor=browerai-api")

# 2. 启用debug日志
RUST_LOG=debug cargo run

# 3. 测试单个端点
curl -v http://localhost:8080/api/v1/health

# 4. 检查数据库连接
grep DATABASE_URL .env

# 5. 验证所有依赖服务正在运行
docker-compose ps
```

---

## 部署问题

### ❌ GitHub Secrets未配置

**症状**:
```
Error: Secrets not found: DOCKER_USERNAME, DOCKER_PASSWORD
```

**原因**: GitHub Actions需要Docker凭证

**解决方案**:
```bash
# 1. 访问仓库设置
# https://github.com/vistone/BrowerAI/settings/secrets/actions

# 2. 添加新secret:
Name: DOCKER_USERNAME
Value: [your-docker-username]

Name: DOCKER_PASSWORD
Value: [your-docker-password-or-pat]

# 3. 验证secrets已保存 (不能查看值,仅显示存在)

# 4. 重新推送标签触发工作流
git tag v1.0.0
git push origin v1.0.0
```

---

### ❌ Docker镜像构建失败

**症状**:
```
Error: failed to build image: dockerfile parse error
```

**原因**: Dockerfile语法错误或缺少依赖

**解决方案**:
```bash
# 1. 验证Dockerfile
docker build -f Dockerfile.prod --no-cache -t browerai:test .

# 2. 检查错误的具体行
grep -n "FROM\|RUN\|COPY" Dockerfile.prod | head -10

# 3. 手动测试构建步骤
docker build --target=base -t browerai-base .
docker build --target=builder -t browerai-builder .

# 4. 查看详细输出
docker build -f Dockerfile.prod --progress=plain .

# 5. 检查磁盘空间
docker system df
docker builder du
```

---

### ❌ Kubernetes部署失败

**症状**:
```
Error: CrashLoopBackOff or ImagePullBackOff
```

**原因**: Pod无法启动或无法拉取镜像

**解决方案**:
```bash
# 1. 检查Pod状态
kubectl get pods -l app=browerai

# 2. 查看Pod日志
kubectl logs -l app=browerai --tail=100

# 3. 描述Pod获取详细信息
kubectl describe pod [pod-name]

# 4. 检查镜像可用性
kubectl get events

# 5. 重启部署
kubectl rollout restart deployment browerai

# 6. 检查镜像仓库凭证
kubectl get secrets
kubectl describe secret docker-registry-secret
```

---

## 数据库问题

### ❌ 迁移失败错误

**症状**:
```
Error: migration failed: relation "xxx" already exists
```

**原因**: 数据库中已存在该表

**解决方案**:
```bash
# 1. 检查已运行的迁移
sqlx migrate info

# 2. 查视实际数据库状态
psql $DATABASE_URL -c "\dt"

# 3. 手动回滚迁移
sqlx migrate revert

# 4. 重新运行迁移
sqlx migrate run

# 5. 如果问题持续,完全重置
docker-compose down -v
docker-compose up -d postgres
sqlx database create
sqlx migrate run
```

---

### ❌ 数据量过大导致查询缓慢

**症状**:
```
Query execution timeout after 30s
```

**原因**: 缺少索引或数据量太大

**解决方案**:
```bash
# 1. 检查表大小
psql $DATABASE_URL -c "SELECT schemaname, tablename, 
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) 
  FROM pg_base tables ORDER BY 1 DESC;"

# 2. 分析查询计划
psql $DATABASE_URL -c "EXPLAIN ANALYZE SELECT * FROM slow_table WHERE ..."

# 3. 检查和创建索引
psql $DATABASE_URL -c "\d slow_table"  # 查看现有索引
psql $DATABASE_URL -c "CREATE INDEX idx_column ON table(column);"

# 4. 清理过期数据
psql $DATABASE_URL -c "DELETE FROM logs WHERE created_at < NOW() - INTERVAL '30 days';"
psql $DATABASE_URL -c "VACUUM ANALYZE;"
```

---

## 开发环境问题

### ❌ 环境变量未加载

**症状**:
```
Error: DATABASE_URL environment variable not set
```

**原因**: .env文件未被加载

**解决方案**:
```bash
# 1. 创建.env文件 (如果不存在)
cp .env.example .env

# 2. 验证.env内容
cat .env | grep DATABASE_URL

# 3. 源文件加载环境变量
source .env
echo $DATABASE_URL

# 4. 对于cargo run,使用dotenv crate (已包含)
cargo run

# 5. 验证所有必需的变量
required_vars=("DATABASE_URL" "REDIS_URL" "API_KEY")
for var in "${required_vars[@]}"; do
  [ -z "${!var}" ] && echo "Missing: $var" || echo "✓ $var"
done
```

---

### ❌ Git历史冲突

**症状**:
```
error: Your local changes to 'src/main.rs' would be overwritten
```

**原因**: 本地修改与远程分支冲突

**解决方案**:
```bash
# 1. 检查修改状态
git status

# 2. 暂存本地修改
git stash

# 3. 拉取最新代码
git pull origin main

# 4. 恢复本地修改
git stash pop

# 5. 解决冲突 (如果存在)
git add .
git commit -m "Resolve merge conflicts"

# 6. 推送修改
git push origin branch-name
```

---

### ❌ Node.js版本不兼容

**症状**:
```
error: requires node v18.0.0+, but found v16.x.x
```

**原因**: Node.js版本过旧

**解决方案**:
```bash
# 1. 检查当前版本
node --version
npm --version

# 2. 使用nvm安装新版本
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# 3. 或使用brew升级
brew upgrade node

# 4. 验证版本
node --version  # 应该 >= v18.0.0

# 5. 重新安装npm依赖
rm -rf node_modules package-lock.json
npm install
```

---

## 监控和性能问题

### ❌ Prometheus无法查询指标

**症状**:
```
Error: no data points found for metric 'xxx'
```

**原因**: 应用未暴露指标或Prometheus未正确配置

**解决方案**:
```bash
# 1. 检查Prometheus配置
cat config/prometheus.yml | grep -A 5 "scrape_configs"

# 2. 验证目标健康状态
# 访问 http://localhost:9090/targets

# 3. 检查应用是否暴露指标
curl http://localhost:8080/metrics

# 4. 验证指标格式
curl http://localhost:8080/metrics | head -20

# 5. 重启Prometheus
docker-compose restart prometheus

# 6. 测试查询
curl 'http://localhost:9090/api/v1/query?query=up'
```

---

### ❌ Grafana面板显示无数据

**症状**:
```
No data in selected time range
```

**原因**: 数据尚未收集或查询错误

**解决方案**:
```bash
# 1. 检查时间范围
# Grafana界面: 右上角调整时间范围到过去几小时

# 2. 验证数据源连接
# Grafana > Settings > Data Sources > Prometheus
# 点击"Test" 确保连接正常

# 3. 检查指标名称
# 在data sources页面运行test query

# 4. 确认应用在运行
docker ps | grep browerai

# 5. 手动查询Prometheus
curl 'http://localhost:9090/api/v1/query?query=http_requests_total'
```

---

## 获取帮助

### 收集诊断信息

遇到问题时,请收集以下信息:

```bash
# 系统信息
echo "=== 系统信息 ==="
uname -a
docker --version
rustc --version
node --version

# 应用日志
echo "=== 应用日志 ==="
docker logs -f $(docker ps -q -f "ancestor=browerai-api")

# Docker状态
echo "=== Docker状态 ==="
docker-compose ps
docker system df

# 网络检查
echo "=== 网络检查 ==="
curl -v http://localhost:8080/api/v1/health
curl -v http://localhost:5432  # PostgreSQL (应该失败)

# 依赖检查
echo "=== 依赖检查 ==="
cargo --version
npm --version
sqlx --version
```

### 提交问题

在GitHub上提交issue时:

1. **标题**: 简洁描述问题
2. **描述**: 复现步骤和预期行为
3. **日志**: 完整的错误日志 (用代码块)
4. **环境**: 上述诊断信息
5. **已尝试**: 列出尝试的解决方案

---

## 相关文档

- [快速开始](QUICK_START_CARD.md)
- [部署指南](DEPLOYMENT_QUICKSTART.md)
- [开发指南](../DEVELOPMENT_GUIDE.md)
- [API文档](../api/ENDPOINTS.md)
