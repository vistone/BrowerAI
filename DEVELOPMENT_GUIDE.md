# 📖 BrowerAI 开发指南

**最后更新**: 2026-02-17  
**针对**: 贡献者和开发团队

---

## 📋 目录

1. [项目概览](#项目概览)
2. [环境设置](#环境设置)
3. [开发工作流](#开发工作流)
4. [代码组织](#代码组织)
5. [编码规范](#编码规范)
6. [测试](#测试)
7. [构建和部署](#构建和部署)

---

## 🏗️ 项目概览

### 架构

**BrowerAI** 是一个AI驱动的浏览器引擎，集成了：

- **Rust后端**: 高性能解析和渲染引擎
- **React前端**: 现代Web界面
- **AI集成**: ONNX模型推理
- **Container化**: Docker/K8s支持

### 关键特性

✅ HTML/CSS/JS智能解析  
✅ AI优化的渲染管道  
✅ 完整的测试套件  
✅ CI/CD自动化部署  
✅ 可扩展的插件系统  

---

## ⚙️ 环境设置

### 系统要求

```
操作系统: Linux 5.10+ / macOS 11+ / Windows 10+
CPU: x86_64 (2核)
内存: 4GB最小 (8GB推荐)
磁盘: 10GB可用空间
```

### Rust环境

```bash
# 安装Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 验证
rustc --version        # 1.70+
cargo --version        # 1.70+

# 更新
rustup update
```

### Node.js环境

```bash
# 安装Node.js 18+
nvm install 18
nvm use 18

# 验证
node --version         # v18.0+
npm --version          # 9.0+
```

### 依赖工具

```bash
# Docker (可选，用于容器化)
docker --version       # 24.0+

# Docker Compose
docker-compose --version  # 2.0+

# Git
git --version          # 2.30+
```

### 本地项目设置

```bash
# 1. 克隆项目
git clone https://github.com/vistone/BrowerAI.git
cd BrowerAI

# 2. 获取依赖
cargo fetch            # 预下载Rust依赖

# 3. 检查项目结构
ls -la crates/         # 查看所有crates
```

---

## 🔄 开发工作流

### 分支策略

```
main                   # 生产分支，每个提交都是稳定版本
  ↑
  ├── week5-postgresql-persistence  # 特性分支
  ├── week6-ai-integration
  └── feature/your-feature           # 个人特性分支
```

### 常见工作流

#### 1. 开始新特性

```bash
# 1. 从main分支创建特性分支
git checkout main
git pull origin main
git checkout -b feature/your-feature

# 2. 开发...
# 修改代码
cargo test             # 本地测试
git add .
git commit -m "feat: add new feature"

# 3. 推送并创建PR
git push origin feature/your-feature
```

#### 2. 代码审查流程

```bash
# 在GitHub创建PR并等待：
1. ✅ 自动化测试通过 (CI/CD)
2. ✅ 代码审查通过
3. ✅ 检查列表完成
4. ✅ 合并到main
```

#### 3. 本地调试

```bash
# 启用调试输出
RUST_LOG=debug cargo run

# 运行单个测试
cargo test --test phase4_e2e_tests

# 查看完整输出
cargo test -- --nocapture
```

---

## 📁 代码组织

### 项目结构

```
crates/
├── browerai/                       # 主库 (大部分逻辑)
├── browerai-api-server/            # API服务器
├── browerai-html-parser/           # HTML解析
├── browerai-css-parser/            # CSS解析
├── browerai-js-parser/             # JavaScript解析
├── browerai-js-analyzer/           # JS分析工具
├── browerai-renderer/              # 渲染引擎
├── browerai-ai-core/               # AI推理核心
├── browerai-learning/              # 学习模块
├── browerai-webclient/             # React前端
└── ... (共27个crates)

docs/
├── guides/                         # 技术指南
├── api/                            # API文档
├── architecture/                   # 架构设计
├── development/                    # 开发规范
├── learning/                       # AI学习
├── phases/                         # 项目历程
└── archived/                       # 历史存档

tests/                              # 集成测试
models/                             # AI模型
data/                               # 训练数据
```

### 添加新模块

```bash
# 1. 创建新crate
cargo new crates/browerai-{module}

# 2. 在workspace Cargo.toml中添加
[workspace]
members = [
    "crates/browerai-{module}",
    # ...
]

# 3. 添加依赖到主库
[dependencies]
browerai-{module} = { path = "crates/browerai-{module}" }

# 4. 开发并测试
cargo test
```

---

## 📝 编码规范

### Rust风格指南

```rust
// ✅ 好的做法

// 1. 使用anyhow处理错误
use anyhow::{Context, Result};

fn read_file(path: &str) -> Result<String> {
    std::fs::read_to_string(path)
        .context("Failed to read file")?
}

// 2. 使用log macro
log::info!("Processing started");
log::debug!("Debug info: {:?}", value);
log::warn!("Warning: something unusual");
log::error!("Error occurred: {}", err);

// 3. 公共API需要文档
/// 解析HTML文档
///
/// # Arguments
/// * `html` - HTML源代码
///
/// # Returns
/// 解析后的DOM树
///
/// # Examples
/// ```
/// let dom = parse_html("<p>Hello</p>")?;
/// ```
pub fn parse_html(html: &str) -> Result<Dom> {
    // ...
}

// 4. 使用match处理枚举
match result {
    Ok(value) => { /* handle success */ },
    Err(e) => { /* handle error */ },
}

// ❌ 避免

// unwrap() - 会panic
let value = result.unwrap();  // 不好！

// 显式panic
panic!("Something went wrong");  // 避免

// 无意义的克隆
let copy = expensive_value.clone();  // 检查是否必需
```

### TypeScript风格指南

```typescript
// ✅ 好的做法

// 1. 完整的类型注解
interface ApiResponse<T> {
  data: T;
  status: "success" | "error";
  timestamp: Date;
}

// 2. 错误处理
try {
  const response = await fetchApi("/endpoint");
} catch (error) {
  if (error instanceof NetworkError) {
    // 处理网络错误
  } else {
    // 处理其他错误
  }
}

// 3. 使用const而不是let
const value = 42;  // 优先使用

// 4. 箭头函数和类型
const processData = (items: string[]): Promise<Result[]> => {
  // ...
};

// ❌ 避免

// any类型
const value: any = response.data;  // 避免

// 忽略类型错误
// @ts-ignore
const x = wrongType;

// 不必要的类型转换
const num = response.data as number;
```

### 文档注释

所有公共API必须有文档：

```rust
/// 简要说明（一行）
///
/// 详细说明（如需要）
///
/// # Arguments
/// * `param1` - 说明
///
/// # Returns
/// 返回值说明
///
/// # Errors
/// 可能的错误情况
///
/// # Examples
/// ```
/// let result = function(arg)?;
/// assert_eq!(result, expected);
/// ```
pub fn function(param1: &str) -> Result<Output> {
    // ...
}
```

---

## 🧪 测试

### 测试结构

```
tests/                              # 集成测试
├── api_integration_tests.rs
├── parser_tests.rs
├── renderer_tests.rs
└── end_to_end_tests.rs

src/
└── lib.rs                          # 单元测试 (#[cfg(test)])
```

### 编写测试

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsing() {
        let html = "<div>Hello</div>";
        let dom = parse_html(html).unwrap();
        assert_eq!(dom.children.len(), 1);
    }

    #[test]
    #[should_panic(expected = "invalid")]
    fn test_invalid_input() {
        parse_html("").unwrap();
    }
}
```

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行核心模块测试（推荐）
cargo test -p browerai-core -p browerai-html-parser -p browerai-css-parser \
           -p browerai-js-parser -p browerai-js-analyzer -p browerai-ai-core \
           -p browerai-renderer-core -p browerai-devtools

# 运行特定crate测试
cargo test -p browerai-js-analyzer

# 运行特定测试
cargo test parser::html

# 单线程运行（避免竞态）
cargo test -- --test-threads=1

# 显示println!输出
cargo test -- --nocapture

# 生成覆盖率 (需要tarpaulin)
cargo tarpaulin --out Html
```

### 测试状态

| 模块 | 测试数 | 状态 |
|------|--------|------|
| browerai-core | 25 | ✅ 通过 |
| browerai-html-parser | 24 | ✅ 通过 |
| browerai-css-parser | 8 | ✅ 通过 |
| browerai-js-parser | 9 | ✅ 通过 |
| browerai-js-analyzer | 33 | ✅ 通过 |
| browerai-ai-core | 27 | ✅ 通过 |
| browerai-renderer-core | 22 | ✅ 通过 |
| browerai-devtools | 10 | ✅ 通过 |
| **总计** | **168** | **✅ 100%通过** |

### 测试最佳实践

✅ 测试一个概念  
✅ 使用清晰的名称  
✅ 包含肯定和否定测试用例  
✅ 测试边界条件  
✅ 使用fixtures 反复设置  

---

## 🏗️ 构建和部署

### 本地构建

#### 构建全部模块

```bash
# Debug构建（快速，但无优化）
cargo build

# Release构建（慢，但优化）
cargo build --release

# 仅构建核心模块（推荐用于快速验证）
cargo build -p browerai-core -p browerai-html-parser -p browerai-css-parser \
            -p browerai-js-parser -p browerai-js-analyzer -p browerai-ai-core \
            -p browerai-renderer-core -p browerai-devtools
```

#### 构建特定模块

```bash
# 单个crate
cargo build -p browerai-html-parser

# 带特性标志
cargo build -p browerai --features "ai,v8,db"

# 检查代码（无生成二进制）
cargo check

# 格式化代码
cargo fmt

# 静态检查
cargo clippy
```

#### 构建状态

| 模块组 | 命令 | 时间 |
|--------|------|------|
| 核心8模块 | `cargo build -p browerai-core ...` | ~35s |
| 全部模块 | `cargo build --workspace` | ~2-3min |
| Release | `cargo build --release` | ~5-10min |

### Docker构建

```bash
# 构建镜像
docker build -t browerai:latest .
docker build -t browerai:v1.0.0 .

# 运行容器
docker run -p 3000:3000 browerai:latest

# 查看日志
docker logs -f <container-id>
```

### CI/CD流程

推送代码后自动执行：

```
git push
  ↓
GitHub Actions triggered
  ├─ Compile Rust
  ├─ Run Tests
  ├─ Build Docker
  ├─ Security Scan
  ├─ Push To Registry
  └─ Deploy
```

查看状态: https://github.com/vistone/BrowerAI/actions

---

## 📞 常见问题

**Q: 如何处理Cargo.lock?**
```
A: 提交Cargo.lock以确保可重复构建
```

**Q: 如何使用ONNX模型?**
```
A: 参考 docs/learning/MODEL_TRAINING.md
```

**Q: 前端如何与后端通信?**
```
A: 查看 docs/api/EXAMPLES.md
```

**Q: 如何调试JavaScript?**
```
A: 使用 RUST_LOG=debug 并查询 src/parser/js_analyzer/
```

---

## 🔗 更多资源

- [项目规范](docs/PROJECT_STANDARDS.md)
- [API文档](docs/api/)
- [架构设计](docs/architecture/)
- [测试指南](docs/guides/TESTING.md)
- [CI/CD指南](docs/guides/CI_CD.md)

---

**祝开发愉快!** 🚀

