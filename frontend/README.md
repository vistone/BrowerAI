# BrowerAI Web 客户端

真实的React + TypeScript Web应用，与Rust后端API集成。

## 功能

- ✅ HTML和CSS代码编辑
- ✅ 与BrowerAI后端API实时通信
- ✅ 实时性能指标展示
- ✅ 错误处理和重试逻辑
- ✅ 响应式设计

## 快速开始

### 安装依赖

```bash
npm install
```

### 开发模式

```bash
npm run dev
```

访问 http://localhost:5173

### 生产构建

```bash
npm run build
npm run preview
```

## API集成

通过 `src/api/client.ts` 与后端通信：

```typescript
import { apiClient } from './api/client';

// 解析HTML
const html = await apiClient.parseHtml({ html: '<div>test</div>' });

// 解析CSS
const css = await apiClient.parseCss({ css: 'body { color: red; }' });

// 完整渲染
const render = await apiClient.render({
  html: '<div>test</div>',
  css: 'body { color: red; }',
  use_ai: false
});
```

## 项目结构

```
src/
├── api/              # API客户端
├── components/       # React组件
├── styles/          # CSS样式
├── App.tsx          # 主应用
├── main.tsx         # 入口
└── index.css        # 全局样式
```

## 与后端集成

该应用需要BrowerAI后端运行在 `http://localhost:3000`

启动后端：

```bash
cd /home/stone/BrowerAI
cargo run --release -p browerai-api-server
```

然后启动前端：

```bash
npm run dev
```

## 生产部署

通过Docker部署：

```bash
docker build -t browerai-webclient .
docker run -p 80:80 browerai-webclient
```
