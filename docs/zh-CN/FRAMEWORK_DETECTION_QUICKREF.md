# 框架检测快速参考 (Framework Detection Quick Reference)

## 🚀 快速开始

```rust
use browerai::learning::advanced_deobfuscation::AdvancedDeobfuscator;

let deobfuscator = AdvancedDeobfuscator::new();
let analysis = deobfuscator.analyze(js_code)?;

// 检测置信度
println!("置信度: {:.1}%", analysis.confidence * 100.0);

// 检测到的框架
for framework in &analysis.framework_patterns {
    let info = deobfuscator.get_framework_info(framework);
    println!("  {} ({})", info.name, info.origin);
}

// 生成报告
let report = deobfuscator.generate_report(&analysis);
```

---

## 📊 框架分类速查

| 类别 | 数量 | 代表框架 |
|------|------|----------|
| 打包器 | 9 | Webpack, Rollup, Vite, esbuild |
| 前端框架 | 19 | React, Vue, Angular, Svelte |
| 元框架 | 9 | Next.js, Nuxt, Gatsby, Remix |
| 移动开发 | 7 | React Native, Ionic, Capacitor |
| 🇨🇳 中国框架 | 11 | Taro, Uni-app, Rax, Qiankun |
| 状态管理 | 9 | Redux, MobX, Vuex, Pinia |
| UI 库 | 9 | Ant Design, Element UI, MUI |
| SSR | 4 | Express, Koa, Fastify, Hono |
| 混淆工具 | 10 | Terser, UglifyJS, Obfuscator |
| 模块系统 | 4 | ESM, CJS, AMD, UMD |
| 微前端 | 4 | single-spa, Module Federation |
| 测试 | 2 | Jest, Vitest |
| **总计** | **100+** | |

---

## 🇨🇳 中国框架完整列表

### 多端框架
1. **Taro** (京东) - `@tarojs`, `Taro.Component`
2. **Uni-app** (DCloud) - `uni-app`, `uni.request`, `@dcloudio`
3. **Remax** (阿里) - `remax`, `@remax`
4. **Kbone** (腾讯) - `mp-webpack-plugin`
5. **Chameleon** (滴滴) - `chameleon`, `cml`

### 前端框架
6. **Rax** (阿里) - `rax`, `createElement`
7. **Omi** (腾讯) - `omi`, `WeElement`
8. **San** (百度) - `san`, `defineComponent`

### 微前端
9. **Qiankun** (阿里) - `qiankun`, `registerMicroApps`
10. **Micro-app** (京东) - `@micro-zoe/micro-app`
11. **Icestark** (阿里) - `@ice/stark`

### 其他
12. **Mpvue** (美团)
13. **WePY** (腾讯)
14. **Lynx** (字节跳动)

---

## 🔍 检测模式速查表

### Webpack
```javascript
// 特征 1: Chunk 加载
(self["webpackChunk"] = self["webpackChunk"] || []).push([[...], {...}])

// 特征 2: 模块加载器
__webpack_require__(moduleId)

// 特征 3: JSONP 回调
webpackJsonpCallback(data)
```

### React
```javascript
// 特征 1: createElement
React.createElement("div", null, "Hello")

// 特征 2: JSX Runtime
import { jsx, jsxs } from 'react/jsx-runtime'

// 特征 3: Hooks
import { useState, useEffect } from 'react'
```

### Vue
```javascript
// 特征 1: Composition API
import { createVNode, createElementVNode } from 'vue'

// 特征 2: 编译后变量
const _hoisted_1 = { class: "container" }

// 特征 3: Render 函数
_createVNode("div", _hoisted_1)
```

### Angular
```javascript
// 特征 1: Ivy 编译指令
import { ɵɵelementStart, ɵɵtext } from '@angular/core'

// 特征 2: NgFactory
ɵɵdefineComponent({ /* ... */ })

// 特征 3: 平台启动
import { platformBrowser } from '@angular/platform-browser'
```

### Next.js
```javascript
// 特征 1: 内部模块
import { __next } from 'next'

// 特征 2: 数据获取
export async function getServerSideProps(context) { }
export async function getStaticProps(context) { }

// 特征 3: API 路由
export default function handler(req, res) { }
```

### Taro
```javascript
// 特征 1: 导入
import Taro from '@tarojs/taro'
import { View, Text } from '@tarojs/components'

// 特征 2: 组件
class MyComponent extends Taro.Component { }

// 特征 3: API
Taro.navigateTo({ url: '/pages/index/index' })
```

### Uni-app
```javascript
// 特征 1: uni API
uni.request({ url: 'https://api.example.com' })
uni.navigateTo({ url: '/pages/index/index' })

// 特征 2: 导入
import { uni } from '@dcloudio/uni-app'

// 特征 3: 组件
<view class="container">Hello Uni-app</view>
```

### Qiankun
```javascript
// 特征 1: 微应用注册
import { registerMicroApps, start } from 'qiankun'

// 特征 2: 配置
registerMicroApps([
  { name: 'app1', entry: '//localhost:8080', container: '#container' }
])

// 特征 3: 启动
start()
```

---

## 🛠️ 反混淆策略

### Webpack 解包
```rust
let clean = deobfuscator.unwrap_webpack(bundle)?;
```
**提取内容**:
- 模块代码
- 依赖关系
- 动态导入

### React 反编译
```rust
let jsx = deobfuscator.deobfuscate_react(compiled)?;
```
**转换**:
- `React.createElement` → JSX-like
- `_jsx` → 可读组件
- Props 提取

### Vue 模板还原
```rust
let template = deobfuscator.deobfuscate_vue(compiled)?;
```
**还原**:
- `_createVNode` → 模板
- `_hoisted_` → 静态内容
- 指令恢复

### Angular Ivy 逆向
```rust
let component = deobfuscator.deobfuscate_angular(ivy)?;
```
**逆向**:
- `ɵɵ` 指令 → 模板
- 组件元数据
- 依赖注入

### Taro 转换
```rust
let web = deobfuscator.deobfuscate_taro(taro_code)?;
```
**转换**:
- 小程序语法 → Web 标准
- Taro API → 标准 API
- 组件适配

### Uni-app 标准化
```rust
let standard = deobfuscator.deobfuscate_uniapp(uniapp_code)?;
```
**标准化**:
- `uni.request` → `fetch`
- `uni.navigateTo` → `router.push`
- 平台 API → Web API

---

## 📈 置信度评分

| 分数 | 含义 | 说明 |
|------|------|------|
| 90-100% | 非常确定 | 多个强特征匹配 |
| 70-89% | 较为确定 | 主要特征匹配 |
| 50-69% | 可能 | 部分特征匹配 |
| 30-49% | 不太确定 | 弱特征匹配 |
| 0-29% | 基本无框架 | 普通 JavaScript |

**计算公式**:
```
置信度 = (检测到的特征数 × 权重) / 总可能特征数
```

---

## 🧪 测试用例

### 检测单一框架
```rust
#[test]
fn test_react_detection() {
    let deobfuscator = AdvancedDeobfuscator::new();
    let code = r#"
        import React from 'react';
        const App = () => React.createElement("div", null, "Hello");
    "#;
    let analysis = deobfuscator.analyze(code).unwrap();
    assert!(analysis.framework_patterns.contains(&FrameworkObfuscation::ReactCompiled));
}
```

### 检测多框架
```rust
#[test]
fn test_multiple_frameworks() {
    let deobfuscator = AdvancedDeobfuscator::new();
    let code = r#"
        // Webpack + React + Next.js
        (self["webpackChunk"] = self["webpackChunk"] || []).push([[123], {
            456: function(module, exports, __webpack_require__) {
                const React = __webpack_require__(1);
                const { __next } = __webpack_require__(2);
            }
        }]);
    "#;
    let analysis = deobfuscator.analyze(code).unwrap();
    assert!(analysis.framework_patterns.len() >= 2);
}
```

### 检测中国框架
```rust
#[test]
fn test_taro_detection() {
    let deobfuscator = AdvancedDeobfuscator::new();
    let code = r#"
        import Taro from '@tarojs/taro';
        class MyComponent extends Taro.Component {
            render() {
                return <View>Hello Taro</View>;
            }
        }
    "#;
    let analysis = deobfuscator.analyze(code).unwrap();
    assert!(analysis.framework_patterns.contains(&FrameworkObfuscation::TaroFramework));
    
    let info = deobfuscator.get_framework_info(&FrameworkObfuscation::TaroFramework);
    assert_eq!(info.name, "Taro");
    assert!(info.origin.contains("JD.com"));
}
```

---

## 🎯 常见问题

### Q: 如何提高检测准确率？
**A**: 提供更完整的代码上下文，包括导入语句和主要逻辑。

### Q: 支持混淆后的代码吗？
**A**: 支持大部分混淆，但极端混淆可能降低准确率。先用通用反混淆再检测。

### Q: 如何添加新框架？
**A**: 
1. 在 `FrameworkObfuscation` enum 添加变体
2. 在 `detect_framework_patterns()` 添加检测逻辑
3. 在 `get_framework_info()` 添加元数据
4. 添加测试用例

### Q: 检测到错误的框架怎么办？
**A**: 检查置信度分数。低于 50% 的结果需要人工确认。

### Q: 支持框架版本检测吗？
**A**: 当前版本仅检测框架类型，版本检测在路线图中。

### Q: 性能如何？
**A**: 平均检测时间 <10ms，内存开销 <5MB。

---

## 🔗 相关文档

- [完整文档](./GLOBAL_FRAMEWORK_DETECTION.md)
- [测试套件](../tests/framework_detection_tests.rs)
- [实现代码](../src/learning/advanced_deobfuscation.rs)
- [AI 学习指南](./zh-CN/AI_LEARNING_IMPLEMENTATION.md)

---

## 📝 更新日志

### v2.0.0 (2024)
- ✨ 新增 100+ 全球框架检测
- 🇨🇳 完整中国框架生态系统支持
- 🛠️ 6 种专用反混淆策略
- 📊 框架元数据系统
- 🧪 全面测试覆盖
- 📖 中英文文档

### v1.0.0
- ✅ 基础框架检测（8 种）
- ✅ 简单反混淆

---

**Made with ❤️ for AI-powered web parsing**
