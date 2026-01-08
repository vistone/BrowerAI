# React 学习快速参考指南

## 🎯 30 秒快速总结

```
React = 声明式 UI 库
      + Fiber 调度引擎
      + Hooks 状态管理
      + Virtual DOM 协调
      + 事件代理系统
```

---

## 📚 核心概念速查

### 1. Component (组件)
```javascript
// Class Component
class Counter extends React.Component {
  state = { count: 0 };
  render() { return <div>{this.state.count}</div>; }
}

// Function Component
function Counter() {
  const [count, setCount] = useState(0);
  return <div>{count}</div>;
}
```

### 2. Virtual Element (虚拟元素)
```javascript
// Element: {type, props, children}
const el = React.createElement('div', {className: 'app'}, 'Hello');
// 等价于: <div className="app">Hello</div>
```

### 3. Fiber (纤程)
```javascript
// Fiber 节点结构
{
  type: ComponentType,        // 组件类型
  props: Props,               // 属性
  parent: Fiber,              // 父节点
  child: Fiber,               // 第一个子节点
  sibling: Fiber,             // 兄弟节点
  effectTag: 'PLACEMENT' | 'UPDATE' | 'DELETION'
}
```

### 4. Hook (钩子)
```javascript
// 最常用的 7 个 Hooks
useState(initialValue)              // 状态
useEffect(callback, deps)           // 副作用
useContext(Context)                 // 全局状态
useReducer(reducer, initialState)   // 复杂状态
useMemo(callback, deps)             // 计算缓存
useCallback(callback, deps)         // 函数缓存
useRef(initialValue)                // 持久化引用
```

### 5. Data Flow (数据流)
```
User Input → onClick Handler
    ↓
setState/dispatch
    ↓
Reconciliation (协调)
    ↓
commit (提交)
    ↓
DOM Update (更新 DOM)
    ↓
Re-render (重新渲染)
```

---

## 🚀 常用代码片段

### Fragment (无包装器)
```javascript
// ❌ 不好
function App() {
  return <div><h1>Title</h1><p>Content</p></div>; // 多余 div
}

// ✅ 好
function App() {
  return <>
    <h1>Title</h1>
    <p>Content</p>
  </>;
}
```

### Conditional Rendering (条件渲染)
```javascript
{condition ? <Component /> : null}
{isVisible && <Component />}
```

### List Rendering (列表渲染)
```javascript
// ❌ 不好 - 没有 key
list.map(item => <Item>{item}</Item>)

// ✅ 好 - 有 key
list.map(item => <Item key={item.id}>{item}</Item>)
```

### State Update (状态更新)
```javascript
// 函数式更新 - 推荐！
const [count, setCount] = useState(0);
setCount(prev => prev + 1);

// 直接更新
setCount(5);
```

### Effect Cleanup (副作用清理)
```javascript
useEffect(() => {
  const subscription = subscribe();
  
  return () => {
    // 清理函数 - 组件卸载时执行
    subscription.unsubscribe();
  };
}, []);
```

### Custom Hook (自定义 Hook)
```javascript
function useAsync(asyncFn) {
  const [state, setState] = useState('idle');
  
  useEffect(() => {
    asyncFn().then(() => setState('success'));
  }, [asyncFn]);
  
  return state;
}
```

---

## ⚡ 性能优化清单

| 优化技巧 | 何时使用 | 效果 |
|---------|---------|------|
| React.memo | 纯组件，props 不经常变 | ⭐⭐ |
| useMemo | 昂贵的计算 | ⭐⭐⭐ |
| useCallback | 子组件依赖 callback | ⭐⭐ |
| 代码分割 | 大型应用 | ⭐⭐⭐⭐ |
| 虚拟列表 | 10K+ 数据项 | ⭐⭐⭐⭐⭐ |
| Fragment | 减少 DOM | ⭐⭐ |

---

## 🔍 常见问题速解

### Q1: 为什么 Hooks 要在顶层调用？
```javascript
// ❌ 错误
if (condition) {
  const [state, setState] = useState(0); // 顺序改变！
}

// ✅ 正确
const [state, setState] = useState(0);
if (condition) {
  // 使用 state
}

原理: React 通过调用顺序识别 Hook
     如果顺序变化，状态映射就会混乱
```

### Q2: useEffect 的依赖项怎么写？
```javascript
// ❌ 每次都运行
useEffect(() => {
  fetchData();
}); // 没有依赖项

// ✅ 仅挂载时运行
useEffect(() => {
  fetchData();
}, []); // 空数组

// ✅ 依赖改变时运行
useEffect(() => {
  fetchData(id);
}, [id]); // 当 id 改变时运行
```

### Q3: Key 有什么作用？
```javascript
// 没有 Key - 重新排序会重新创建
[<A/>, <B/>, <C/>]
// 变为 [<C/>, <B/>, <A/>]
// React 认为 A 变成了 C！

// 有 Key - 可以正确复用
[<A key="a"/>, <B key="b"/>, <C key="c"/>]
// React 知道哪个是哪个
```

### Q4: useState vs useReducer 怎么选？
```javascript
// useState: 简单状态
const [count, setCount] = useState(0);

// useReducer: 复杂状态逻辑
const [state, dispatch] = useReducer(reducer, initialState);

规则:
- 单个值 → useState
- 多个关联值 → useReducer
- 依赖其他状态 → useReducer
```

### Q5: Context 会导致全部重新渲染吗？
```javascript
// ✅ 其他不依赖 value 的组件不会重新渲染
const AppContext = createContext();

function App() {
  const [count, setCount] = useState(0);
  
  return (
    <AppContext.Provider value={{ count }}>
      <Expensive /> {/* 只有使用 Context 的组件会重新渲染 */}
    </AppContext.Provider>
  );
}
```

---

## 🎯 优化决策树

```
应用性能慢？
    ↓
├─ 是否有 10K+ 列表?
│  ├─ 是 → 使用虚拟列表
│  └─ 否 → 继续检查
│
├─ 是否计算复杂操作?
│  ├─ 是 → 使用 useMemo
│  └─ 否 → 继续检查
│
├─ 是否传递 callback 给子组件?
│  ├─ 是 → 使用 useCallback
│  └─ 否 → 继续检查
│
├─ 是否有大量 DOM 节点?
│  ├─ 是 → 使用代码分割
│  └─ 否 → 继续检查
│
└─ 使用 React DevTools Profiler 分析
```

---

## 📊 React 版本功能速查

| 功能 | 推出版本 | 使用方式 |
|------|--------|---------|
| Hooks | 16.8 | useState, useEffect 等 |
| Context | 16.3 | createContext, useContext |
| Suspense | 16.6 | React.lazy, Suspense |
| Concurrent | 18.0 | startTransition, useTransition |
| Automatic Batching | 18.0 | 自动批处理更新 |

---

## 🔗 模式对比表

### 状态管理方案对比

| 方案 | 学习难度 | 代码量 | 适用场景 | 性能 |
|------|--------|--------|---------|------|
| useState | ⭐ | 少 | 简单状态 | 优 |
| useReducer | ⭐⭐ | 中 | 复杂逻辑 | 优 |
| Context | ⭐⭐ | 中 | 全局状态 | 良 |
| Redux | ⭐⭐⭐ | 多 | 大型应用 | 优 |
| Zustand | ⭐⭐ | 少 | 中型应用 | 优 |
| Jotai | ⭐⭐ | 少 | 原子状态 | 优 |

### 代码复用方案对比

| 方案 | 优点 | 缺点 | 何时用 |
|------|------|------|--------|
| Custom Hooks | 简单，高度复用 | 需理解 Hooks | 推荐 |
| HOC | 灵活，支持旧版本 | Wrapper Hell | 旧项目 |
| Render Props | 显式数据流 | 代码嵌套 | 特殊场景 |

---

## 🚨 常见陷阱

| 陷阱 | 描述 | 解决 |
|------|------|------|
| 闭包陷阱 | useEffect 中拿不到最新 state | 加入依赖项 |
| 无限循环 | useEffect 没有依赖项 | 加 [] |
| 过度渲染 | 没用 memo/useMemo | 使用优化 |
| 内存泄漏 | 没有清理 effect | 返回清理函数 |
| Key 问题 | 使用 index 作为 key | 使用唯一 id |

---

## 💾 快速复制代码

### Form 处理
```javascript
function Form() {
  const [form, setForm] = useState({ name: '', email: '' });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log(form);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input name="name" value={form.name} onChange={handleChange} />
      <input name="email" value={form.email} onChange={handleChange} />
      <button type="submit">Submit</button>
    </form>
  );
}
```

### API 数据加载
```javascript
function useApi(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then(data => { setData(data); setLoading(false); })
      .catch(err => { setError(err); setLoading(false); });
  }, [url]);

  return { data, loading, error };
}
```

### 防抖 Hook
```javascript
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}
```

---

## 📖 推荐阅读顺序

1. **官方文档** (1 天)
   - Concepts: Components, Props, State
   - Hooks: useState, useEffect

2. **高级主题** (2 天)
   - Advanced: Context, Code Splitting, Performance
   - API Reference

3. **源代码** (1 周)
   - React Core (packages/react)
   - Scheduler (packages/scheduler)
   - Reconciler (packages/react-reconciler)

4. **实战项目** (2 周)
   - 构建自己的 React-like 库
   - 学习真实应用案例

---

## 🎓 成为 React 专家的 4 个级别

```
Level 1: React 用户 ⭐
  能用 useState, useEffect, 编写组件
  预计时间: 1 周

Level 2: React 开发者 ⭐⭐⭐
  理解 Fiber, Hooks 原理
  能优化性能，实现复杂模式
  预计时间: 1 月

Level 3: React 架构师 ⭐⭐⭐⭐⭐
  深入理解源代码
  能参与开源贡献
  预计时间: 6 月

Level 4: React 核心贡献者 ⭐⭐⭐⭐⭐⭐
  参与 React 开发
  提出新特性和改进
  预计时间: 1-2 年
```

---

## ✨ 最后的话

> "React 是学习现代前端的最好教材。
> 理解了 React，你就理解了大多数现代框架。"

**当你遇到不懂的概念时，记住:**
1. 读官方文档
2. 看源代码
3. 写测试代码
4. 看 DevTools
5. 问 AI（现在可以）

**学习路径总结:**
```
React 基础 → Hooks → Fiber → 源代码 → 实现自己的 → 贡献开源
   1 周      1 周     1 周     2 周      2 周          持续
```

---

**此指南定期更新**  
**最后更新**: 2026-01-07  
**适用版本**: React 18.0+
