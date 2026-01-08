// React 高级模式识别和架构学习示例
// 深度分析 React 的设计模式、优化技巧和架构决策

use browerai_learning::CodeUnderstandingSystem;
use browerai_learning::VisualizationFormat;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     React 高级模式识别和架构深度学习分析                      ║");
    println!("║          Advanced React Patterns & Architecture Analysis       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let system = CodeUnderstandingSystem::new();

    // ===== 分析 1: 自定义 Hooks 模式 =====
    println!("═══════════════════════════════════════════════════════════════");
    println!("🎣 分析 1: 自定义 Hooks 库 (Custom Hooks Patterns)");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_custom_hooks(&system)?;

    // ===== 分析 2: 高阶组件模式 =====
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("🔄 分析 2: 高阶组件 (Higher-Order Components)");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_hoc_pattern(&system)?;

    // ===== 分析 3: Render Props 模式 =====
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("📦 分析 3: Render Props 模式");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_render_props(&system)?;

    // ===== 分析 4: 状态管理架构 =====
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("🎛️ 分析 4: 全局状态管理架构 (Redux-like)");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_state_management(&system)?;

    // ===== 分析 5: 性能优化模式 =====
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("⚡ 分析 5: 性能优化模式 (Optimization Techniques)");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_optimization_patterns(&system)?;

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  ✅ React 高级模式分析完成！                                  ║");
    println!("║                                                               ║");
    println!("║  📖 高级设计模式总结：                                        ║");
    println!("║  • 自定义 Hooks: 逻辑复用的现代方式                          ║");
    println!("║  • 高阶组件: 功能增强和交叉关注点处理                        ║");
    println!("║  • Render Props: 灵活的组件间通信                            ║");
    println!("║  • 状态管理: 可预测的数据流和单向数据绑定                    ║");
    println!("║  • 性能优化: 记忆化、按需加载、虚拟化列表                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

fn analyze_custom_hooks(system: &CodeUnderstandingSystem) -> anyhow::Result<()> {
    let custom_hooks = r#"
// 自定义 Hooks 库 - React 逻辑复用的新范式

// Hook 1: useAsync - 处理异步操作
function useAsync(asyncFunction, immediate = true) {
  const [status, setStatus] = React.useState('idle');
  const [value, setValue] = React.useState(null);
  const [error, setError] = React.useState(null);

  const execute = React.useCallback(async () => {
    setStatus('pending');
    setValue(null);
    setError(null);
    try {
      const response = await asyncFunction();
      setValue(response);
      setStatus('success');
    } catch (error) {
      setError(error);
      setStatus('error');
    }
  }, [asyncFunction]);

  React.useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [execute, immediate]);

  return { execute, status, value, error };
}

// Hook 2: useFetch - 数据获取
function useFetch(url, options = {}) {
  return useAsync(async () => {
    const response = await fetch(url, options);
    if (!response.ok) throw new Error(response.statusText);
    return response.json();
  }, true);
}

// Hook 3: useLocalStorage - 本地存储
function useLocalStorage(key, initialValue) {
  const [storedValue, setStoredValue] = React.useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.log(error);
      return initialValue;
    }
  });

  const setValue = (value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.log(error);
    }
  };

  return [storedValue, setValue];
}

// Hook 4: useWindowSize - 响应式窗口尺寸
function useWindowSize() {
  const [windowSize, setWindowSize] = React.useState({
    width: undefined,
    height: undefined,
  });

  React.useEffect(() => {
    function handleResize() {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }

    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return windowSize;
}

// Hook 5: useDebounce - 防抖
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = React.useState(value);

  React.useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
}

// Hook 6: useThrottle - 节流
function useThrottle(value, interval) {
  const [throttledValue, setThrottledValue] = React.useState(value);
  const lastUpdated = React.useRef(null);

  React.useEffect(() => {
    const now = Date.now();
    if (lastUpdated.current && now >= lastUpdated.current + interval) {
      lastUpdated.current = now;
      setThrottledValue(value);
    }
  }, [value, interval]);

  return throttledValue;
}

// Hook 7: usePrevious - 获取前一个值
function usePrevious(value) {
  const ref = React.useRef();
  
  React.useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
}

// Hook 8: useToggle - 布尔值开关
function useToggle(initialValue = false) {
  const [value, setValue] = React.useState(initialValue);
  return [value, () => setValue(!value)];
}

// Hook 9: useCounter - 计数器
function useCounter(initialValue = 0) {
  const [count, setCount] = React.useState(initialValue);
  
  return {
    count,
    increment: () => setCount(count + 1),
    decrement: () => setCount(count - 1),
    reset: () => setCount(initialValue),
  };
}

// Hook 10: useMountedEffect - 仅在挂载后执行
function useMountedEffect(callback, deps) {
  const isMounted = React.useRef(false);

  React.useEffect(() => {
    if (!isMounted.current) {
      isMounted.current = true;
      return;
    }
    return callback();
  }, deps);
}

export { 
  useAsync, useFetch, useLocalStorage, useWindowSize, 
  useDebounce, useThrottle, usePrevious, useToggle, 
  useCounter, useMountedEffect 
};
"#;

    let report = system.analyze(custom_hooks, "React Custom Hooks Library v1.0")?;

    println!("📊 自定义 Hooks 库分析：\n");
    println!("{}", system.generate_report(&report));

    println!("\n🔍 关键发现：");
    println!("  ✓ 10 个高频 Hooks: useAsync, useFetch, useLocalStorage 等");
    println!("  ✓ 异步处理: useAsync/useFetch 处理数据加载");
    println!("  ✓ 性能优化: useDebounce/useThrottle 控制更新频率");
    println!("  ✓ 状态管理: useLocalStorage 持久化数据");
    println!("  ✓ 响应式: useWindowSize 处理窗口变化");

    println!("\n💡 最佳实践：");
    println!("  1. 每个 Hook 只做一件事 (单一职责)");
    println!("  2. 使用 useRef 保存不需要触发重新渲染的值");
    println!("  3. useCallback 包装函数避免无限循环");
    println!("  4. 正确管理依赖项避免过度调用");

    Ok(())
}

fn analyze_hoc_pattern(system: &CodeUnderstandingSystem) -> anyhow::Result<()> {
    let hoc_code = r#"
// 高阶组件 (Higher-Order Component) 模式
// 核心思想: 组件是函数，可以接收组件作为参数并返回新组件

// HOC 1: withTheme - 注入主题
function withTheme(Component) {
  return function ThemedComponent(props) {
    const [theme, setTheme] = React.useState('light');

    const toggleTheme = () => {
      setTheme(theme === 'light' ? 'dark' : 'light');
    };

    return (
      <div className={`theme-${theme}`}>
        <Component {...props} theme={theme} toggleTheme={toggleTheme} />
      </div>
    );
  };
}

// HOC 2: withRouter - 注入路由
function withRouter(Component) {
  return function RouterComponent(props) {
    const [location, setLocation] = React.useState(window.location.pathname);

    React.useEffect(() => {
      const handlePopState = () => setLocation(window.location.pathname);
      window.addEventListener('popstate', handlePopState);
      return () => window.removeEventListener('popstate', handlePopState);
    }, []);

    const navigate = (path) => {
      window.history.pushState({}, '', path);
      setLocation(path);
    };

    return <Component {...props} location={location} navigate={navigate} />;
  };
}

// HOC 3: withAuth - 验证和授权
function withAuth(Component) {
  return function AuthComponent(props) {
    const [isAuthenticated, setIsAuthenticated] = React.useState(false);

    React.useEffect(() => {
      // 检查认证状态
      const checkAuth = async () => {
        const response = await fetch('/api/auth/check');
        setIsAuthenticated(response.ok);
      };
      checkAuth();
    }, []);

    if (!isAuthenticated) {
      return <div>Please log in</div>;
    }

    return <Component {...props} />;
  };
}

// HOC 4: withDataFetching - 数据加载
function withDataFetching(url) {
  return function WithDataComponent(Component) {
    return function DataComponent(props) {
      const [data, setData] = React.useState(null);
      const [loading, setLoading] = React.useState(true);
      const [error, setError] = React.useState(null);

      React.useEffect(() => {
        fetch(url)
          .then(res => res.json())
          .then(data => { setData(data); setLoading(false); })
          .catch(err => { setError(err); setLoading(false); });
      }, [url]);

      return (
        <Component 
          {...props} 
          data={data} 
          loading={loading} 
          error={error} 
        />
      );
    };
  };
}

// HOC 5: withLogger - 日志记录
function withLogger(Component) {
  return function LoggingComponent(props) {
    React.useEffect(() => {
      console.log(`Component mounted: ${Component.name || 'Unknown'}`);
      return () => {
        console.log(`Component unmounted: ${Component.name || 'Unknown'}`);
      };
    }, []);

    return <Component {...props} />;
  };
}

// HOC 6: compose - HOC 组合
function compose(...hocs) {
  return (Component) => {
    return hocs.reduceRight((acc, hoc) => hoc(acc), Component);
  };
}

// 使用示例:
// const EnhancedComponent = compose(
//   withTheme,
//   withRouter,
//   withAuth,
//   withLogger
// )(MyComponent);

// HOC 7: withMemo - 性能优化
function withMemo(Component) {
  return React.memo(Component, (prevProps, nextProps) => {
    return JSON.stringify(prevProps) === JSON.stringify(nextProps);
  });
}

// HOC 8: withForwardRef - 转发 Ref
function withForwardRef(Component) {
  return React.forwardRef((props, ref) => {
    return <Component {...props} forwardedRef={ref} />;
  });
}

export { 
  withTheme, withRouter, withAuth, withDataFetching, 
  withLogger, withMemo, withForwardRef, compose 
};
"#;

    let report = system.analyze(hoc_code, "React HOC Patterns v1.0")?;

    println!("📊 高阶组件模式分析：\n");
    println!("{}", system.generate_report(&report));

    println!("\n🔍 关键发现：");
    println!("  ✓ 8 个常用 HOC: withTheme, withRouter, withAuth 等");
    println!("  ✓ 功能类型:");
    println!("    - 属性代理: withTheme, withLogger");
    println!("    - 反向继承: withAuth, withDataFetching");
    println!("    - 组合: compose 函数");
    println!("  ✓ 性能优化: withMemo, React.memo");

    println!("\n⚠️ HOC vs Hooks:");
    println!("  HOC 优势:");
    println!("    • 支持旧版本 React");
    println!("    • 灵活的组件包装");
    println!("  HOC 劣势:");
    println!("    • 造成 'wrapper hell'");
    println!("    • 难以调试");
    println!("  → 现代 React 优先使用 Hooks！");

    Ok(())
}

fn analyze_render_props(system: &CodeUnderstandingSystem) -> anyhow::Result<()> {
    let render_props = r#"
// Render Props 模式
// 核心: 将组件逻辑作为函数通过 props 传递

// Render Prop 1: Mouse Tracker
class MouseTracker extends React.Component {
  constructor(props) {
    super(props);
    this.state = { x: 0, y: 0 };
  }

  componentDidMount() {
    document.addEventListener('mousemove', this.handleMouseMove);
  }

  componentWillUnmount() {
    document.removeEventListener('mousemove', this.handleMouseMove);
  }

  handleMouseMove = (event) => {
    this.setState({
      x: event.clientX,
      y: event.clientY
    });
  }

  render() {
    return this.props.render(this.state);
  }
}

// 使用: <MouseTracker render={({x, y}) => <div>x: {x}, y: {y}</div>} />

// Render Prop 2: DataProvider
class DataProvider extends React.Component {
  constructor(props) {
    super(props);
    this.state = { data: null, loading: true };
  }

  componentDidMount() {
    fetch(this.props.url)
      .then(res => res.json())
      .then(data => this.setState({ data, loading: false }))
      .catch(err => this.setState({ error: err, loading: false }));
  }

  render() {
    return this.props.children(this.state);
  }
}

// 使用: <DataProvider url="/api/data">
//        {({data, loading}) => loading ? <div>Loading</div> : <div>{data}</div>}
//      </DataProvider>

// Render Prop 3: RenderIfAdmin
class RenderIfAdmin extends React.Component {
  render() {
    const isAdmin = this.props.user?.role === 'admin';
    return this.props.children(isAdmin);
  }
}

// Render Prop 4: Intersection Observer
class InView extends React.Component {
  constructor(props) {
    super(props);
    this.state = { inView: false };
  }

  componentDidMount() {
    const observer = new IntersectionObserver(([entry]) => {
      this.setState({ inView: entry.isIntersecting });
    });
    observer.observe(this.ref);
  }

  render() {
    return (
      <div ref={ref => this.ref = ref}>
        {this.props.render(this.state.inView)}
      </div>
    );
  }
}

// Render Prop 5: Toggle
class Toggle extends React.Component {
  constructor(props) {
    super(props);
    this.state = { on: false };
  }

  toggle = () => this.setState(prev => ({ on: !prev.on }));

  render() {
    return this.props.children({
      on: this.state.on,
      toggle: this.toggle
    });
  }
}

// 使用: <Toggle>
//        {({on, toggle}) => (
//          <button onClick={toggle}>{on ? 'ON' : 'OFF'}</button>
//        )}
//      </Toggle>

export { MouseTracker, DataProvider, RenderIfAdmin, InView, Toggle };
"#;

    let report = system.analyze(render_props, "React Render Props v1.0")?;

    println!("📊 Render Props 模式分析：\n");
    println!("{}", system.generate_report(&report));

    println!("\n🔍 关键发现：");
    println!("  ✓ 5 个常用 Render Props: MouseTracker, DataProvider 等");
    println!("  ✓ 灵活的数据传递机制");
    println!("  ✓ 支持组件间逻辑共享");
    println!("  ✓ children as function 的变体");

    println!("\n📊 模式对比：");
    println!("  Render Props vs HOC vs Hooks:");
    println!("  ┌─────────────┬──────────┬─────┬────────┐");
    println!("  │ 特性       │ Hooks    │ HOC │ Render │");
    println!("  ├─────────────┼──────────┼─────┼────────┤");
    println!("  │ 调试难度   │ 简单     │ 难  │ 中等   │");
    println!("  │ 代码复用   │ 优秀     │ 好  │ 好     │");
    println!("  │ 性能       │ 最优     │ 良  │ 良     │");
    println!("  │ 学习曲线   │ 简单     │ 难  │ 中等   │");
    println!("  └─────────────┴──────────┴─────┴────────┘");

    Ok(())
}

fn analyze_state_management(system: &CodeUnderstandingSystem) -> anyhow::Result<()> {
    let state_mgmt = r#"
// 全局状态管理架构 (Redux-like Pattern)

// Action Creators
const Actions = {
  addTodo: (text) => ({ type: 'ADD_TODO', payload: text }),
  deleteTodo: (id) => ({ type: 'DELETE_TODO', payload: id }),
  toggleTodo: (id) => ({ type: 'TOGGLE_TODO', payload: id }),
  setFilter: (filter) => ({ type: 'SET_FILTER', payload: filter })
};

// Reducer - 纯函数，无副作用
function rootReducer(state = initialState, action) {
  switch (action.type) {
    case 'ADD_TODO':
      return {
        ...state,
        todos: [...state.todos, { id: Date.now(), text: action.payload, done: false }]
      };
    
    case 'DELETE_TODO':
      return {
        ...state,
        todos: state.todos.filter(todo => todo.id !== action.payload)
      };
    
    case 'TOGGLE_TODO':
      return {
        ...state,
        todos: state.todos.map(todo =>
          todo.id === action.payload ? { ...todo, done: !todo.done } : todo
        )
      };
    
    case 'SET_FILTER':
      return { ...state, filter: action.payload };
    
    default:
      return state;
  }
}

// Store - 中央存储
class Store {
  constructor(reducer, initialState) {
    this.reducer = reducer;
    this.state = initialState;
    this.listeners = [];
  }

  getState() {
    return this.state;
  }

  dispatch(action) {
    this.state = this.reducer(this.state, action);
    this.listeners.forEach(listener => listener(this.state));
  }

  subscribe(listener) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }
}

// Middleware - 扩展 dispatch
function applyMiddleware(...middlewares) {
  return (Store) => {
    return class EnhancedStore extends Store {
      dispatch(action) {
        const chain = middlewares.map(middleware => 
          middleware(this.getState, this.dispatch.bind(this))
        );
        const enhancedDispatch = chain.reduce((f, g) => f(g));
        return enhancedDispatch(action);
      }
    };
  };
}

// 常用中间件
const logger = (getState, dispatch) => (next) => (action) => {
  console.log('Dispatching:', action);
  const result = next(action);
  console.log('Next state:', getState());
  return result;
};

const asyncMiddleware = (getState, dispatch) => (next) => (action) => {
  if (typeof action === 'function') {
    return action(dispatch, getState);
  }
  return next(action);
};

// Selectors - 获取状态的特定部分
const Selectors = {
  getTodos: (state) => state.todos,
  getFilter: (state) => state.filter,
  getFilteredTodos: (state) => {
    const filter = state.filter;
    if (filter === 'active') return state.todos.filter(t => !t.done);
    if (filter === 'completed') return state.todos.filter(t => t.done);
    return state.todos;
  },
  getTodoCount: (state) => state.todos.length
};

// 组件连接
function connect(mapStateToProps, mapDispatchToProps) {
  return (Component) => {
    return (props) => {
      const [state, setState] = React.useState(store.getState());

      React.useEffect(() => {
        const unsubscribe = store.subscribe(setState);
        return unsubscribe;
      }, []);

      const stateProps = mapStateToProps(state);
      const dispatchProps = mapDispatchToProps(store.dispatch);

      return <Component {...props} {...stateProps} {...dispatchProps} />;
    };
  };
}

// Context API 简化版本
const StoreContext = React.createContext();

function Provider({ store, children }) {
  return (
    <StoreContext.Provider value={store}>
      {children}
    </StoreContext.Provider>
  );
}

function useStore() {
  return React.useContext(StoreContext);
}

function useSelector(selector) {
  const store = useStore();
  const [state, setState] = React.useState(() => selector(store.getState()));

  React.useEffect(() => {
    return store.subscribe(() => {
      setState(selector(store.getState()));
    });
  }, [selector, store]);

  return state;
}

function useDispatch() {
  const store = useStore();
  return store.dispatch.bind(store);
}

export { 
  Actions, rootReducer, Store, applyMiddleware, 
  logger, asyncMiddleware, Selectors, connect, 
  Provider, useStore, useSelector, useDispatch 
};
"#;

    let report = system.analyze(state_mgmt, "React State Management v1.0")?;

    println!("📊 全局状态管理架构分析：\n");
    println!("{}", system.generate_report(&report));

    println!("\n🔍 关键发现：");
    println!("  ✓ Store 中央存储");
    println!("  ✓ Actions 和 Reducers 的单向数据流");
    println!("  ✓ Middleware 支持异步和日志记录");
    println!("  ✓ Selectors 优化状态访问");
    println!("  ✓ Context API 集成");

    println!("\n🏗️ 架构层次：");
    println!("  ┌──────────────────────┐");
    println!("  │  UI Components       │");
    println!("  ├──────────────────────┤");
    println!("  │ connect/useDispatch  │");
    println!("  ├──────────────────────┤");
    println!("  │ Middleware Chain     │");
    println!("  ├──────────────────────┤");
    println!("  │ Store (单一真实源)  │");
    println!("  ├──────────────────────┤");
    println!("  │ Reducer (状态机)     │");
    println!("  └──────────────────────┘");

    Ok(())
}

fn analyze_optimization_patterns(system: &CodeUnderstandingSystem) -> anyhow::Result<()> {
    let optimization = r#"
// React 性能优化模式

// 优化 1: 代码分割和懒加载
const LazyComponent = React.lazy(() => import('./LazyComponent'));

function App() {
  return (
    <React.Suspense fallback={<div>Loading...</div>}>
      <LazyComponent />
    </React.Suspense>
  );
}

// 优化 2: 虚拟列表
class VirtualList extends React.Component {
  constructor(props) {
    super(props);
    this.state = { scrollTop: 0 };
  }

  handleScroll = (event) => {
    this.setState({ scrollTop: event.target.scrollTop });
  }

  render() {
    const { items, itemHeight, height } = this.props;
    const startIndex = Math.floor(this.state.scrollTop / itemHeight);
    const endIndex = startIndex + Math.ceil(height / itemHeight);
    const visibleItems = items.slice(startIndex, endIndex);

    return (
      <div onScroll={this.handleScroll} style={{ height, overflow: 'auto' }}>
        <div style={{ height: items.length * itemHeight }}>
          {visibleItems.map((item, i) => (
            <div key={startIndex + i} style={{ height: itemHeight }}>
              {item}
            </div>
          ))}
        </div>
      </div>
    );
  }
}

// 优化 3: 批量更新
class BatchedUpdates extends React.Component {
  handleClick = async () => {
    // 自动批处理
    this.setState({ count: this.state.count + 1 });
    this.setState({ label: 'clicked' });
    // 只触发一次渲染
  }

  handleAsyncClick = async () => {
    await new Promise(resolve => setTimeout(resolve, 0));
    // React 18+: 自动批处理异步更新
    this.setState({ count: this.state.count + 1 });
  }

  render() {
    return <button onClick={this.handleClick}>Click</button>;
  }
}

// 优化 4: 记忆化结果
function expensiveComponent(data) {
  const memoizedValue = React.useMemo(() => {
    return data.items
      .filter(item => item.active)
      .map(item => item.value)
      .reduce((sum, val) => sum + val, 0);
  }, [data.items]);

  return <div>{memoizedValue}</div>;
}

// 优化 5: 缓存回调函数
function CallbackCache() {
  const [count, setCount] = React.useState(0);

  const memoizedCallback = React.useCallback(() => {
    console.log('Count:', count);
  }, [count]);

  return <Child onCallback={memoizedCallback} />;
}

// 优化 6: 按需加载大数据集
function DataGrid({ data }) {
  const [visibleRange, setVisibleRange] = React.useState({ start: 0, end: 50 });

  const handleScroll = React.useCallback((index) => {
    setVisibleRange({
      start: Math.max(0, index - 25),
      end: Math.min(data.length, index + 75)
    });
  }, [data.length]);

  const visibleData = data.slice(visibleRange.start, visibleRange.end);

  return (
    <div onScroll={() => handleScroll(Math.floor(visibleRange.start / 50))}>
      {visibleData.map(row => <Row key={row.id} data={row} />)}
    </div>
  );
}

// 优化 7: 条件渲染避免 DOM 污染
function ConditionalRender({ showDetail }) {
  return (
    <div>
      {showDetail && <DetailComponent />}
    </div>
  );
}

// 优化 8: 样式优化 - 避免内联对象
const buttonStyle = { padding: '10px', background: 'blue' };

function Button() {
  return <button style={buttonStyle}>Click</button>;
}

// 优化 9: Fragment 减少 DOM 节点
function MultipleElements() {
  return (
    <>
      <header>Header</header>
      <main>Main</main>
      <footer>Footer</footer>
    </>
  );
}

// 优化 10: 事件委托
function EventDelegation() {
  const handleClick = (e) => {
    if (e.target.matches('.item')) {
      console.log('Item clicked:', e.target.id);
    }
  };

  return (
    <ul onClick={handleClick}>
      <li className="item" id="1">Item 1</li>
      <li className="item" id="2">Item 2</li>
      <li className="item" id="3">Item 3</li>
    </ul>
  );
}

export { 
  LazyComponent, VirtualList, BatchedUpdates, 
  expensiveComponent, CallbackCache, DataGrid, 
  ConditionalRender, Button, EventDelegation 
};
"#;

    let report = system.analyze(optimization, "React Optimization Patterns v1.0")?;

    println!("📊 性能优化模式分析：\n");
    println!("{}", system.generate_report(&report));

    println!("\n🔍 关键发现：");
    println!("  ✓ 10 个优化技巧");
    println!("  ✓ 代码分割和懒加载");
    println!("  ✓ 虚拟列表处理大数据");
    println!("  ✓ 批量更新减少渲染");
    println!("  ✓ 记忆化和缓存优化");

    println!("\n⚡ 优化优先级：");
    println!("  Priority 1 (必做):");
    println!("    • 虚拟列表 - 处理大数据集");
    println!("    • 代码分割 - 减少初始加载");
    println!("    • 记忆化 - 避免不必要的计算");
    println!();
    println!("  Priority 2 (推荐):");
    println!("    • useCallback 缓存回调");
    println!("    • Fragment 减少 DOM");
    println!("    • 事件委托");
    println!();
    println!("  Priority 3 (微优化):");
    println!("    • 样式常量");
    println!("    • 条件渲染");

    Ok(())
}
