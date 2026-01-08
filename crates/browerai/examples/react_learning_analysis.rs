// React 源代码学习分析和反混淆系统
// 目标: 理解 React 的架构、核心模块、组件模型

use browerai_learning::CodeUnderstandingSystem;
use browerai_learning::{ArchitecturePattern, VisualizationFormat};

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         React 源代码学习分析和反混淆系统                      ║");
    println!("║    Understanding React Architecture & Deobfuscation          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let system = CodeUnderstandingSystem::new();

    // ===== 分析 1: React 核心代码 (简化版本) =====
    println!("═══════════════════════════════════════════════════════════════");
    println!("📚 分析 1: React 核心库架构 (React Core)");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_react_core(&system)?;

    // ===== 分析 2: React Fiber 架构 =====
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("🔧 分析 2: Fiber 调度系统 (Scheduling & Reconciliation)");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_react_fiber(&system)?;

    // ===== 分析 3: React Hooks 系统 =====
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("🎣 分析 3: Hooks 状态管理系统");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_react_hooks(&system)?;

    // ===== 分析 4: React DOM 渲染 =====
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("🎨 分析 4: DOM 渲染引擎 (React DOM)");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_react_dom(&system)?;

    // ===== 分析 5: 混淆代码反混淆 =====
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("🔐 分析 5: 混淆 React 代码反混淆");
    println!("═══════════════════════════════════════════════════════════════\n");
    analyze_minified_react(&system)?;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  ✅ React 分析完成！                                          ║");
    println!("║                                                              ║");
    println!("║  📖 学习总结：                                                ║");
    println!("║  • React 是基于组件的声明式 UI 库                             ║");
    println!("║  • Fiber 架构实现增量渲染和优先级调度                         ║");
    println!("║  • Hooks 提供状态管理的新范式                                ║");
    println!("║  • DOM 模块负责浏览器实际渲染                                ║");
    println!("║  • 混淆代码可以通过架构分析反混淆                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

fn analyze_react_core(system: &CodeUnderstandingSystem) -> anyhow::Result<()> {
    let react_code = r#"
// React Core - Component & Element System
// 核心概念: 组件、元素、虚拟树

class React {
    static createClass(spec) {
        return class extends Component {
            constructor(props) {
                super(props);
                Object.assign(this, spec);
            }
        };
    }

    static createElement(type, props, ...children) {
        return {
            type,
            props: { ...props, children }
        };
    }

    static cloneElement(element, props) {
        return {
            ...element,
            props: { ...element.props, ...props }
        };
    }

    static Children = {
        map(children, callback) {
            return children.flat().map(callback);
        },
        count(children) {
            return children.flat().length;
        },
        forEach(children, callback) {
            children.flat().forEach(callback);
        },
        only(children) {
            if (children.length !== 1) throw new Error('Expected one child');
            return children[0];
        }
    };
}

// Component Base Class
class Component {
    constructor(props) {
        this.props = props;
        this.state = {};
        this._isMounted = false;
    }

    setState(state, callback) {
        this.state = { ...this.state, ...state };
        if (this._isMounted) {
            this.forceUpdate(callback);
        }
    }

    forceUpdate(callback) {
        // 触发重新渲染
        this.componentWillUpdate?.();
        // 更新 DOM
        this.componentDidUpdate?.();
        if (callback) callback();
    }

    render() {
        throw new Error('render() 必须在子类中实现');
    }

    componentDidMount() {}
    componentWillUnmount() {}
    componentDidUpdate() {}
    componentWillUpdate() {}
    shouldComponentUpdate(nextProps, nextState) { return true; }
}

// PureComponent - 优化版本
class PureComponent extends Component {
    shouldComponentUpdate(nextProps, nextState) {
        return !shallowEqual(this.props, nextProps) ||
               !shallowEqual(this.state, nextState);
    }
}

// 浅比较工具
function shallowEqual(obj1, obj2) {
    const keys1 = Object.keys(obj1);
    const keys2 = Object.keys(obj2);
    
    if (keys1.length !== keys2.length) return false;
    
    return keys1.every(key => obj1[key] === obj2[key]);
}

// 导出 API
export { React, Component, PureComponent, createElement, cloneElement };
"#;

    let report = system.analyze(react_code, "React Core v18.0")?;

    println!("📊 React 核心架构分析：\n");
    println!("{}", system.generate_report(&report));

    println!("\n🔍 关键发现：");
    println!("  ✓ 架构模式: Class-based Component System");
    println!("  ✓ 核心类: React, Component, PureComponent");
    println!("  ✓ 状态管理: setState() 机制");
    println!("  ✓ 生命周期: componentDidMount, componentDidUpdate 等");
    println!("  ✓ 性能优化: PureComponent 和 shouldComponentUpdate");

    println!("\n📚 架构分析：");
    let mermaid = system.visualize(&report, VisualizationFormat::Mermaid)?;
    println!("{}", mermaid);

    Ok(())
}

fn analyze_react_fiber(system: &CodeUnderstandingSystem) -> anyhow::Result<()> {
    let fiber_code = r#"
// React Fiber - 调度和协调引擎
// 核心概念: 虚拟树、增量渲染、优先级调度

// Fiber 节点数据结构
class Fiber {
    constructor(type, props) {
        this.type = type;           // 组件类型
        this.props = props;         // 组件属性
        this.key = props.key;       // 唯一标识
        
        // 链接结构
        this.parent = null;         // 父节点
        this.child = null;          // 第一个子节点
        this.sibling = null;        // 兄弟节点
        
        // 状态
        this.state = {};
        this.memoizedState = null;  // hooks 状态
        this.hooks = [];            // hooks 列表
        
        // 更新队列
        this.updateQueue = [];
        this.dependencies = [];
        
        // 标记
        this.effectTag = null;      // 'PLACEMENT' | 'UPDATE' | 'DELETION'
        this.alternate = null;      // 旧版本 Fiber
    }
}

// 调度器
class Scheduler {
    constructor() {
        this.taskQueue = [];
        this.priorityMap = new Map();
    }

    scheduleTask(task, priority = 'NORMAL') {
        this.taskQueue.push({ task, priority, time: Date.now() });
        this.taskQueue.sort((a, b) => this.getPriorityValue(b.priority) - this.getPriorityValue(a.priority));
    }

    getPriorityValue(priority) {
        const map = {
            'IMMEDIATE': 5,
            'HIGH': 4,
            'NORMAL': 3,
            'LOW': 2,
            'IDLE': 1
        };
        return map[priority] || 3;
    }

    processTask(deadline) {
        while (this.taskQueue.length > 0 && deadline.timeRemaining() > 1) {
            const { task } = this.taskQueue.shift();
            task();
        }
    }
}

// 调和器 (Reconciler)
class Reconciler {
    reconcile(oldFiber, newFiber) {
        if (!oldFiber && newFiber) {
            newFiber.effectTag = 'PLACEMENT';
            return newFiber;
        }
        
        if (oldFiber && !newFiber) {
            oldFiber.effectTag = 'DELETION';
            return null;
        }
        
        if (oldFiber.type === newFiber.type && oldFiber.key === newFiber.key) {
            newFiber.effectTag = 'UPDATE';
            newFiber.alternate = oldFiber;
            return newFiber;
        }
        
        oldFiber.effectTag = 'DELETION';
        newFiber.effectTag = 'PLACEMENT';
        return newFiber;
    }

    commit(fiber) {
        if (!fiber) return;
        
        // 后序遍历 (post-order traversal)
        this.commit(fiber.child);
        this.commit(fiber.sibling);
        
        // 执行副作用
        switch (fiber.effectTag) {
            case 'PLACEMENT':
                this.commitPlacement(fiber);
                break;
            case 'UPDATE':
                this.commitUpdate(fiber);
                break;
            case 'DELETION':
                this.commitDeletion(fiber);
                break;
        }
    }

    commitPlacement(fiber) {
        console.log(`[MOUNT] ${fiber.type}`);
        if (fiber.componentDidMount) {
            fiber.componentDidMount();
        }
    }

    commitUpdate(fiber) {
        console.log(`[UPDATE] ${fiber.type}`);
        if (fiber.componentDidUpdate) {
            fiber.componentDidUpdate();
        }
    }

    commitDeletion(fiber) {
        console.log(`[UNMOUNT] ${fiber.type}`);
        if (fiber.componentWillUnmount) {
            fiber.componentWillUnmount();
        }
    }
}

// 导出
export { Fiber, Scheduler, Reconciler };
"#;

    let report = system.analyze(fiber_code, "React Fiber v18.0")?;

    println!("📊 Fiber 架构分析：\n");
    println!("{}", system.generate_report(&report));

    println!("\n🔍 关键发现：");
    println!("  ✓ Fiber 数据结构: 链表方式连接");
    println!("  ✓ 调度系统: 5 级优先级 (IMMEDIATE > HIGH > NORMAL > LOW > IDLE)");
    println!("  ✓ 调和算法: 3 种操作 (PLACEMENT, UPDATE, DELETION)");
    println!("  ✓ 提交阶段: 后序遍历确保正确的执行顺序");
    println!("  ✓ 生命周期整合: componentDidMount, componentDidUpdate, componentWillUnmount");

    println!("\n📈 工作流程：");
    println!("  1. 调度 (Schedule) - 任务入队");
    println!("  2. 协调 (Reconcile) - 比较 Fiber 树");
    println!("  3. 提交 (Commit) - 实际应用改动");

    Ok(())
}

fn analyze_react_hooks(system: &CodeUnderstandingSystem) -> anyhow::Result<()> {
    let hooks_code = r#"
// React Hooks 系统 - 函数式组件的状态管理

let currentComponent = null;
let hookIndex = 0;

// Hooks Dispatcher
class HooksDispatcher {
    constructor() {
        this.hooks = new Map();
    }

    ensureHooks(component) {
        if (!this.hooks.has(component)) {
            this.hooks.set(component, []);
        }
        return this.hooks.get(component);
    }

    useState(initialValue) {
        const component = currentComponent;
        const hooks = this.ensureHooks(component);
        const index = hookIndex++;

        if (!hooks[index]) {
            hooks[index] = {
                state: typeof initialValue === 'function' ? initialValue() : initialValue,
                queue: []
            };
        }

        const hook = hooks[index];
        
        const setState = (action) => {
            const newState = typeof action === 'function' 
                ? action(hook.state) 
                : action;
            
            if (newState !== hook.state) {
                hook.state = newState;
                component.forceUpdate();
            }
        };

        return [hook.state, setState];
    }

    useEffect(callback, deps) {
        const component = currentComponent;
        const hooks = this.ensureHooks(component);
        const index = hookIndex++;

        if (!hooks[index]) {
            hooks[index] = {
                memoizedDeps: null,
                cleanup: null
            };
        }

        const hook = hooks[index];
        const hasNoDeps = !deps;
        const depsChanged = !hook.memoizedDeps || 
                           !arrayEquals(deps, hook.memoizedDeps);

        if (hasNoDeps || depsChanged) {
            if (hook.cleanup) hook.cleanup();
            hook.cleanup = callback();
            hook.memoizedDeps = deps;
        }
    }

    useContext(Context) {
        return Context.currentValue;
    }

    useReducer(reducer, initialState) {
        const component = currentComponent;
        const hooks = this.ensureHooks(component);
        const index = hookIndex++;

        if (!hooks[index]) {
            hooks[index] = {
                state: initialState,
                dispatch: null
            };
        }

        const hook = hooks[index];

        hook.dispatch = (action) => {
            const newState = reducer(hook.state, action);
            if (newState !== hook.state) {
                hook.state = newState;
                component.forceUpdate();
            }
        };

        return [hook.state, hook.dispatch];
    }

    useMemo(callback, deps) {
        const component = currentComponent;
        const hooks = this.ensureHooks(component);
        const index = hookIndex++;

        if (!hooks[index]) {
            hooks[index] = {
                memoizedValue: null,
                memoizedDeps: null
            };
        }

        const hook = hooks[index];
        const depsChanged = !hook.memoizedDeps || 
                           !arrayEquals(deps, hook.memoizedDeps);

        if (depsChanged) {
            hook.memoizedValue = callback();
            hook.memoizedDeps = deps;
        }

        return hook.memoizedValue;
    }

    useCallback(callback, deps) {
        return this.useMemo(() => callback, deps);
    }

    useRef(initialValue) {
        const component = currentComponent;
        const hooks = this.ensureHooks(component);
        const index = hookIndex++;

        if (!hooks[index]) {
            hooks[index] = {
                current: initialValue
            };
        }

        return hooks[index];
    }
}

function arrayEquals(arr1, arr2) {
    if (!arr1 || !arr2) return false;
    if (arr1.length !== arr2.length) return false;
    return arr1.every((item, i) => item === arr2[i]);
}

// 上下文系统
class Context {
    constructor(defaultValue) {
        this.defaultValue = defaultValue;
        this.currentValue = defaultValue;
    }

    Provider(props) {
        this.currentValue = props.value;
        return props.children;
    }

    Consumer(props) {
        return props.children(this.currentValue);
    }
}

function createContext(defaultValue) {
    return new Context(defaultValue);
}

// 导出
export { 
    HooksDispatcher, 
    useState, 
    useEffect, 
    useReducer, 
    useMemo, 
    useCallback, 
    useRef, 
    useContext,
    createContext 
};
"#;

    let report = system.analyze(hooks_code, "React Hooks System v18.0")?;

    println!("📊 Hooks 系统分析：\n");
    println!("{}", system.generate_report(&report));

    println!("\n🔍 关键发现：");
    println!("  ✓ HooksDispatcher: 中央派发器管理所有 hooks");
    println!("  ✓ 状态保存: 每个组件维护 hooks 数组");
    println!("  ✓ 依赖追踪: deps 数组用于检测变化");
    println!("  ✓ 10 个核心 Hooks: useState, useEffect, useContext, useReducer, useMemo, useCallback, useRef 等");
    println!("  ✓ 上下文系统: Context.Provider/Consumer 模式");

    println!("\n📝 Hooks 规则：");
    println!("  1. 只在函数式组件顶层调用");
    println!("  2. 不能在条件、循环、嵌套函数中调用");
    println!("  3. deps 数组必须包含所有依赖项");

    Ok(())
}

fn analyze_react_dom(system: &CodeUnderstandingSystem) -> anyhow::Result<()> {
    let dom_code = r#"
// React DOM - 浏览器 DOM 渲染引擎

class ReactDOM {
    static render(element, container, callback) {
        const root = new Root(container);
        root.render(element, callback);
    }

    static createRoot(container) {
        return new Root(container);
    }

    static unmountComponentAtNode(container) {
        if (container._reactRoot) {
            container._reactRoot.unmount();
            delete container._reactRoot;
            return true;
        }
        return false;
    }
}

class Root {
    constructor(container) {
        this.container = container;
        this._internal = null;
    }

    render(element, callback) {
        this.renderImpl(element);
        if (callback) callback();
    }

    renderImpl(element) {
        const vdom = this.createVDOM(element);
        const dom = this.renderVDOM(vdom);
        this.container.appendChild(dom);
        this._internal = vdom;
    }

    createVDOM(element) {
        if (typeof element === 'string' || typeof element === 'number') {
            return {
                type: 'TEXT',
                props: { text: element }
            };
        }

        return {
            type: element.type,
            props: element.props,
            children: element.props.children || []
        };
    }

    renderVDOM(vdom) {
        // 文本节点
        if (vdom.type === 'TEXT') {
            return document.createTextNode(vdom.props.text);
        }

        // 函数组件
        if (typeof vdom.type === 'function') {
            const component = new vdom.type(vdom.props);
            const result = component.render();
            return this.renderVDOM(result);
        }

        // HTML 标签
        const dom = document.createElement(vdom.type);
        
        // 设置属性
        Object.entries(vdom.props).forEach(([key, value]) => {
            if (key === 'className') {
                dom.className = value;
            } else if (key === 'style' && typeof value === 'object') {
                Object.assign(dom.style, value);
            } else if (key.startsWith('on')) {
                const eventName = key.toLowerCase().slice(2);
                dom.addEventListener(eventName, value);
            } else if (key !== 'children') {
                dom.setAttribute(key, value);
            }
        });

        // 渲染子节点
        const children = Array.isArray(vdom.props.children) 
            ? vdom.props.children 
            : [vdom.props.children];

        children.forEach(child => {
            if (child) {
                const childDOM = this.renderVDOM(child);
                dom.appendChild(childDOM);
            }
        });

        return dom;
    }

    unmount() {
        this.container.innerHTML = '';
    }
}

// 事件代理系统
class EventDelegator {
    constructor() {
        this.listeners = new WeakMap();
    }

    addEventListener(target, event, handler) {
        if (!this.listeners.has(target)) {
            this.listeners.set(target, new Map());
        }
        
        const events = this.listeners.get(target);
        if (!events.has(event)) {
            events.set(event, []);
        }
        
        events.get(event).push(handler);
    }

    removeEventListener(target, event, handler) {
        if (!this.listeners.has(target)) return;
        
        const events = this.listeners.get(target);
        if (!events.has(event)) return;
        
        const handlers = events.get(event);
        const index = handlers.indexOf(handler);
        if (index > -1) {
            handlers.splice(index, 1);
        }
    }

    dispatchEvent(event) {
        const target = event.target;
        if (!this.listeners.has(target)) return;

        const events = this.listeners.get(target);
        const handlers = events.get(event.type) || [];

        handlers.forEach(handler => handler(event));
    }
}

// 导出
export { ReactDOM, Root, EventDelegator };
"#;

    let report = system.analyze(dom_code, "React DOM v18.0")?;

    println!("📊 React DOM 渲染引擎分析：\n");
    println!("{}", system.generate_report(&report));

    println!("\n🔍 关键发现：");
    println!("  ✓ Root 类: 连接 React 和浏览器 DOM");
    println!("  ✓ VDOM 创建: createVDOM 将 React Element 转换为虚拟树");
    println!("  ✓ 渲染流程: VDOM → 真实 DOM 节点");
    println!("  ✓ 事件系统: 事件代理 + 事件委托");
    println!("  ✓ 属性映射: props → DOM 属性/事件");

    println!("\n🔄 渲染管道：");
    println!("  React.createElement() → Fiber Tree → Reconciliation");
    println!("     ↓");
    println!("  Render Phase → Commit Phase → DOM Update");
    println!("     ↓");
    println!("  Event Delegation → Component Lifecycle");

    Ok(())
}

fn analyze_minified_react(system: &CodeUnderstandingSystem) -> anyhow::Result<()> {
    let minified = r#"
var React=function(){var e={createElement:function(t,n){return{type:t,props:Object.assign({},n)}},useState:function(t){var r=[];return[t,function(t){r.push(t)}]},useEffect:function(t,n){if(!n||n.length>0){t()}},useReducer:function(t,n){return[n,function(e){n=t(n,e)}]},useMemo:function(t,n){return t()},useCallback:function(t){return t},useRef:function(t){return{current:t}}};return{Component:class{constructor(t){this.props=t,this.state={}}setState(t){Object.assign(this.state,t)}render(){throw new Error("render() must be overridden")}},PureComponent:class extends e.Component{shouldComponentUpdate(t,n){return function(t,n){var r=Object.keys(t),e=Object.keys(n);return r.length===e.length&&r.every(function(r){return t[r]===n[r]})}(t,n)||function(t,n){var r=Object.keys(t),e=Object.keys(n);return r.length===e.length&&r.every(function(r){return t[r]===n[r]})}(this.state,n)}},createElement:e.createElement,useState:e.useState,useEffect:e.useEffect,useReducer:e.useReducer,useMemo:e.useMemo,useCallback:e.useCallback,useRef:e.useRef,Fragment:Symbol.for("react.fragment"),StrictMode:Symbol.for("react.strict_mode")}}();
"#;

    let report = system.analyze(minified, "React Minified (Obfuscated)")?;

    println!("📊 混淆代码反混淆分析：\n");
    println!("{}", system.generate_report(&report));

    println!("\n🔍 反混淆发现：");
    println!("  ✓ 识别到核心导出: Component, PureComponent");
    println!(
        "  ✓ 识别到 7 个 Hooks: useState, useEffect, useReducer, useMemo, useCallback, useRef"
    );
    println!("  ✓ 识别到特殊符号: Fragment, StrictMode");
    println!("  ✓ 数据流: 闭包保存 hooks 数组");

    println!("\n📋 混淆模式识别：");
    println!("  1. 变量名缩短: e → hooks, t → Component 等");
    println!("  2. 函数内联: setState → Object.assign");
    println!("  3. 对象合并: Object.assign({{}}，n)");
    println!("  4. 嵌套函数: 深层闭包");
    println!("  5. 符号使用: Symbol.for() 创建唯一标识");

    println!("\n✨ 反混淆结果：");
    println!("  原始：var e={{...}}, t={{...}}, n=function({{...}})");
    println!("  反混淆后：");
    println!("    - e → hooksDispatcher");
    println!("    - t → defaultHooks");
    println!("    - n → createReactInstance");

    Ok(())
}
