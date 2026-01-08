//! 开源项目分析演示
//!
//! 这个示例展示如何使用代码理解系统分析真实的开源项目
//! （这里以常见库为例）
//!
//! 运行: cargo run --example analyze_opensource_projects

use browerai_learning::CodeUnderstandingSystem;

fn main() -> anyhow::Result<()> {
    println!("🔬 开源项目深度分析演示");
    println!("=============================================\n");

    // 示例 1: 分析类似 Lodash 的工具库
    analyze_lodash_like()?;

    println!("\n");

    // 示例 2: 分析类似 Express 的框架
    analyze_express_like()?;

    println!("\n");

    // 示例 3: 分析类似 Vue 的前端框架
    analyze_vue_like()?;

    Ok(())
}

fn analyze_lodash_like() -> anyhow::Result<()> {
    println!("📚 示例 1: Lodash-Like 工具库分析\n");

    let code = r#"
        // lodash-like 工具集
        
        export function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }

        export function throttle(func, limit) {
            let inThrottle;
            return function(...args) {
                if (!inThrottle) {
                    func.apply(this, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        }

        export function curry(func) {
            const arity = func.length;
            return function $curry(...args) {
                if (args.length < arity) {
                    return $curry.bind(null, ...args);
                }
                return func.call(null, ...args);
            };
        }

        export function compose(...funcs) {
            return x => funcs.reduceRight((acc, func) => func(acc), x);
        }

        export function pipe(...funcs) {
            return x => funcs.reduce((acc, func) => func(acc), x);
        }

        export function memoize(func, resolver) {
            const cache = new Map();
            return function memoized(...args) {
                const key = resolver ? resolver(...args) : JSON.stringify(args);
                if (cache.has(key)) {
                    return cache.get(key);
                }
                const result = func.apply(this, args);
                cache.set(key, result);
                return result;
            };
        }
    "#;

    let system = CodeUnderstandingSystem::new();
    let report = system.analyze(code, "lodash-like v4.17.0")?;

    println!("✅ 架构特征:");
    for char in &report.architecture.characteristics {
        println!("   • {}", char);
    }

    println!("\n✅ 导出的 API 函数库:");
    for (i, api) in report.apis.iter().enumerate() {
        println!("   {}. {}", i + 1, api.signature);
    }

    println!("\n✅ 代码统计:");
    println!("   • 函数数量: {}", report.statistics.function_count);
    println!("   • 复杂度等级: {}", report.statistics.complexity_level);

    println!("\n💡 分析结论:");
    println!("   这是一个纯函数工具库，提供高阶函数如:");
    println!("   - 控制流: debounce, throttle");
    println!("   - 函数组合: compose, pipe");
    println!("   - 函数变换: curry, memoize");
    println!("   没有内部依赖，完全模块化设计");

    Ok(())
}

fn analyze_express_like() -> anyhow::Result<()> {
    println!("🚀 示例 2: Express-Like 服务器框架分析\n");

    let code = r#"
        // express-like 框架核心
        
        import { EventEmitter } from 'events';
        import { Router } from './router.js';
        import { Layer } from './layer.js';

        export class Application extends EventEmitter {
            constructor() {
                super();
                this.router = new Router();
                this.layers = [];
                this.settings = {};
            }

            use(path, handler) {
                const layer = new Layer(path, {
                    sensitive: this.get('case sensitive routing'),
                    strict: this.get('strict routing')
                }, handler);
                this.layers.push(layer);
                return this;
            }

            get(path, ...handlers) {
                return this.route(path).get(...handlers);
            }

            post(path, ...handlers) {
                return this.route(path).post(...handlers);
            }

            route(path) {
                return this.router.route(path);
            }

            async handle(req, res) {
                let layerIndex = 0;
                
                const next = async () => {
                    if (layerIndex >= this.layers.length) {
                        return;
                    }
                    const layer = this.layers[layerIndex++];
                    if (layer.match(req.path)) {
                        await layer.handler(req, res, next);
                    } else {
                        await next();
                    }
                };
                
                await next();
            }

            listen(port, callback) {
                const server = require('http').createServer((req, res) => {
                    this.handle(req, res);
                });
                return server.listen(port, callback);
            }
        }

        export function createApplication() {
            return new Application();
        }
    "#;

    let system = CodeUnderstandingSystem::new();
    let report = system.analyze(code, "express-like v4.18.0")?;

    println!("✅ 架构特征:");
    for char in &report.architecture.characteristics {
        println!("   • {}", char);
    }

    println!("\n✅ 核心模块:");
    for module in &report.modules {
        println!("   📦 {}", module.name);
        println!("      职责: {}", module.responsibility);
        if !module.dependencies.is_empty() {
            println!("      依赖: {}", module.dependencies.join(", "));
        }
    }

    println!("\n✅ 数据流分析");
    println!("   发现 {} 条数据流", report.dataflows.len());
    for flow in report.dataflows.iter().take(5) {
        println!(
            "   • {} → {} ({})",
            flow.source, flow.target, flow.description
        );
    }

    println!("\n💡 分析结论:");
    println!("   这是一个事件驱动的 Web 框架:");
    println!("   - 核心: Application 类（继承 EventEmitter）");
    println!("   - 路由: 支持 GET/POST 等 HTTP 方法");
    println!("   - 中间件: 通过 use() 链式注册");
    println!("   - 请求处理: 异步中间件执行链");

    Ok(())
}

fn analyze_vue_like() -> anyhow::Result<()> {
    println!("⚛️  示例 3: Vue-Like 前端框架分析\n");

    let code = r#"
        // vue-like 框架核心
        
        export class Component {
            constructor(options) {
                this.data = typeof options.data === 'function' 
                    ? options.data() 
                    : options.data || {};
                
                this.computed = options.computed || {};
                this.methods = options.methods || {};
                this.watchers = options.watch || {};
                this.el = options.el;
                
                this.init();
            }

            init() {
                this.setupReactivity();
                this.setupComputedProperties();
                this.setupWatchers();
                this.mount();
            }

            setupReactivity() {
                this.data = new Proxy(this.data, {
                    set: (target, key, value) => {
                        target[key] = value;
                        this.update();
                        return true;
                    }
                });
            }

            setupComputedProperties() {
                for (const [key, getter] of Object.entries(this.computed)) {
                    Object.defineProperty(this, key, {
                        get: () => getter.call(this)
                    });
                }
            }

            setupWatchers() {
                for (const [key, callback] of Object.entries(this.watchers)) {
                    this.watch(key, callback);
                }
            }

            watch(key, callback) {
                let prevValue = this.data[key];
                this.watchers[key] = () => {
                    const newValue = this.data[key];
                    if (newValue !== prevValue) {
                        callback(newValue, prevValue);
                        prevValue = newValue;
                    }
                };
            }

            update() {
                this.render();
            }

            render() {
                if (this.el && this.$render) {
                    this.el.innerHTML = this.$render();
                }
            }

            mount() {
                const template = document.querySelector(this.el);
                if (template) {
                    this.render();
                }
            }
        }

        export function createComponent(options) {
            return new Component(options);
        }

        export class VueApp {
            constructor() {
                this.components = new Map();
                this.globalData = {};
            }

            component(name, definition) {
                this.components.set(name, definition);
                return this;
            }

            mount(selector) {
                const root = document.querySelector(selector);
                const app = new Component({ el: selector });
                return app;
            }
        }
    "#;

    let system = CodeUnderstandingSystem::new();
    let report = system.analyze(code, "vue-like v3.3.0")?;

    println!("✅ 架构特征:");
    for char in &report.architecture.characteristics {
        println!("   • {}", char);
    }

    println!("\n✅ 核心类设计:");
    for module in &report.modules {
        if module.name.contains("Component") || module.name.contains("App") {
            println!("   🏗️  {}", module.name);
            println!("      职责: {}", module.responsibility);
            if !module.functions.is_empty() {
                println!(
                    "      方法: {}",
                    module
                        .functions
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
    }

    println!("\n✅ MVVM 架构组件:");
    println!("   • 响应式数据 (Reactive Data)");
    println!("   • 计算属性 (Computed Properties)");
    println!("   • 监听器 (Watchers)");
    println!("   • 组件生命周期 (Lifecycle)");
    println!("   • 模板渲染 (Template Rendering)");

    println!("\n💡 分析结论:");
    println!("   这是一个 MVVM 前端框架:");
    println!("   - Model: 响应式 data 对象");
    println!("   - View: DOM 模板和渲染");
    println!("   - ViewModel: Component 类连接 M 和 V");
    println!("   - 核心特性: Proxy 响应性、Computed、Watchers");

    Ok(())
}
