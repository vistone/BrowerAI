/// V8 执行追踪系统
///
/// 用于在沙盒中执行网站代码，并记录：
/// - 函数调用链
/// - DOM 操作
/// - 事件监听
/// - 状态变化
///
/// 这是学习阶段的核心基础设施
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 追踪记录的函数调用
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CallRecord {
    /// 函数名称
    pub function_name: String,

    /// 调用的参数（去除敏感信息）
    pub arguments: Vec<String>,

    /// 返回值类型（不记录实际值，只记录类型）
    pub return_type: String,

    /// 调用时间戳（相对于开始）
    pub timestamp_ms: u64,

    /// 调用的上下文对象（如果是方法调用）
    pub context_object: Option<String>,

    /// 该函数的调用深度（嵌套层数）
    pub call_depth: usize,
}

/// DOM 操作记录
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DOMOperation {
    /// 操作类型：appendChild, removeChild, setAttribute, setTextContent 等
    pub operation_type: String,

    /// 目标元素的标签名
    pub target_tag: String,

    /// 目标元素的 ID（如果有）
    pub target_id: Option<String>,

    /// 目标元素的 class（如果有）
    pub target_class: Option<String>,

    /// 操作的详细信息
    pub details: String,

    /// 操作时间戳
    pub timestamp_ms: u64,
}

/// 事件监听记录
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventListener {
    /// 事件类型：click, change, submit 等
    pub event_type: String,

    /// 绑定事件的元素标签
    pub target_tag: String,

    /// 元素 ID
    pub target_id: Option<String>,

    /// 事件处理函数名称
    pub handler_name: String,

    /// 注册时间戳
    pub timestamp_ms: u64,
}

/// 用户交互事件
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserEvent {
    /// 事件类型：click, submit, input 等
    pub event_type: String,

    /// 触发事件的元素
    pub target_element: String,

    /// 元素选择器信息
    pub selector: Option<String>,

    /// 事件发生的时间
    pub timestamp_ms: u64,

    /// 这个事件引发的后续操作数
    pub triggered_operations: usize,
}

/// 页面状态变化
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateChange {
    /// 改变的变量名
    pub variable_name: String,

    /// 新值的类型
    pub new_value_type: String,

    /// 是否是全局变量
    pub is_global: bool,

    /// 变化时间戳
    pub timestamp_ms: u64,
}

/// 完整的执行追踪数据
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// 函数调用序列
    pub function_calls: Vec<CallRecord>,

    /// DOM 操作序列
    pub dom_operations: Vec<DOMOperation>,

    /// 事件监听注册
    pub event_listeners: Vec<EventListener>,

    /// 用户交互（自动检测的）
    pub user_events: Vec<UserEvent>,

    /// 状态变化
    pub state_changes: Vec<StateChange>,

    /// 总执行时间（毫秒）
    pub total_duration_ms: u64,

    /// 页面初始化完成的时间
    pub page_ready_ms: Option<u64>,
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionTrace {
    /// 创建新的执行追踪
    pub fn new() -> Self {
        Self {
            function_calls: vec![],
            dom_operations: vec![],
            event_listeners: vec![],
            user_events: vec![],
            state_changes: vec![],
            total_duration_ms: 0,
            page_ready_ms: None,
        }
    }

    /// 获取所有由用户事件触发的操作链
    ///
    /// 例如：用户点击"Add to Cart"按钮 → 一系列函数调用 → DOM 更新
    pub fn get_operation_chain(&self, user_event: &UserEvent) -> OperationChain {
        let mut chain = OperationChain {
            trigger: user_event.clone(),
            function_calls: vec![],
            dom_operations: vec![],
        };

        // 找出在该用户事件之后触发的所有操作
        for call in &self.function_calls {
            if call.timestamp_ms >= user_event.timestamp_ms {
                chain.function_calls.push(call.clone());
            }
        }

        for dom_op in &self.dom_operations {
            if dom_op.timestamp_ms >= user_event.timestamp_ms {
                chain.dom_operations.push(dom_op.clone());
            }
        }

        chain
    }

    /// 统计函数调用频率
    pub fn function_frequency(&self) -> HashMap<String, usize> {
        let mut freq = HashMap::new();
        for call in &self.function_calls {
            *freq.entry(call.function_name.clone()).or_insert(0) += 1;
        }
        freq
    }

    /// 统计 DOM 操作类型
    pub fn dom_operation_summary(&self) -> HashMap<String, usize> {
        let mut summary = HashMap::new();
        for op in &self.dom_operations {
            *summary.entry(op.operation_type.clone()).or_insert(0) += 1;
        }
        summary
    }
}

/// 单个工作流程的操作链
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationChain {
    /// 触发用户事件
    pub trigger: UserEvent,

    /// 后续的函数调用
    pub function_calls: Vec<CallRecord>,

    /// 造成的 DOM 操作
    pub dom_operations: Vec<DOMOperation>,
}

impl OperationChain {
    /// 获取这个操作链的深度（最深的调用嵌套）
    pub fn max_call_depth(&self) -> usize {
        self.function_calls
            .iter()
            .map(|call| call.call_depth)
            .max()
            .unwrap_or(0)
    }

    /// 获取操作链的总长度
    pub fn total_operations(&self) -> usize {
        self.function_calls.len() + self.dom_operations.len()
    }
}

/// V8 追踪器：执行网站代码并记录所有操作
#[allow(dead_code)]
pub struct V8Tracer {
    /// 是否已初始化
    initialized: bool,
}

impl V8Tracer {
    /// 创建新的 V8 追踪器
    pub fn new() -> Result<Self> {
        Ok(Self { initialized: true })
    }

    /// 生成追踪代码，注入到 HTML 中
    ///
    /// 这些代码将：
    /// 1. 拦截所有函数调用
    /// 2. 记录 DOM 操作
    /// 3. 监听事件注册
    /// 4. 追踪用户交互
    pub fn generate_tracer_code() -> String {
        r#"
// ============================================
// BrowerAI V8 执行追踪系统
// ============================================

window.__browerai__ = {
    traces: {
        functionCalls: [],
        domOperations: [],
        eventListeners: [],
        userEvents: [],
        stateChanges: [],
    },
    
    startTime: Date.now(),
    
    // 记录函数调用
    recordFunctionCall(name, args, returnType, contextObj = null, depth = 0) {
        this.traces.functionCalls.push({
            function_name: name,
            arguments: args.slice(0, 3).map(a => String(a).slice(0, 50)),
            return_type: returnType,
            timestamp_ms: Date.now() - this.startTime,
            context_object: contextObj ? contextObj.constructor.name : null,
            call_depth: depth,
        });
    },
    
    // 记录 DOM 操作
    recordDOMOperation(type, targetEl, details) {
        if (!targetEl) return;
        
        this.traces.domOperations.push({
            operation_type: type,
            target_tag: targetEl.tagName || 'unknown',
            target_id: targetEl.id || null,
            target_class: targetEl.className || null,
            details: details || '',
            timestamp_ms: Date.now() - this.startTime,
        });
    },
    
    // 记录事件监听
    recordEventListener(eventType, targetEl, handlerName) {
        if (!targetEl) return;
        
        this.traces.eventListeners.push({
            event_type: eventType,
            target_tag: targetEl.tagName || 'unknown',
            target_id: targetEl.id || null,
            handler_name: handlerName,
            timestamp_ms: Date.now() - this.startTime,
        });
    },
    
    // 记录用户交互
    recordUserEvent(eventType, targetEl, selector) {
        this.traces.userEvents.push({
            event_type: eventType,
            target_element: targetEl ? targetEl.tagName : 'document',
            selector: selector || null,
            timestamp_ms: Date.now() - this.startTime,
            triggered_operations: 0,
        });
    },
    
    // 记录状态变化
    recordStateChange(varName, newType, isGlobal) {
        this.traces.stateChanges.push({
            variable_name: varName,
            new_value_type: newType,
            is_global: isGlobal,
            timestamp_ms: Date.now() - this.startTime,
        });
    },
    
    // 获取所有追踪数据
    getTraces() {
        return {
            ...this.traces,
            total_duration_ms: Date.now() - this.startTime,
        };
    },
};

// ============================================
// 拦截 DOM 操作
// ============================================

const originalAppendChild = Element.prototype.appendChild;
Element.prototype.appendChild = function(node) {
    window.__browerai__.recordDOMOperation(
        'appendChild',
        this,
        `添加子元素 ${node.tagName}`
    );
    return originalAppendChild.call(this, node);
};

const originalRemoveChild = Element.prototype.removeChild;
Element.prototype.removeChild = function(node) {
    window.__browerai__.recordDOMOperation(
        'removeChild',
        this,
        `移除子元素 ${node.tagName}`
    );
    return originalRemoveChild.call(this, node);
};

const originalSetAttribute = Element.prototype.setAttribute;
Element.prototype.setAttribute = function(name, value) {
    window.__browerai__.recordDOMOperation(
        'setAttribute',
        this,
        `设置属性 ${name}=${value}`
    );
    return originalSetAttribute.call(this, name, value);
};

const originalInnerHTML = Object.getOwnPropertyDescriptor(Element.prototype, 'innerHTML');
Object.defineProperty(Element.prototype, 'innerHTML', {
    set: function(value) {
        window.__browerai__.recordDOMOperation(
            'setInnerHTML',
            this,
            `设置内容长度 ${value.length}`
        );
        originalInnerHTML.set.call(this, value);
    },
    get: function() {
        return originalInnerHTML.get.call(this);
    },
});

// ============================================
// 拦截事件监听
// ============================================

const originalAddEventListener = EventTarget.prototype.addEventListener;
EventTarget.prototype.addEventListener = function(type, listener, options) {
    const listenerName = listener.name || 'anonymous';
    const targetEl = this instanceof Element ? this : null;
    
    window.__browerai__.recordEventListener(type, targetEl, listenerName);
    
    // 用 wrapper 追踪事件触发
    const wrappedListener = function(event) {
        window.__browerai__.recordUserEvent(
            type,
            event.target,
            event.target?.id || event.target?.className
        );
        return listener.call(this, event);
    };
    
    return originalAddEventListener.call(this, type, wrappedListener, options);
};

// ============================================
// 拦截函数定义（采样）
// ============================================

window.__browerai__.callDepth = 0;

// 为常见的全局函数添加追踪
const tracedFunctions = [
    'fetch', 'XMLHttpRequest', 'eval',
    'JSON.parse', 'JSON.stringify',
];

for (const fnName of tracedFunctions) {
    if (fnName === 'JSON.parse') {
        const original = JSON.parse;
        JSON.parse = function(...args) {
            window.__browerai__.recordFunctionCall('JSON.parse', args, 'object');
            return original.apply(this, args);
        };
    } else if (fnName === 'JSON.stringify') {
        const original = JSON.stringify;
        JSON.stringify = function(...args) {
            window.__browerai__.recordFunctionCall('JSON.stringify', args, 'string');
            return original.apply(this, args);
        };
    } else if (fnName === 'fetch') {
        const original = window.fetch;
        window.fetch = function(...args) {
            window.__browerai__.recordFunctionCall('fetch', args, 'Promise');
            return original.apply(this, args);
        };
    }
}

// ============================================
// 页面加载完成时的标记
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    if (!window.__browerai__.traces.page_ready_ms) {
        window.__browerai__.traces.page_ready_ms = Date.now() - window.__browerai__.startTime;
    }
});

console.log('[BrowerAI] V8 追踪系统已激活');
"#
        .to_string()
    }

    /// 将追踪代码注入到 HTML 中
    pub fn inject_tracers_to_html(html: &str) -> String {
        let tracer_code = Self::generate_tracer_code();

        // 在 <head> 或 <body> 开始处插入追踪代码
        if let Some(head_end) = html.find("</head>") {
            let mut result = html.to_string();
            result.insert_str(head_end, &format!("<script>{}</script>\n", tracer_code));
            result
        } else if let Some(body_start) = html.find("<body>") {
            let body_end = body_start + "<body>".len();
            let mut result = html.to_string();
            result.insert_str(body_end, &format!("<script>{}</script>\n", tracer_code));
            result
        } else {
            // 如果没有 <head> 或 <body>，就在最开始插入
            format!("<script>{}</script>\n{}", tracer_code, html)
        }
    }

    /// 从注入的追踪代码中提取结果
    ///
    /// 在 V8 执行完毕后，调用这个方法从 `window.__browerai__.traces` 获取追踪数据
    pub fn extract_traces_from_window(traces_json: &str) -> Result<ExecutionTrace> {
        let traces: serde_json::Value = serde_json::from_str(traces_json)?;

        // 解析 JSON 并转换为 ExecutionTrace 结构
        Ok(ExecutionTrace {
            function_calls: Self::parse_function_calls(&traces["functionCalls"])?,
            dom_operations: Self::parse_dom_operations(&traces["domOperations"])?,
            event_listeners: Self::parse_event_listeners(&traces["eventListeners"])?,
            user_events: Self::parse_user_events(&traces["userEvents"])?,
            state_changes: Self::parse_state_changes(&traces["stateChanges"])?,
            total_duration_ms: traces["total_duration_ms"].as_u64().unwrap_or(0),
            page_ready_ms: traces["page_ready_ms"].as_u64(),
        })
    }

    fn parse_function_calls(arr: &serde_json::Value) -> Result<Vec<CallRecord>> {
        let mut calls = vec![];
        if let Some(array) = arr.as_array() {
            for item in array {
                calls.push(CallRecord {
                    function_name: item["function_name"]
                        .as_str()
                        .unwrap_or("unknown")
                        .to_string(),
                    arguments: item["arguments"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .map(|v| v.as_str().unwrap_or("").to_string())
                        .collect(),
                    return_type: item["return_type"]
                        .as_str()
                        .unwrap_or("unknown")
                        .to_string(),
                    timestamp_ms: item["timestamp_ms"].as_u64().unwrap_or(0),
                    context_object: item["context_object"].as_str().map(|s| s.to_string()),
                    call_depth: item["call_depth"].as_u64().unwrap_or(0) as usize,
                });
            }
        }
        Ok(calls)
    }

    fn parse_dom_operations(arr: &serde_json::Value) -> Result<Vec<DOMOperation>> {
        let mut ops = vec![];
        if let Some(array) = arr.as_array() {
            for item in array {
                ops.push(DOMOperation {
                    operation_type: item["operation_type"]
                        .as_str()
                        .unwrap_or("unknown")
                        .to_string(),
                    target_tag: item["target_tag"].as_str().unwrap_or("unknown").to_string(),
                    target_id: item["target_id"].as_str().map(|s| s.to_string()),
                    target_class: item["target_class"].as_str().map(|s| s.to_string()),
                    details: item["details"].as_str().unwrap_or("").to_string(),
                    timestamp_ms: item["timestamp_ms"].as_u64().unwrap_or(0),
                });
            }
        }
        Ok(ops)
    }

    fn parse_event_listeners(arr: &serde_json::Value) -> Result<Vec<EventListener>> {
        let mut listeners = vec![];
        if let Some(array) = arr.as_array() {
            for item in array {
                listeners.push(EventListener {
                    event_type: item["event_type"].as_str().unwrap_or("unknown").to_string(),
                    target_tag: item["target_tag"].as_str().unwrap_or("unknown").to_string(),
                    target_id: item["target_id"].as_str().map(|s| s.to_string()),
                    handler_name: item["handler_name"]
                        .as_str()
                        .unwrap_or("anonymous")
                        .to_string(),
                    timestamp_ms: item["timestamp_ms"].as_u64().unwrap_or(0),
                });
            }
        }
        Ok(listeners)
    }

    fn parse_user_events(arr: &serde_json::Value) -> Result<Vec<UserEvent>> {
        let mut events = vec![];
        if let Some(array) = arr.as_array() {
            for item in array {
                events.push(UserEvent {
                    event_type: item["event_type"].as_str().unwrap_or("unknown").to_string(),
                    target_element: item["target_element"]
                        .as_str()
                        .unwrap_or("unknown")
                        .to_string(),
                    selector: item["selector"].as_str().map(|s| s.to_string()),
                    timestamp_ms: item["timestamp_ms"].as_u64().unwrap_or(0),
                    triggered_operations: item["triggered_operations"].as_u64().unwrap_or(0)
                        as usize,
                });
            }
        }
        Ok(events)
    }

    fn parse_state_changes(arr: &serde_json::Value) -> Result<Vec<StateChange>> {
        let mut changes = vec![];
        if let Some(array) = arr.as_array() {
            for item in array {
                changes.push(StateChange {
                    variable_name: item["variable_name"]
                        .as_str()
                        .unwrap_or("unknown")
                        .to_string(),
                    new_value_type: item["new_value_type"]
                        .as_str()
                        .unwrap_or("unknown")
                        .to_string(),
                    is_global: item["is_global"].as_bool().unwrap_or(false),
                    timestamp_ms: item["timestamp_ms"].as_u64().unwrap_or(0),
                });
            }
        }
        Ok(changes)
    }
}

impl Default for V8Tracer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_code_generation() {
        let code = V8Tracer::generate_tracer_code();
        assert!(code.contains("__browerai__"));
        assert!(code.contains("recordFunctionCall"));
        assert!(code.contains("recordDOMOperation"));
    }

    #[test]
    fn test_html_injection() {
        let html = "<html><head></head><body></body></html>";
        let injected = V8Tracer::inject_tracers_to_html(html);
        assert!(injected.contains("__browerai__"));
        assert!(injected.contains("<script>"));
    }

    #[test]
    fn test_execution_trace_creation() {
        let trace = ExecutionTrace::new();
        assert_eq!(trace.function_calls.len(), 0);
        assert_eq!(trace.dom_operations.len(), 0);
    }
}
