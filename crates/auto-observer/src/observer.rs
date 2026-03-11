//! 行为观察器
//! 在页面中注入JavaScript来观察运行时行为

use crate::*;

/// 行为观察器
pub struct BehaviorObserver {
    observation_script: String,
}

impl BehaviorObserver {
    pub fn new() -> Self {
        Self {
            observation_script: Self::generate_observation_script(),
        }
    }

    /// 生成观察脚本
    fn generate_observation_script() -> String {
        r#"
(function() {
    'use strict';
    
    // 观察数据存储
    window.__behaviorObservations = [];
    
    // 工具函数
    function getElementInfo(element) {
        if (!element) return null;
        
        const rect = element.getBoundingClientRect();
        return {
            tag: element.tagName.toLowerCase(),
            id: element.id || null,
            classes: Array.from(element.classList),
            attributes: Object.fromEntries(
                Array.from(element.attributes).map(a => [a.name, a.value])
            ),
            text: element.textContent?.substring(0, 100),
            selector: getUniqueSelector(element),
            boundingBox: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            },
            isVisible: isVisible(element),
            isInteractive: isInteractive(element)
        };
    }
    
    function getUniqueSelector(element) {
        if (element.id) return '#' + element.id;
        if (element.dataset.testid) return `[data-testid="${element.dataset.testid}"]`;
        
        let selector = element.tagName.toLowerCase();
        if (element.className) {
            selector += '.' + Array.from(element.classList).join('.');
        }
        
        // 添加nth-child以提高唯一性
        const siblings = Array.from(element.parentElement?.children || []);
        const index = siblings.indexOf(element) + 1;
        if (siblings.length > 1) {
            selector += `:nth-child(${index})`;
        }
        
        return selector;
    }
    
    function isVisible(element) {
        const style = window.getComputedStyle(element);
        return style.display !== 'none' && 
               style.visibility !== 'hidden' && 
               style.opacity !== '0';
    }
    
    function isInteractive(element) {
        const interactiveTags = ['button', 'a', 'input', 'select', 'textarea'];
        const interactiveRoles = ['button', 'link', 'menuitem', 'tab'];
        
        if (interactiveTags.includes(element.tagName.toLowerCase())) return true;
        
        const role = element.getAttribute('role');
        if (role && interactiveRoles.includes(role)) return true;
        
        if (element.onclick || element.getAttribute('onclick')) return true;
        
        return false;
    }
    
    function recordObservation(type, target, details = {}) {
        const observation = {
            timestamp: Date.now(),
            type: type,
            target: getElementInfo(target),
            url: window.location.href,
            details: details,
            pageState: capturePageState()
        };
        
        window.__behaviorObservations.push(observation);
        
        // 触发自定义事件供外部监听
        window.dispatchEvent(new CustomEvent('behavior-observed', { 
            detail: observation 
        }));
    }
    
    function capturePageState() {
        return {
            url: window.location.href,
            title: document.title,
            scrollX: window.scrollX,
            scrollY: window.scrollY,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            }
        };
    }
    
    // 监听所有事件类型
    const eventTypes = [
        'click', 'dblclick', 'mousedown', 'mouseup',
        'keydown', 'keyup', 'keypress',
        'focus', 'blur', 'change', 'input',
        'submit', 'scroll', 'resize',
        'mouseenter', 'mouseleave', 'mouseover', 'mouseout',
        'touchstart', 'touchend', 'touchmove',
        'dragstart', 'dragend', 'dragover', 'drop'
    ];
    
    eventTypes.forEach(type => {
        document.addEventListener(type, function(e) {
            const details = {
                bubbles: e.bubbles,
                cancelable: e.cancelable,
                composed: e.composed,
                defaultPrevented: e.defaultPrevented
            };
            
            // 添加特定事件类型的详情
            if (e instanceof MouseEvent) {
                details.clientX = e.clientX;
                details.clientY = e.clientY;
                details.button = e.button;
            }
            
            if (e instanceof KeyboardEvent) {
                details.key = e.key;
                details.code = e.code;
                details.ctrlKey = e.ctrlKey;
                details.altKey = e.altKey;
                details.shiftKey = e.shiftKey;
                details.metaKey = e.metaKey;
            }
            
            if (e instanceof InputEvent || type === 'input' || type === 'change') {
                details.value = e.target.value;
                details.inputType = e.inputType;
            }
            
            recordObservation(type, e.target, details);
        }, true);
    });
    
    // 观察DOM变化
    const observer = new MutationObserver((mutations) => {
        mutations.forEach(mutation => {
            const details = {
                type: mutation.type,
                target: getElementInfo(mutation.target)
            };
            
            if (mutation.type === 'childList') {
                details.addedCount = mutation.addedNodes.length;
                details.removedCount = mutation.removedNodes.length;
                details.addedNodes = Array.from(mutation.addedNodes)
                    .filter(n => n.nodeType === 1)
                    .map(n => getElementInfo(n));
            } else if (mutation.type === 'attributes') {
                details.attributeName = mutation.attributeName;
                details.oldValue = mutation.oldValue;
                details.newValue = mutation.target.getAttribute(mutation.attributeName);
            }
            
            recordObservation('mutation', mutation.target, details);
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeOldValue: true,
        characterData: true
    });
    
    // 拦截fetch
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        const [url, options] = args;
        
        recordObservation('fetch', null, {
            url: url,
            method: options?.method || 'GET',
            headers: options?.headers
        });
        
        return originalFetch.apply(this, args).then(response => {
            recordObservation('fetch-response', null, {
                url: url,
                status: response.status,
                statusText: response.statusText
            });
            return response;
        }).catch(error => {
            recordObservation('fetch-error', null, {
                url: url,
                error: error.message
            });
            throw error;
        });
    };
    
    // 拦截XMLHttpRequest
    const OriginalXHR = window.XMLHttpRequest;
    window.XMLHttpRequest = function() {
        const xhr = new OriginalXHR();
        let requestInfo = {};
        
        const originalOpen = xhr.open;
        xhr.open = function(method, url) {
            requestInfo = { method, url };
            return originalOpen.apply(this, arguments);
        };
        
        const originalSend = xhr.send;
        xhr.send = function(body) {
            recordObservation('xhr', null, {
                method: requestInfo.method,
                url: requestInfo.url,
                body: body ? 'present' : 'none'
            });
            
            xhr.addEventListener('load', () => {
                recordObservation('xhr-response', null, {
                    url: requestInfo.url,
                    status: xhr.status,
                    statusText: xhr.statusText
                });
            });
            
            xhr.addEventListener('error', () => {
                recordObservation('xhr-error', null, {
                    url: requestInfo.url
                });
            });
            
            return originalSend.apply(this, arguments);
        };
        
        return xhr;
    };
    
    // 监听URL变化
    let lastUrl = location.href;
    new MutationObserver(() => {
        const url = location.href;
        if (url !== lastUrl) {
            recordObservation('navigation', null, {
                from: lastUrl,
                to: url
            });
            lastUrl = url;
        }
    }).observe(document, { subtree: true, childList: true });
    
    // 拦截history API
    const originalPushState = history.pushState;
    history.pushState = function(...args) {
        recordObservation('history-pushstate', null, {
            state: args[0],
            title: args[1],
            url: args[2]
        });
        return originalPushState.apply(this, args);
    };
    
    const originalReplaceState = history.replaceState;
    history.replaceState = function(...args) {
        recordObservation('history-replacestate', null, {
            state: args[0],
            title: args[1],
            url: args[2]
        });
        return originalReplaceState.apply(this, args);
    };
    
    // 监听错误
    window.addEventListener('error', (e) => {
        recordObservation('error', e.target, {
            message: e.message,
            filename: e.filename,
            lineno: e.lineno,
            colno: e.colno
        });
    });
    
    window.addEventListener('unhandledrejection', (e) => {
        recordObservation('unhandledrejection', null, {
            reason: e.reason?.toString()
        });
    });
    
    // 提供API供外部使用
    window.BehaviorObserver = {
        getObservations: () => window.__behaviorObservations,
        clearObservations: () => { window.__behaviorObservations = []; },
        getSummary: () => ({
            total: window.__behaviorObservations.length,
            byType: window.__behaviorObservations.reduce((acc, o) => {
                acc[o.type] = (acc[o.type] || 0) + 1;
                return acc;
            }, {}),
            uniqueElements: new Set(
                window.__behaviorObservations.map(o => o.target?.selector).filter(Boolean)
            ).size
        })
    };
    
    console.log('[BehaviorObserver] 初始化完成');
})();
"#
        .to_string()
    }

    /// 获取观察脚本
    pub fn get_script(&self) -> &str {
        &self.observation_script
    }

    /// 分析观察数据
    pub fn analyze_observations(&self, observations: &[serde_json::Value]) -> ObservationAnalysis {
        let mut analysis = ObservationAnalysis::default();

        for obs in observations {
            let event_type = obs
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            *analysis
                .event_counts
                .entry(event_type.to_string())
                .or_insert(0) += 1;

            // 分析点击行为
            if event_type == "click" {
                analysis.click_patterns.push(self.analyze_click(obs));
            }

            // 分析输入行为
            if event_type == "input" {
                analysis.input_patterns.push(self.analyze_input(obs));
            }

            // 分析导航
            if event_type == "navigation" || event_type == "history-pushstate" {
                if let Some(to) = obs
                    .get("details")
                    .and_then(|d| d.get("to"))
                    .and_then(|v| v.as_str())
                {
                    analysis.navigation_targets.push(to.to_string());
                }
            }

            // 分析网络请求
            if event_type == "fetch" || event_type == "xhr" {
                analysis.api_endpoints.push(self.extract_endpoint(obs));
            }

            // 分析DOM变化
            if event_type == "mutation" {
                analysis.dom_mutations.push(self.analyze_mutation(obs));
            }
        }

        analysis
    }

    fn analyze_click(&self, obs: &serde_json::Value) -> ClickPattern {
        ClickPattern {
            target_selector: obs
                .get("target")
                .and_then(|t| t.get("selector"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            timestamp: obs.get("timestamp").and_then(|v| v.as_u64()).unwrap_or(0),
            coordinates: obs
                .get("details")
                .and_then(|d| Some((d.get("clientX")?, d.get("clientY")?)))
                .and_then(|(x, y)| Some((x.as_f64()?, y.as_f64()?))),
        }
    }

    fn analyze_input(&self, obs: &serde_json::Value) -> InputPattern {
        InputPattern {
            target_selector: obs
                .get("target")
                .and_then(|t| t.get("selector"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            input_type: obs
                .get("details")
                .and_then(|d| d.get("inputType"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            has_value: obs.get("details").and_then(|d| d.get("value")).is_some(),
        }
    }

    fn extract_endpoint(&self, obs: &serde_json::Value) -> ApiEndpoint {
        ApiEndpoint {
            url: obs
                .get("details")
                .and_then(|d| d.get("url"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_default(),
            method: obs
                .get("details")
                .and_then(|d| d.get("method"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "GET".to_string()),
        }
    }

    fn analyze_mutation(&self, obs: &serde_json::Value) -> DomMutation {
        DomMutation {
            mutation_type: obs
                .get("details")
                .and_then(|d| d.get("type"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_default(),
            target: obs
                .get("details")
                .and_then(|d| d.get("target"))
                .and_then(|t| t.get("selector"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            added_nodes: obs
                .get("details")
                .and_then(|d| d.get("addedCount"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
        }
    }
}

impl Default for BehaviorObserver {
    fn default() -> Self {
        Self::new()
    }
}

/// 观察分析结果
#[derive(Debug, Default)]
pub struct ObservationAnalysis {
    pub event_counts: HashMap<String, usize>,
    pub click_patterns: Vec<ClickPattern>,
    pub input_patterns: Vec<InputPattern>,
    pub navigation_targets: Vec<String>,
    pub api_endpoints: Vec<ApiEndpoint>,
    pub dom_mutations: Vec<DomMutation>,
}

#[derive(Debug)]
pub struct ClickPattern {
    pub target_selector: Option<String>,
    pub timestamp: u64,
    pub coordinates: Option<(f64, f64)>,
}

#[derive(Debug)]
pub struct InputPattern {
    pub target_selector: Option<String>,
    pub input_type: Option<String>,
    pub has_value: bool,
}

#[derive(Debug)]
pub struct ApiEndpoint {
    pub url: String,
    pub method: String,
}

#[derive(Debug)]
pub struct DomMutation {
    pub mutation_type: String,
    pub target: Option<String>,
    pub added_nodes: usize,
}
