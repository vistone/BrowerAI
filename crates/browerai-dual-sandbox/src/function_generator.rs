//! 功能生成器 - 基于学习到的意图，重新实现等价功能
//!
//! 核心：保持功能行为一致，但用全新代码实现

use crate::js_understander::{
    FunctionIntents, InteractionIntent, BehaviorType, TriggerType,
    ApiIntent, HttpMethod
};
use crate::component_extractor::ComponentLibrary;

/// 功能实现生成器
pub struct FunctionGenerator {
    intents: FunctionIntents,
    _components: ComponentLibrary,
}

/// 生成的功能代码
#[derive(Debug, Clone)]
pub struct GeneratedFunctions {
    /// 事件处理代码
    pub event_handlers: String,
    /// API调用代码
    pub api_functions: String,
    /// 状态管理代码
    pub state_functions: String,
    /// 动画代码
    pub animation_functions: String,
    /// 工具函数
    pub utility_functions: String,
    /// 初始化代码
    pub init_code: String,
}

/// 目标框架
#[derive(Debug, Clone, Copy)]
pub enum TargetFramework {
    VanillaJs,  // 原生JavaScript
    React,      // React + Hooks
    Vue,        // Vue 3
    Svelte,     // Svelte
}

impl FunctionGenerator {
    pub fn new(intents: FunctionIntents, components: ComponentLibrary) -> Self {
        Self {
            intents,
            _components: components,
        }
    }

    /// 生成功能代码
    pub fn generate(&self, framework: TargetFramework) -> GeneratedFunctions {
        match framework {
            TargetFramework::VanillaJs => self.generate_vanilla_js(),
            TargetFramework::React => self.generate_react(),
            TargetFramework::Vue => self.generate_vue(),
            TargetFramework::Svelte => self.generate_svelte(),
        }
    }

    /// 生成原生JavaScript实现
    fn generate_vanilla_js(&self) -> GeneratedFunctions {
        GeneratedFunctions {
            event_handlers: self.generate_event_handlers(),
            api_functions: self.generate_api_functions(),
            state_functions: self.generate_state_functions(),
            animation_functions: self.generate_animations(),
            utility_functions: self.generate_utilities(),
            init_code: self.generate_init(),
        }
    }

    /// 生成事件处理器
    fn generate_event_handlers(&self) -> String {
        let mut code = String::new();
        
        code.push_str("// ============================================\n");
        code.push_str("// 事件处理器 - 从学习到的交互意图重新实现\n");
        code.push_str("// ============================================\n\n");
        
        for (i, intent) in self.intents.interactions.iter().enumerate() {
            code.push_str(&self.generate_single_handler(intent, i));
            code.push_str("\n\n");
        }
        
        // 如果没有学习到交互，生成默认的
        if self.intents.interactions.is_empty() {
            code.push_str(&self.generate_default_handlers());
        }
        
        code
    }

    fn generate_single_handler(&self, intent: &InteractionIntent, index: usize) -> String {
        let _func_name = format!("handle_interaction_{}", index);
        let _trigger = format!("{:?}", intent.trigger).to_lowercase();
        
        let mut code = format!(
            "// 处理 {:?} 事件\n",
            intent.behavior.behavior_type
        );
        
        // 生成事件监听
        code.push_str(&format!(
            "document.addEventListener('{}', function(event) {{\n",
            self.map_trigger_to_event(&intent.trigger)
        ));
        
        // 生成目标选择
        code.push_str(&format!(
            "    const target = event.target.closest('{}');\n",
            intent.target.value
        ));
        code.push_str("    if (!target) return;\n\n");
        
        // 生成行为处理
        code.push_str(&self.generate_behavior(&intent.behavior));
        
        code.push_str("});\n");
        
        code
    }

    fn generate_behavior(&self, behavior: &crate::js_understander::BehaviorDescription) -> String {
        let mut code = String::new();
        
        match &behavior.behavior_type {
            BehaviorType::NavigateTo => {
                code.push_str("    // 导航到指定页面\n");
                code.push_str("    window.location.href = target.getAttribute('href') || '/';\n");
            }
            BehaviorType::ToggleVisibility => {
                code.push_str("    // 切换元素可见性\n");
                code.push_str("    const element = document.querySelector(target.dataset.target);\n");
                code.push_str("    if (element) {\n");
                code.push_str("        element.classList.toggle('hidden');\n");
                code.push_str("    }\n");
            }
            BehaviorType::SubmitForm => {
                code.push_str("    // 提交表单\n");
                code.push_str("    const form = target.closest('form');\n");
                code.push_str("    if (form && validateForm(form)) {\n");
                code.push_str("        submitForm(form);\n");
                code.push_str("    }\n");
            }
            BehaviorType::ValidateInput => {
                code.push_str("    // 验证输入\n");
                code.push_str("    const input = target;\n");
                code.push_str("    const isValid = input.checkValidity();\n");
                code.push_str("    input.classList.toggle('invalid', !isValid);\n");
            }
            BehaviorType::FetchData => {
                code.push_str("    // 获取数据\n");
                code.push_str("    fetchData();\n");
            }
            BehaviorType::ToggleClass => {
                code.push_str("    // 切换CSS类\n");
                code.push_str("    target.classList.toggle('active');\n");
            }
            _ => {
                code.push_str("    // 自定义行为\n");
                code.push_str("    console.log('Action triggered:', event);\n");
            }
        }
        
        code
    }

    fn generate_default_handlers(&self) -> String {
        let mut code = String::new();
        
        // 按钮点击处理
        code.push_str("// 默认按钮点击处理\n");
        code.push_str("document.addEventListener('click', function(e) {\n");
        code.push_str("    const btn = e.target.closest('button, .btn, [role=\"button\"]');\n");
        code.push_str("    if (!btn) return;\n\n");
        code.push_str("    // 添加点击反馈动画\n");
        code.push_str("    btn.classList.add('clicked');\n");
        code.push_str("    setTimeout(() => btn.classList.remove('clicked'), 200);\n\n");
        code.push_str("    // 处理不同按钮类型\n");
        code.push_str("    if (btn.dataset.action === 'submit') {\n");
        code.push_str("        handleSubmit(btn);\n");
        code.push_str("    } else if (btn.dataset.action === 'toggle') {\n");
        code.push_str("        handleToggle(btn);\n");
        code.push_str("    } else if (btn.dataset.action === 'navigate') {\n");
        code.push_str("        window.location.href = btn.dataset.href || '/';\n");
        code.push_str("    }\n");
        code.push_str("});\n\n");
        
        // 表单处理
        code.push_str("// 表单验证和提交\n");
        code.push_str("document.addEventListener('submit', function(e) {\n");
        code.push_str("    const form = e.target;\n");
        code.push_str("    if (!validateForm(form)) {\n");
        code.push_str("        e.preventDefault();\n");
        code.push_str("        showFormErrors(form);\n");
        code.push_str("    }\n");
        code.push_str("});\n\n");
        
        // 输入验证
        code.push_str("// 实时输入验证\n");
        code.push_str("document.addEventListener('input', function(e) {\n");
        code.push_str("    const input = e.target;\n");
        code.push_str("    if (input.tagName === 'INPUT' || input.tagName === 'TEXTAREA') {\n");
        code.push_str("        validateInput(input);\n");
        code.push_str("    }\n");
        code.push_str("});\n");
        
        code
    }

    /// 生成API函数
    fn generate_api_functions(&self) -> String {
        let mut code = String::new();
        
        code.push_str("// ============================================\n");
        code.push_str("// API调用函数\n");
        code.push_str("// ============================================\n\n");
        
        for (i, api) in self.intents.api_intents.iter().enumerate() {
            code.push_str(&self.generate_api_function(api, i));
            code.push_str("\n\n");
        }
        
        // 默认API函数
        if self.intents.api_intents.is_empty() {
            code.push_str(&self.generate_default_api_functions());
        }
        
        code
    }

    fn generate_api_function(&self, api: &ApiIntent, index: usize) -> String {
        let func_name = format!("api_call_{}", index);
        let method = format!("{:?}", api.method).to_uppercase();
        
        let mut code = format!(
            "// API调用: {}\n",
            api.name
        );
        
        code.push_str(&format!(
            "async function {}(data) {{\n",
            func_name
        ));
        
        code.push_str("    try {\n");
        code.push_str(&format!(
            "        const response = await fetch('{}', {{\n",
            api.endpoint
        ));
        code.push_str(&format!(
            "            method: '{}',\n",
            method
        ));
        code.push_str("            headers: {\n");
        code.push_str("                'Content-Type': 'application/json',\n");
        for (key, value) in &api.headers {
            code.push_str(&format!("                '{}': '{}',\n", key, value));
        }
        code.push_str("            },\n");
        
        if api.method != HttpMethod::Get {
            code.push_str("            body: JSON.stringify(data)\n");
        }
        
        code.push_str("        });\n\n");
        code.push_str("        if (!response.ok) {\n");
        code.push_str("            throw new Error(`HTTP ${response.status}`);\n");
        code.push_str("        }\n\n");
        code.push_str("        const result = await response.json();\n");
        code.push_str(&format!(
            "        {};\n",
            self.generate_success_handler(&api.response_handling.on_success)
        ));
        code.push_str("        return result;\n");
        
        code.push_str("    } catch (error) {\n");
        code.push_str(&format!(
            "        {};\n",
            self.generate_error_handler(&api.response_handling.on_error)
        ));
        code.push_str("        throw error;\n");
        code.push_str("    }\n");
        
        code.push_str("}\n");
        
        code
    }

    fn generate_default_api_functions(&self) -> String {
        let mut code = String::new();
        
        code.push_str("// 通用API请求函数\n");
        code.push_str("async function apiRequest(endpoint, options = {}) {\n");
        code.push_str("    const defaultOptions = {\n");
        code.push_str("        headers: {\n");
        code.push_str("            'Content-Type': 'application/json',\n");
        code.push_str("        },\n");
        code.push_str("    };\n\n");
        code.push_str("    const response = await fetch(endpoint, { ...defaultOptions, ...options });\n\n");
        code.push_str("    if (!response.ok) {\n");
        code.push_str("        throw new Error(`API Error: ${response.status}`);\n");
        code.push_str("    }\n\n");
        code.push_str("    return response.json();\n");
        code.push_str("}\n\n");
        
        code.push_str("// 表单数据提交\n");
        code.push_str("async function submitFormData(form) {\n");
        code.push_str("    const formData = new FormData(form);\n");
        code.push_str("    const data = Object.fromEntries(formData);\n\n");
        code.push_str("    return apiRequest(form.action, {\n");
        code.push_str("        method: form.method || 'POST',\n");
        code.push_str("        body: JSON.stringify(data),\n");
        code.push_str("    });\n");
        code.push_str("}\n");
        
        code
    }

    /// 生成状态管理函数
    fn generate_state_functions(&self) -> String {
        let mut code = String::new();
        
        code.push_str("// ============================================\n");
        code.push_str("// 状态管理\n");
        code.push_str("// ============================================\n\n");
        
        code.push_str("// 简单状态管理器\n");
        code.push_str("const AppState = {\n");
        code.push_str("    _state: {},\n");
        code.push_str("    _listeners: [],\n\n");
        code.push_str("    get(key) {\n");
        code.push_str("        return this._state[key];\n");
        code.push_str("    },\n\n");
        code.push_str("    set(key, value) {\n");
        code.push_str("        this._state[key] = value;\n");
        code.push_str("        this._notify(key, value);\n");
        code.push_str("    },\n\n");
        code.push_str("    subscribe(listener) {\n");
        code.push_str("        this._listeners.push(listener);\n");
        code.push_str("        return () => {\n");
        code.push_str("            this._listeners = this._listeners.filter(l => l !== listener);\n");
        code.push_str("        };\n");
        code.push_str("    },\n\n");
        code.push_str("    _notify(key, value) {\n");
        code.push_str("        this._listeners.forEach(listener => listener(key, value));\n");
        code.push_str("    }\n");
        code.push_str("};\n");
        
        code
    }

    /// 生成动画函数
    fn generate_animations(&self) -> String {
        let mut code = String::new();
        
        code.push_str("// ============================================\n");
        code.push_str("// 动画效果\n");
        code.push_str("// ============================================\n\n");
        
        code.push_str("// 淡入动画\n");
        code.push_str("function fadeIn(element, duration = 300) {\n");
        code.push_str("    element.style.opacity = '0';\n");
        code.push_str("    element.style.display = 'block';\n");
        code.push_str("    element.style.transition = `opacity ${duration}ms`;\n");
        code.push_str("    requestAnimationFrame(() => {\n");
        code.push_str("        element.style.opacity = '1';\n");
        code.push_str("    });\n");
        code.push_str("}\n\n");
        
        code.push_str("// 淡出动画\n");
        code.push_str("function fadeOut(element, duration = 300) {\n");
        code.push_str("    element.style.transition = `opacity ${duration}ms`;\n");
        code.push_str("    element.style.opacity = '0';\n");
        code.push_str("    setTimeout(() => {\n");
        code.push_str("        element.style.display = 'none';\n");
        code.push_str("    }, duration);\n");
        code.push_str("}\n\n");
        
        code.push_str("// 滑动动画\n");
        code.push_str("function slideToggle(element, duration = 300) {\n");
        code.push_str("    const isHidden = window.getComputedStyle(element).display === 'none';\n");
        code.push_str("    if (isHidden) {\n");
        code.push_str("        slideDown(element, duration);\n");
        code.push_str("    } else {\n");
        code.push_str("        slideUp(element, duration);\n");
        code.push_str("    }\n");
        code.push_str("}\n");
        
        code
    }

    /// 生成工具函数
    fn generate_utilities(&self) -> String {
        let mut code = String::new();
        
        code.push_str("// ============================================\n");
        code.push_str("// 工具函数\n");
        code.push_str("// ============================================\n\n");
        
        code.push_str("// 表单验证\n");
        code.push_str("function validateForm(form) {\n");
        code.push_str("    const inputs = form.querySelectorAll('input, textarea, select');\n");
        code.push_str("    let isValid = true;\n\n");
        code.push_str("    inputs.forEach(input => {\n");
        code.push_str("        if (!validateInput(input)) {\n");
        code.push_str("            isValid = false;\n");
        code.push_str("        }\n");
        code.push_str("    });\n\n");
        code.push_str("    return isValid;\n");
        code.push_str("}\n\n");
        
        code.push_str("// 输入验证\n");
        code.push_str("function validateInput(input) {\n");
        code.push_str("    const value = input.value.trim();\n");
        code.push_str("    const type = input.type;\n");
        code.push_str("    let isValid = true;\n");
        code.push_str("    let errorMessage = '';\n\n");
        code.push_str("    // 必填验证\n");
        code.push_str("    if (input.required && !value) {\n");
        code.push_str("        isValid = false;\n");
        code.push_str("        errorMessage = '此字段为必填项';\n");
        code.push_str("    }\n\n");
        code.push_str("    // 邮箱验证\n");
        code.push_str("    if (isValid && type === 'email' && value) {\n");
        code.push_str("        const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n");
        code.push_str("        isValid = emailRegex.test(value);\n");
        code.push_str("        errorMessage = '请输入有效的邮箱地址';\n");
        code.push_str("    }\n\n");
        code.push_str("    // 更新UI\n");
        code.push_str("    input.classList.toggle('invalid', !isValid);\n");
        code.push_str("    const errorEl = input.parentElement.querySelector('.error-message');\n");
        code.push_str("    if (errorEl) {\n");
        code.push_str("        errorEl.textContent = isValid ? '' : errorMessage;\n");
        code.push_str("    }\n\n");
        code.push_str("    return isValid;\n");
        code.push_str("}\n\n");
        
        code.push_str("// 防抖函数\n");
        code.push_str("function debounce(func, wait) {\n");
        code.push_str("    let timeout;\n");
        code.push_str("    return function executedFunction(...args) {\n");
        code.push_str("        const later = () => {\n");
        code.push_str("            clearTimeout(timeout);\n");
        code.push_str("            func(...args);\n");
        code.push_str("        };\n");
        code.push_str("        clearTimeout(timeout);\n");
        code.push_str("        timeout = setTimeout(later, wait);\n");
        code.push_str("    };\n");
        code.push_str("}\n\n");
        
        code.push_str("// 节流函数\n");
        code.push_str("function throttle(func, limit) {\n");
        code.push_str("    let inThrottle;\n");
        code.push_str("    return function(...args) {\n");
        code.push_str("        if (!inThrottle) {\n");
        code.push_str("            func.apply(this, args);\n");
        code.push_str("            inThrottle = true;\n");
        code.push_str("            setTimeout(() => inThrottle = false, limit);\n");
        code.push_str("        }\n");
        code.push_str("    };\n");
        code.push_str("}\n");
        
        code
    }

    /// 生成初始化代码
    fn generate_init(&self) -> String {
        let mut code = String::new();
        
        code.push_str("// ============================================\n");
        code.push_str("// 初始化\n");
        code.push_str("// ============================================\n\n");
        
        code.push_str("document.addEventListener('DOMContentLoaded', function() {\n");
        code.push_str("    console.log('🚀 Website initialized');\n\n");
        
        code.push_str("    // 初始化所有交互\n");
        code.push_str("    initEventHandlers();\n\n");
        
        code.push_str("    // 初始化表单验证\n");
        code.push_str("    initFormValidation();\n\n");
        
        code.push_str("    // 初始化动画\n");
        code.push_str("    initAnimations();\n");
        
        code.push_str("});\n\n");
        
        code.push_str("function initEventHandlers() {\n");
        code.push_str("    // 事件处理器已在上面注册\n");
        code.push_str("    console.log('✓ Event handlers initialized');\n");
        code.push_str("}\n\n");
        
        code.push_str("function initFormValidation() {\n");
        code.push_str("    const forms = document.querySelectorAll('form');\n");
        code.push_str("    forms.forEach(form => {\n");
        code.push_str("        form.setAttribute('novalidate', true);\n");
        code.push_str("    });\n");
        code.push_str("    console.log('✓ Form validation initialized');\n");
        code.push_str("}\n\n");
        
        code.push_str("function initAnimations() {\n");
        code.push_str("    // 初始化滚动动画\n");
        code.push_str("    const animatedElements = document.querySelectorAll('[data-animate]');\n");
        code.push_str("    const observer = new IntersectionObserver((entries) => {\n");
        code.push_str("        entries.forEach(entry => {\n");
        code.push_str("            if (entry.isIntersecting) {\n");
        code.push_str("                entry.target.classList.add('animated');\n");
        code.push_str("            }\n");
        code.push_str("        });\n");
        code.push_str("    });\n");
        code.push_str("    animatedElements.forEach(el => observer.observe(el));\n");
        code.push_str("    console.log('✓ Animations initialized');\n");
        code.push_str("}\n");
        
        code
    }

    // 辅助方法
    fn map_trigger_to_event(&self, trigger: &TriggerType) -> String {
        match trigger {
            TriggerType::Click => "click".to_string(),
            TriggerType::DoubleClick => "dblclick".to_string(),
            TriggerType::RightClick => "contextmenu".to_string(),
            TriggerType::Hover => "mouseenter".to_string(),
            TriggerType::Focus => "focus".to_string(),
            TriggerType::Blur => "blur".to_string(),
            TriggerType::Input(_) => "input".to_string(),
            TriggerType::Change => "change".to_string(),
            TriggerType::Submit => "submit".to_string(),
            TriggerType::KeyPress(_) => "keydown".to_string(),
            TriggerType::Scroll => "scroll".to_string(),
            TriggerType::Resize => "resize".to_string(),
            TriggerType::Load => "load".to_string(),
            TriggerType::Custom(s) => s.clone(),
        }
    }

    fn generate_success_handler(&self, behavior: &crate::js_understander::BehaviorDescription) -> String {
        match &behavior.behavior_type {
            BehaviorType::UpdateState => "AppState.set('data', result)".to_string(),
            BehaviorType::NavigateTo => "window.location.href = result.redirect".to_string(),
            _ => "console.log('Success:', result)".to_string(),
        }
    }

    fn generate_error_handler(&self, behavior: &crate::js_understander::BehaviorDescription) -> String {
        match &behavior.behavior_type {
            BehaviorType::ShowError => "showNotification(error.message, 'error')".to_string(),
            _ => "console.error('Error:', error)".to_string(),
        }
    }

    // 其他框架生成器（简化版）
    fn generate_react(&self) -> GeneratedFunctions {
        // React实现...
        self.generate_vanilla_js()
    }

    fn generate_vue(&self) -> GeneratedFunctions {
        // Vue实现...
        self.generate_vanilla_js()
    }

    fn generate_svelte(&self) -> GeneratedFunctions {
        // Svelte实现...
        self.generate_vanilla_js()
    }
}

/// 生成完整的JS文件
pub fn generate_js_file(functions: &GeneratedFunctions) -> String {
    let mut code = String::new();
    
    code.push_str("/*!\n");
    code.push_str(" * Generated Website JavaScript\n");
    code.push_str(" * Based on learned interaction intents\n");
    code.push_str(" * Functionally equivalent to original, with new implementation\n");
    code.push_str(" */\n\n");
    
    code.push_str(&functions.state_functions);
    code.push_str("\n\n");
    
    code.push_str(&functions.utility_functions);
    code.push_str("\n\n");
    
    code.push_str(&functions.api_functions);
    code.push_str("\n\n");
    
    code.push_str(&functions.event_handlers);
    code.push_str("\n\n");
    
    code.push_str(&functions.animation_functions);
    code.push_str("\n\n");
    
    code.push_str(&functions.init_code);
    
    code
}
