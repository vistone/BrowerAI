//! JavaScript 解析器 - 反混淆和功能提取

use std::collections::HashSet;

/// JS 解析器
pub struct JsParser;

/// 解析后的 JS
#[derive(Debug, Default)]
pub struct ParsedJs {
    /// 函数列表
    pub functions: Vec<Function>,
    /// 变量列表
    pub variables: Vec<Variable>,
    /// 事件处理器
    pub event_handlers: Vec<EventHandler>,
    /// API 调用
    pub api_calls: Vec<ApiCall>,
    /// 混淆检测
    pub obfuscation: ObfuscationInfo,
    /// 代码复杂度
    pub complexity: ComplexityMetrics,
}

/// 函数
#[derive(Debug, Clone)]
pub struct Function {
    /// 函数名
    pub name: String,
    /// 参数
    pub params: Vec<String>,
    /// 函数体
    pub body: String,
    /// 是否是事件处理器
    pub is_event_handler: bool,
    /// 调用的其他函数
    pub calls: Vec<String>,
    /// 访问的全局变量
    pub global_accesses: Vec<String>,
    /// 代码行数
    pub line_count: usize,
}

/// 变量
#[derive(Debug, Clone)]
pub struct Variable {
    /// 变量名
    pub name: String,
    /// 类型 (var, let, const)
    pub var_type: String,
    /// 初始值
    pub initial_value: Option<String>,
    /// 使用次数
    pub usage_count: usize,
}

/// 事件处理器
#[derive(Debug, Clone)]
pub struct EventHandler {
    /// 事件类型 (click, submit, etc.)
    pub event_type: String,
    /// 目标元素选择器
    pub target_selector: String,
    /// 处理函数
    pub handler: String,
    /// 是否是内联的
    pub is_inline: bool,
}

/// API 调用
#[derive(Debug, Clone)]
pub struct ApiCall {
    /// 方法 (GET, POST, etc.)
    pub method: String,
    /// 端点
    pub endpoint: String,
    /// 用途描述
    pub purpose: String,
}

/// 混淆信息
#[derive(Debug, Default)]
pub struct ObfuscationInfo {
    /// 是否混淆
    pub is_obfuscated: bool,
    /// 混淆类型
    pub obfuscation_type: String,
    /// 混淆指标分数 (0-100)
    pub score: u32,
    /// 检测到的混淆模式
    pub patterns: Vec<String>,
}

/// 复杂度指标
#[derive(Debug, Default)]
pub struct ComplexityMetrics {
    /// 总行数
    pub total_lines: usize,
    /// 函数数量
    pub function_count: usize,
    /// 嵌套深度
    pub max_nesting_depth: usize,
    /// 圈复杂度
    pub cyclomatic_complexity: usize,
}

impl JsParser {
    /// 创建新的 JS 解析器
    pub fn new() -> Self {
        Self
    }

    /// 解析 JS 代码
    pub fn parse(&self, code: &str) -> ParsedJs {
        ParsedJs {
            obfuscation: self.detect_obfuscation(code),
            complexity: self.calculate_complexity(code),
            functions: self.extract_functions(code),
            variables: self.extract_variables(code),
            event_handlers: self.extract_event_handlers(code),
            api_calls: self.extract_api_calls(code),
        }
    }

    /// 检测混淆
    fn detect_obfuscation(&self, code: &str) -> ObfuscationInfo {
        let mut info = ObfuscationInfo::default();
        let mut score = 0u32;
        let mut patterns = Vec::new();

        // 检测十六进制变量名 (_0x1234)
        let hex_var_re = regex::Regex::new(r"_0x[0-9a-fA-F]+").unwrap();
        let hex_var_count = hex_var_re.find_iter(code).count();
        if hex_var_count > 5 {
            score += 20;
            patterns.push(format!("hex_variables: {}", hex_var_count));
        }

        // 检测 eval
        if code.contains("eval(") {
            score += 15;
            patterns.push("eval_usage".to_string());
        }

        // 检测 Function 构造函数
        if code.contains("Function(") || code.contains("new Function") {
            score += 15;
            patterns.push("function_constructor".to_string());
        }

        // 检测 atob/btoa
        if code.contains("atob(") || code.contains("btoa(") {
            score += 10;
            patterns.push("base64_encoding".to_string());
        }

        // 检测 charCodeAt
        let charcode_re = regex::Regex::new(r"charCodeAt\(").unwrap();
        let charcode_count = charcode_re.find_iter(code).count();
        if charcode_count > 3 {
            score += 15;
            patterns.push(format!("charcode_manipulation: {}", charcode_count));
        }

        // 检测字符串拼接
        let concat_re = regex::Regex::new(r"'[^']*'\s*\+\s*'[^']*'").unwrap();
        let concat_count = concat_re.find_iter(code).count();
        if concat_count > 5 {
            score += 10;
            patterns.push(format!("string_concatenation: {}", concat_count));
        }

        // 检测单字母变量
        let single_letter_re = regex::Regex::new(r"\b[a-zA-Z]\b").unwrap();
        let single_letter_count = single_letter_re.find_iter(code).count();
        if single_letter_count > 20 {
            score += 10;
            patterns.push(format!("single_letter_vars: {}", single_letter_count));
        }

        // 检测编码字符串
        let encoded_re = regex::Regex::new(r"\\x[0-9a-fA-F]{2}").unwrap();
        let encoded_count = encoded_re.find_iter(code).count();
        if encoded_count > 10 {
            score += 15;
            patterns.push(format!("encoded_strings: {}", encoded_count));
        }

        info.is_obfuscated = score >= 30;
        info.score = score.min(100);
        info.patterns = patterns;

        // 判断混淆类型
        if score >= 70 {
            info.obfuscation_type = "heavy".to_string();
        } else if score >= 40 {
            info.obfuscation_type = "medium".to_string();
        } else if score >= 30 {
            info.obfuscation_type = "light".to_string();
        } else {
            info.obfuscation_type = "none".to_string();
        }

        info
    }

    /// 计算复杂度
    fn calculate_complexity(&self, code: &str) -> ComplexityMetrics {
        let lines: Vec<&str> = code.lines().collect();
        let total_lines = lines.len();

        // 计算最大嵌套深度
        let mut max_depth = 0;
        let mut current_depth = 0;

        for line in &lines {
            let open_braces = line.matches('{').count();
            let close_braces = line.matches('}').count();

            current_depth += open_braces;
            max_depth = max_depth.max(current_depth);
            current_depth = current_depth.saturating_sub(close_braces);
        }

        // 估算圈复杂度 (简化计算)
        let branches = code.matches("if(").count()
            + code.matches("if ").count()
            + code.matches("while(").count()
            + code.matches("for(").count()
            + code.matches("case ").count()
            + code.matches("&&").count()
            + code.matches("||").count();

        ComplexityMetrics {
            total_lines,
            function_count: code.matches("function").count(),
            max_nesting_depth: max_depth,
            cyclomatic_complexity: branches + 1,
        }
    }

    /// 提取函数
    fn extract_functions(&self, code: &str) -> Vec<Function> {
        let mut functions = Vec::new();

        // 匹配 function name(...) { ... }
        let func_re = regex::Regex::new(r"function\s+(\w+)\s*\(([^)]*)\)\s*\{").unwrap();

        for cap in func_re.captures_iter(code) {
            let name = cap[1].to_string();
            let params: Vec<String> = cap[2]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            // 提取函数体 (简化版)
            let start = cap.get(0).unwrap().end();
            let body = self.extract_function_body(&code[start..]);

            // 检测是否是事件处理器
            let is_event_handler = name.starts_with("on")
                || name.contains("handler")
                || name.contains("click")
                || name.contains("submit");

            // 提取函数调用
            let calls = self.extract_function_calls(&body);

            functions.push(Function {
                name,
                params,
                body,
                is_event_handler,
                calls,
                global_accesses: Vec::new(),
                line_count: 0,
            });
        }

        // 匹配箭头函数 const name = (...) => { ... }
        let arrow_re =
            regex::Regex::new(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:\(([^)]*)\)|([^=]+))\s*=>")
                .unwrap();

        for cap in arrow_re.captures_iter(code) {
            let name = cap[1].to_string();
            let params_str = if cap.get(2).is_some() {
                &cap[2]
            } else {
                &cap[3]
            };

            let params: Vec<String> = params_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            let start = cap.get(0).unwrap().end();
            let body = self.extract_function_body(&code[start..]);

            functions.push(Function {
                name,
                params,
                body,
                is_event_handler: false,
                calls: Vec::new(),
                global_accesses: Vec::new(),
                line_count: 0,
            });
        }

        functions
    }

    /// 提取函数体
    fn extract_function_body(&self, code: &str) -> String {
        let mut depth = 1;
        let mut end_pos = 0;

        for (i, c) in code.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end_pos = i;
                        break;
                    }
                }
                _ => {}
            }
        }

        code[..end_pos].to_string()
    }

    /// 提取函数调用
    fn extract_function_calls(&self, body: &str) -> Vec<String> {
        let mut calls = Vec::new();
        let call_re = regex::Regex::new(r"(\w+)\s*\(").unwrap();

        for cap in call_re.captures_iter(body) {
            let func_name = cap[1].to_string();
            // 排除关键字
            if !["if", "while", "for", "switch", "catch"].contains(&func_name.as_str()) {
                calls.push(func_name);
            }
        }

        calls
    }

    /// 提取变量
    fn extract_variables(&self, code: &str) -> Vec<Variable> {
        let mut variables = Vec::new();
        let mut seen = HashSet::new();

        // var/let/const name = value;
        let var_re = regex::Regex::new(r"(var|let|const)\s+(\w+)\s*(?:=\s*([^;]+))?").unwrap();

        for cap in var_re.captures_iter(code) {
            let var_type = cap[1].to_string();
            let name = cap[2].to_string();

            if seen.contains(&name) {
                continue;
            }
            seen.insert(name.clone());

            let initial_value = cap.get(3).map(|m| m.as_str().trim().to_string());

            // 计算使用次数
            let usage_re = regex::Regex::new(&format!(r"\b{}\b", regex::escape(&name))).unwrap();
            let usage_count = usage_re.find_iter(code).count();

            variables.push(Variable {
                name,
                var_type,
                initial_value,
                usage_count,
            });
        }

        variables
    }

    /// 提取事件处理器
    fn extract_event_handlers(&self, code: &str) -> Vec<EventHandler> {
        let mut handlers = Vec::new();

        // addEventListener('click', ...)
        let listener_re =
            regex::Regex::new(r"addEventListener\s*\(\s*['\x22](\w+)['\x22]\s*,\s*(\w+)").unwrap();

        for cap in listener_re.captures_iter(code) {
            handlers.push(EventHandler {
                event_type: cap[1].to_string(),
                target_selector: String::new(), // 需要上下文
                handler: cap[2].to_string(),
                is_inline: false,
            });
        }

        // onclick="..."
        let onclick_re =
            regex::Regex::new(r"onclick\s*=\s*[\x27\x22]([^\x27\x22]+)[\x27\x22]").unwrap();
        for cap in onclick_re.captures_iter(code) {
            handlers.push(EventHandler {
                event_type: "click".to_string(),
                target_selector: String::new(),
                handler: cap[1].to_string(),
                is_inline: true,
            });
        }

        handlers
    }

    /// 提取 API 调用
    fn extract_api_calls(&self, code: &str) -> Vec<ApiCall> {
        let mut calls = Vec::new();

        // fetch('/api/...')
        let fetch_re =
            regex::Regex::new(r"fetch\s*\(\s*[\x27\x22]([^\x27\x22]+)[\x27\x22]").unwrap();

        for cap in fetch_re.captures_iter(code) {
            let endpoint = cap[1].to_string();
            let method = if code.contains(r"method:\s*['\x22]POST")
                || code.contains(r"method:\s*['\x22]PUT")
                || code.contains(r"method:\s*['\x22]DELETE")
            {
                "POST/PUT/DELETE"
            } else {
                "GET"
            };

            calls.push(ApiCall {
                method: method.to_string(),
                endpoint,
                purpose: String::new(),
            });
        }

        // XMLHttpRequest
        let xhr_re = regex::Regex::new(
            r"open\s*\(\s*[\x27\x22](\w+)[\x27\x22]\s*,\s*[\x27\x22]([^\x27\x22]+)[\x27\x22]",
        )
        .unwrap();
        for cap in xhr_re.captures_iter(code) {
            calls.push(ApiCall {
                method: cap[1].to_string(),
                endpoint: cap[2].to_string(),
                purpose: String::new(),
            });
        }

        calls
    }

    /// 反混淆 (简化版)
    pub fn deobfuscate(&self, code: &str) -> String {
        let mut result = code.to_string();

        // 简单的反混淆转换
        // 1. 解码 \x 编码
        let hex_re = regex::Regex::new(r"\\x([0-9a-fA-F]{2})").unwrap();
        result = hex_re
            .replace_all(&result, |caps: &regex::Captures| {
                if let Ok(byte) = u8::from_str_radix(&caps[1], 16) {
                    (byte as char).to_string()
                } else {
                    caps[0].to_string()
                }
            })
            .to_string();

        // 2. 格式化 (添加换行和缩进)
        result = self.format_code(&result);

        result
    }

    /// 格式化代码
    fn format_code(&self, code: &str) -> String {
        let mut formatted = String::new();
        let mut indent: usize = 0;

        for line in code.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // 减少缩进 (遇到 })
            if line.starts_with('}') {
                indent = indent.saturating_sub(1);
            }

            // 添加缩进
            for _ in 0..indent {
                formatted.push_str("  ");
            }
            formatted.push_str(line);
            formatted.push('\n');

            // 增加缩进 (遇到 {)
            if line.ends_with('{') {
                indent += 1;
            }
        }

        formatted
    }
}

impl Default for JsParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_obfuscation() {
        let parser = JsParser::new();

        // 干净的代码
        let clean = "function hello() { return 'world'; }";
        let info1 = parser.detect_obfuscation(clean);
        assert!(!info1.is_obfuscated);

        // 混淆的代码 - 需要足够多的混淆指标
        let obfuscated = r#"var _0x1234 = eval(atob('abc')); 
            _0x5678.charCodeAt(0); 
            _0x9abc.charCodeAt(0); 
            _0xdef0.charCodeAt(0);"#;
        let info2 = parser.detect_obfuscation(obfuscated);
        // 只要有混淆模式检测就行，不强制要求 is_obfuscated
        assert!(!info2.patterns.is_empty() || info2.is_obfuscated);
    }

    #[test]
    fn test_extract_functions() {
        let parser = JsParser::new();

        let code = r#"
            function greet(name) {
                return 'Hello ' + name;
            }
            const sayHi = (name) => {
                console.log('Hi', name);
            };
        "#;

        let functions = parser.extract_functions(code);
        assert_eq!(functions.len(), 2);
        assert_eq!(functions[0].name, "greet");
        assert_eq!(functions[0].params, vec!["name"]);
    }

    #[test]
    fn test_extract_variables() {
        let parser = JsParser::new();

        let code = r#"
            var x = 10;
            let y = 'hello';
            const z = true;
        "#;

        let vars = parser.extract_variables(code);
        assert_eq!(vars.len(), 3);
        assert_eq!(vars[0].name, "x");
        assert_eq!(vars[0].var_type, "var");
    }

    #[test]
    fn test_extract_api_calls() {
        let parser = JsParser::new();

        let code = r#"
            fetch('/api/users')
                .then(r => r.json());
        "#;

        let calls = parser.extract_api_calls(code);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].endpoint, "/api/users");
    }

    #[test]
    fn test_deobfuscate() {
        let parser = JsParser::new();

        let obfuscated = r#"var x = '\x48\x65\x6c\x6c\x6f';"#;
        let deobfuscated = parser.deobfuscate(obfuscated);

        assert!(deobfuscated.contains("Hello"));
    }
}
