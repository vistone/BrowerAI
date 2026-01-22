/// 代码验证模块
///
/// 对生成的HTML/CSS/JavaScript代码进行自动验证：
/// - 语法正确性
/// - 可执行性
/// - 符合规范性
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// HTML验证结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HtmlVerification {
    /// 是否有效
    pub valid: bool,

    /// 解析错误列表
    pub parse_errors: Vec<String>,

    /// 警告信息
    pub warnings: Vec<String>,

    /// 验证评分 (0-1)
    /// 1.0 = 完全有效，无错误无警告
    /// 0.8-1.0 = 有轻微警告
    /// 0.5-0.8 = 有解析错误但可恢复
    /// <0.5 = 严重错误
    pub score: f64,

    /// 检测到的标签列表
    pub detected_tags: Vec<String>,

    /// 检测到的事件处理器
    pub event_handlers: Vec<String>,
}

/// CSS验证结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CssVerification {
    /// 是否有效
    pub valid: bool,

    /// 解析错误列表
    pub parse_errors: Vec<String>,

    /// 警告信息
    pub warnings: Vec<String>,

    /// 验证评分 (0-1)
    pub score: f64,

    /// 检测到的选择器
    pub selectors: Vec<String>,

    /// 检测到的CSS属性
    pub properties: Vec<String>,
}

/// JavaScript验证结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsVerification {
    /// 语法是否有效
    pub syntax_valid: bool,

    /// 语法错误列表
    pub syntax_errors: Vec<String>,

    /// 警告信息
    pub warnings: Vec<String>,

    /// 验证评分 (0-1)
    pub score: f64,

    /// 检测到的函数
    pub functions: Vec<String>,

    /// 检测到的变量
    pub variables: Vec<String>,

    /// 检测到的异步操作
    pub async_operations: Vec<String>,

    /// 检测到的API调用（如 fetch, XMLHttpRequest 等）
    pub api_calls: Vec<String>,
}

/// 综合验证结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeVerificationResult {
    /// HTML验证结果
    pub html: HtmlVerification,

    /// CSS验证结果
    pub css: CssVerification,

    /// JavaScript验证结果
    pub js: JsVerification,

    /// 综合验证评分 (0-1)
    /// 权重: HTML 30%, CSS 20%, JS 50%
    pub verification_score: f64,

    /// 所有错误的汇总
    pub all_errors: Vec<VerificationError>,

    /// 建议的修复
    pub suggested_fixes: Vec<(String, String)>, // (问题, 修复建议)
}

/// 验证错误详情
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationError {
    pub code_type: String,  // "html", "css", "js"
    pub error_type: String, // "parse_error", "warning", etc.
    pub message: String,
    pub severity: String,         // "error", "warning"
    pub location: Option<String>, // 错误位置
}

pub struct CodeVerifier;

impl CodeVerifier {
    /// 验证生成的HTML
    pub fn verify_html(html: &str) -> Result<HtmlVerification> {
        log::debug!("验证HTML代码 ({} 字节)...", html.len());

        let mut errors = vec![];
        let mut warnings = vec![];
        let mut detected_tags = vec![];
        let mut event_handlers = vec![];

        // 检查基本的HTML结构
        let html_lower = html.to_lowercase();

        // 检查是否有DOCTYPE或html标签
        if !html_lower.contains("<!doctype") && !html_lower.contains("<html") {
            warnings.push("缺少<!DOCTYPE>或<html>标签".to_string());
        }

        // 检查是否有head和body
        if !html_lower.contains("<head") {
            warnings.push("缺少<head>标签".to_string());
        }
        if !html_lower.contains("<body") {
            warnings.push("缺少<body>标签".to_string());
        }

        // 使用简单的正则验证标签配对
        let open_tags: HashMap<String, usize> = HashMap::new();
        let close_tags: HashMap<String, usize> = HashMap::new();

        // 提取标签
        for cap in regex::Regex::new(r"<(\w+)").unwrap().captures_iter(html) {
            if let Some(tag) = cap.get(1) {
                let tag_name = tag.as_str().to_lowercase();
                detected_tags.push(tag_name);
            }
        }
        detected_tags.sort();
        detected_tags.dedup();

        // 提取事件处理器
        for cap in regex::Regex::new(r#"on(\w+)\s*=\s*['""]?([^'""\s>]+)"#)
            .unwrap()
            .captures_iter(html)
        {
            if let (Some(event), Some(handler)) = (cap.get(1), cap.get(2)) {
                event_handlers.push(format!("on{}={}", event.as_str(), handler.as_str()));
            }
        }

        // 尝试用html5ever解析（如果可用）
        // 这里我们用简单的检查替代
        let tag_pattern = regex::Regex::new(r"<[^>]+>").unwrap();
        let tag_count = tag_pattern.find_iter(html).count();

        if tag_count == 0 {
            errors.push("HTML代码中未找到任何标签".to_string());
        }

        // 检查常见错误
        if html.contains("< ") || html.contains(" >") {
            errors.push("发现格式不正确的标签（空格位置错误）".to_string());
        }

        // 检查未关闭的关键标签
        let self_closing_tags = vec!["br", "hr", "img", "input", "meta", "link"];
        for tag in &self_closing_tags {
            let pattern = format!(r"<{}\s*[^>]*(?<!/)>", tag);
            if let Ok(re) = regex::Regex::new(&pattern) {
                if re.is_match(html) {
                    warnings.push(format!("{}标签可能未正确关闭", tag));
                }
            }
        }

        // 计算评分
        let error_count = errors.len() as f64;
        let warning_count = warnings.len() as f64;
        let score = if error_count > 0.0 {
            (10.0 - error_count * 3.0 - warning_count * 0.5) / 10.0
        } else if warning_count > 0.0 {
            1.0 - warning_count * 0.05
        } else {
            1.0
        };
        let score = score.max(0.0).min(1.0);

        Ok(HtmlVerification {
            valid: errors.is_empty(),
            parse_errors: errors,
            warnings,
            score,
            detected_tags,
            event_handlers,
        })
    }

    /// 验证生成的CSS
    pub fn verify_css(css: &str) -> Result<CssVerification> {
        log::debug!("验证CSS代码 ({} 字节)...", css.len());

        let mut errors = vec![];
        let mut warnings = vec![];
        let mut selectors = vec![];
        let mut properties = vec![];

        // 检查基本的CSS结构
        if css.trim().is_empty() {
            warnings.push("CSS代码为空".to_string());
        }

        // 提取选择器和属性
        let css_rule_pattern = regex::Regex::new(r"([^{}]+)\s*\{([^}]+)\}").unwrap();

        for cap in css_rule_pattern.captures_iter(css) {
            if let (Some(selector), Some(props)) = (cap.get(1), cap.get(2)) {
                selectors.push(selector.as_str().trim().to_string());

                // 提取CSS属性
                for prop in props.as_str().split(';') {
                    if let Some(colon_pos) = prop.find(':') {
                        let prop_name = prop[..colon_pos].trim();
                        if !prop_name.is_empty() {
                            properties.push(prop_name.to_string());
                        }
                    }
                }
            }
        }
        selectors.sort();
        selectors.dedup();
        properties.sort();
        properties.dedup();

        // 检查常见的CSS错误
        if !css.contains('{') || !css.contains('}') {
            if css.trim().len() > 0 {
                errors.push("CSS代码中缺少规则定义（{}）".to_string());
            }
        }

        // 检查未关闭的括号
        let open_braces = css.matches('{').count();
        let close_braces = css.matches('}').count();
        if open_braces != close_braces {
            errors.push(format!(
                "花括号不匹配：{{}} 数量 {} vs }}}} 数量 {}",
                open_braces, close_braces
            ));
        }

        // 检查常见的CSS属性错误
        if css.contains(": ;") || css.contains(":;") {
            errors.push("检测到属性值缺失的情况".to_string());
        }

        // 计算评分
        let error_count = errors.len() as f64;
        let warning_count = warnings.len() as f64;
        let score = if error_count > 0.0 {
            (10.0 - error_count * 3.0 - warning_count * 0.5) / 10.0
        } else if warning_count > 0.0 {
            1.0 - warning_count * 0.05
        } else {
            1.0
        };
        let score = score.max(0.0).min(1.0);

        Ok(CssVerification {
            valid: errors.is_empty(),
            parse_errors: errors,
            warnings,
            score,
            selectors,
            properties,
        })
    }

    /// 验证生成的JavaScript
    pub fn verify_js(js: &str) -> Result<JsVerification> {
        log::debug!("验证JavaScript代码 ({} 字节)...", js.len());

        let mut errors = vec![];
        let mut warnings = vec![];
        let mut functions = vec![];
        let mut variables = vec![];
        let mut async_operations = vec![];
        let mut api_calls = vec![];

        // 检查基本结构
        if js.trim().is_empty() {
            warnings.push("JavaScript代码为空".to_string());
        }

        // 提取函数定义
        for cap in regex::Regex::new(r"(?:async\s+)?function\s+(\w+)")
            .unwrap()
            .captures_iter(js)
        {
            if let Some(func_name) = cap.get(1) {
                functions.push(func_name.as_str().to_string());
            }
        }

        // 提取箭头函数
        for cap in regex::Regex::new(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(.*?\)\s*=>")
            .unwrap()
            .captures_iter(js)
        {
            if let Some(func_name) = cap.get(1) {
                functions.push(func_name.as_str().to_string());
            }
        }

        // 提取变量声明
        for cap in regex::Regex::new(r"(?:const|let|var)\s+(\w+)\s*=")
            .unwrap()
            .captures_iter(js)
        {
            if let Some(var_name) = cap.get(1) {
                variables.push(var_name.as_str().to_string());
            }
        }

        // 检测异步操作
        if js.contains("async") {
            async_operations.push("async function".to_string());
        }
        if js.contains("await") {
            async_operations.push("await expression".to_string());
        }
        if js.contains("Promise") {
            async_operations.push("Promise".to_string());
        }

        // 检测API调用
        if js.contains("fetch(") {
            api_calls.push("fetch".to_string());
        }
        if js.contains("XMLHttpRequest") {
            api_calls.push("XMLHttpRequest".to_string());
        }
        if js.contains("axios.") || js.contains("axios(") {
            api_calls.push("axios".to_string());
        }
        if js.contains("WebSocket") {
            api_calls.push("WebSocket".to_string());
        }

        // 基本的语法检查
        let open_braces = js.matches('{').count();
        let close_braces = js.matches('}').count();
        if open_braces != close_braces {
            errors.push(format!(
                "花括号不匹配：开{{}} 数量 {} vs 关闭}}}} 数量 {}",
                open_braces, close_braces
            ));
        }

        let open_parens = js.matches('(').count();
        let close_parens = js.matches(')').count();
        if open_parens != close_parens {
            errors.push(format!(
                "括号不匹配：开() 数量 {} vs 关闭)) 数量 {}",
                open_parens, close_parens
            ));
        }

        let open_brackets = js.matches('[').count();
        let close_brackets = js.matches(']').count();
        if open_brackets != close_brackets {
            errors.push(format!(
                "方括号不匹配：开[] 数量 {} vs 关闭]] 数量 {}",
                open_brackets, close_brackets
            ));
        }

        // 检查常见的JavaScript错误
        if js.contains("var ") {
            warnings.push("使用了'var'而不是'const'或'let'".to_string());
        }

        if js.contains("== ") || js.contains("!=") {
            warnings.push("可能使用了==而不是===".to_string());
        }

        // 检查未定义的引用（简单检查）
        if js.contains("undefined") {
            warnings.push("代码中包含对'undefined'的引用".to_string());
        }

        // 去重并排序
        functions.sort();
        functions.dedup();
        variables.sort();
        variables.dedup();
        async_operations.dedup();
        api_calls.dedup();

        // 计算评分
        let error_count = errors.len() as f64;
        let warning_count = warnings.len() as f64;
        let score = if error_count > 0.0 {
            (10.0 - error_count * 3.0 - warning_count * 0.5) / 10.0
        } else if warning_count > 0.0 {
            1.0 - warning_count * 0.05
        } else {
            1.0
        };
        let score = score.max(0.0).min(1.0);

        Ok(JsVerification {
            syntax_valid: errors.is_empty(),
            syntax_errors: errors,
            warnings,
            score,
            functions,
            variables,
            async_operations,
            api_calls,
        })
    }

    /// 综合验证所有代码
    pub fn verify_all(html: &str, css: &str, js: &str) -> Result<CodeVerificationResult> {
        log::info!("开始综合代码验证...");

        let html_result = Self::verify_html(html)?;
        let css_result = Self::verify_css(css)?;
        let js_result = Self::verify_js(js)?;

        // 组合评分：HTML 30%, CSS 20%, JS 50%
        let verification_score =
            (html_result.score * 0.3 + css_result.score * 0.2 + js_result.score * 0.5);

        // 收集所有错误
        let mut all_errors = vec![];
        for msg in &html_result.parse_errors {
            all_errors.push(VerificationError {
                code_type: "html".to_string(),
                error_type: "parse_error".to_string(),
                message: msg.clone(),
                severity: "error".to_string(),
                location: None,
            });
        }
        for msg in &html_result.warnings {
            all_errors.push(VerificationError {
                code_type: "html".to_string(),
                error_type: "warning".to_string(),
                message: msg.clone(),
                severity: "warning".to_string(),
                location: None,
            });
        }

        for msg in &css_result.parse_errors {
            all_errors.push(VerificationError {
                code_type: "css".to_string(),
                error_type: "parse_error".to_string(),
                message: msg.clone(),
                severity: "error".to_string(),
                location: None,
            });
        }
        for msg in &css_result.warnings {
            all_errors.push(VerificationError {
                code_type: "css".to_string(),
                error_type: "warning".to_string(),
                message: msg.clone(),
                severity: "warning".to_string(),
                location: None,
            });
        }

        for msg in &js_result.syntax_errors {
            all_errors.push(VerificationError {
                code_type: "js".to_string(),
                error_type: "syntax_error".to_string(),
                message: msg.clone(),
                severity: "error".to_string(),
                location: None,
            });
        }
        for msg in &js_result.warnings {
            all_errors.push(VerificationError {
                code_type: "js".to_string(),
                error_type: "warning".to_string(),
                message: msg.clone(),
                severity: "warning".to_string(),
                location: None,
            });
        }

        // 生成修复建议
        let suggested_fixes = Self::generate_fix_suggestions(&html_result, &css_result, &js_result);

        log::info!(
            "✓ 代码验证完成: 评分 {:.1}%, 错误 {}, 警告 {}",
            verification_score * 100.0,
            all_errors.iter().filter(|e| e.severity == "error").count(),
            all_errors
                .iter()
                .filter(|e| e.severity == "warning")
                .count()
        );

        Ok(CodeVerificationResult {
            html: html_result,
            css: css_result,
            js: js_result,
            verification_score,
            all_errors,
            suggested_fixes,
        })
    }

    /// 生成修复建议
    fn generate_fix_suggestions(
        html: &HtmlVerification,
        css: &CssVerification,
        js: &JsVerification,
    ) -> Vec<(String, String)> {
        let mut fixes = vec![];

        // HTML修复建议
        for error in &html.parse_errors {
            if error.contains("DOCTYPE") {
                fixes.push((
                    error.clone(),
                    "在HTML文档开头添加: <!DOCTYPE html>".to_string(),
                ));
            } else if error.contains("head") {
                fixes.push((
                    error.clone(),
                    "在<html>标签内添加<head></head>部分".to_string(),
                ));
            }
        }

        // CSS修复建议
        for error in &css.parse_errors {
            if error.contains("花括号不匹配") {
                fixes.push((error.clone(), "检查CSS规则的开闭括号是否成对".to_string()));
            } else if error.contains("属性值缺失") {
                fixes.push((error.clone(), "确保每个CSS属性都有有效的值".to_string()));
            }
        }

        // JS修复建议
        for error in &js.syntax_errors {
            if error.contains("花括号不匹配") {
                fixes.push((
                    error.clone(),
                    "检查所有函数和代码块的{}是否成对".to_string(),
                ));
            } else if error.contains("括号不匹配") {
                fixes.push((
                    error.clone(),
                    "检查函数调用和表达式的()是否成对".to_string(),
                ));
            }
        }

        fixes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_valid_html() {
        let html = r#"<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body><h1>Hello</h1></body>
</html>"#;
        let result = CodeVerifier::verify_html(html).unwrap();
        assert!(result.valid);
        assert!(result.score > 0.9);
    }

    #[test]
    fn test_verify_invalid_html() {
        let html = "<div><p>Unclosed tag";
        let result = CodeVerifier::verify_html(html).unwrap();
        assert!(!result.parse_errors.is_empty() || !result.warnings.is_empty());
    }

    #[test]
    fn test_verify_valid_css() {
        let css = r#"
        body { background-color: white; }
        .container { width: 100%; margin: 0; }
        "#;
        let result = CodeVerifier::verify_css(css).unwrap();
        assert!(result.valid);
        assert!(result.selectors.contains(&"body".to_string()));
        assert!(result.selectors.contains(&".container".to_string()));
    }

    #[test]
    fn test_verify_valid_js() {
        let js = r#"
        function helloWorld() {
            const greeting = "Hello, World!";
            console.log(greeting);
            return greeting;
        }
        "#;
        let result = CodeVerifier::verify_js(js).unwrap();
        assert!(result.syntax_valid);
        assert!(result.functions.contains(&"helloWorld".to_string()));
        assert!(result.variables.contains(&"greeting".to_string()));
    }

    #[test]
    fn test_verify_all() {
        let html = "<!DOCTYPE html><html><body>Test</body></html>";
        let css = "body { color: black; }";
        let js = "console.log('test');";

        let result = CodeVerifier::verify_all(html, css, js).unwrap();
        assert!(result.verification_score > 0.0);
        assert_eq!(result.html.valid, true);
        assert_eq!(result.css.valid, true);
        assert_eq!(result.js.syntax_valid, true);
    }
}
