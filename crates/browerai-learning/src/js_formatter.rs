//! JavaScript 代码格式化器
//!
//! 使用 SWC 解析 AST 并重新排版，保证语法正确且换行缩进合理。

use std::io;

use anyhow::Result;
use swc_core::common::errors::Handler;
use swc_core::common::{sync::Lrc, FileName, SourceMap, GLOBALS};
use swc_core::ecma::ast::EsVersion;
use swc_core::ecma::ast::Module;
use swc_core::ecma::codegen::text_writer::JsWriter;
use swc_core::ecma::codegen::{Config as CodegenConfig, Emitter};
use swc_core::ecma::parser::{EsSyntax, Parser, StringInput, Syntax};

/// JavaScript 格式化器
pub struct JsFormatter;

impl JsFormatter {
    pub fn new() -> Self {
        Self
    }

    /// 格式化 JavaScript 代码。优先使用 SWC，失败则降级到简单规则。
    pub fn format(&self, code: &str) -> String {
        match self.format_with_swc(code) {
            Ok(formatted) => formatted,
            Err(_) => self.format_simple(code),
        }
    }

    /// 使用 SWC AST 重新排版，保证语法正确。
    fn format_with_swc(&self, code: &str) -> Result<String> {
        let cm: Lrc<SourceMap> = Default::default();
        let handler = Handler::with_emitter_writer(Box::new(io::sink()), Some(cm.clone()));
        let fm = cm.new_source_file(FileName::Custom("input.js".into()), code.to_string());

        let res: Result<String> = GLOBALS.set(&swc_core::common::Globals::new(), || {
            let mut parser = Parser::new(
                Syntax::Es(EsSyntax::default()),
                StringInput::from(&*fm),
                None,
            );

            let module: Module = parser.parse_module().map_err(|e| {
                e.into_diagnostic(&handler).emit();
                anyhow::anyhow!("SWC parse failed")
            })?;

            let mut buf = Vec::new();
            {
                let writer = JsWriter::new(cm.clone(), "\n", &mut buf, None);
                let mut cfg = CodegenConfig::default();
                cfg.minify = false;
                cfg.ascii_only = false;
                cfg.target = EsVersion::Es2022;

                let mut emitter = Emitter {
                    cfg,
                    comments: None,
                    cm: cm.clone(),
                    wr: writer,
                };
                emitter.emit_module(&module)?;
            }

            Ok(String::from_utf8(buf)?)
        });

        res
    }

    /// 简单的基于规则的格式化（备用方案）
    fn format_simple(&self, code: &str) -> String {
        let mut result = String::with_capacity(code.len() * 2);
        let mut indent = 0;
        let mut in_string = false;
        let mut string_char = ' ';
        let mut in_regex = false;
        let mut last_non_space = ' ';

        let chars: Vec<char> = code.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];
            let prev = if i > 0 { chars[i - 1] } else { ' ' };
            let next = if i + 1 < chars.len() {
                chars[i + 1]
            } else {
                ' '
            };

            // 处理转义
            if prev == '\\' && (in_string || in_regex) {
                result.push(ch);
                i += 1;
                continue;
            }

            // 字符串处理
            if (ch == '"' || ch == '\'' || ch == '`') && !in_regex {
                if !in_string {
                    in_string = true;
                    string_char = ch;
                } else if ch == string_char {
                    in_string = false;
                }
                result.push(ch);
                i += 1;
                continue;
            }

            if in_string {
                result.push(ch);
                i += 1;
                continue;
            }

            // 正则表达式检测和处理
            if ch == '/' && !in_string {
                if !in_regex
                    && [
                        '=', '(', ',', '[', ':', '!', '&', '|', '?', '+', '-', '*', '%', '^', '~',
                        ';', '\n', ' ',
                    ]
                    .contains(&last_non_space)
                {
                    in_regex = true;
                    result.push(ch);
                    i += 1;
                    continue;
                }

                if in_regex && prev != '\\' {
                    in_regex = false;
                    result.push(ch);
                    i += 1;
                    while i < chars.len() && chars[i].is_alphabetic() {
                        result.push(chars[i]);
                        i += 1;
                    }
                    continue;
                }
            }

            if in_regex {
                result.push(ch);
                i += 1;
                continue;
            }

            match ch {
                '{' => {
                    result.push(' ');
                    result.push(ch);
                    indent += 1;
                    result.push('\n');
                    result.push_str(&"  ".repeat(indent));
                }
                '}' => {
                    indent = indent.saturating_sub(1);
                    result.push('\n');
                    result.push_str(&"  ".repeat(indent));
                    result.push(ch);
                    if next != ';' && next != ',' && next != ')' && next != '}' {
                        result.push('\n');
                        result.push_str(&"  ".repeat(indent));
                    }
                }
                ';' => {
                    result.push(ch);
                    if next != '}' && next != ')' {
                        result.push('\n');
                        result.push_str(&"  ".repeat(indent));
                    }
                }
                ',' => {
                    result.push(ch);
                    if last_non_space != '}' {
                        result.push(' ');
                    }
                }
                _ => {
                    result.push(ch);
                }
            }

            if ch != ' ' && ch != '\n' && ch != '\t' {
                last_non_space = ch;
            }

            i += 1;
        }

        result.trim().to_string()
    }
}

impl Default for JsFormatter {
    fn default() -> Self {
        Self::new()
    }
}

/// 便捷函数：格式化 JavaScript 代码
pub fn format_js(code: &str) -> String {
    JsFormatter::new().format(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_formatting() {
        let input = "function foo(){return 42;}";
        let formatter = JsFormatter::new();
        let output = formatter.format(input);
        assert!(output.contains('\n'));
        assert!(output.lines().count() > 1);
    }

    #[test]
    fn test_nested_blocks() {
        let input = "function foo(){if(true){return 42;}else{return 0;}}";
        let formatter = JsFormatter::new();
        let output = formatter.format(input);
        assert!(output.contains('\n'));
        println!("Formatted:\n{}", output);
    }

    #[test]
    fn test_string_preservation() {
        let input = r#"var x="hello;world";var y='test{data}';"#;
        let formatter = JsFormatter::new();
        let output = formatter.format(input);
        assert!(output.contains("hello;world"));
        assert!(output.contains("test{data}"));
    }

    #[test]
    fn test_comma_separation() {
        let input = "var a=1,b=2,c=3;";
        let formatter = JsFormatter::new();
        let output = formatter.format(input);
        println!("Comma test:\n{}", output);
        assert!(output.contains(", ") || output.contains(",\n"));
    }
}
