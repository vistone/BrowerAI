//! BrowerAI JavaScript Parser
//!
//! 基于 Boa 的 JavaScript 解析器，提供：
//! - ES2022 标准解析
//! - AST 生成
//!
//! # 示例
//! ```
//! use browerai_js_parser::JsParser;
//! use browerai_core::traits::Parser;
//!
//! let parser = JsParser::new();
//! let js = "function hello() { return 'world'; }";
//! let ast = parser.parse(js).unwrap();
//! ```

#![warn(missing_docs)]

use browerai_core::{traits::Parser, BrowserError, CodeType, Result};
use boa_parser::{Parser as BoaParser, Source};
use boa_interner::ToInternedString;
use boa_ast::scope::Scope;

pub mod ast;

pub use ast::{AstNode, FunctionDecl, JsAst, VariableDecl};

/// JavaScript 解析器
pub struct JsParser {
    /// 是否解析 TypeScript
    parse_typescript: bool,
    /// 是否解析 JSX
    parse_jsx: bool,
    /// 目标 ECMAScript 版本
    target_version: EcmaVersion,
    /// 内部 interner 用于字符串解析
    interner: boa_interner::Interner,
}

/// ECMAScript 版本
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcmaVersion {
    /// ES5
    ES5,
    /// ES2015
    ES2015,
    /// ES2017
    ES2017,
    /// ES2019
    ES2019,
    /// ES2020
    ES2020,
    /// ES2021
    ES2021,
    /// ES2022
    ES2022,
    /// ESNext
    ESNext,
}

impl Default for EcmaVersion {
    fn default() -> Self {
        EcmaVersion::ES2022
    }
}

impl JsParser {
    /// 创建新的 JS 解析器
    pub fn new() -> Self {
        Self {
            parse_typescript: true,
            parse_jsx: true,
            target_version: EcmaVersion::ES2022,
            interner: boa_interner::Interner::default(),
        }
    }

    /// 设置是否解析 TypeScript
    pub fn parse_typescript(mut self, enable: bool) -> Self {
        self.parse_typescript = enable;
        self
    }

    /// 设置是否解析 JSX
    pub fn parse_jsx(mut self, enable: bool) -> Self {
        self.parse_jsx = enable;
        self
    }

    /// 设置目标 ECMAScript 版本
    pub fn target_version(mut self, version: EcmaVersion) -> Self {
        self.target_version = version;
        self
    }

    /// 解析 JavaScript 字符串
    pub fn parse_string(&mut self, js: impl AsRef<str>) -> Result<JsAst> {
        let js = js.as_ref();
        
        // 使用 Boa 解析器
        let mut parser = BoaParser::new(Source::from_bytes(js));
        let scope = Scope::new_global();
        
        let script = parser.parse_script(&scope, &mut self.interner)
            .map_err(|e| BrowserError::parse(format!("JS parse error: {:?}", e)))?;
        
        // 转换为内部 AST 表示
        let ast = self.convert_ast(&script);
        
        Ok(ast)
    }

    /// 将 Boa AST 转换为内部表示
    fn convert_ast(&self, script: &boa_ast::Script) -> JsAst {
        let mut ast = JsAst::new();
        
        // 遍历语句列表
        for stmt in script.statements().as_ref() {
            self.convert_statement_list_item(stmt, &mut ast);
        }
        
        ast
    }

    /// 转换语句列表项
    fn convert_statement_list_item(&self, stmt: &boa_ast::StatementListItem, ast: &mut JsAst) {
        use boa_ast::StatementListItem;
        
        match stmt {
            StatementListItem::Statement(s) => self.convert_statement(s, ast),
            StatementListItem::Declaration(d) => self.convert_declaration(d, ast),
        }
    }

    /// 转换声明
    fn convert_declaration(&self, decl: &boa_ast::Declaration, ast: &mut JsAst) {
        use boa_ast::declaration::Declaration;
        
        match decl {
            Declaration::FunctionDeclaration(func) => {
                let name = Some(func.name().to_interned_string(&self.interner));
                let params: Vec<String> = func.parameters().as_ref().iter()
                    .map(|p: &boa_ast::function::FormalParameter| {
                        p.to_interned_string(&self.interner)
                    })
                    .collect();
                ast.function_decls.push(FunctionDecl {
                    name,
                    params,
                    is_async: false,
                    is_generator: false,
                    body: None,
                });
            }
            Declaration::GeneratorDeclaration(func) => {
                let name = Some(func.name().to_interned_string(&self.interner));
                let params: Vec<String> = func.parameters().as_ref().iter()
                    .map(|p: &boa_ast::function::FormalParameter| {
                        p.to_interned_string(&self.interner)
                    })
                    .collect();
                ast.function_decls.push(FunctionDecl {
                    name,
                    params,
                    is_async: false,
                    is_generator: true,
                    body: None,
                });
            }
            Declaration::AsyncFunctionDeclaration(func) => {
                let name = Some(func.name().to_interned_string(&self.interner));
                let params: Vec<String> = func.parameters().as_ref().iter()
                    .map(|p: &boa_ast::function::FormalParameter| {
                        p.to_interned_string(&self.interner)
                    })
                    .collect();
                ast.function_decls.push(FunctionDecl {
                    name,
                    params,
                    is_async: true,
                    is_generator: false,
                    body: None,
                });
            }
            Declaration::AsyncGeneratorDeclaration(func) => {
                let name = Some(func.name().to_interned_string(&self.interner));
                let params: Vec<String> = func.parameters().as_ref().iter()
                    .map(|p: &boa_ast::function::FormalParameter| {
                        p.to_interned_string(&self.interner)
                    })
                    .collect();
                ast.function_decls.push(FunctionDecl {
                    name,
                    params,
                    is_async: true,
                    is_generator: true,
                    body: None,
                });
            }
            Declaration::Lexical(lex) => {
                let kind = if lex.is_const() { "const" } else { "let" };
                for var in lex.variable_list().as_ref() {
                    if let boa_ast::declaration::Binding::Identifier(id) = var.binding() {
                        ast.variable_decls.push(VariableDecl {
                            name: id.to_interned_string(&self.interner),
                            kind: kind.to_string(),
                            init: var.init().map(|_| "expression".to_string()),
                        });
                    }
                }
            }
            _ => {}
        }
    }

    /// 转换语句
    fn convert_statement(&self, stmt: &boa_ast::Statement, ast: &mut JsAst) {
        use boa_ast::Statement;
        
        match stmt {
            Statement::Var(var_decl) => {
                // 变量声明 - VarDeclaration 是一个 tuple struct
                for var in var_decl.0.as_ref() {
                    if let boa_ast::declaration::Binding::Identifier(id) = var.binding() {
                        ast.variable_decls.push(VariableDecl {
                            name: id.to_interned_string(&self.interner),
                            kind: "var".to_string(),
                            init: var.init().map(|_| "expression".to_string()),
                        });
                    }
                }
            }
            Statement::Block(block) => {
                // 块语句，递归处理
                for stmt in block.statement_list().as_ref() {
                    self.convert_statement_list_item(stmt, ast);
                }
            }
            Statement::If(if_stmt) => {
                // if 语句，递归处理
                self.convert_statement(if_stmt.body(), ast);
                if let Some(else_body) = if_stmt.else_node() {
                    self.convert_statement(else_body, ast);
                }
            }
            Statement::WhileLoop(while_stmt) => {
                self.convert_statement(while_stmt.body(), ast);
            }
            Statement::ForLoop(for_stmt) => {
                self.convert_statement(for_stmt.body(), ast);
            }
            // 其他语句类型...
            _ => {}
        }
    }

    /// 提取所有函数
    pub fn extract_functions<'a>(&self, ast: &'a JsAst) -> Vec<&'a FunctionDecl> {
        ast.function_decls.iter().collect()
    }

    /// 提取所有变量
    pub fn extract_variables<'a>(&self, ast: &'a JsAst) -> Vec<&'a VariableDecl> {
        ast.variable_decls.iter().collect()
    }

    /// 检测代码类型
    pub fn detect_code_type(&self, code: &str) -> CodeType {
        // TypeScript 特征检测
        let ts_patterns = [
            ": string", ": number", ": boolean", ": void", ": any",
            "interface ", "type ", "enum ", "namespace ",
            "as ", "readonly ", "abstract ", "implements ",
        ];
        
        for pattern in &ts_patterns {
            if code.contains(pattern) {
                return CodeType::TypeScript;
            }
        }
        
        CodeType::JavaScript
    }

    /// 检查是否是模块
    pub fn is_module(&self, code: &str) -> bool {
        code.contains("import ") || code.contains("export ")
    }

    /// 提取导入
    pub fn extract_imports(&self, _code: &str) -> Vec<String> {
        // 简化实现
        Vec::new()
    }

    /// 提取导出
    pub fn extract_exports(&self, _code: &str) -> Vec<String> {
        // 简化实现
        Vec::new()
    }
}

impl Default for JsParser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser for JsParser {
    type Input = str;
    type Output = JsAst;

    fn parse(&self, input: &Self::Input) -> Result<Self::Output> {
        // 创建一个新的解析器实例，因为parse_string需要&mut self
        let mut parser = Self::new();
        parser.parse_string(input)
    }
}

/// JS 解析统计
#[derive(Debug, Clone, Default)]
pub struct JsParseStats {
    /// 函数数量
    pub function_count: usize,
    /// 变量声明数量
    pub variable_count: usize,
    /// 语句数量
    pub statement_count: usize,
    /// 最大嵌套深度
    pub max_nesting_depth: usize,
    /// 代码行数
    pub line_count: usize,
}

impl JsParseStats {
    /// 从 AST 计算统计
    pub fn from_ast(ast: &JsAst) -> Self {
        Self {
            function_count: ast.function_decls.len(),
            variable_count: ast.variable_decls.len(),
            statement_count: ast.statement_count(),
            max_nesting_depth: ast.max_nesting_depth(),
            line_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_js() {
        let mut parser = JsParser::new();
        let js = "function hello() { return 'world'; }";
        let ast = parser.parse_string(js).unwrap();
        
        assert!(!ast.function_decls.is_empty());
    }

    #[test]
    fn test_extract_functions() {
        let mut parser = JsParser::new();
        let js = r#"
            function foo() {}
            function bar() {}
            const baz = () => {};
        "#;
        let ast = parser.parse_string(js).unwrap();
        let functions = parser.extract_functions(&ast);
        
        assert!(!functions.is_empty());
    }

    #[test]
    fn test_detect_code_type() {
        let parser = JsParser::new();
        
        let js = "function test() {}";
        assert_eq!(parser.detect_code_type(js), CodeType::JavaScript);
        
        let ts = "function test(): string {}";
        assert_eq!(parser.detect_code_type(ts), CodeType::TypeScript);
    }

    #[test]
    fn test_is_module() {
        let parser = JsParser::new();
        
        let script = "function test() {}";
        assert!(!parser.is_module(script));
        
        let module = "import { foo } from 'bar';";
        assert!(parser.is_module(module));
    }
}
