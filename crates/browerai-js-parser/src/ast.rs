//! JavaScript AST 类型

/// JavaScript AST
#[derive(Debug, Clone, Default)]
pub struct JsAst {
    /// 类型
    pub kind: AstKind,
    /// 语句列表
    pub statements: Vec<AstNode>,
    /// 函数声明
    pub function_decls: Vec<FunctionDecl>,
    /// 变量声明
    pub variable_decls: Vec<VariableDecl>,
    /// 类声明
    pub class_decls: Vec<ClassDecl>,
    /// 导入声明
    pub import_decls: Vec<ImportDecl>,
    /// 导出声明
    pub export_decls: Vec<ExportDecl>,
}

impl JsAst {
    /// 创建新的 AST
    pub fn new() -> Self {
        Self {
            kind: AstKind::Script,
            statements: Vec::new(),
            function_decls: Vec::new(),
            variable_decls: Vec::new(),
            class_decls: Vec::new(),
            import_decls: Vec::new(),
            export_decls: Vec::new(),
        }
    }

    /// 创建模块 AST
    pub fn module() -> Self {
        Self {
            kind: AstKind::Module,
            ..Self::new()
        }
    }

    /// 添加语句
    pub fn add_statement(&mut self, stmt: AstNode) {
        self.statements.push(stmt);
    }

    /// 获取语句数量
    pub fn statement_count(&self) -> usize {
        self.statements.len()
    }

    /// 获取最大嵌套深度
    pub fn max_nesting_depth(&self) -> usize {
        self.statements.iter()
            .map(|s| s.nesting_depth())
            .max()
            .unwrap_or(0)
    }

    /// 获取所有标识符（函数名、变量名、参数名等）
    pub fn all_identifiers(&self) -> Vec<String> {
        let mut ids = Vec::new();
        
        for func in &self.function_decls {
            if let Some(ref name) = func.name {
                ids.push(name.clone());
            }
            // 添加参数名
            for param in &func.params {
                ids.push(param.clone());
            }
        }
        
        for var in &self.variable_decls {
            ids.push(var.name.clone());
        }
        
        for class in &self.class_decls {
            if let Some(ref name) = class.name {
                ids.push(name.clone());
            }
        }
        
        ids
    }

    /// 序列化为 JSON (需要启用 serde feature)
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
    
    /// 序列化为 JSON (简化版本，不需要 serde)
    #[cfg(not(feature = "serde"))]
    pub fn to_json(&self) -> String {
        format!("{{\"kind\":\"{:?}\",\"statements\":[],\"function_decls\":[{}],\"variable_decls\":[{}]}}",
            self.kind,
            self.function_decls.len(),
            self.variable_decls.len()
        )
    }
}

/// AST 类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstKind {
    /// 脚本
    Script,
    /// 模块
    Module,
}

impl Default for AstKind {
    fn default() -> Self {
        AstKind::Script
    }
}

/// AST 节点
#[derive(Debug, Clone)]
pub enum AstNode {
    /// 表达式语句
    Expression(Expression),
    /// 变量声明
    Variable(VariableDecl),
    /// 函数声明
    Function(FunctionDecl),
    /// 类声明
    Class(ClassDecl),
    /// 块语句
    Block(Vec<AstNode>),
    /// If 语句
    If {
        condition: Expression,
        then_branch: Box<AstNode>,
        else_branch: Option<Box<AstNode>>,
    },
    /// While 语句
    While {
        condition: Expression,
        body: Box<AstNode>,
    },
    /// For 语句
    For {
        init: Option<Box<AstNode>>,
        condition: Option<Expression>,
        update: Option<Expression>,
        body: Box<AstNode>,
    },
    /// Return 语句
    Return(Option<Expression>),
    /// Break 语句
    Break(Option<String>),
    /// Continue 语句
    Continue(Option<String>),
    /// Try-Catch
    Try {
        try_block: Box<AstNode>,
        catch_clause: Option<CatchClause>,
        finally_block: Option<Box<AstNode>>,
    },
    /// Throw 语句
    Throw(Expression),
    /// 导入
    Import(ImportDecl),
    /// 导出
    Export(ExportDecl),
}

impl AstNode {
    /// 获取嵌套深度
    pub fn nesting_depth(&self) -> usize {
        match self {
            AstNode::Block(stmts) => {
                1 + stmts.iter().map(|s| s.nesting_depth()).max().unwrap_or(0)
            }
            AstNode::If { then_branch, else_branch, .. } => {
                let then_depth = then_branch.nesting_depth();
                let else_depth = else_branch.as_ref().map(|e| e.nesting_depth()).unwrap_or(0);
                1 + then_depth.max(else_depth)
            }
            AstNode::While { body, .. } => 1 + body.nesting_depth(),
            AstNode::For { body, .. } => 1 + body.nesting_depth(),
            AstNode::Try { try_block, finally_block, .. } => {
                let try_depth = try_block.nesting_depth();
                let finally_depth = finally_block.as_ref().map(|f| f.nesting_depth()).unwrap_or(0);
                1 + try_depth.max(finally_depth)
            }
            _ => 1,
        }
    }
}

/// 表达式
#[derive(Debug, Clone)]
pub enum Expression {
    /// 标识符
    Identifier(String),
    /// 字面量
    Literal(Literal),
    /// 二元表达式
    Binary {
        left: Box<Expression>,
        operator: BinaryOperator,
        right: Box<Expression>,
    },
    /// 一元表达式
    Unary {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    /// 函数调用
    Call {
        callee: Box<Expression>,
        arguments: Vec<Expression>,
    },
    /// 成员访问
    Member {
        object: Box<Expression>,
        property: Box<Expression>,
        computed: bool,
    },
    /// 箭头函数
    ArrowFunction {
        params: Vec<String>,
        body: Box<AstNode>,
    },
    /// 对象字面量
    Object(Vec<Property>),
    /// 数组字面量
    Array(Vec<Expression>),
    /// 条件表达式
    Conditional {
        condition: Box<Expression>,
        consequent: Box<Expression>,
        alternate: Box<Expression>,
    },
    /// 赋值表达式
    Assignment {
        left: Box<Expression>,
        operator: AssignmentOperator,
        right: Box<Expression>,
    },
}

/// 字面量
#[derive(Debug, Clone)]
pub enum Literal {
    /// 字符串
    String(String),
    /// 数字
    Number(f64),
    /// 布尔
    Boolean(bool),
    /// Null
    Null,
    /// Undefined
    Undefined,
    /// 正则表达式
    Regex { pattern: String, flags: String },
    /// 模板字符串
    Template(Vec<TemplateElement>),
}

/// 模板元素
#[derive(Debug, Clone)]
pub struct TemplateElement {
    /// 字符串部分
    pub raw: String,
    /// 是否有表达式
    pub tail: bool,
}

/// 二元运算符
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    /// +
    Add,
    /// -
    Subtract,
    /// *
    Multiply,
    /// /
    Divide,
    /// %
    Modulo,
    /// ==
    Equal,
    /// !=
    NotEqual,
    /// ===
    StrictEqual,
    /// !==
    StrictNotEqual,
    /// <
    LessThan,
    /// <=
    LessThanOrEqual,
    /// >
    GreaterThan,
    /// >=
    GreaterThanOrEqual,
    /// &&
    LogicalAnd,
    /// ||
    LogicalOr,
    /// |
    BitwiseOr,
    /// &
    BitwiseAnd,
    /// ^
    BitwiseXor,
    /// <<
    ShiftLeft,
    /// >>
    ShiftRight,
    /// >>>
    UnsignedShiftRight,
    /// in
    In,
    /// instanceof
    InstanceOf,
    /// **
    Exponent,
}

/// 一元运算符
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    /// -
    Minus,
    /// +
    Plus,
    /// !
    Not,
    /// ~
    BitwiseNot,
    /// typeof
    TypeOf,
    /// void
    Void,
    /// delete
    Delete,
    /// ++
    PreIncrement,
    /// --
    PreDecrement,
}

/// 赋值运算符
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentOperator {
    /// =
    Assign,
    /// +=
    AddAssign,
    /// -=
    SubtractAssign,
    /// *=
    MultiplyAssign,
    /// /=
    DivideAssign,
    /// %=
    ModuloAssign,
    /// **=
    ExponentAssign,
    /// <<=
    ShiftLeftAssign,
    /// >>=
    ShiftRightAssign,
    /// >>>=
    UnsignedShiftRightAssign,
    /// &=
    BitwiseAndAssign,
    /// |=
    BitwiseOrAssign,
    /// ^=
    BitwiseXorAssign,
}

/// 属性
#[derive(Debug, Clone)]
pub struct Property {
    /// 键
    pub key: PropertyKey,
    /// 值
    pub value: Expression,
    /// 是否是方法
    pub is_method: bool,
    /// 是否是 getter
    pub is_getter: bool,
    /// 是否是 setter
    pub is_setter: bool,
}

/// 属性键
#[derive(Debug, Clone)]
pub enum PropertyKey {
    /// 标识符
    Identifier(String),
    /// 字符串
    String(String),
    /// 数字
    Number(f64),
    /// 计算属性
    Computed(Expression),
}

/// 函数声明
#[derive(Debug, Clone, Default)]
pub struct FunctionDecl {
    /// 函数名
    pub name: Option<String>,
    /// 参数列表
    pub params: Vec<String>,
    /// 是否是异步函数
    pub is_async: bool,
    /// 是否是生成器函数
    pub is_generator: bool,
    /// 函数体
    pub body: Option<Vec<AstNode>>,
}

/// 变量声明
#[derive(Debug, Clone)]
pub struct VariableDecl {
    /// 变量名
    pub name: String,
    /// 类型（var/let/const）
    pub kind: String,
    /// 初始化表达式
    pub init: Option<String>,
}

/// 类声明
#[derive(Debug, Clone)]
pub struct ClassDecl {
    /// 类名
    pub name: Option<String>,
    /// 父类
    pub super_class: Option<String>,
    /// 方法
    pub methods: Vec<MethodDef>,
    /// 属性
    pub properties: Vec<PropertyDef>,
}

/// 方法定义
#[derive(Debug, Clone)]
pub struct MethodDef {
    /// 方法名
    pub name: String,
    /// 是否是静态方法
    pub is_static: bool,
    /// 是否是异步方法
    pub is_async: bool,
    /// 是否是生成器方法
    pub is_generator: bool,
    /// 参数
    pub params: Vec<String>,
}

/// 属性定义
#[derive(Debug, Clone)]
pub struct PropertyDef {
    /// 属性名
    pub name: String,
    /// 是否是静态属性
    pub is_static: bool,
}

/// 导入声明
#[derive(Debug, Clone)]
pub struct ImportDecl {
    /// 导入的标识符
    pub specifiers: Vec<ImportSpecifier>,
    /// 来源模块
    pub source: String,
}

/// 导入说明符
#[derive(Debug, Clone)]
pub enum ImportSpecifier {
    /// 默认导入
    Default(String),
    /// 命名导入
    Named { local: String, imported: String },
    /// 命名空间导入
    Namespace(String),
}

/// 导出声明
#[derive(Debug, Clone)]
pub struct ExportDecl {
    /// 导出的标识符
    pub specifiers: Vec<ExportSpecifier>,
    /// 来源模块（如果是重导出）
    pub source: Option<String>,
    /// 是否是默认导出
    pub is_default: bool,
}

/// 导出说明符
#[derive(Debug, Clone)]
pub struct ExportSpecifier {
    /// 本地名称
    pub local: String,
    /// 导出名称
    pub exported: String,
}

/// Catch 子句
#[derive(Debug, Clone)]
pub struct CatchClause {
    /// 参数名
    pub param: Option<String>,
    /// 主体
    pub body: Box<AstNode>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_creation() {
        let mut ast = JsAst::new();
        ast.function_decls.push(FunctionDecl {
            name: Some("test".to_string()),
            params: vec!["a".to_string()],
            is_async: false,
            is_generator: false,
            body: None,
        });
        
        assert_eq!(ast.function_decls.len(), 1);
        assert_eq!(ast.all_identifiers(), vec!["test", "a"]);
    }

    #[test]
    fn test_nesting_depth() {
        let block = AstNode::Block(vec![
            AstNode::Block(vec![
                AstNode::Return(None),
            ]),
        ]);
        
        assert_eq!(block.nesting_depth(), 3);
    }

    #[test]
    fn test_ast_kind() {
        let script = JsAst::new();
        assert_eq!(script.kind, AstKind::Script);
        
        let module = JsAst::module();
        assert_eq!(module.kind, AstKind::Module);
    }
}
