use serde::{Deserialize, Serialize};
/// JavaScript AST分析的核心数据结构
///
/// 这个模块定义了所有用于表示JavaScript代码语义信息的类型。
/// 设计目标：
/// - 与具体的AST实现解耦（不依赖boa_ast）
/// - 便于序列化和传输
/// - 支持增量更新
use std::collections::{HashMap, HashSet};

/// AST元数据：统计和基本信息
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JsAstMetadata {
    /// 代码行数
    pub line_count: usize,

    /// 字符数
    pub char_count: usize,

    /// 语句数
    pub statement_count: usize,

    /// 表达式数
    pub expression_count: usize,

    /// 是否为有效的JavaScript
    pub is_valid: bool,

    /// 代码复杂度评分（0-100）
    pub complexity_score: u32,

    /// 代码长度分类
    pub code_size: CodeSizeCategory,
}

/// 代码长度分类
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CodeSizeCategory {
    /// < 100行
    #[default]
    Tiny,
    /// 100-1000行
    Small,
    /// 1000-10000行
    Medium,
    /// > 10000行
    Large,
}

// ============================================================================
// Phase 2: Location Information
// ============================================================================

/// 位置信息结构 - 记录代码中的精确位置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocationInfo {
    /// 行号 (1-based)
    pub line: usize,
    /// 列号 (0-based)
    pub column: usize,
    /// 字符偏移 (从文件开始)
    pub start: usize,
    /// 字符偏移 (范围结束)
    pub end: usize,
}

impl LocationInfo {
    pub fn new(line: usize, column: usize, start: usize, end: usize) -> Self {
        Self {
            line,
            column,
            start,
            end,
        }
    }
}

/// 函数声明信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsFunctionInfo {
    /// 函数唯一ID
    pub id: String,

    /// 函数名称（可能为空，表示匿名函数）
    pub name: Option<String>,

    /// 函数的作用域级别（全局=0）
    pub scope_level: u32,

    /// 参数列表
    pub parameters: Vec<JsParameter>,

    /// 返回值类型提示（基于静态分析）
    pub return_type_hint: Option<String>,

    /// 函数体的语句数
    pub statement_count: usize,

    /// 函数复杂度
    pub cyclomatic_complexity: u32,

    /// 是否异步函数
    pub is_async: bool,

    /// 是否生成器函数
    pub is_generator: bool,

    /// 捕获的外部变量
    pub captured_vars: Vec<String>,

    /// 定义的局部变量
    pub local_vars: Vec<String>,

    /// 调用的其他函数
    pub called_functions: Vec<String>,

    /// 出现的行号范围
    pub start_line: usize,
    pub end_line: usize,
}

/// 函数参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsParameter {
    pub name: String,
    pub has_default: bool,
    pub is_rest: bool,
    pub type_hint: Option<String>,
}

/// 类声明信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsClassInfo {
    /// 类唯一ID
    pub id: String,

    /// 类名
    pub name: String,

    /// 父类（如果有）
    pub parent_class: Option<Box<JsClassInfo>>,

    /// 实现的接口
    pub implements: Vec<String>,

    /// 属性列表
    pub properties: Vec<JsProperty>,

    /// 方法列表
    pub methods: Vec<JsMethod>,

    /// 静态方法列表
    pub static_methods: Vec<JsMethod>,

    /// 构造函数信息
    pub constructor: Option<Box<JsFunctionInfo>>,

    /// 出现的行号范围
    pub start_line: usize,
    pub end_line: usize,
}

/// 类属性
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsProperty {
    pub name: String,
    pub is_static: bool,
    pub is_private: bool,
    pub type_hint: Option<String>,
}

/// 类方法（简化版函数信息）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsMethod {
    pub name: String,
    pub is_static: bool,
    pub is_private: bool,
    pub is_async: bool,
    pub parameters: Vec<JsParameter>,
}

/// 事件处理器信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsEventHandler {
    /// 事件处理器ID
    pub id: String,

    /// 事件类型（click, change, submit等）
    pub event_type: String,

    /// 处理器函数ID
    pub handler_function_id: String,

    /// 目标选择器（如果可识别）
    pub target_selector: Option<String>,

    /// 是否为事件委托
    pub is_delegated: bool,

    /// 事件监听方式（addEventListener, onclick属性等）
    pub binding_method: EventBindingMethod,
}

/// 事件绑定方式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventBindingMethod {
    /// addEventListener('event', handler)
    AddEventListener,
    /// element.onclick = handler
    DirectProperty,
    /// <div onclick="handler">
    HtmlAttribute,
    /// jQuery: $(selector).on('event', handler)
    JqueryOn,
    /// jQuery: $(selector).click(handler)
    JqueryShorthand,
    /// 其他方式
    Other,
}

/// 调用图节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsCallNode {
    /// 函数ID
    pub id: String,

    /// 函数名
    pub name: Option<String>,

    /// 调用的其他函数
    pub callees: Vec<String>,

    /// 被调用的地点（函数ID）
    pub callers: Vec<String>,

    /// 调用深度
    pub depth: u32,

    /// 是否为入口点（通常指全局作用域调用的函数）
    pub is_entry_point: bool,
}

/// 调用图
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JsCallGraph {
    /// 所有调用节点
    pub nodes: Vec<JsCallNode>,

    /// 循环调用集合（如果存在，表示存在递归）
    pub cycles: Vec<Vec<String>>,

    /// 最大调用深度
    pub max_depth: u32,
}

impl JsCallGraph {
    /// 检查是否存在循环调用
    pub fn has_cycles(&self) -> bool {
        !self.cycles.is_empty()
    }

    /// 获取指定函数的调用链
    pub fn get_call_chain(&self, func_id: &str) -> Option<Vec<String>> {
        // BFS遍历调用图
        let mut visited = HashSet::new();
        let mut queue = vec![func_id.to_string()];
        let mut chain = vec![func_id.to_string()];

        while !queue.is_empty() {
            let current = queue.remove(0);
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(node) = self.nodes.iter().find(|n| n.id == current) {
                for callee in &node.callees {
                    if !visited.contains(callee) {
                        queue.push(callee.clone());
                        chain.push(callee.clone());
                    }
                }
            }
        }

        Some(chain)
    }
}

/// 依赖图节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsDependencyNode {
    /// 模块/文件ID
    pub id: String,

    /// 依赖的其他模块
    pub dependencies: Vec<String>,

    /// 被依赖的模块
    pub dependents: Vec<String>,
}

/// 依赖图
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JsDependencyGraph {
    /// 所有依赖节点
    pub nodes: Vec<JsDependencyNode>,

    /// 循环依赖集合
    pub cycles: Vec<Vec<String>>,
}

impl JsDependencyGraph {
    /// 检查是否有循环依赖
    pub fn has_circular_dependencies(&self) -> bool {
        !self.cycles.is_empty()
    }
}

/// 导入信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsImportInfo {
    /// 导入的模块路径
    pub module_path: String,

    /// 导入的符号列表
    pub symbols: Vec<String>,

    /// 是否为默认导入
    pub is_default: bool,

    /// 导入的别名
    pub alias: Option<String>,
}

/// 导出信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsExportInfo {
    /// 导出的符号名
    pub symbol_name: String,

    /// 是否为默认导出
    pub is_default: bool,

    /// 导出的实际值（如果可识别）
    pub export_value: Option<String>,
}

/// 模块信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsModuleInfo {
    /// 模块ID（通常是文件路径）
    pub id: String,

    /// 模块类型
    pub module_type: ModuleType,

    /// 导入列表
    pub imports: Vec<JsImportInfo>,

    /// 导出列表
    pub exports: Vec<JsExportInfo>,

    /// 引用的全局变量
    pub global_references: Vec<String>,
}

/// 模块类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModuleType {
    /// ES6模块
    Es6Module,
    /// CommonJS模块
    CommonJs,
    /// AMD模块
    Amd,
    /// UMD模块
    Umd,
    /// 全局脚本（非模块）
    Global,
}

/// 完整的语义信息
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JsSemanticInfo {
    /// 所有函数声明
    pub functions: Vec<JsFunctionInfo>,

    /// 所有类声明
    pub classes: Vec<JsClassInfo>,

    /// 所有事件处理器
    pub event_handlers: Vec<JsEventHandler>,

    /// 全局变量列表
    pub global_vars: Vec<String>,

    /// 是否使用了eval
    pub uses_eval: bool,

    /// 是否使用了动态require
    pub uses_dynamic_require: bool,

    /// 检测到的框架
    pub detected_frameworks: Vec<String>,

    /// 其他值得注意的特性
    pub special_features: Vec<String>,
}

impl JsSemanticInfo {
    /// 查找指定名称的函数
    pub fn find_function(&self, name: &str) -> Option<&JsFunctionInfo> {
        self.functions
            .iter()
            .find(|f| f.name.as_deref() == Some(name))
    }

    /// 查找指定名称的类
    pub fn find_class(&self, name: &str) -> Option<&JsClassInfo> {
        self.classes.iter().find(|c| c.name == name)
    }

    /// 获取函数总数
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// 获取类总数
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_default() {
        let metadata = JsAstMetadata::default();
        assert_eq!(metadata.complexity_score, 0);
        assert!(!metadata.is_valid);
    }

    #[test]
    fn test_call_graph_operations() {
        let mut graph = JsCallGraph::default();
        graph.nodes.push(JsCallNode {
            id: "func1".to_string(),
            name: Some("func1".to_string()),
            callees: vec!["func2".to_string()],
            callers: vec![],
            depth: 0,
            is_entry_point: true,
        });

        assert!(!graph.has_cycles());
        assert!(graph.get_call_chain("func1").is_some());
    }
}

// ============================================================================
// Phase 3: Scope Analysis Types
// ============================================================================

/// Variable binding kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingKind {
    /// let binding
    Let,
    /// const binding
    Const,
    /// var binding
    Var,
    /// function parameter
    Parameter,
    /// function declaration
    Function,
    /// class declaration
    Class,
    /// import binding
    Import,
}

/// Variable binding information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableBinding {
    /// Variable name
    pub name: String,

    /// Binding kind
    pub kind: BindingKind,

    /// Scope ID where this variable is defined
    pub scope_id: String,

    /// Whether this variable is captured by a closure
    pub is_captured: bool,

    /// Whether this variable shadows another variable
    pub is_shadowing: bool,

    /// Location information
    pub location: Option<LocationInfo>,
}

/// Closure capture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosureInfo {
    /// Function ID that creates the closure
    pub function_id: String,

    /// Variables captured from outer scope
    pub captured_variables: Vec<String>,

    /// Parent scope ID
    pub parent_scope_id: String,
}

/// Scope information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    /// Unique scope identifier
    pub id: String,

    /// Parent scope ID (None for global scope)
    pub parent: Option<String>,

    /// Variables defined in this scope
    pub variables: HashMap<String, VariableBinding>,

    /// Functions defined in this scope
    pub functions: Vec<String>,

    /// Closures created in this scope
    pub closures: Vec<ClosureInfo>,

    /// Nested child scopes
    pub children: Vec<String>,

    /// Scope type
    pub scope_type: ScopeType,
}

/// Scope type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScopeType {
    /// Global scope
    Global,
    /// Function scope
    Function,
    /// Block scope (if, for, etc.)
    Block,
    /// Module scope
    Module,
    /// Class scope
    Class,
}

/// Scope tree - complete scope hierarchy
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopeTree {
    /// All scopes in the tree
    pub scopes: HashMap<String, Scope>,

    /// Global scope ID
    pub global_scope_id: String,

    /// Variable shadowing warnings
    pub shadowing_warnings: Vec<String>,
}

impl ScopeTree {
    /// Create a new scope tree with a global scope
    pub fn new() -> Self {
        let global_id = "scope_0".to_string();
        let mut scopes = HashMap::new();

        scopes.insert(
            global_id.clone(),
            Scope {
                id: global_id.clone(),
                parent: None,
                variables: HashMap::new(),
                functions: Vec::new(),
                closures: Vec::new(),
                children: Vec::new(),
                scope_type: ScopeType::Global,
            },
        );

        Self {
            scopes,
            global_scope_id: global_id,
            shadowing_warnings: Vec::new(),
        }
    }

    /// Look up a variable in the scope tree
    pub fn lookup_variable(&self, var_name: &str) -> Option<&VariableBinding> {
        // Search from global scope
        self.lookup_in_scope(&self.global_scope_id, var_name)
    }

    /// Look up a variable in a specific scope and its parents
    fn lookup_in_scope(&self, scope_id: &str, var_name: &str) -> Option<&VariableBinding> {
        if let Some(scope) = self.scopes.get(scope_id) {
            if let Some(binding) = scope.variables.get(var_name) {
                return Some(binding);
            }

            // Search parent scope
            if let Some(parent_id) = &scope.parent {
                return self.lookup_in_scope(parent_id, var_name);
            }
        }
        None
    }

    /// Get all shadowing warnings
    pub fn get_shadowing_warnings(&self) -> &[String] {
        &self.shadowing_warnings
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;

    #[test]
    fn test_scope_tree_creation() {
        let tree = ScopeTree::new();
        assert!(tree.scopes.contains_key(&tree.global_scope_id));
        assert_eq!(tree.scopes.len(), 1);
    }

    #[test]
    fn test_variable_lookup() {
        let mut tree = ScopeTree::new();

        // Add a variable to global scope
        if let Some(global_scope) = tree.scopes.get_mut(&tree.global_scope_id) {
            global_scope.variables.insert(
                "x".to_string(),
                VariableBinding {
                    name: "x".to_string(),
                    kind: BindingKind::Let,
                    scope_id: tree.global_scope_id.clone(),
                    is_captured: false,
                    is_shadowing: false,
                    location: None,
                },
            );
        }

        let binding = tree.lookup_variable("x");
        assert!(binding.is_some());
        assert_eq!(binding.unwrap().name, "x");
    }
}

// ============================================================================
// Phase 3 Day 3-4: Data Flow Analysis Types
// ============================================================================

/// Data flow node type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataFlowNodeType {
    /// Variable definition
    Definition,
    /// Variable use/read
    Use,
    /// Variable assignment
    Assignment,
    /// Function return
    Return,
    /// Parameter binding
    Parameter,
}

/// Data flow node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowNode {
    /// Unique node ID
    pub id: String,

    /// Node type
    pub node_type: DataFlowNodeType,

    /// Variable name
    pub variable: String,

    /// Location in source code
    pub location: Option<LocationInfo>,

    /// Function scope where this occurs
    pub scope_id: String,

    /// Line number for ordering
    pub line: usize,
}

/// Data flow edge connecting def-use chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowEdge {
    /// Source node ID (definition or assignment)
    pub from: String,

    /// Target node ID (use)
    pub to: String,

    /// Whether this edge is reachable
    pub is_reachable: bool,
}

/// Data flow graph
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataFlowGraph {
    /// All nodes in the graph
    pub nodes: HashMap<String, DataFlowNode>,

    /// All edges in the graph
    pub edges: Vec<DataFlowEdge>,

    /// Variables with multiple definitions (potential issues)
    pub multi_def_variables: Vec<String>,

    /// Unused variables
    pub unused_variables: Vec<String>,

    /// Potentially constant variables
    pub constant_candidates: Vec<String>,
}

impl DataFlowGraph {
    /// Create a new empty data flow graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Find all uses of a variable
    pub fn find_uses(&self, var_name: &str) -> Vec<&DataFlowNode> {
        self.nodes
            .values()
            .filter(|n| n.variable == var_name && n.node_type == DataFlowNodeType::Use)
            .collect()
    }

    /// Find all definitions of a variable
    pub fn find_definitions(&self, var_name: &str) -> Vec<&DataFlowNode> {
        self.nodes
            .values()
            .filter(|n| {
                n.variable == var_name
                    && (n.node_type == DataFlowNodeType::Definition
                        || n.node_type == DataFlowNodeType::Assignment)
            })
            .collect()
    }

    /// Check if a variable is unused
    pub fn is_unused(&self, var_name: &str) -> bool {
        self.unused_variables.contains(&var_name.to_string())
    }

    /// Get constant candidates (variables never reassigned)
    pub fn get_constant_candidates(&self) -> &[String] {
        &self.constant_candidates
    }
}

#[cfg(test)]
mod dataflow_tests {
    use super::*;

    #[test]
    fn test_dataflow_graph_creation() {
        let graph = DataFlowGraph::new();
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_dataflow_node_creation() {
        let node = DataFlowNode {
            id: "node_1".to_string(),
            node_type: DataFlowNodeType::Definition,
            variable: "x".to_string(),
            location: None,
            scope_id: "scope_0".to_string(),
            line: 1,
        };

        assert_eq!(node.variable, "x");
        assert_eq!(node.node_type, DataFlowNodeType::Definition);
    }
}

// ============================================================================
// Phase 3 Week 2: Control Flow Analysis Types
// ============================================================================

/// Control flow node type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CFGNodeType {
    /// Entry node
    Entry,
    /// Exit node
    Exit,
    /// Regular statement
    Statement,
    /// Conditional branch
    Branch,
    /// Loop node
    Loop,
    /// Return statement
    Return,
    /// Throw statement
    Throw,
}

/// Control flow node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CFGNode {
    /// Unique node identifier
    pub id: String,

    /// Node type
    pub node_type: CFGNodeType,

    /// Original statement (if applicable)
    pub statement: Option<String>,

    /// Line number
    pub line: usize,

    /// Column number
    pub column: usize,

    /// Whether this node is reachable
    pub is_reachable: bool,
}

/// Control flow edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CFGEdge {
    /// Source node ID
    pub from: String,

    /// Target node ID
    pub to: String,

    /// Edge type
    pub edge_type: EdgeType,
}

/// Type of control flow edge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Unconditional edge
    Unconditional,
    /// True branch
    True,
    /// False branch
    False,
    /// Exception edge
    Exception,
    /// Back edge (loop)
    BackEdge,
}

/// Loop information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopInfo {
    /// Loop header node ID
    pub header: String,

    /// Loop latch node ID (where back edge originates)
    pub latch: Option<String>,

    /// Nodes in the loop body
    pub body_nodes: Vec<String>,

    /// Loop type
    pub loop_type: LoopType,
}

/// Loop type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopType {
    /// While loop
    While,
    /// Do-While loop
    DoWhile,
    /// For loop
    For,
    /// For-in/for-of loop
    ForIn,
    /// Other loop
    Other,
}

/// Control flow graph for a function
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControlFlowGraph {
    /// Entry node ID
    pub entry: Option<String>,

    /// Exit node ID
    pub exit: Option<String>,

    /// All nodes in the graph
    pub nodes: Vec<CFGNode>,

    /// All edges in the graph
    pub edges: Vec<CFGEdge>,

    /// Detected loops
    pub loops: Vec<LoopInfo>,

    /// Unreachable nodes
    pub unreachable_nodes: Vec<String>,

    /// Strongly connected components (for loop detection)
    pub sccs: Vec<Vec<String>>,
}

impl ControlFlowGraph {
    /// Create a new empty CFG
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a node is reachable
    pub fn is_reachable(&self, node_id: &str) -> bool {
        self.nodes
            .iter()
            .find(|n| n.id == node_id)
            .map(|n| n.is_reachable)
            .unwrap_or(false)
    }

    /// Get all successors of a node
    pub fn get_successors(&self, node_id: &str) -> Vec<&CFGNode> {
        let successor_ids: Vec<_> = self
            .edges
            .iter()
            .filter(|e| e.from == node_id)
            .map(|e| &e.to)
            .collect();

        self.nodes
            .iter()
            .filter(|n| successor_ids.contains(&&n.id))
            .collect()
    }

    /// Get all predecessors of a node
    pub fn get_predecessors(&self, node_id: &str) -> Vec<&CFGNode> {
        let predecessor_ids: Vec<_> = self
            .edges
            .iter()
            .filter(|e| e.to == node_id)
            .map(|e| &e.from)
            .collect();

        self.nodes
            .iter()
            .filter(|n| predecessor_ids.contains(&&n.id))
            .collect()
    }

    /// Find unreachable code
    pub fn find_unreachable_code(&self) -> Vec<&CFGNode> {
        self.nodes
            .iter()
            .filter(|n| !n.is_reachable && n.node_type != CFGNodeType::Exit)
            .collect()
    }
}

#[cfg(test)]
mod cfg_tests {
    use super::*;

    #[test]
    fn test_cfg_creation() {
        let cfg = ControlFlowGraph::new();
        assert!(cfg.nodes.is_empty());
        assert!(cfg.edges.is_empty());
    }

    #[test]
    fn test_cfg_node_creation() {
        let node = CFGNode {
            id: "node_1".to_string(),
            node_type: CFGNodeType::Statement,
            statement: Some("let x = 10;".to_string()),
            line: 1,
            column: 0,
            is_reachable: true,
        };

        assert_eq!(node.node_type, CFGNodeType::Statement);
        assert!(node.is_reachable);
    }

    #[test]
    fn test_cfg_edge_creation() {
        let edge = CFGEdge {
            from: "node_1".to_string(),
            to: "node_2".to_string(),
            edge_type: EdgeType::Unconditional,
        };

        assert_eq!(edge.edge_type, EdgeType::Unconditional);
    }
}
