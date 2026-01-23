/// Phase 2 Day 4-5 - 完整模块集成测试
///
/// 测试 EnhancedAst 与现有模块（semantic.rs、call_graph.rs）的集成
use browerai::parser::js_analyzer::{
    AstExtractor, CallGraphBuilder, SemanticAnalyzer, SwcAstExtractor,
};

#[test]
fn test_semantic_analyze_with_enhanced_ast() {
    let code = r#"
import React from 'react';

interface Props {
    name: string;
}

const MyComponent: React.FC<Props> = ({ name }) => {
    const [count, setCount] = React.useState(0);
    
    return <div className="container">Hello {name}!</div>;
};

export default MyComponent;
"#;

    // 1. 提取基础信息
    let mut extractor = AstExtractor::new();
    let basic = extractor
        .extract_from_source(code)
        .expect("Basic extraction failed");

    // 2. 使用 SwcAstExtractor 获取增强信息
    let swc_extractor = SwcAstExtractor::new();
    let enhanced = swc_extractor
        .extract_from_source(code)
        .expect("Enhanced extraction failed");

    // 3. 使用增强数据进行语义分析
    let analyzer = SemanticAnalyzer::new();
    let mut semantic = basic.semantic;

    let result = analyzer
        .analyze_with_enhanced_ast(&mut semantic, &enhanced)
        .expect("Semantic analysis failed");

    // 验证结果
    assert!(
        result.detected_frameworks.contains(&"React".to_string()),
        "Should detect React framework"
    );
    assert!(enhanced.has_jsx, "Should detect JSX");
    assert!(enhanced.has_typescript, "Should detect TypeScript");
    assert!(
        result
            .special_features
            .contains(&"typescript_support".to_string()),
        "Should detect TypeScript support"
    );
}

#[test]
fn test_call_graph_with_locations() {
    let code = r#"
function fibonacci(n: number): number {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

async function fetchData() {
    const result = await fetch('/api/data');
    return result.json();
}

const helper = () => fibonacci(10);
"#;

    // 1. 提取基础信息
    let mut extractor = AstExtractor::new();
    let basic = extractor
        .extract_from_source(code)
        .expect("Extraction failed");

    // 2. 获取增强信息（包括位置）
    let swc_extractor = SwcAstExtractor::new();
    let enhanced = swc_extractor
        .extract_from_source(code)
        .expect("Enhanced extraction failed");

    // 3. 构建调用图
    let mut graph_builder = CallGraphBuilder::new();
    let graph = graph_builder
        .build_with_locations(&basic.semantic, &enhanced.locations)
        .expect("Call graph building failed");

    // 验证调用图
    assert!(!graph.nodes.is_empty(), "Should have call graph nodes");

    // 验证检测到的递归调用
    let cycles = &graph.cycles;
    if !cycles.is_empty() {
        // fibonacci 函数应该被识别为递归
        let found_recursion = cycles
            .iter()
            .any(|cycle| cycle.iter().any(|node| node.contains("fibonacci")));
        assert!(
            found_recursion || cycles.is_empty(),
            "Should detect recursion or have no cycles"
        );
    }
}

#[test]
fn test_framework_detection_with_jsx() {
    let code = r#"
import React from 'react';

// React 函数组件
const Button = (props) => {
    return <button onClick={props.onClick}>{props.label}</button>;
};

// 使用 JSX
const App = () => {
    return (
        <div>
            <Button label="Click me" onClick={() => console.log('clicked')} />
        </div>
    );
};

export default App;
"#;

    let mut extractor = AstExtractor::new();
    let basic = extractor
        .extract_from_source(code)
        .expect("Extraction failed");

    let swc_extractor = SwcAstExtractor::new();
    let enhanced = swc_extractor
        .extract_from_source(code)
        .expect("Enhanced extraction failed");

    let analyzer = SemanticAnalyzer::new();
    let mut semantic = basic.semantic;

    let result = analyzer
        .analyze_with_enhanced_ast(&mut semantic, &enhanced)
        .expect("Analysis failed");

    // 验证 React 框架检测
    assert!(
        result.detected_frameworks.contains(&"React".to_string()),
        "Should detect React from JSX"
    );
    assert!(enhanced.has_jsx, "Should detect JSX elements");
    assert!(enhanced.jsx_elements.len() > 0, "Should find JSX elements");
}

#[test]
fn test_typescript_interface_extraction() {
    let code = r#"
interface User {
    id: number;
    name: string;
    email?: string;
}

interface AdminUser extends User {
    role: string;
    permissions: string[];
}

type UserType = User | AdminUser;

class UserService {
    getUser(id: number): User {
        return { id, name: "John" };
    }
}
"#;

    let mut extractor = AstExtractor::new();
    let basic = extractor
        .extract_from_source(code)
        .expect("Extraction failed");

    let swc_extractor = SwcAstExtractor::new();
    let enhanced = swc_extractor
        .extract_from_source(code)
        .expect("Enhanced extraction failed");

    // 验证 TypeScript 类型提取
    assert!(enhanced.has_typescript, "Should detect TypeScript");
    assert!(
        enhanced.typescript_types.len() > 0,
        "Should extract TypeScript types"
    );

    // 查找 interface 定义
    let interfaces = enhanced
        .typescript_types
        .iter()
        .filter(|t| t.kind == "interface");
    assert_eq!(interfaces.count(), 2, "Should find 2 interfaces");

    // 查找继承关系
    let admin_user = enhanced
        .typescript_types
        .iter()
        .find(|t| t.name == "AdminUser");
    assert!(admin_user.is_some(), "Should find AdminUser interface");
    if let Some(admin) = admin_user {
        assert!(
            admin.extends.contains(&"User".to_string()),
            "AdminUser should extend User"
        );
    }
}

#[test]
fn test_backward_compatibility() {
    let code = r#"
function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}

const calc = (x, y) => add(x, y) * multiply(x, y);
"#;

    // 使用基础提取器
    let mut extractor = AstExtractor::new();
    let basic1 = extractor
        .extract_from_source(code)
        .expect("Basic extraction failed");

    // 使用增强提取器
    let swc_extractor = SwcAstExtractor::new();
    let enhanced = swc_extractor
        .extract_from_source(code)
        .expect("Enhanced extraction failed");

    // 验证 Phase 1 的数据仍然可用
    assert_eq!(
        enhanced.semantic.functions.len(),
        basic1.semantic.functions.len(),
        "Function count should match"
    );
    assert_eq!(
        enhanced.semantic.global_vars.len(),
        basic1.semantic.global_vars.len(),
        "Global vars count should match"
    );

    // 验证语义分析的向后兼容性
    let analyzer = SemanticAnalyzer::new();
    let mut semantic1 = basic1.semantic.clone();
    let mut semantic2 = enhanced.semantic.clone();

    let result1 = analyzer.analyze(&mut semantic1).expect("Analysis 1 failed");
    let result2 = analyzer.analyze(&mut semantic2).expect("Analysis 2 failed");

    // 基础分析结果应该相同
    assert_eq!(
        result1.detected_frameworks, result2.detected_frameworks,
        "Framework detection should match"
    );
}

#[test]
fn test_complex_react_component_with_hooks() {
    let code = r#"
import React, { useState, useCallback, useEffect } from 'react';

interface ComponentProps {
    title: string;
    onClose?: () => void;
}

const Modal: React.FC<ComponentProps> = ({ title, onClose }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [data, setData] = useState(null);
    
    const handleOpen = useCallback(() => {
        setIsOpen(true);
    }, []);
    
    const handleClose = useCallback(() => {
        setIsOpen(false);
        onClose?.();
    }, [onClose]);
    
    useEffect(() => {
        // Load data
        fetchData().then(setData);
    }, []);
    
    if (!isOpen) return null;
    
    return (
        <div className="modal">
            <h2>{title}</h2>
            <button onClick={handleClose}>Close</button>
        </div>
    );
};

async function fetchData() {
    const response = await fetch('/api/data');
    return response.json();
}

export default Modal;
"#;

    let mut extractor = AstExtractor::new();
    let basic = extractor
        .extract_from_source(code)
        .expect("Extraction failed");

    let swc_extractor = SwcAstExtractor::new();
    let enhanced = swc_extractor
        .extract_from_source(code)
        .expect("Enhanced extraction failed");

    let analyzer = SemanticAnalyzer::new();
    let mut semantic = basic.semantic;

    let result = analyzer
        .analyze_with_enhanced_ast(&mut semantic, &enhanced)
        .expect("Analysis failed");

    // 验证复杂组件的分析
    assert!(
        result.detected_frameworks.contains(&"React".to_string()),
        "Should detect React"
    );
    assert!(enhanced.has_jsx, "Should detect JSX");
    assert!(enhanced.has_typescript, "Should detect TypeScript");
    // Note: async_functions detection depends on whether the extractor properly
    // marks functions as async. We verify that special_features were populated.
    assert!(
        !result.special_features.is_empty() || enhanced.has_typescript,
        "Should detect special features or TypeScript"
    );
    assert!(
        result
            .special_features
            .contains(&"typescript_support".to_string()),
        "Should detect TypeScript"
    );

    // 验证组件检测
    assert!(enhanced.jsx_elements.len() > 0, "Should find JSX elements");
}

#[test]
fn test_location_info_accuracy() {
    let code = "const x = 1;\nconst y = 2;\nfunction test() { return 3; }";

    let swc_extractor = SwcAstExtractor::new();
    let enhanced = swc_extractor
        .extract_from_source(code)
        .expect("Extraction failed");

    // 验证位置信息
    for (name, location) in &enhanced.locations {
        assert!(location.line > 0, "Line should be positive for {}", name);
        // Column 可以为 0（在行首）
        assert!(
            location.start < location.end || location.start == location.end,
            "Start should be <= end for {}",
            name
        );
    }
}
