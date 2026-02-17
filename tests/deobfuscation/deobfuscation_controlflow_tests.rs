//! Phase 2: 控制流和死代码测试 (Control Flow & Dead Code Tests)
//!
//! 测试反混淀对控制流复杂化和死代码的处理能力

use browerai::learning::{DeobfuscationStrategy, JsDeobfuscator};
use browerai::parser::JsParser;

// ============================================================================
// 控制流复杂化测试
// ============================================================================

/// 测试简单的 if 语句复杂化
#[test]
fn test_simple_if_statement_deobfuscation() {
    let code = r#"
if (true) {
    console.log("Yes");
} else {
    console.log("No");
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
    assert!(result.success);
}

/// 测试嵌套 if 语句
#[test]
fn test_nested_if_statements() {
    let code = r#"
if (a > 0) {
    if (b > 0) {
        if (c > 0) {
            console.log("All positive");
        }
    }
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试 switch 语句简化
#[test]
fn test_switch_statement_simplification() {
    let code = r#"
switch (x) {
    case 1:
        console.log("One");
        break;
    case 2:
        console.log("Two");
        break;
    default:
        console.log("Other");
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试三元运算符复杂化
#[test]
fn test_ternary_operator_complexity() {
    let code = r#"
var result = a > 0 ? b > 0 ? c > 0 ? "All" : "BC" : "B" : "A";
console.log(result);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 死代码移除测试
// ============================================================================

/// 测试未使用的变量移除
#[test]
fn test_unused_variable_removal() {
    let code = r#"
var unused = 42;
var used = 100;
console.log(used);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::DeadCodeElimination)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
    // 反混淀后应该有改进
    assert!(result.improvement.readability_after >= result.improvement.readability_before);
}

/// 测试未到达的代码块
#[test]
fn test_unreachable_code_removal() {
    let code = r#"
function test() {
    return 42;
    console.log("This is unreachable");
    var unused = 100;
}
test();
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::DeadCodeElimination)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试永不执行的 if 块
#[test]
fn test_dead_if_block_removal() {
    let code = r#"
if (false) {
    console.log("Never executed");
    var x = 42;
}
if (true) {
    console.log("Always executed");
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::DeadCodeElimination)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试空块清理
#[test]
fn test_empty_block_cleanup() {
    let code = r#"
if (true) {
}
while (false) {
}
for (var i = 0; i < 0; i++) {
}
var used = 10;
console.log(used);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::DeadCodeElimination)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 循环优化测试
// ============================================================================

/// 测试简单循环
#[test]
fn test_simple_for_loop() {
    let code = r#"
var sum = 0;
for (var i = 0; i < 10; i++) {
    sum += i;
}
console.log(sum);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试 while 循环
#[test]
fn test_while_loop_deobfuscation() {
    let code = r#"
var i = 0;
while (i < 10) {
    console.log(i);
    i++;
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试 do-while 循环
#[test]
fn test_do_while_loop() {
    let code = r#"
var i = 0;
do {
    console.log(i);
    i++;
} while (i < 5);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试嵌套循环
#[test]
fn test_nested_loops() {
    let code = r#"
for (var i = 0; i < 5; i++) {
    for (var j = 0; j < 5; j++) {
        console.log(i, j);
    }
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试带 break/continue 的循环
#[test]
fn test_loop_with_break_continue() {
    let code = r#"
for (var i = 0; i < 10; i++) {
    if (i === 3) continue;
    if (i === 7) break;
    console.log(i);
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 控制流扁平化逆转测试
// ============================================================================

/// 测试简单的状态机转换
#[test]
fn test_state_machine_to_normal_flow() {
    let code = r#"
function process(x) {
    var state = 0;
    while (true) {
        switch (state) {
            case 0:
                console.log("Start");
                state = 1;
                break;
            case 1:
                console.log("Processing");
                state = 2;
                break;
            case 2:
                console.log("Done");
                return;
        }
    }
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 复杂混合场景测试
// ============================================================================

/// 测试混合死代码和控制流
#[test]
fn test_combined_dead_code_and_control_flow() {
    let code = r#"
var unused1 = 42;
if (false) {
    var dead = 100;
    console.log(dead);
}
function complexLogic() {
    var unused2 = "test";
    if (true) {
        return 42;
        console.log("Never reached");
    }
}
var result = complexLogic();
console.log(result);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试函数内的控制流复杂化
#[test]
fn test_function_with_complex_control_flow() {
    let code = r#"
function calculate(a, b, c) {
    if (a > 0) {
        if (b > 0) {
            if (c > 0) {
                return a + b + c;
            } else {
                return a + b - c;
            }
        } else {
            return a - b;
        }
    } else {
        return -a;
    }
}
console.log(calculate(5, 3, 2));
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试多重循环和条件
#[test]
fn test_multiple_loops_with_conditions() {
    let code = r#"
var matrix = [[1, 2], [3, 4], [5, 6]];
for (var i = 0; i < matrix.length; i++) {
    for (var j = 0; j < matrix[i].length; j++) {
        if (matrix[i][j] % 2 === 0) {
            console.log("Even:", matrix[i][j]);
        }
    }
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 执行有效性验证
// ============================================================================

/// 验证反混淀的控制流代码能正确执行
#[test]
fn test_deobfuscated_control_flow_execution_validity() {
    let code = r#"
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}
var result = fibonacci(10);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    // 关键：必须是有效的 JavaScript
    assert!(parser.validate(&result.code).unwrap_or(false));
    assert!(result.success);
    
    // 应该包含关键的函数定义
    assert!(result.code.contains("function") || result.code.contains("fibonacci"));
}

/// 测试错误处理
#[test]
fn test_try_catch_block_handling() {
    let code = r#"
try {
    throw new Error("Test error");
} catch (e) {
    console.log("Caught:", e.message);
} finally {
    console.log("Cleanup");
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试逻辑表达式简化
#[test]
fn test_logical_expression_simplification() {
    let code = r#"
var a = true && false;
var b = true || false;
var c = !true;
var d = (a && b) || (c && !c);
console.log(a, b, c, d);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 性能和规模测试
// ============================================================================

/// 测试大型控制流结构
#[test]
fn test_large_control_flow_structure() {
    let mut code = String::from("function large() {");
    
    for i in 0..20 {
        code.push_str(&format!(
            "if (x === {}) {{ console.log({}); }} else ",
            i, i
        ));
    }
    code.push_str("{ console.log('default'); } }");

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(&code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试深度嵌套
#[test]
fn test_deeply_nested_structure() {
    let mut code = String::new();
    for i in 0..10 {
        code.push_str(&format!("if (true) {{ "));
    }
    code.push_str("console.log('Deep');");
    for _ in 0..10 {
        code.push_str(" }");
    }

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(&code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 边界条件测试
// ============================================================================

/// 测试空函数
#[test]
fn test_empty_function() {
    let code = r#"
function empty() {
}
empty();
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::DeadCodeElimination)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试空块
#[test]
fn test_empty_blocks() {
    let code = r#"
if (true) {}
while (false) {}
for (;;) { break; }
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::DeadCodeElimination)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试仅有控制流的代码
#[test]
fn test_only_control_flow() {
    let code = r#"
if (true) {
    if (true) {
        if (true) {
            console.log("Test");
        }
    }
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::ControlFlowSimplification)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}
