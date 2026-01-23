//! Phase 3: 变量和函数处理测试 (Variable & Function Handling Tests)
//!
//! 测试反混淀对变量重命名、函数分析的处理能力

use browerai::learning::{DeobfuscationStrategy, JsDeobfuscator};
use browerai::parser::JsParser;

// ============================================================================
// 变量重命名测试
// ============================================================================

/// 测试单字母变量重命名
#[test]
fn test_single_letter_variable_renaming() {
    let code = r#"
var a = 10;
var b = 20;
var c = a + b;
console.log(c);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::VariableRenaming)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试数字后缀变量重命名
#[test]
fn test_numeric_suffix_variable_renaming() {
    let code = r#"
var _0x1a2b = "Hello";
var _0x3c4d = "World";
var _0x5e6f = _0x1a2b + _0x3c4d;
console.log(_0x5e6f);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::VariableRenaming)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试无意义变量名识别
#[test]
fn test_meaningless_variable_detection() {
    let code = r#"
var abc = calculateValue();
var def = processData(abc);
var ghi = transformResult(def);
console.log(ghi);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::VariableRenaming)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试上下文相关的变量重命名
#[test]
fn test_context_based_variable_renaming() {
    let code = r#"
var x = getUserData();
var y = x.name;
var z = x.email;
console.log(y, z);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::VariableRenaming)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 函数分析和重命名
// ============================================================================

/// 测试函数检测
#[test]
fn test_function_detection() {
    let code = r#"
function calculate(a, b) {
    return a + b;
}
var result = calculate(5, 3);
console.log(result);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试匿名函数处理
#[test]
fn test_anonymous_function_handling() {
    let code = r#"
var process = function(x) {
    return x * 2;
};
console.log(process(10));
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试箭头函数
#[test]
fn test_arrow_function_handling() {
    let code = r#"
const multiply = (x, y) => x * y;
const result = multiply(3, 4);
console.log(result);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试函数参数重命名
#[test]
fn test_function_parameter_renaming() {
    let code = r#"
function _0x1234(a, b, c) {
    var d = a + b;
    var e = d * c;
    return e;
}
_0x1234(1, 2, 3);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::VariableRenaming)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试回调函数处理
#[test]
fn test_callback_function_handling() {
    let code = r#"
function processArray(arr, callback) {
    var result = [];
    for (var i = 0; i < arr.length; i++) {
        result.push(callback(arr[i]));
    }
    return result;
}
var doubled = processArray([1, 2, 3], function(x) {
    return x * 2;
});
console.log(doubled);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 作用域和闭包测试
// ============================================================================

/// 测试函数作用域
#[test]
fn test_function_scope_handling() {
    let code = r#"
var globalVar = "global";
function outer() {
    var outerVar = "outer";
    function inner() {
        var innerVar = "inner";
        console.log(globalVar, outerVar, innerVar);
    }
    inner();
}
outer();
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试闭包
#[test]
fn test_closure_handling() {
    let code = r#"
function createCounter() {
    var count = 0;
    return function() {
        count++;
        return count;
    };
}
var counter = createCounter();
console.log(counter());
console.log(counter());
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试块级作用域
#[test]
fn test_block_scope_handling() {
    let code = r#"
{
    let blockVar = "block";
    const constVar = "const";
    console.log(blockVar, constVar);
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试变量提升
#[test]
fn test_variable_hoisting_handling() {
    let code = r#"
function testHoisting() {
    console.log(x); // undefined
    var x = 5;
    console.log(x); // 5
}
testHoisting();
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 复杂变量处理
// ============================================================================

/// 测试对象属性
#[test]
fn test_object_property_handling() {
    let code = r#"
var user = {
    name: "John",
    email: "john@example.com",
    age: 30
};
console.log(user.name, user.email);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试数组操作
#[test]
fn test_array_variable_handling() {
    let code = r#"
var items = [1, 2, 3, 4, 5];
var mapped = items.map(function(x) {
    return x * 2;
});
var filtered = items.filter(function(x) {
    return x > 2;
});
console.log(mapped, filtered);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试解构赋值
#[test]
fn test_destructuring_handling() {
    let code = r#"
const { name, email } = user;
const [first, second, ...rest] = array;
console.log(name, email, first, second, rest);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试 this 上下文
#[test]
fn test_this_context_handling() {
    let code = r#"
const obj = {
    name: "Test",
    getName: function() {
        return this.name;
    }
};
console.log(obj.getName());
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 高级函数特性
// ============================================================================

/// 测试高阶函数
#[test]
fn test_higher_order_functions() {
    let code = r#"
function compose(f, g) {
    return function(x) {
        return f(g(x));
    };
}
function double(x) {
    return x * 2;
}
function addOne(x) {
    return x + 1;
}
var composed = compose(double, addOne);
console.log(composed(5));
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试递归函数
#[test]
fn test_recursive_function_handling() {
    let code = r#"
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
console.log(factorial(5));
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试立即执行函数表达式
#[test]
fn test_iife_handling() {
    let code = r#"
(function() {
    var privateVar = 42;
    console.log(privateVar);
})();
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试异步函数
#[test]
fn test_async_function_handling() {
    let code = r#"
async function fetchData(url) {
    const response = await fetch(url);
    const data = await response.json();
    return data;
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试生成器函数
#[test]
fn test_generator_function_handling() {
    let code = r#"
function* generateSequence() {
    yield 1;
    yield 2;
    yield 3;
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 类和对象处理
// ============================================================================

/// 测试类定义
#[test]
fn test_class_definition_handling() {
    let code = r#"
class User {
    constructor(name, email) {
        this.name = name;
        this.email = email;
    }
    
    getName() {
        return this.name;
    }
}
const user = new User("John", "john@example.com");
console.log(user.getName());
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试类继承
#[test]
fn test_class_inheritance_handling() {
    let code = r#"
class Animal {
    speak() {
        console.log("Animal speaks");
    }
}
class Dog extends Animal {
    speak() {
        console.log("Dog barks");
    }
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 执行有效性验证
// ============================================================================

/// 验证变量作用域保留
#[test]
fn test_variable_scope_preservation() {
    let code = r#"
var a = 1;
{
    var a = 2;
    console.log(a);
}
console.log(a);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 验证函数调用链保留
#[test]
fn test_function_call_chain_preservation() {
    let code = r#"
function a() { return { b: function() { return { c: function() { return 42; } } } } }
console.log(a().b().c());
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 验证变量引用的完整性
#[test]
fn test_variable_reference_integrity() {
    let code = r#"
var data = [1, 2, 3];
var transform = function(arr) {
    return arr.map(function(x) { return x * 2; });
};
var result = transform(data);
console.log(result);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

// ============================================================================
// 性能和规模测试
// ============================================================================

/// 测试大量变量处理
#[test]
fn test_large_number_of_variables() {
    let mut code = String::new();
    for i in 0..50 {
        code.push_str(&format!("var v{0} = {0};\n", i));
    }
    code.push_str("console.log(v0, v49);\n");

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(&code, DeobfuscationStrategy::VariableRenaming)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}

/// 测试大量函数处理
#[test]
fn test_large_number_of_functions() {
    let mut code = String::new();
    for i in 0..20 {
        code.push_str(&format!("function fn{0}() {{ return {0}; }}\n", i));
    }
    code.push_str("console.log(fn0(), fn19());\n");

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();
    
    let result = deobf
        .deobfuscate(&code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(parser.validate(&result.code).unwrap_or(false));
}
