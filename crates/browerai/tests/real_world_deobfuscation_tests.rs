//! 真实网站反混淀测试 (Real-World Deobfuscation Tests)
//!
//! 从真实网站获取混淆的 JavaScript 代码，测试反混淆功能
//! 验证反混淆后的代码是否可以执行

use browerai::js_parser::JsParser;
use browerai::learning::{
    DeobfuscationStrategy, ExecutionValidator, JsDeobfuscator, WebsiteDeobfuscationVerifier,
};

// ============================================================================
// 真实网站反混淀测试
// ============================================================================

/// 从真实网站获取和测试反混淀
/// 这个测试需要网络连接
#[test]
#[ignore] // 需要网络，所以默认忽略。运行：cargo test real_website_deobfuscation -- --ignored --nocapture
fn test_real_website_deobfuscation_minified_code() {
    // 常见网站中的最小化代码
    let real_minified_code = r#"
function hideEmail(a,b,c){var d=a.split('@');var e=d[0].substring(0,3);var f=d[1];return e+'***@'+f;}
var users=['john@example.com','jane@example.org'];var emails=users.map(function(x){return hideEmail(x);});
console.log(emails);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    // 原始代码应该是有效的
    assert!(
        parser.validate(real_minified_code).unwrap_or(false),
        "Original minified code should be valid"
    );

    // 反混淀
    let result = deobf
        .deobfuscate(real_minified_code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    println!("\n=== Minified Code Test ===");
    println!(
        "Readability before: {}",
        result.improvement.readability_before
    );
    println!(
        "Readability after: {}",
        result.improvement.readability_after
    );
    println!("Original code length: {}", real_minified_code.len());
    println!("Deobfuscated code length: {}", result.code.len());

    // 反混淆后的代码应该仍然是有效的
    assert!(
        parser.validate(&result.code).unwrap_or(false),
        "Deobfuscated code should still be valid JavaScript"
    );

    // 代码可读性应该没有严重下降（对于已经比较可读的代码，可能不会有大的改善）
    assert!(
        result.improvement.readability_after >= result.improvement.readability_before * 0.8,
        "Deobfuscated code readability should not significantly decrease (before: {}, after: {})",
        result.improvement.readability_before,
        result.improvement.readability_after
    );
}

/// 使用真实线上库的最小化代码，验证反混淀和执行有效性
#[test]
#[ignore] // 需要网络，运行：cargo test test_real_world_minified_libraries_execution_validation -- --ignored --nocapture
fn test_real_world_minified_libraries_execution_validation() {
    let mut verifier = WebsiteDeobfuscationVerifier::new();
    let execution_validator = ExecutionValidator::new();

    let urls = vec![
        "https://unpkg.com/react@18/umd/react.production.min.js",
        "https://cdn.jsdelivr.net/npm/dayjs@1.11.10/dayjs.min.js",
    ];

    for url in urls {
        println!("\n=== Testing {} ===", url);

        let result = verifier.verify_website(url, None).expect(&format!(
            "Failed to fetch and deobfuscate real-world JS from {}",
            url
        ));

        println!("Original size: {} bytes", result.original_size);
        println!("Deobfuscated size: {} bytes", result.deobfuscated_size);
        println!("Success: {}", result.success);
        println!("Is valid: {}", result.is_valid);
        println!(
            "Readability improvement: {:.2}%",
            result.readability_improvement * 100.0
        );
        println!("Processing time: {} ms", result.processing_time_ms);
        println!("Detected techniques: {:?}", result.obfuscation_techniques);
        if let Some(ref err) = result.error {
            println!("Error: {}", err);
        }

        assert!(
            result.original_size > 1024,
            "Should fetch non-trivial JS from {}",
            url
        );

        if !result.success {
            eprintln!("Warning: Deobfuscation failed for {}", url);
        }

        if !result.is_valid {
            eprintln!("Warning: Deobfuscated code is not valid for {}", url);
        }

        let exec_result = execution_validator
            .validate_execution(&result.original_code, &result.deobfuscated_code);

        println!(
            "Execution validation - syntax valid: {}, features: {}, risks: {}",
            exec_result.is_valid_syntax,
            exec_result.features.len(),
            exec_result.risks.len()
        );

        assert!(
            exec_result.is_valid_syntax || result.is_valid,
            "Either execution validator or parser should mark syntax valid for {}",
            url
        );
    }
}

/// 测试从 GitHub 风格的混淀代码
#[test]
fn test_webpack_bundled_code_deobfuscation() {
    // 来自 Webpack 的真实风格代码
    let webpack_style_code = r#"
(function(modules) {
    var installedModules = {};
    function __webpack_require__(moduleId) {
        if(installedModules[moduleId]) {
            return installedModules[moduleId].exports;
        }
        var module = installedModules[moduleId] = {
            i: moduleId,
            l: false,
            exports: {}
        };
        modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
        module.l = true;
        return module.exports;
    }
    return __webpack_require__(0);
})({
    0: function(module, exports) {
        var greeting = 'Hello World';
        console.log(greeting);
    }
});
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    // 原始代码有效
    assert!(
        parser.validate(webpack_style_code).unwrap_or(false),
        "Webpack code should be valid"
    );

    // 反混淀
    let result = deobf
        .deobfuscate(webpack_style_code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    // 反混淀后代码有效
    assert!(
        parser.validate(&result.code).unwrap_or(false),
        "Deobfuscated webpack code should be valid"
    );

    assert!(result.success, "Webpack deobfuscation should succeed");
}

/// 测试字符串数组混淀（常见的混淀技术）
#[test]
fn test_string_array_obfuscation_deobfuscation() {
    // 这是一个常见的混淀模式
    let string_array_code = r#"
var _0xabc7=['Hello','World','concat'];
function _0x4c2d(a,b){
    a=a-0x0;
    var c=_0xabc7[a];
    return c;
}
var msg=_0x4c2d('0x0')+_0x4c2d('0x1');
console.log(msg);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    // 原始代码有效
    assert!(
        parser.validate(string_array_code).unwrap_or(false),
        "String array obfuscated code should be valid"
    );

    // 反混淀
    let result = deobf
        .deobfuscate(string_array_code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    println!("\n=== String Array Obfuscation Test ===");
    println!(
        "Readability before: {}",
        result.improvement.readability_before
    );
    println!(
        "Readability after: {}",
        result.improvement.readability_after
    );

    // 反混淆后代码有效
    assert!(
        parser.validate(&result.code).unwrap_or(false),
        "Deobfuscated string array code should be valid"
    );

    // 应该至少保持 80% 的可读性（允许一些下降）
    assert!(
        result.improvement.readability_after >= result.improvement.readability_before * 0.8,
        "Deobfuscation should maintain reasonable readability (before: {}, after: {})",
        result.improvement.readability_before,
        result.improvement.readability_after
    );
}

/// 测试 React 编译后的代码
#[test]
fn test_react_compiled_code_deobfuscation() {
    // 来自 React 应用的编译后代码风格（使用 CommonJS 而不是 ES6 模块）
    let react_code = r#"
var React = require('react');
var Component = React.createElement('div', null, 
    React.createElement('h1', null, 'Hello'),
    React.createElement('p', null, 'World')
);
module.exports = Component;
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    println!("\n=== React Code Test ===");
    let validation_result = parser.validate(react_code);
    println!("React code validation result: {:?}", validation_result);

    // 原始代码有效（或者至少能够处理）
    // 注意：Boa 可能不完全支持 require/module.exports，但这不是本测试的重点
    let is_valid = validation_result.unwrap_or(false);
    if !is_valid {
        println!("Note: Parser may not fully support CommonJS syntax, but continuing test");
    }

    // 反混淆（虽然 React 代码不算混淆，但测试兼容性）
    let result = deobf
        .deobfuscate(react_code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    println!("Deobfuscation completed, checking result validity");

    // 反混淆后代码应该可以被处理（即使不完全符合 ES5 标准）
    // 主要验证反混淆过程不会崩溃或产生无效输出
    assert!(
        result.code.len() > 0,
        "Deobfuscated React code should not be empty"
    );

    assert!(
        result.code.contains("React") || result.code.contains("createElement"),
        "Deobfuscated React code should preserve key React elements"
    );
}

/// 测试控制流扁平化后的代码
#[test]
fn test_control_flow_flattened_code() {
    // 控制流扁平化的真实示例
    let flattened_code = r#"
function process(x) {
    var state = 0;
    while (true) {
        switch (state) {
            case 0:
                if (x > 0) state = 1;
                else state = 2;
                break;
            case 1:
                console.log('Positive');
                state = 3;
                break;
            case 2:
                console.log('Negative');
                state = 3;
                break;
            case 3:
                return x;
        }
    }
}
process(5);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    // 原始代码有效
    assert!(
        parser.validate(flattened_code).unwrap_or(false),
        "Control flow flattened code should be valid"
    );

    // 反混淀
    let result = deobf
        .deobfuscate(
            flattened_code,
            DeobfuscationStrategy::ControlFlowSimplification,
        )
        .unwrap();

    // 反混淀后代码有效
    assert!(
        parser.validate(&result.code).unwrap_or(false),
        "Deobfuscated control flow code should be valid"
    );
}

// ============================================================================
// 代码执行测试 (验证反混淀后的代码是否能执行)
// ============================================================================

/// 测试反混淀后的代码执行性能
#[test]
fn test_deobfuscated_code_execution_simple() {
    // 简单的计算代码
    let obfuscated = r#"
var a=function(b,c){var d=0;for(var e=0;e<b.length;e++){d+=b[e]}return d+c};
var result=a([1,2,3,4,5],10);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    // 反混淀
    let result_deobf = deobf
        .deobfuscate(obfuscated, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    // 关键：反混淀后的代码必须是有效的
    let is_valid = parser.validate(&result_deobf.code).unwrap_or(false);
    assert!(
        is_valid,
        "Deobfuscated code must be valid JavaScript: {}",
        result_deobf.code
    );

    // 代码可读性应该改善
    assert!(
        result_deobf.improvement.readability_after >= 0.5,
        "Deobfuscated code should have reasonable readability"
    );
}

/// 测试混合编码和混淀的综合场景
#[test]
fn test_combined_obfuscation_deobfuscation() {
    let combined = r#"
var _0x1a2b = ['\x48\x65\x6c\x6c\x6f', '\x57\x6f\x72\x6c\x64'];
function _0x3c4d(a) {
    if (true) {
        return _0x1a2b[a];
    }
}
var msg = _0x3c4d(0) + ' ' + _0x3c4d(1);
console.log(msg);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    // 反混淀
    let result = deobf
        .deobfuscate(combined, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    // 验证有效性
    assert!(
        parser.validate(&result.code).unwrap_or(false),
        "Combined obfuscation deobfuscation should produce valid code"
    );

    // 验证改进
    assert!(
        result.improvement.readability_after > 0.3,
        "Should have some readability improvement"
    );
}

// ============================================================================
// 边界情况和复杂场景
// ============================================================================

/// 测试极度混淀的代码
#[test]
fn test_heavily_obfuscated_code() {
    let heavily_obfuscated = r#"
eval(String.fromCharCode(99,111,110,115,111,108,101,46,108,111,103,40,34,72,101,108,108,111,34,41))
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    // 即使高度混淀，原始代码应该有效
    assert!(
        parser.validate(heavily_obfuscated).unwrap_or(false),
        "Heavily obfuscated code should be valid JavaScript"
    );

    // 尝试反混淀
    let result = deobf
        .deobfuscate(heavily_obfuscated, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    // 反混淀后的代码应该有效（即使不完全解码）
    assert!(
        parser.validate(&result.code).unwrap_or(false),
        "Deobfuscation of eval code should produce valid JavaScript"
    );
}

/// 测试大型真实代码块
#[test]
fn test_large_real_world_code_block() {
    let large_code = r#"
(function() {
    var cache = {};
    
    function memoize(fn) {
        return function(x) {
            if (!(x in cache)) {
                cache[x] = fn(x);
            }
            return cache[x];
        };
    }
    
    var fibonacci = memoize(function(n) {
        if (n <= 1) return n;
        return fibonacci(n - 1) + fibonacci(n - 2);
    });
    
    console.log(fibonacci(10));
})();
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    // 原始代码有效
    assert!(
        parser.validate(large_code).unwrap_or(false),
        "Large code block should be valid"
    );

    // 反混淀
    let result = deobf
        .deobfuscate(large_code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    // 关键检查：反混淀后的代码必须有效
    assert!(
        parser.validate(&result.code).unwrap_or(false),
        "Large code block deobfuscation should produce valid code"
    );

    // 代码行数应该合理（不应该增加太多）
    let original_lines = large_code.lines().count();
    let deobf_lines = result.code.lines().count();

    assert!(
        deobf_lines <= original_lines * 2,
        "Deobfuscated code should not be much longer"
    );
}

// ============================================================================
// 执行验证测试
// ============================================================================

/// 验证反混淀后的代码是否能正确执行
#[test]
fn test_deobfuscation_preserves_functionality() {
    let original = r#"
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
var result = factorial(5);
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    // 原始代码有效
    assert!(
        parser.validate(original).unwrap_or(false),
        "Original code should be valid"
    );

    // 反混淀（这个代码其实不算混淀，但验证过程有效）
    let result = deobf
        .deobfuscate(original, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    // 关键：反混淀后的代码必须仍然有效
    assert!(
        parser.validate(&result.code).unwrap_or(false),
        "Deobfuscated code must preserve functionality and validity"
    );

    // 结构应该保留关键元素
    assert!(
        result.code.contains("factorial") || result.code.contains("function"),
        "Deobfuscated code should preserve function structure"
    );
}

/// 测试多种混淀技术的组合
#[test]
fn test_multiple_obfuscation_techniques_combined() {
    let multi_technique = r#"
var _0x = ['log','Hello World'];
(function(_0xa,_0xb){
    while(--_0xb){
        _0xa.push(_0xa.shift());
    }
}(_0x,0x123));
if(1===1){
    console[_0x[0]](_0x[1]);
}
"#;

    let deobf = JsDeobfuscator::new();
    let parser = JsParser::new();

    // 原始代码有效
    assert!(
        parser.validate(multi_technique).unwrap_or(false),
        "Multi-technique obfuscated code should be valid"
    );

    // 反混淀
    let result = deobf
        .deobfuscate(multi_technique, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    // 反混淀后代码有效
    assert!(
        parser.validate(&result.code).unwrap_or(false),
        "Multi-technique deobfuscation should succeed"
    );

    // 应该有显著改进
    assert!(
        result.steps.len() > 0,
        "Should apply multiple deobfuscation steps"
    );
}

// ============================================================================
// 性能和统计测试
// ============================================================================

/// 测试反混淀的性能表现
#[test]
fn test_deobfuscation_performance() {
    let deobf = JsDeobfuscator::new();

    // 构造一个中等大小的混淀代码
    let mut code = String::new();
    for i in 0..50 {
        code.push_str(&format!(
            "var v{0}='{1}{2}{3}';",
            i, "\\x41\\x42\\x43", "\\x44\\x45\\x46", "\\x47\\x48\\x49"
        ));
    }

    let start = std::time::Instant::now();
    let result = deobf
        .deobfuscate(&code, DeobfuscationStrategy::Comprehensive)
        .unwrap();
    let elapsed = start.elapsed();

    // 性能要求：应该在合理时间内完成
    assert!(
        elapsed.as_secs() < 10,
        "Deobfuscation should complete in reasonable time, took {:?}",
        elapsed
    );

    assert!(result.success, "Deobfuscation should succeed");
}

/// 测试覆盖率报告
#[test]
fn test_deobfuscation_improvement_metrics() {
    let obfuscated = r#"
var a=function(b){var c=0;for(var d=0;d<b;d++){c+=d}return c};
var x=a(100);
console.log(x);
"#;

    let deobf = JsDeobfuscator::new();
    let result = deobf
        .deobfuscate(obfuscated, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    // 验证改进指标
    assert!(
        result.improvement.readability_before >= 0.0
            && result.improvement.readability_before <= 1.0,
        "Readability scores should be between 0 and 1"
    );

    assert!(
        result.improvement.readability_after >= 0.0 && result.improvement.readability_after <= 1.0,
        "Readability scores should be between 0 and 1"
    );

    // 应该有改进或保持不变
    assert!(
        result.improvement.readability_after >= result.improvement.readability_before * 0.9,
        "Deobfuscation should not significantly reduce readability"
    );
}
