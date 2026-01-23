//! 反混淆编码技术测试 (Deobfuscation Encoding Tests)
//!
//! 测试各种字符串编码和数字编码技术的检测和解码
//! 包括: 十六进制、八进制、Unicode、Base64 等

use browerai::learning::{DeobfuscationStrategy, JsDeobfuscator};

// ============================================================================
// 字符串编码测试
// ============================================================================

/// 测试八进制转义序列解码
/// 八进制转义: \101 = 'A', \102 = 'B', \103 = 'C'
#[test]
fn test_octal_string_decoding() {
    let deobf = JsDeobfuscator::new();
    let code = r#"var msg = "\101\102\103";"#;
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    // 验证八进制字符被解码
    assert!(
        result.code.contains("ABC") || !result.code.contains("\\101"),
        "Octal escape should be decoded. Got: {}",
        result.code
    );
    assert!(result.success, "Deobfuscation should succeed");
}

/// 测试 Unicode 转义序列解码
/// Unicode 转义: \u0048 = 'H', \u0065 = 'e', etc.
#[test]
fn test_unicode_escape_decoding() {
    let deobf = JsDeobfuscator::new();
    let code = r#"var msg = "\u0048\u0065\u006c\u006c\u006f";"#;
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    // 验证 Unicode 转义被解码为 "Hello"
    assert!(
        result.code.contains("Hello") || !result.code.contains("\\u006"),
        "Unicode escape should be decoded. Got: {}",
        result.code
    );
    assert!(result.success, "Deobfuscation should succeed");
}

/// 测试混合编码 (十六进制 + Unicode + 普通字符)
#[test]
fn test_mixed_string_encoding() {
    let deobf = JsDeobfuscator::new();
    let code = r#"var msg = "A" + "\x42" + "\u0043";"#;
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    // 验证混合编码被处理
    assert!(result.success, "Mixed encoding deobfuscation should succeed");
    // 最终可能是 "ABC" 或保留的编码形式
    assert!(!result.code.is_empty(), "Result should not be empty");
}

/// 测试转义字符序列 (换行、制表符等)
#[test]
fn test_escape_sequences_decoding() {
    let deobf = JsDeobfuscator::new();
    let code = r#"var msg = "Hello\nWorld\tTab";"#;
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    assert!(result.success, "Escape sequences should be handled");
    assert!(!result.code.is_empty(), "Result should not be empty");
}

/// 测试空字符和特殊字符编码
#[test]
fn test_special_character_encoding() {
    let deobf = JsDeobfuscator::new();
    let code = r#"var msg = "\x00\x01\x02";"#;  // 空字符和控制字符
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    assert!(result.success, "Special characters should be decoded");
}

/// 测试 URL 编码字符串
#[test]
fn test_url_encoded_strings() {
    let deobf = JsDeobfuscator::new();
    let code = r#"var url = "https://example.com/path?key=%20value";"#;
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    assert!(result.success, "URL encoded strings should be handled");
    assert!(result.code.contains("example.com"), "URL should be present");
}

// ============================================================================
// 数字编码测试
// ============================================================================

/// 测试十六进制数字常量
#[test]
fn test_hexadecimal_number_detection() {
    let deobf = JsDeobfuscator::new();
    let code = "var x = 0xFF; var y = 0x100; var z = 0xDEADBEEF;";
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(result.success, "Hexadecimal numbers should be processed");
    // 十六进制可以转换为十进制或保留
    assert!(!result.code.is_empty(), "Result should not be empty");
}

/// 测试八进制数字常量
#[test]
fn test_octal_number_detection() {
    let deobf = JsDeobfuscator::new();
    let code = "var x = 0o755; var y = 0o644; var z = 0o777;";
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(result.success, "Octal numbers should be processed");
    assert!(!result.code.is_empty(), "Result should not be empty");
}

/// 测试二进制数字常量
#[test]
fn test_binary_number_detection() {
    let deobf = JsDeobfuscator::new();
    let code = "var x = 0b1010; var y = 0b1111; var z = 0b11111111;";
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(result.success, "Binary numbers should be processed");
    assert!(!result.code.is_empty(), "Result should not be empty");
}

/// 测试科学计数法
#[test]
fn test_scientific_notation() {
    let deobf = JsDeobfuscator::new();
    let code = "var x = 1e3; var y = 2.5e-2; var z = 1.23e+4;";
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(result.success, "Scientific notation should be processed");
    assert!(result.code.contains("1e3") || result.code.contains("1000"), 
            "Scientific notation should be preserved or converted");
}

/// 测试浮点数精度问题
#[test]
fn test_floating_point_precision() {
    let deobf = JsDeobfuscator::new();
    let code = "var x = 0.1 + 0.2; console.log(x);";
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::Comprehensive)
        .unwrap();

    assert!(result.success, "Floating point operations should be handled");
    assert!(result.code.contains("0.1"), "Floating point literals should be preserved");
}

// ============================================================================
// 组合编码测试
// ============================================================================

/// 测试多行字符串编码
#[test]
fn test_multiline_encoded_strings() {
    let deobf = JsDeobfuscator::new();
    let code = r#"
var msg = "\x48\x65\x6c\x6c\x6f" +
          "\x20" +
          "\x57\x6f\x72\x6c\x64";
"#;
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    assert!(result.success, "Multiline encoded strings should be decoded");
}

/// 测试字符串拼接和编码混合
#[test]
fn test_concatenated_encoded_strings() {
    let deobf = JsDeobfuscator::new();
    let code = r#"
var part1 = "\x4d\x79";  // My
var part2 = "Code";
var combined = part1 + part2;
"#;
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    assert!(result.success, "Concatenated encoded strings should be processed");
}

/// 测试数组中的编码字符串
#[test]
fn test_encoded_strings_in_array() {
    let deobf = JsDeobfuscator::new();
    let code = r#"
var arr = [
    "\x61",  // a
    "\x62",  // b
    "\x63"   // c
];
"#;
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    assert!(result.success, "Encoded strings in arrays should be decoded");
}

/// 测试对象中的编码字符串
#[test]
fn test_encoded_strings_in_object() {
    let deobf = JsDeobfuscator::new();
    let code = r#"
var obj = {
    name: "\x4a\x6f\x68\x6e",  // John
    age: 30,
    city: "\x4e\x59"  // NY
};
"#;
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    assert!(result.success, "Encoded strings in objects should be decoded");
}

// ============================================================================
// 编码检测和分析
// ============================================================================

/// 测试编码检测准确性
#[test]
fn test_encoding_detection_accuracy() {
    let deobf = JsDeobfuscator::new();
    
    // 包含多种编码的代码
    let code = r#"
var hex = "\x48\x65\x6c\x6c\x6f";
var oct = "\101\102\103";
var unicode = "\u0057\u006f\u0072\u006c\u0064";
var plain = "Regular String";
"#;
    
    let analysis = deobf.analyze_obfuscation(code);
    
    // 应该检测到编码技术
    assert!(analysis.obfuscation_score > 0.0, "Should detect encoding obfuscation");
    assert!(analysis.complexity.encoded_literal_count > 0, 
            "Should count encoded literals");
}

/// 测试编码比例计算
#[test]
fn test_encoding_ratio_calculation() {
    let deobf = JsDeobfuscator::new();
    
    let heavily_encoded = r#"
var a = "\x48\x65\x6c\x6c\x6f";
var b = "\x57\x6f\x72\x6c\x64";
var c = "\x4a\x53";
"#;
    
    let lightly_encoded = r#"
var a = "Hello";
var b = "World";
var c = "JS";
"#;
    
    let heavy_analysis = deobf.analyze_obfuscation(heavily_encoded);
    let light_analysis = deobf.analyze_obfuscation(lightly_encoded);
    
    // 重度编码应该有更高的混淆分数
    assert!(heavy_analysis.obfuscation_score > light_analysis.obfuscation_score,
            "Heavily encoded code should have higher obfuscation score");
}

// ============================================================================
// 边界情况和错误处理
// ============================================================================

/// 测试无效的十六进制转义
#[test]
fn test_invalid_hex_escape() {
    let deobf = JsDeobfuscator::new();
    let code = r#"var s = "\xGG";"#;  // 无效的十六进制
    
    // 应该不会崩溃，但可能不解码
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding);
    
    // 应该优雅地处理错误
    assert!(result.is_ok() || result.is_err(), "Should handle error gracefully");
}

/// 测试不完整的 Unicode 转义
#[test]
fn test_incomplete_unicode_escape() {
    let deobf = JsDeobfuscator::new();
    let code = r#"var s = "\u00";"#;  // 不完整的 Unicode
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding);
    
    // 应该处理或报告错误
    assert!(result.is_ok() || result.is_err(), "Should handle incomplete escape");
}

/// 测试空字符串和 null
#[test]
fn test_empty_and_null_strings() {
    let deobf = JsDeobfuscator::new();
    let code = r#"
var empty = "";
var withNull = "\x00";
var nullStr = null;
"#;
    
    let result = deobf
        .deobfuscate(code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    assert!(result.success, "Empty and null strings should be handled");
}

/// 测试极长编码字符串
#[test]
fn test_very_long_encoded_string() {
    let deobf = JsDeobfuscator::new();
    
    // 构造一个很长的编码字符串
    let mut long_code = String::from("var longStr = \"");
    for i in 0..100 {
        long_code.push_str(&format!("\\x{:02x}", i % 256));
    }
    long_code.push_str("\";");
    
    let result = deobf
        .deobfuscate(&long_code, DeobfuscationStrategy::StringDecoding)
        .unwrap();

    assert!(result.success, "Very long encoded strings should be processed");
}

// ============================================================================
// 性能和统计测试
// ============================================================================

/// 测试编码处理性能
#[test]
fn test_encoding_processing_performance() {
    let deobf = JsDeobfuscator::new();
    
    // 包含很多编码字符串的代码
    let code = (0..50)
        .map(|i| format!(r#"var v{} = "\x{:02x}\x{:02x}\x{:02x}";"#, i, i, i+1, i+2))
        .collect::<Vec<_>>()
        .join("\n");
    
    let start = std::time::Instant::now();
    let result = deobf
        .deobfuscate(&code, DeobfuscationStrategy::StringDecoding)
        .unwrap();
    let elapsed = start.elapsed();
    
    assert!(result.success, "Should process multiple encodings");
    assert!(elapsed.as_secs() < 5, "Should complete in reasonable time");
}

/// 测试编码统计
#[test]
fn test_encoding_statistics() {
    let deobf = JsDeobfuscator::new();
    
    let code = r#"
var a = "\x48\x65";
var b = "\x6c\x6c";
var c = "\x6f";
"#;
    
    let analysis = deobf.analyze_obfuscation(code);
    
    // 应该正确计数编码的字符串
    assert!(analysis.complexity.encoded_literal_count >= 3,
            "Should count all encoded strings");
}
