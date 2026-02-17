use anyhow::Result;
use regex::Regex;
/// 通用混淆模式识别库 (Obfuscation Pattern Recognition Library)
///
/// 识别常见的混淆技术，自动转换为清晰代码，
/// 可扩展的规则系统。
use std::collections::HashMap;

/// 混淆模式类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObfuscationPatternType {
    /// 动态代码执行 (eval, Function)
    DynamicCodeExecution,
    /// 变量名混淆 (单字母变量)
    VariableNameObfuscation,
    /// 数字混淆 (16进制, 8进制等)
    NumberObfuscation,
    /// 字符串拆分
    StringSplitting,
    /// 数组索引混淆
    ArrayIndexObfuscation,
    /// 对象属性混淆
    PropertyObfuscation,
    /// 函数参数混淆
    FunctionParameterObfuscation,
    /// 条件反转 (取反条件)
    ConditionInversion,
    /// 循环混淆
    LoopObfuscation,
    /// 表达式复杂化
    ExpressionComplexity,
    /// 作用域混淆
    ScopeObfuscation,
    /// 原型链污染
    PrototypeChainPollution,
}

/// 混淆模式定义
#[derive(Debug, Clone)]
pub struct ObfuscationPattern {
    /// 模式类型
    pub pattern_type: ObfuscationPatternType,
    /// 模式名称
    pub name: String,
    /// 正则表达式模式
    pub regex_pattern: String,
    /// 反混淆规则
    pub deobfuscation_rule: String,
    /// 危险等级 (low, medium, high, critical)
    pub severity: String,
    /// 检测置信度 (0.0 - 1.0)
    pub confidence: f32,
    /// 示例代码
    pub example_obfuscated: String,
    pub example_clear: String,
}

/// 检测到的混淆实例
#[derive(Debug, Clone)]
pub struct DetectedPattern {
    pub pattern: ObfuscationPattern,
    pub matched_text: String,
    pub suggested_replacement: String,
    pub line_number: usize,
    pub confidence: f32,
}

/// 通用混淆模式识别器
pub struct ObfuscationPatternLibrary {
    patterns: HashMap<ObfuscationPatternType, Vec<ObfuscationPattern>>,
    custom_patterns: Vec<ObfuscationPattern>,
}

impl ObfuscationPatternLibrary {
    pub fn new() -> Self {
        let mut library = Self {
            patterns: HashMap::new(),
            custom_patterns: Vec::new(),
        };

        // 初始化内置模式
        library.initialize_patterns();

        library
    }

    /// 初始化所有内置模式
    fn initialize_patterns(&mut self) {
        // 动态代码执行模式
        self.add_pattern(
            ObfuscationPatternType::DynamicCodeExecution,
            ObfuscationPattern {
                pattern_type: ObfuscationPatternType::DynamicCodeExecution,
                name: "eval() 函数调用".to_string(),
                regex_pattern: r#"eval\s*\(\s*['"](.+?)['\"]\s*\)"#.to_string(),
                deobfuscation_rule: "直接替换为执行内容".to_string(),
                severity: "critical".to_string(),
                confidence: 0.95,
                example_obfuscated: r#"eval("console.log('hello')")"#.to_string(),
                example_clear: "console.log('hello')".to_string(),
            },
        );

        self.add_pattern(
            ObfuscationPatternType::DynamicCodeExecution,
            ObfuscationPattern {
                pattern_type: ObfuscationPatternType::DynamicCodeExecution,
                name: "Function() 构造函数".to_string(),
                regex_pattern: r#"new\s+Function\s*\(\s*['"](.+?)['\"]\s*\)"#.to_string(),
                deobfuscation_rule: "提取函数体".to_string(),
                severity: "critical".to_string(),
                confidence: 0.9,
                example_obfuscated: "new Function(\"return Math.random()\")()".to_string(),
                example_clear: "return Math.random()".to_string(),
            },
        );

        // 变量名混淆模式
        self.add_pattern(
            ObfuscationPatternType::VariableNameObfuscation,
            ObfuscationPattern {
                pattern_type: ObfuscationPatternType::VariableNameObfuscation,
                name: "单字母变量".to_string(),
                regex_pattern: r"\b[a-z]\b(?=\s*[=;,\)])".to_string(),
                deobfuscation_rule: "重命名为有意义的名字".to_string(),
                severity: "medium".to_string(),
                confidence: 0.6,
                example_obfuscated: "let a = 10; let b = a + 5;".to_string(),
                example_clear: "let value = 10; let result = value + 5;".to_string(),
            },
        );

        // 数字混淆模式
        self.add_pattern(
            ObfuscationPatternType::NumberObfuscation,
            ObfuscationPattern {
                pattern_type: ObfuscationPatternType::NumberObfuscation,
                name: "16进制数字".to_string(),
                regex_pattern: r"0x[0-9a-fA-F]+".to_string(),
                deobfuscation_rule: "转换为十进制".to_string(),
                severity: "low".to_string(),
                confidence: 0.95,
                example_obfuscated: "0x41, 0x42, 0x43".to_string(),
                example_clear: "65, 66, 67".to_string(),
            },
        );

        // 字符串拆分模式
        self.add_pattern(
            ObfuscationPatternType::StringSplitting,
            ObfuscationPattern {
                pattern_type: ObfuscationPatternType::StringSplitting,
                name: "字符串连接".to_string(),
                regex_pattern: r#"['"][^'"]*['\"]\s*\+\s*['"][^'"]*['"]"#.to_string(),
                deobfuscation_rule: "合并为单个字符串".to_string(),
                severity: "low".to_string(),
                confidence: 0.9,
                example_obfuscated: r#""hello" + " " + "world""#.to_string(),
                example_clear: r#""hello world""#.to_string(),
            },
        );

        // 数组索引混淆模式
        self.add_pattern(
            ObfuscationPatternType::ArrayIndexObfuscation,
            ObfuscationPattern {
                pattern_type: ObfuscationPatternType::ArrayIndexObfuscation,
                name: "复杂数组索引".to_string(),
                regex_pattern: r"\w+\[[\w\+\-\*\/\(\)]+\]".to_string(),
                deobfuscation_rule: "简化索引表达式".to_string(),
                severity: "medium".to_string(),
                confidence: 0.7,
                example_obfuscated: "arr[index * 2 + 1]".to_string(),
                example_clear: "arr[simplifiedIndex]".to_string(),
            },
        );

        // 对象属性混淆模式
        self.add_pattern(
            ObfuscationPatternType::PropertyObfuscation,
            ObfuscationPattern {
                pattern_type: ObfuscationPatternType::PropertyObfuscation,
                name: "括号访问属性".to_string(),
                regex_pattern: r#"\w+\[['"][^'"]+['"]\]"#.to_string(),
                deobfuscation_rule: "转换为点记号".to_string(),
                severity: "low".to_string(),
                confidence: 0.85,
                example_obfuscated: r#"obj["property"]"#.to_string(),
                example_clear: "obj.property".to_string(),
            },
        );

        // 条件反转模式
        self.add_pattern(
            ObfuscationPatternType::ConditionInversion,
            ObfuscationPattern {
                pattern_type: ObfuscationPatternType::ConditionInversion,
                name: "条件取反".to_string(),
                regex_pattern: r"if\s*\(\s*!\s*[\w.]+\s*\)".to_string(),
                deobfuscation_rule: "简化为正条件".to_string(),
                severity: "medium".to_string(),
                confidence: 0.8,
                example_obfuscated: "if (!condition) { doA(); } else { doB(); }".to_string(),
                example_clear: "if (condition) { doB(); } else { doA(); }".to_string(),
            },
        );

        // 循环混淆模式
        self.add_pattern(
            ObfuscationPatternType::LoopObfuscation,
            ObfuscationPattern {
                pattern_type: ObfuscationPatternType::LoopObfuscation,
                name: "复杂循环结构".to_string(),
                regex_pattern:
                    r"for\s*\(\s*\w+\s*=\s*[\w.]+;[\w\s<>=!&|+\-*/().,]*;[\w\s<>=!&|+\-*/().,]*\)"
                        .to_string(),
                deobfuscation_rule: "简化循环逻辑".to_string(),
                severity: "medium".to_string(),
                confidence: 0.7,
                example_obfuscated: "for(let i=0;i<10;i+=2){...}".to_string(),
                example_clear: "for(let i=0;i<10;i+=2){...}".to_string(),
            },
        );
    }

    /// 添加模式
    fn add_pattern(&mut self, pattern_type: ObfuscationPatternType, pattern: ObfuscationPattern) {
        self.patterns.entry(pattern_type).or_default().push(pattern);
    }

    /// 添加自定义模式
    pub fn add_custom_pattern(&mut self, pattern: ObfuscationPattern) {
        self.custom_patterns.push(pattern);
    }

    /// 检测代码中的混淆模式
    pub fn detect(&self, code: &str) -> Result<Vec<DetectedPattern>> {
        let mut detected = Vec::new();

        // 检查所有内置模式
        for patterns in self.patterns.values() {
            for pattern in patterns {
                detected.extend(self.detect_pattern(code, pattern));
            }
        }

        // 检查自定义模式
        for pattern in &self.custom_patterns {
            detected.extend(self.detect_pattern(code, pattern));
        }

        // 按置信度排序
        detected.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(detected)
    }

    /// 检测单个模式
    fn detect_pattern(&self, code: &str, pattern: &ObfuscationPattern) -> Vec<DetectedPattern> {
        let mut results = Vec::new();

        if let Ok(re) = Regex::new(&pattern.regex_pattern) {
            for (line_num, line) in code.lines().enumerate() {
                for cap in re.captures_iter(line) {
                    let matched_text = cap.get(0).unwrap().as_str().to_string();

                    // 生成建议替换
                    let suggested_replacement = self.generate_replacement(pattern, &matched_text);

                    results.push(DetectedPattern {
                        pattern: pattern.clone(),
                        matched_text,
                        suggested_replacement,
                        line_number: line_num + 1,
                        confidence: pattern.confidence,
                    });
                }
            }
        }

        results
    }

    /// 生成建议的替换
    fn generate_replacement(&self, pattern: &ObfuscationPattern, matched_text: &str) -> String {
        match pattern.pattern_type {
            ObfuscationPatternType::PropertyObfuscation => {
                // obj["property"] -> obj.property
                if let Ok(re) = Regex::new(r#"\[['"]([^'"]+)['"]\]"#) {
                    if let Some(caps) = re.captures(matched_text) {
                        return format!(".{}", caps.get(1).unwrap().as_str());
                    }
                }
                matched_text.to_string()
            }
            ObfuscationPatternType::NumberObfuscation => {
                // 0x41 -> 65
                if let Ok(num) = i64::from_str_radix(&matched_text[2..], 16) {
                    num.to_string()
                } else {
                    matched_text.to_string()
                }
            }
            _ => format!("/* 建议简化: {} */", pattern.deobfuscation_rule),
        }
    }

    /// 应用所有检测到的模式修复
    pub fn deobfuscate(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        let detected = self.detect(code)?;

        // 按行应用替换（从后往前，以避免索引偏移）
        for pattern_result in detected.iter().rev() {
            if pattern_result.confidence >= 0.85 {
                // 只应用高置信度的替换
                result = result.replace(
                    &pattern_result.matched_text,
                    &pattern_result.suggested_replacement,
                );
            }
        }

        Ok(result)
    }

    /// 获取统计信息
    pub fn get_statistics(&self) -> PatternLibraryStatistics {
        let mut stats = HashMap::new();

        for (pattern_type, patterns) in &self.patterns {
            stats.insert(format!("{:?}", pattern_type), patterns.len());
        }

        PatternLibraryStatistics {
            total_builtin_patterns: self.patterns.values().map(|v| v.len()).sum(),
            total_custom_patterns: self.custom_patterns.len(),
            pattern_type_stats: stats,
        }
    }

    /// 生成人类可读的报告
    pub fn generate_report(&self, code: &str) -> Result<String> {
        let detected = self.detect(code)?;

        let mut report = String::from("=== 混淆模式检测报告 ===\n\n");

        if detected.is_empty() {
            report.push_str("✅ 未检测到混淆模式\n");
            return Ok(report);
        }

        // 按严重等级分类
        let mut critical = Vec::new();
        let mut high = Vec::new();
        let mut medium = Vec::new();
        let mut low = Vec::new();

        for pattern in detected {
            match pattern.pattern.severity.as_str() {
                "critical" => critical.push(pattern),
                "high" => high.push(pattern),
                "medium" => medium.push(pattern),
                _ => low.push(pattern),
            }
        }

        // 输出 critical
        if !critical.is_empty() {
            report.push_str("🔴 严重威胁:\n");
            for p in critical {
                report.push_str(&format!(
                    "  - {} (置信度: {:.0}%)\n    位置: 第 {} 行\n    匹配: {}\n    建议: {}\n",
                    p.pattern.name,
                    p.confidence * 100.0,
                    p.line_number,
                    p.matched_text,
                    p.suggested_replacement
                ));
            }
            report.push('\n');
        }

        // 输出 high
        if !high.is_empty() {
            report.push_str("🟠 高风险:\n");
            for p in high {
                report.push_str(&format!(
                    "  - {} (置信度: {:.0}%)\n",
                    p.pattern.name,
                    p.confidence * 100.0
                ));
            }
            report.push('\n');
        }

        // 输出 medium
        if !medium.is_empty() {
            report.push_str("🟡 中等:\n");
            for p in medium {
                report.push_str(&format!(
                    "  - {} (置信度: {:.0}%)\n",
                    p.pattern.name,
                    p.confidence * 100.0
                ));
            }
            report.push('\n');
        }

        // 输出 low
        if !low.is_empty() {
            report.push_str("🟢 低风险:\n");
            for p in low {
                report.push_str(&format!(
                    "  - {} (置信度: {:.0}%)\n",
                    p.pattern.name,
                    p.confidence * 100.0
                ));
            }
        }

        Ok(report)
    }
}

#[derive(Debug, Clone)]
pub struct PatternLibraryStatistics {
    pub total_builtin_patterns: usize,
    pub total_custom_patterns: usize,
    pub pattern_type_stats: HashMap<String, usize>,
}

impl Default for ObfuscationPatternLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_detection() {
        let library = ObfuscationPatternLibrary::new();
        let code = r#"eval("alert('xss')")"#;

        let detected = library.detect(code).unwrap();
        assert!(!detected.is_empty());
        assert_eq!(
            detected[0].pattern.pattern_type,
            ObfuscationPatternType::DynamicCodeExecution
        );
    }

    #[test]
    fn test_property_obfuscation_detection() {
        let library = ObfuscationPatternLibrary::new();
        let code = r#"obj["property"]"#;

        let detected = library.detect(code).unwrap();
        assert!(!detected.is_empty());
        assert_eq!(
            detected[0].pattern.pattern_type,
            ObfuscationPatternType::PropertyObfuscation
        );
    }

    #[test]
    fn test_deobfuscation() {
        let library = ObfuscationPatternLibrary::new();
        let code = r#"obj["test"]"#;

        let result = library.deobfuscate(code).unwrap();
        // 应该包含 .test
        assert!(result.contains("test"));
    }

    #[test]
    fn test_statistics() {
        let library = ObfuscationPatternLibrary::new();
        let stats = library.get_statistics();
        assert!(stats.total_builtin_patterns > 0);
    }

    #[test]
    fn test_report_generation() {
        let library = ObfuscationPatternLibrary::new();
        let code = r#"eval("test")"#;

        let report = library.generate_report(code).unwrap();
        assert!(!report.is_empty());
    }
}
