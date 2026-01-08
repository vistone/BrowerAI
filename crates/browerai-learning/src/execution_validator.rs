//! 反混淀代码执行验证框架
//! Deobfuscated Code Execution Verification Framework
//!
//! 这个模块验证反混淀后的代码是否能正确执行

use std::collections::HashMap;

use browerai_js_parser::JsParser;

/// 代码执行验证结果
#[derive(Debug, Clone)]
pub struct ExecutionValidationResult {
    /// 原始代码
    pub original_code: String,

    /// 反混淀后的代码
    pub deobfuscated_code: String,

    /// 是否有效的 JavaScript 语法
    pub is_valid_syntax: bool,

    /// 是否成功反混淀
    pub deobfuscation_success: bool,

    /// 估计的执行结果（基于静态分析）
    pub execution_estimate: String,

    /// 检测到的执行风险
    pub risks: Vec<ExecutionRisk>,

    /// 验证的代码特性
    pub features: Vec<CodeFeature>,
}

/// 执行风险
#[derive(Debug, Clone)]
pub struct ExecutionRisk {
    /// 风险级别：low, medium, high
    pub level: String,

    /// 风险描述
    pub description: String,
}

/// 代码特性
#[derive(Debug, Clone)]
pub struct CodeFeature {
    /// 特性名称
    pub name: String,

    /// 是否检测到
    pub detected: bool,
}

/// 执行验证框架
pub struct ExecutionValidator {
    /// 验证规则缓存
    validation_rules: HashMap<String, Box<dyn Fn(&str) -> bool>>,
}

impl ExecutionValidator {
    /// 创建新的执行验证器
    pub fn new() -> Self {
        Self {
            validation_rules: HashMap::new(),
        }
    }

    /// 验证代码是否可以执行
    ///
    /// # 参数
    /// - `original_code`: 原始混淀代码
    /// - `deobfuscated_code`: 反混淀后的代码
    ///
    /// # 返回
    /// 执行验证结果
    pub fn validate_execution(
        &self,
        original_code: &str,
        deobfuscated_code: &str,
    ) -> ExecutionValidationResult {
        let mut risks = Vec::new();
        let mut features = Vec::new();

        // 检查基本语法
        let is_valid_syntax = self.check_syntax(deobfuscated_code);

        // 检查执行风险
        if deobfuscated_code.contains("eval") {
            risks.push(ExecutionRisk {
                level: "high".to_string(),
                description: "Code uses eval() which is dangerous".to_string(),
            });
        }

        if deobfuscated_code.contains("document.write") {
            risks.push(ExecutionRisk {
                level: "medium".to_string(),
                description: "Code modifies DOM with document.write".to_string(),
            });
        }

        // 检测代码特性
        if deobfuscated_code.contains("async") || deobfuscated_code.contains("await") {
            features.push(CodeFeature {
                name: "async/await".to_string(),
                detected: true,
            });
        }

        if deobfuscated_code.contains("class ") {
            features.push(CodeFeature {
                name: "classes".to_string(),
                detected: true,
            });
        }

        if deobfuscated_code.contains("=>") {
            features.push(CodeFeature {
                name: "arrow_functions".to_string(),
                detected: true,
            });
        }

        ExecutionValidationResult {
            original_code: original_code.to_string(),
            deobfuscated_code: deobfuscated_code.to_string(),
            is_valid_syntax,
            deobfuscation_success: !deobfuscated_code.is_empty(),
            execution_estimate: self.estimate_execution(deobfuscated_code),
            risks,
            features,
        }
    }

    /// 检查代码语法
    fn check_syntax(&self, code: &str) -> bool {
        // 优先使用真实解析器进行验证
        let parser = JsParser::new();
        if let Ok(valid) = parser.validate(code) {
            if valid {
                return true;
            }
        }

        // 回退到基本的括号匹配检查
        let open_braces = code.matches('{').count();
        let close_braces = code.matches('}').count();
        let open_parens = code.matches('(').count();
        let close_parens = code.matches(')').count();

        open_braces == close_braces && open_parens == close_parens
    }

    /// 估计代码执行结果
    fn estimate_execution(&self, code: &str) -> String {
        if code.contains("console.log") {
            "Will output to console".to_string()
        } else if code.contains("return") {
            "Function returns a value".to_string()
        } else if code.contains("document") {
            "Modifies DOM".to_string()
        } else {
            "Performs computation or side effects".to_string()
        }
    }

    /// 报告执行安全性
    pub fn report_safety(&self, result: &ExecutionValidationResult) -> SafetyReport {
        let mut score: f64 = 100.0;

        // 根据风险等级降分
        for risk in &result.risks {
            match risk.level.as_str() {
                "high" => score -= 40.0,
                "medium" => score -= 20.0,
                "low" => score -= 5.0,
                _ => {}
            }
        }

        SafetyReport {
            safety_score: score.max(0.0),
            is_safe: score >= 70.0,
            total_risks: result.risks.len(),
            high_risks: result.risks.iter().filter(|r| r.level == "high").count(),
            medium_risks: result.risks.iter().filter(|r| r.level == "medium").count(),
        }
    }

    /// 生成执行报告
    pub fn generate_report(&self, result: &ExecutionValidationResult) -> String {
        let safety = self.report_safety(result);
        let mut report = String::new();

        report.push_str("=== Execution Validation Report ===\n\n");

        report.push_str(&format!("Valid Syntax: {}\n", result.is_valid_syntax));
        report.push_str(&format!(
            "Deobfuscation Success: {}\n",
            result.deobfuscation_success
        ));
        report.push_str(&format!(
            "Execution Estimate: {}\n\n",
            result.execution_estimate
        ));

        report.push_str(&format!("Safety Score: {:.1}/100\n", safety.safety_score));
        report.push_str(&format!("Is Safe: {}\n\n", safety.is_safe));

        report.push_str(&format!("Total Risks: {}\n", safety.total_risks));
        report.push_str(&format!("  High: {}\n", safety.high_risks));
        report.push_str(&format!("  Medium: {}\n", safety.medium_risks));

        if !result.features.is_empty() {
            report.push_str("\nDetected Features:\n");
            for feature in &result.features {
                report.push_str(&format!("  - {}\n", feature.name));
            }
        }

        report
    }
}

impl Default for ExecutionValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// 安全报告
#[derive(Debug, Clone)]
pub struct SafetyReport {
    /// 安全分数 (0-100)
    pub safety_score: f64,

    /// 是否安全
    pub is_safe: bool,

    /// 总风险数
    pub total_risks: usize,

    /// 高风险数
    pub high_risks: usize,

    /// 中风险数
    pub medium_risks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = ExecutionValidator::new();
        assert_eq!(validator.validation_rules.len(), 0);
    }

    #[test]
    fn test_syntax_check_valid() {
        let validator = ExecutionValidator::new();
        assert!(validator.check_syntax("if (true) { console.log('test'); }"));
    }

    #[test]
    fn test_syntax_check_invalid() {
        let validator = ExecutionValidator::new();
        assert!(!validator.check_syntax("if (true { console.log('test'); }"));
    }

    #[test]
    fn test_execution_estimation() {
        let validator = ExecutionValidator::new();

        let result = validator.estimate_execution("console.log('test')");
        assert!(result.contains("console"));

        let result = validator.estimate_execution("return 42");
        assert!(result.contains("Function"));

        let result = validator.estimate_execution("var x = 5");
        assert!(result.contains("Performs"));
    }

    #[test]
    fn test_safety_report() {
        let validator = ExecutionValidator::new();

        let mut result = ExecutionValidationResult {
            original_code: "obfuscated".to_string(),
            deobfuscated_code: "console.log('hello')".to_string(),
            is_valid_syntax: true,
            deobfuscation_success: true,
            execution_estimate: "Output to console".to_string(),
            risks: vec![],
            features: vec![],
        };

        let safety = validator.report_safety(&result);
        assert_eq!(safety.safety_score, 100.0);
        assert!(safety.is_safe);

        // 添加高风险
        result.risks.push(ExecutionRisk {
            level: "high".to_string(),
            description: "Test risk".to_string(),
        });

        let safety = validator.report_safety(&result);
        assert!(safety.safety_score < 100.0);
        assert_eq!(safety.high_risks, 1);
    }

    #[test]
    fn test_report_generation() {
        let validator = ExecutionValidator::new();

        let result = ExecutionValidationResult {
            original_code: "obfuscated".to_string(),
            deobfuscated_code: "console.log('hello')".to_string(),
            is_valid_syntax: true,
            deobfuscation_success: true,
            execution_estimate: "Output to console".to_string(),
            risks: vec![],
            features: vec![CodeFeature {
                name: "arrow_functions".to_string(),
                detected: true,
            }],
        };

        let report = validator.generate_report(&result);
        assert!(report.contains("Execution Validation Report"));
        assert!(report.contains("Safety Score"));
        assert!(report.contains("arrow_functions"));
    }
}
