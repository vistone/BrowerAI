/// Advanced JavaScript deobfuscation module
use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};

lazy_static! {
    static ref VAR_PATTERN: Regex = Regex::new(r"\b([a-z])\b").unwrap();
    static ref HEX_PATTERN: Regex = Regex::new(r"\\x([0-9a-fA-F]{2})").unwrap();
    static ref IF_FALSE_PATTERN: Regex = Regex::new(r"if\s*\(\s*false\s*\)\s*\{[^}]*\}").unwrap();
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObfuscationAnalysis {
    pub obfuscation_score: f32,
    pub techniques: Vec<ObfuscationTechnique>,
    pub complexity: ComplexityMetrics,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObfuscationTechnique {
    NameMangling,
    StringEncoding,
    ControlFlowFlattening,
    DeadCodeInjection,
    ExpressionObfuscation,
    DataObfuscation,
    CodeSplitting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub variable_count: usize,
    pub avg_var_name_length: f32,
    pub function_count: usize,
    pub max_nesting_depth: usize,
    pub string_literal_count: usize,
    pub encoded_literal_count: usize,
}

#[derive(Debug, Clone)]
pub enum DeobfuscationStrategy {
    Basic,
    VariableRenaming,
    StringDecoding,
    ControlFlowSimplification,
    Comprehensive,
}

#[derive(Debug, Clone)]
pub struct DeobfuscationResult {
    pub code: String,
    pub original_code: String,
    pub success: bool,
    pub steps: Vec<String>,
    pub improvement: ImprovementMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementMetrics {
    pub readability_before: f32,
    pub readability_after: f32,
    pub loc_reduction_percent: f32,
    pub complexity_reduction: f32,
}

pub struct JsDeobfuscator {
    _aggressive: bool,
    max_iterations: usize,
}

impl JsDeobfuscator {
    pub fn new() -> Self {
        Self {
            _aggressive: false,
            max_iterations: 3,
        }
    }

    pub fn new_aggressive() -> Self {
        Self {
            _aggressive: true,
            max_iterations: 5,
        }
    }

    pub fn analyze_obfuscation(&self, code: &str) -> ObfuscationAnalysis {
        let mut techniques = Vec::new();

        if self.detect_name_mangling(code) {
            techniques.push(ObfuscationTechnique::NameMangling);
        }
        if self.detect_string_encoding(code) {
            techniques.push(ObfuscationTechnique::StringEncoding);
        }
        if self.detect_control_flow_flattening(code) {
            techniques.push(ObfuscationTechnique::ControlFlowFlattening);
        }
        if self.detect_dead_code(code) {
            techniques.push(ObfuscationTechnique::DeadCodeInjection);
        }

        let complexity = self.calculate_complexity(code);
        let obfuscation_score = self.calculate_obfuscation_score(&techniques, &complexity);
        let suggestions = self.generate_suggestions(&techniques);

        ObfuscationAnalysis {
            obfuscation_score,
            techniques,
            complexity,
            suggestions,
        }
    }

    pub fn deobfuscate(
        &self,
        code: &str,
        strategy: DeobfuscationStrategy,
    ) -> Result<DeobfuscationResult> {
        let original_code = code.to_string();
        let mut current_code = code.to_string();
        let mut steps = Vec::new();

        let readability_before = self.calculate_readability(&current_code);

        match strategy {
            DeobfuscationStrategy::Basic => {
                current_code = self.apply_basic_cleanup(&current_code);
                steps.push("Basic cleanup".to_string());
            }
            DeobfuscationStrategy::VariableRenaming => {
                current_code = self.apply_variable_renaming(&current_code);
                steps.push("Variable renaming".to_string());
            }
            DeobfuscationStrategy::StringDecoding => {
                current_code = self.apply_string_decoding(&current_code);
                steps.push("String decoding".to_string());
            }
            DeobfuscationStrategy::ControlFlowSimplification => {
                current_code = self.apply_control_flow_simplification(&current_code);
                steps.push("Control flow simplification".to_string());
            }
            DeobfuscationStrategy::Comprehensive => {
                for i in 0..self.max_iterations {
                    let prev_code = current_code.clone();
                    current_code = self.apply_basic_cleanup(&current_code);
                    current_code = self.apply_string_decoding(&current_code);
                    current_code = self.apply_variable_renaming(&current_code);
                    current_code = self.apply_control_flow_simplification(&current_code);
                    steps.push(format!("Comprehensive pass {}", i + 1));
                    if current_code == prev_code {
                        break;
                    }
                }
            }
        }

        let readability_after = self.calculate_readability(&current_code);
        let complexity_before = self.calculate_complexity(&original_code);
        let complexity_after = self.calculate_complexity(&current_code);

        let improvement = ImprovementMetrics {
            readability_before,
            readability_after,
            loc_reduction_percent: self.calculate_loc_reduction(&original_code, &current_code),
            complexity_reduction: complexity_before.max_nesting_depth as f32
                - complexity_after.max_nesting_depth as f32,
        };

        Ok(DeobfuscationResult {
            code: current_code,
            original_code,
            success: true,
            steps,
            improvement,
        })
    }

    fn detect_name_mangling(&self, code: &str) -> bool {
        let single_char_vars = code
            .matches(|c: char| c.is_ascii_lowercase() && c.is_alphabetic())
            .count();
        let total_tokens = code.split_whitespace().count();
        total_tokens > 0 && (single_char_vars as f32 / total_tokens as f32) > 0.15
    }

    fn detect_string_encoding(&self, code: &str) -> bool {
        code.contains("\\x") || code.contains("\\u") || code.contains("String.fromCharCode")
    }

    fn detect_control_flow_flattening(&self, code: &str) -> bool {
        let switch_count = code.matches("switch").count();
        let case_count = code.matches("case").count();
        switch_count > 0 && (case_count as f32 / switch_count as f32) > 10.0
    }

    fn detect_dead_code(&self, code: &str) -> bool {
        code.contains("if (false)") || code.contains("while (false)")
    }

    fn calculate_complexity(&self, code: &str) -> ComplexityMetrics {
        let variable_count = code.matches("var ").count()
            + code.matches("let ").count()
            + code.matches("const ").count();
        let function_count = code.matches("function").count();
        let max_nesting_depth = self.calculate_nesting_depth(code);
        let string_literal_count = code.matches('"').count() / 2;

        ComplexityMetrics {
            variable_count,
            avg_var_name_length: 2.0,
            function_count,
            max_nesting_depth,
            string_literal_count,
            encoded_literal_count: code.matches("0x").count(),
        }
    }

    fn calculate_nesting_depth(&self, code: &str) -> usize {
        let mut max_depth: usize = 0;
        let mut current_depth: usize = 0;
        for c in code.chars() {
            match c {
                '{' => {
                    current_depth += 1;
                    max_depth = max_depth.max(current_depth);
                }
                '}' => {
                    current_depth = current_depth.saturating_sub(1);
                }
                _ => {}
            }
        }
        max_depth
    }

    fn calculate_obfuscation_score(
        &self,
        techniques: &[ObfuscationTechnique],
        complexity: &ComplexityMetrics,
    ) -> f32 {
        let technique_score = techniques.len() as f32 * 0.15;
        let complexity_score = (complexity.max_nesting_depth as f32 / 20.0).min(0.5);
        (technique_score + complexity_score).min(1.0)
    }

    fn generate_suggestions(&self, techniques: &[ObfuscationTechnique]) -> Vec<String> {
        let mut suggestions = Vec::new();
        for technique in techniques {
            match technique {
                ObfuscationTechnique::NameMangling => {
                    suggestions.push("Apply variable renaming".to_string())
                }
                ObfuscationTechnique::StringEncoding => {
                    suggestions.push("Decode encoded strings".to_string())
                }
                ObfuscationTechnique::ControlFlowFlattening => {
                    suggestions.push("Simplify control flow".to_string())
                }
                ObfuscationTechnique::DeadCodeInjection => {
                    suggestions.push("Remove dead code".to_string())
                }
                _ => {}
            }
        }
        suggestions
    }

    fn apply_basic_cleanup(&self, code: &str) -> String {
        code.lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn apply_variable_renaming(&self, code: &str) -> String {
        let mut result = code.to_string();
        let mut counter = 0;
        for cap in VAR_PATTERN.captures_iter(code) {
            let var_name = cap.get(1).unwrap().as_str();
            if !["var", "let", "const", "function", "return"].contains(&var_name) {
                result = result.replace(var_name, &format!("var{}", counter));
                counter += 1;
            }
        }
        result
    }

    fn apply_string_decoding(&self, code: &str) -> String {
        HEX_PATTERN
            .replace_all(code, |caps: &regex::Captures| {
                let hex = &caps[1];
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    if byte.is_ascii_graphic() || byte == b' ' {
                        (byte as char).to_string()
                    } else {
                        format!("\\x{}", hex)
                    }
                } else {
                    format!("\\x{}", hex)
                }
            })
            .to_string()
    }

    fn apply_control_flow_simplification(&self, code: &str) -> String {
        let mut result = code.to_string();
        result = IF_FALSE_PATTERN.replace_all(&result, "").to_string();
        result
    }

    fn calculate_readability(&self, code: &str) -> f32 {
        let complexity = self.calculate_complexity(code);
        (complexity.avg_var_name_length / 10.0).min(1.0)
    }

    fn calculate_loc_reduction(&self, original: &str, deobfuscated: &str) -> f32 {
        let original_loc = original.lines().count() as f32;
        let deobf_loc = deobfuscated.lines().count() as f32;
        if original_loc == 0.0 {
            0.0
        } else {
            ((original_loc - deobf_loc) / original_loc * 100.0).max(0.0)
        }
    }
}

impl Default for JsDeobfuscator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_name_mangling() {
        let deobf = JsDeobfuscator::new();
        let obfuscated = "var a=1;var b=2;var c=a+b;";
        assert!(deobf.detect_name_mangling(obfuscated));
    }

    #[test]
    fn test_detect_string_encoding() {
        let deobf = JsDeobfuscator::new();
        let code_with_encoding = r#"var s = "\x48\x65\x6c\x6c\x6f";"#;
        assert!(deobf.detect_string_encoding(code_with_encoding));
    }

    #[test]
    fn test_basic_deobfuscation() {
        let deobf = JsDeobfuscator::new();
        let code = "var   a  =  1  ;  var   b =  2 ;";
        let result = deobf
            .deobfuscate(code, DeobfuscationStrategy::Basic)
            .unwrap();
        assert!(result.success);
        assert!(!result.code.is_empty());
    }

    #[test]
    fn test_nesting_depth_calculation() {
        let deobf = JsDeobfuscator::new();
        let code = "function test() { if (true) { if (true) { return 1; } } }";
        let depth = deobf.calculate_nesting_depth(code);
        assert_eq!(depth, 3);
    }
}
