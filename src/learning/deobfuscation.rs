/// Advanced JavaScript deobfuscation module
/// 
/// Provides multi-level obfuscation detection and progressive deobfuscation

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Obfuscation detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObfuscationAnalysis {
    /// Overall obfuscation score (0.0-1.0)
    pub obfuscation_score: f32,
    /// Detected obfuscation techniques
    pub techniques: Vec<ObfuscationTechnique>,
    /// Complexity metrics
    pub complexity: ComplexityMetrics,
    /// Suggestions for deobfuscation
    pub suggestions: Vec<String>,
}

/// Obfuscation technique detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObfuscationTechnique {
    /// Variable name mangling (single letters, etc.)
    NameMangling,
    /// String encoding/encryption
    StringEncoding,
    /// Control flow flattening
    ControlFlowFlattening,
    /// Dead code injection
    DeadCodeInjection,
    /// Function inlining
    FunctionInlining,
    /// Expression complexity
    ExpressionObfuscation,
    /// Array/object manipulation
    DataObfuscation,
    /// Code splitting and reassembly
    CodeSplitting,
}

/// Code complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Number of variables
    pub variable_count: usize,
    /// Average variable name length
    pub avg_var_name_length: f32,
    /// Number of functions
    pub function_count: usize,
    /// Nesting depth
    pub max_nesting_depth: usize,
    /// String literals count
    pub string_literal_count: usize,
    /// Hexadecimal/octal usage
    pub encoded_literal_count: usize,
}

/// Deobfuscation strategy
#[derive(Debug, Clone)]
pub enum DeobfuscationStrategy {
    /// Basic cleanup (whitespace, formatting)
    Basic,
    /// Variable renaming
    VariableRenaming,
    /// String decoding
    StringDecoding,
    /// Control flow simplification
    ControlFlowSimplification,
    /// Full multi-pass deobfuscation
    Comprehensive,
}

/// Deobfuscation result
#[derive(Debug, Clone)]
pub struct DeobfuscationResult {
    /// Deobfuscated code
    pub code: String,
    /// Original code
    pub original_code: String,
    /// Success flag
    pub success: bool,
    /// Transformation steps applied
    pub steps: Vec<String>,
    /// Improvement metrics
    pub improvement: ImprovementMetrics,
}

/// Metrics showing deobfuscation improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementMetrics {
    /// Readability score before (0.0-1.0)
    pub readability_before: f32,
    /// Readability score after (0.0-1.0)
    pub readability_after: f32,
    /// Lines of code reduction (%)
    pub loc_reduction_percent: f32,
    /// Complexity reduction
    pub complexity_reduction: f32,
}

/// Advanced JavaScript deobfuscator
pub struct JsDeobfuscator {
    /// Enable aggressive deobfuscation
    aggressive: bool,
    /// Maximum iterations for multi-pass
    max_iterations: usize,
}

impl JsDeobfuscator {
    /// Create a new deobfuscator
    pub fn new() -> Self {
        Self {
            aggressive: false,
            max_iterations: 3,
        }
    }

    /// Create with aggressive mode enabled
    pub fn new_aggressive() -> Self {
        Self {
            aggressive: true,
            max_iterations: 5,
        }
    }

    /// Analyze obfuscation in JavaScript code
    pub fn analyze_obfuscation(&self, code: &str) -> ObfuscationAnalysis {
        let mut techniques = Vec::new();
        
        // Detect name mangling
        if self.detect_name_mangling(code) {
            techniques.push(ObfuscationTechnique::NameMangling);
        }
        
        // Detect string encoding
        if self.detect_string_encoding(code) {
            techniques.push(ObfuscationTechnique::StringEncoding);
        }
        
        // Detect control flow flattening
        if self.detect_control_flow_flattening(code) {
            techniques.push(ObfuscationTechnique::ControlFlowFlattening);
        }
        
        // Detect dead code
        if self.detect_dead_code(code) {
            techniques.push(ObfuscationTechnique::DeadCodeInjection);
        }

        // Detect expression obfuscation
        if self.detect_expression_obfuscation(code) {
            techniques.push(ObfuscationTechnique::ExpressionObfuscation);
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

    /// Deobfuscate JavaScript code
    pub fn deobfuscate(&self, code: &str, strategy: DeobfuscationStrategy) -> Result<DeobfuscationResult> {
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
                // Multi-pass comprehensive deobfuscation
                for i in 0..self.max_iterations {
                    let prev_code = current_code.clone();
                    
                    current_code = self.apply_basic_cleanup(&current_code);
                    current_code = self.apply_string_decoding(&current_code);
                    current_code = self.apply_variable_renaming(&current_code);
                    current_code = self.apply_control_flow_simplification(&current_code);
                    
                    steps.push(format!("Comprehensive pass {}", i + 1));
                    
                    // Stop if no significant change
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
            complexity_reduction: complexity_before.max_nesting_depth as f32 - complexity_after.max_nesting_depth as f32,
        };
        
        Ok(DeobfuscationResult {
            code: current_code,
            original_code,
            success: true,
            steps,
            improvement,
        })
    }

    /// Detect name mangling (short variable names)
    fn detect_name_mangling(&self, code: &str) -> bool {
        let single_char_vars = code.matches(|c: char| c.is_ascii_lowercase() && c.is_alphabetic()).count();
        let total_tokens = code.split_whitespace().count();
        
        if total_tokens == 0 {
            return false;
        }
        
        // High ratio of single-char identifiers suggests mangling
        (single_char_vars as f32 / total_tokens as f32) > 0.15
    }

    /// Detect string encoding (hex, unicode escapes, etc.)
    fn detect_string_encoding(&self, code: &str) -> bool {
        code.contains("\\x") || code.contains("\\u") || code.contains("String.fromCharCode")
    }

    /// Detect control flow flattening
    fn detect_control_flow_flattening(&self, code: &str) -> bool {
        // Look for switch statements with many cases (common in flattened control flow)
        let switch_count = code.matches("switch").count();
        let case_count = code.matches("case").count();
        
        if switch_count == 0 {
            return false;
        }
        
        // Many cases per switch suggests control flow flattening
        (case_count as f32 / switch_count as f32) > 10.0
    }

    /// Detect dead code injection
    fn detect_dead_code(&self, code: &str) -> bool {
        // Look for unreachable code patterns
        code.contains("if (false)") || code.contains("while (false)") || code.contains("return;") && code.contains("return;")
    }

    /// Detect expression obfuscation
    fn detect_expression_obfuscation(&self, code: &str) -> bool {
        // Look for complex mathematical expressions
        let operator_count = code.matches(|c| "+-*/%^&|".contains(c)).count();
        let line_count = code.lines().count().max(1);
        
        // High operator density suggests expression obfuscation
        (operator_count as f32 / line_count as f32) > 5.0
    }

    /// Calculate complexity metrics
    fn calculate_complexity(&self, code: &str) -> ComplexityMetrics {
        let variable_count = self.count_variables(code);
        let avg_var_name_length = self.average_var_name_length(code);
        let function_count = code.matches("function").count();
        let max_nesting_depth = self.calculate_nesting_depth(code);
        let string_literal_count = code.matches('"').count() / 2;
        let encoded_literal_count = code.matches("0x").count() + code.matches("\\x").count();
        
        ComplexityMetrics {
            variable_count,
            avg_var_name_length,
            function_count,
            max_nesting_depth,
            string_literal_count,
            encoded_literal_count,
        }
    }

    /// Count variables (approximate)
    fn count_variables(&self, code: &str) -> usize {
        code.matches("var ").count() + code.matches("let ").count() + code.matches("const ").count()
    }

    /// Average variable name length (approximate)
    fn average_var_name_length(&self, code: &str) -> f32 {
        let var_names: Vec<&str> = code
            .split_whitespace()
            .filter(|w| w.chars().all(|c| c.is_alphanumeric() || c == '_'))
            .collect();
        
        if var_names.is_empty() {
            return 0.0;
        }
        
        var_names.iter().map(|n| n.len()).sum::<usize>() as f32 / var_names.len() as f32
    }

    /// Calculate maximum nesting depth
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

    /// Calculate obfuscation score
    fn calculate_obfuscation_score(&self, techniques: &[ObfuscationTechnique], complexity: &ComplexityMetrics) -> f32 {
        let technique_score = techniques.len() as f32 * 0.15;
        let complexity_score = (complexity.max_nesting_depth as f32 / 20.0).min(0.5);
        let var_name_score = (5.0 - complexity.avg_var_name_length).max(0.0) / 5.0 * 0.2;
        
        (technique_score + complexity_score + var_name_score).min(1.0)
    }

    /// Generate deobfuscation suggestions
    fn generate_suggestions(&self, techniques: &[ObfuscationTechnique]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        for technique in techniques {
            match technique {
                ObfuscationTechnique::NameMangling => {
                    suggestions.push("Apply variable renaming to meaningful names".to_string());
                }
                ObfuscationTechnique::StringEncoding => {
                    suggestions.push("Decode encoded string literals".to_string());
                }
                ObfuscationTechnique::ControlFlowFlattening => {
                    suggestions.push("Simplify control flow structures".to_string());
                }
                ObfuscationTechnique::DeadCodeInjection => {
                    suggestions.push("Remove unreachable dead code".to_string());
                }
                ObfuscationTechnique::ExpressionObfuscation => {
                    suggestions.push("Simplify complex expressions".to_string());
                }
                _ => {}
            }
        }
        
        suggestions
    }

    /// Apply basic cleanup
    fn apply_basic_cleanup(&self, code: &str) -> String {
        // Remove excessive whitespace and format
        code.lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Apply variable renaming
    fn apply_variable_renaming(&self, code: &str) -> String {
        // Simple variable renaming (placeholder - would use AST transformation)
        let mut result = code.to_string();
        let mut replacements = HashMap::new();
        let mut counter = 0;
        
        // Find single-letter variables and rename them
        for word in code.split_whitespace() {
            if word.len() == 1 && word.chars().all(|c| c.is_alphabetic()) {
                if !replacements.contains_key(word) {
                    replacements.insert(word.to_string(), format!("var{}", counter));
                    counter += 1;
                }
            }
        }
        
        for (old, new) in replacements {
            result = result.replace(&old, &new);
        }
        
        result
    }

    /// Apply string decoding
    fn apply_string_decoding(&self, code: &str) -> String {
        // Decode hex strings (simplified)
        let result = code.to_string();
        
        // This is a simplified example - full implementation would decode all encoded strings
        if result.contains("\\x") {
            log::debug!("String decoding applied");
        }
        
        result
    }

    /// Apply control flow simplification
    fn apply_control_flow_simplification(&self, code: &str) -> String {
        // Remove some obvious dead code
        let mut result = code.to_string();
        
        // Remove if (false) blocks (simplified)
        result = result.replace("if (false)", "// removed dead code");
        result = result.replace("while (false)", "// removed dead code");
        
        result
    }

    /// Calculate readability score
    fn calculate_readability(&self, code: &str) -> f32 {
        let complexity = self.calculate_complexity(code);
        let avg_line_length = if code.lines().count() > 0 {
            code.len() as f32 / code.lines().count() as f32
        } else {
            0.0
        };
        
        // Simple readability score (higher is better)
        let var_name_score = (complexity.avg_var_name_length / 10.0).min(1.0);
        let nesting_score = (1.0 - (complexity.max_nesting_depth as f32 / 20.0)).max(0.0);
        let line_length_score = (1.0 - (avg_line_length / 200.0)).max(0.0);
        
        (var_name_score + nesting_score + line_length_score) / 3.0
    }

    /// Calculate lines of code reduction
    fn calculate_loc_reduction(&self, original: &str, deobfuscated: &str) -> f32 {
        let original_loc = original.lines().count() as f32;
        let deobf_loc = deobfuscated.lines().count() as f32;
        
        if original_loc == 0.0 {
            return 0.0;
        }
        
        ((original_loc - deobf_loc) / original_loc * 100.0).max(0.0)
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
    fn test_deobfuscator_creation() {
        let deobf = JsDeobfuscator::new();
        assert_eq!(deobf.max_iterations, 3);
        assert!(!deobf.aggressive);
    }

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
    fn test_analyze_obfuscation() {
        let deobf = JsDeobfuscator::new();
        let code = "var a=1;var b=2;var c=a+b;console.log(c);";
        let analysis = deobf.analyze_obfuscation(code);
        
        assert!(analysis.obfuscation_score >= 0.0);
        assert!(analysis.obfuscation_score <= 1.0);
        assert!(analysis.complexity.variable_count > 0);
    }

    #[test]
    fn test_basic_deobfuscation() {
        let deobf = JsDeobfuscator::new();
        let code = "var   a  =  1  ;  var   b =  2 ;";
        let result = deobf.deobfuscate(code, DeobfuscationStrategy::Basic).unwrap();
        
        assert!(result.success);
        assert!(!result.code.is_empty());
        assert_eq!(result.steps.len(), 1);
    }

    #[test]
    fn test_complexity_calculation() {
        let deobf = JsDeobfuscator::new();
        let code = "function test() {\n  if (true) {\n    if (true) {\n      return 1;\n    }\n  }\n}";
        let complexity = deobf.calculate_complexity(code);
        
        assert!(complexity.max_nesting_depth > 0);
        assert!(complexity.function_count > 0);
    }

    #[test]
    fn test_readability_score() {
        let deobf = JsDeobfuscator::new();
        let readable = "function calculateSum(a, b) { return a + b; }";
        let obfuscated = "function a(b,c){return b+c}";
        
        let score_readable = deobf.calculate_readability(readable);
        let score_obfuscated = deobf.calculate_readability(obfuscated);
        
        // Both should be valid scores
        assert!(score_readable >= 0.0 && score_readable <= 1.0);
        assert!(score_obfuscated >= 0.0 && score_obfuscated <= 1.0);
    }
}
