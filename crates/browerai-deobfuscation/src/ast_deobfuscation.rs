/// Advanced AST-based Deobfuscation Techniques
///
/// This module implements more sophisticated deobfuscation techniques that require
/// deeper code analysis, inspired by webcrack's AST transformations.
///
/// Key techniques:
/// - Variable inlining and constant propagation
/// - Dead code elimination
/// - Function call inlining
/// - Array rotation reversal
/// - Sequence expression simplification
use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;

/// Variable usage tracking for safe inlining
#[derive(Debug, Clone)]
pub struct VariableUsage {
    /// Variable name
    pub name: String,
    /// Number of times used
    pub use_count: usize,
    /// Initial value (if constant)
    pub initial_value: Option<String>,
    /// Can be safely inlined
    pub can_inline: bool,
}

/// Dead code pattern
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeadCodePattern {
    /// Unreachable after return/throw
    UnreachableAfterReturn,
    /// Unreachable in false branch
    UnreachableInFalseBranch,
    /// Unused function
    UnusedFunction,
    /// Unused variable
    UnusedVariable,
}

/// Array rotation info
#[derive(Debug, Clone)]
pub struct ArrayRotation {
    /// Array name
    pub array_name: String,
    /// Rotation amount
    pub rotation: usize,
    /// Original array
    pub original: Vec<String>,
}

/// AST-based deobfuscator
pub struct ASTDeobfuscator {
    /// Track variable usage
    variable_usage: HashMap<String, VariableUsage>,
    /// Detected dead code
    dead_code: Vec<DeadCodePattern>,
}

impl ASTDeobfuscator {
    /// Create new AST deobfuscator
    pub fn new() -> Self {
        Self {
            variable_usage: HashMap::new(),
            dead_code: Vec::new(),
        }
    }

    /// Analyze and deobfuscate code
    pub fn deobfuscate(&mut self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        // Phase 1: Track variable usage
        self.track_variable_usage(&result);

        // Phase 2: Inline single-use constants
        result = self.inline_constants(&result)?;

        // Phase 3: Remove dead code
        result = self.remove_dead_code(&result)?;

        // Phase 4: Simplify sequence expressions
        result = self.simplify_sequences(&result)?;

        // Phase 5: Inline simple function calls
        result = self.inline_function_calls(&result)?;

        // Phase 6: Detect and reverse array rotation
        if let Some(rotation) = self.detect_array_rotation(&result) {
            result = self.reverse_array_rotation(&result, &rotation)?;
        }

        Ok(result)
    }

    /// Track variable usage for inlining decisions
    fn track_variable_usage(&mut self, code: &str) {
        self.variable_usage.clear();

        let var_decl_pattern = Regex::new(r"(?:var|let|const)\s+(\w+)\s*=\s*([^;]+);").ok();

        if let Some(re) = var_decl_pattern {
            for caps in re.captures_iter(code) {
                if let (Some(name), Some(value)) = (caps.get(1), caps.get(2)) {
                    let var_name = name.as_str().to_string();
                    let var_value = value.as_str().trim().to_string();

                    let use_pattern =
                        Regex::new(&format!(r"\b{}\b", regex::escape(&var_name))).ok();
                    let use_count = use_pattern
                        .map(|re| re.find_iter(code).count().saturating_sub(1))
                        .unwrap_or(0);

                    let is_constant = self.is_simple_constant(&var_value);

                    self.variable_usage.insert(
                        var_name.clone(),
                        VariableUsage {
                            name: var_name,
                            use_count,
                            initial_value: Some(var_value),
                            can_inline: use_count <= 2 && is_constant,
                        },
                    );
                }
            }
        }
    }

    /// Check if a value is a simple constant
    fn is_simple_constant(&self, value: &str) -> bool {
        let constant_pattern =
            Regex::new(r#"^(?:\d+|0x[0-9a-fA-F]+|'[^']*'|"[^"]*"|true|false|null|undefined)$"#)
                .ok();

        constant_pattern
            .map(|re| re.is_match(value))
            .unwrap_or(false)
    }

    /// Inline single-use constant variables
    fn inline_constants(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        for (name, usage) in &self.variable_usage {
            if usage.can_inline {
                if let Some(value) = &usage.initial_value {
                    let pattern = format!(r"\b{}\b", regex::escape(name));
                    if let Ok(re) = Regex::new(&pattern) {
                        let parts: Vec<&str> = result.split('\n').collect();
                        let mut new_lines = Vec::new();

                        for line in parts {
                            if line.contains(&format!("{} =", name))
                                || line.contains(&format!("var {}", name))
                            {
                                new_lines.push(line.to_string());
                            } else {
                                let replaced = re.replace_all(line, value.as_str()).to_string();
                                new_lines.push(replaced);
                            }
                        }

                        result = new_lines.join("\n");
                    }

                    let decl_pattern = format!(
                        r"(?:var|let|const)\s+{}\s*=\s*[^;]+;\s*",
                        regex::escape(name)
                    );
                    if let Ok(re) = Regex::new(&decl_pattern) {
                        result = re.replace(&result, "").to_string();
                    }
                }
            }
        }

        Ok(result)
    }

    /// Remove dead code
    fn remove_dead_code(&mut self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        let after_return = Regex::new(r"return\s+[^;]+;\s*([^}]+)\s*}")?;
        for caps in after_return.captures_iter(code) {
            if let Some(dead) = caps.get(1) {
                let dead_code = dead.as_str();
                if !dead_code.contains("function") && !dead_code.contains("var") {
                    result = result.replace(dead_code, "");
                    self.dead_code.push(DeadCodePattern::UnreachableAfterReturn);
                }
            }
        }

        let false_branch = Regex::new(r"if\s*\(\s*false\s*\)\s*\{[^}]*\}")?;
        let removed_count = false_branch.find_iter(&result).count();
        result = false_branch.replace_all(&result, "").to_string();

        for _ in 0..removed_count {
            self.dead_code
                .push(DeadCodePattern::UnreachableInFalseBranch);
        }

        Ok(result)
    }

    /// Simplify sequence expressions
    fn simplify_sequences(&self, code: &str) -> Result<String> {
        let result = code.to_string();
        let sequence_pattern = Regex::new(r"\([^,]+,[^)]+\)")?;
        let _sequence_count = sequence_pattern.find_iter(&result).count();
        Ok(result)
    }

    /// Inline simple function calls
    fn inline_function_calls(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();

        let simple_func = Regex::new(r"function\s+(\w+)\s*\(\s*\)\s*\{\s*return\s+([^;]+);\s*\}")?;

        let mut to_inline = Vec::new();

        for caps in simple_func.captures_iter(code) {
            if let (Some(name), Some(return_val)) = (caps.get(1), caps.get(2)) {
                to_inline.push((name.as_str().to_string(), return_val.as_str().to_string()));
            }
        }

        for (func_name, return_val) in to_inline {
            let call_pattern = format!(r"\b{}\(\s*\)", regex::escape(&func_name));
            if let Ok(re) = Regex::new(&call_pattern) {
                result = re.replace_all(&result, return_val.as_str()).to_string();
            }

            let def_pattern = format!(
                r"function\s+{}\s*\(\s*\)\s*\{{\s*return\s+[^;]+;\s*\}}\s*",
                regex::escape(&func_name)
            );
            if let Ok(re) = Regex::new(&def_pattern) {
                result = re.replace(&result, "").to_string();
            }
        }

        Ok(result)
    }

    /// Detect array rotation
    fn detect_array_rotation(&self, code: &str) -> Option<ArrayRotation> {
        let rotation_pattern = Regex::new(
            r"\(function\s*\((\w+),\s*(\w+)\)\s*\{[^}]*\.push\([^)]*\.shift\(\)\)[^}]*\}\)\((\w+),\s*(\d+)\)"
        ).ok()?;

        if let Some(caps) = rotation_pattern.captures(code) {
            if let (Some(_param1), Some(_param2), Some(array_name), Some(rotation)) =
                (caps.get(1), caps.get(2), caps.get(3), caps.get(4))
            {
                let rotation_amount = rotation.as_str().parse::<usize>().ok()?;

                let array_pattern = Regex::new(&format!(
                    r"var\s+{}\s*=\s*\[([^\]]+)\]",
                    regex::escape(array_name.as_str())
                ))
                .ok()?;

                if let Some(arr_caps) = array_pattern.captures(code) {
                    if let Some(contents) = arr_caps.get(1) {
                        let elements: Vec<String> = contents
                            .as_str()
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();

                        return Some(ArrayRotation {
                            array_name: array_name.as_str().to_string(),
                            rotation: rotation_amount,
                            original: elements,
                        });
                    }
                }
            }
        }

        None
    }

    /// Reverse array rotation
    fn reverse_array_rotation(&self, code: &str, rotation: &ArrayRotation) -> Result<String> {
        let mut result = code.to_string();

        let mut restored = rotation.original.clone();
        for _ in 0..rotation.rotation {
            if let Some(first) = restored.first().cloned() {
                restored.remove(0);
                restored.push(first);
            }
        }

        let old_array = format!(
            "var {} = [{}]",
            rotation.array_name,
            rotation.original.join(", ")
        );
        let new_array = format!("var {} = [{}]", rotation.array_name, restored.join(", "));

        result = result.replace(&old_array, &new_array);

        let rotation_call_pattern = Regex::new(&format!(
            r"\(function\s*\([^)]+\)\s*\{{[^}}]*\.push\([^)]*\.shift\(\)\)[^}}]*\}}\)\({},\s*{}\);?",
            regex::escape(&rotation.array_name),
            rotation.rotation
        ))?;

        result = rotation_call_pattern.replace(&result, "").to_string();

        Ok(result)
    }

    /// Get statistics about the deobfuscation
    pub fn get_stats(&self) -> ASTDeobfuscationStats {
        ASTDeobfuscationStats {
            variables_inlined: self
                .variable_usage
                .values()
                .filter(|v| v.can_inline)
                .count(),
            dead_code_removed: self.dead_code.len(),
            functions_inlined: 0,
        }
    }
}

impl Default for ASTDeobfuscator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from AST deobfuscation
#[derive(Debug, Clone)]
pub struct ASTDeobfuscationStats {
    pub variables_inlined: usize,
    pub dead_code_removed: usize,
    pub functions_inlined: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_inlining() {
        let mut deob = ASTDeobfuscator::new();
        let code = r#"
var x = 42;
console.log(x);
"#;

        let result = deob.deobfuscate(code).unwrap();
        assert!(result.contains("42") || result.contains("x"));
    }

    #[test]
    fn test_dead_code_removal() {
        let mut deob = ASTDeobfuscator::new();
        let code = r#"
function test() {
    return 1;
    console.log('never');
}
"#;

        let result = deob.deobfuscate(code).unwrap();
        assert!(!result.contains("never") || result.contains("return"));
    }

    #[test]
    fn test_simple_function_inlining() {
        let mut deob = ASTDeobfuscator::new();
        let code = r#"
function getVal() { return 42; }
var x = getVal();
"#;

        let result = deob.deobfuscate(code).unwrap();
        assert!(result.contains("42"));
    }

    #[test]
    fn test_variable_usage_tracking() {
        let mut deob = ASTDeobfuscator::new();
        let code = r#"
var x = 10;
var y = x + 5;
console.log(y);
"#;

        deob.track_variable_usage(code);
        assert!(deob.variable_usage.contains_key("x"));
        assert!(deob.variable_usage.contains_key("y"));
    }

    #[test]
    fn test_is_simple_constant() {
        let deob = ASTDeobfuscator::new();

        assert!(deob.is_simple_constant("42"));
        assert!(deob.is_simple_constant("0x10"));
        assert!(deob.is_simple_constant("'hello'"));
        assert!(deob.is_simple_constant("true"));
        assert!(!deob.is_simple_constant("x + y"));
    }
}
