/// Enhanced JavaScript Deobfuscation Module
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct StringArray {
    pub name: String,
    pub contents: Vec<String>,
    pub is_rotated: bool,
    pub rotation_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyFunctionType {
    Simple,
    ArrayAccess,
    Arithmetic,
    ObjectProperty,
    Chained,
}

#[derive(Debug, Clone)]
pub struct ProxyFunction {
    pub name: String,
    pub proxy_type: ProxyFunctionType,
    pub target: String,
    pub can_inline: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlFlowPattern {
    SwitchFlattened,
    ObjectBased,
    WhileLoop,
    OpaquePredicates,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelfDefendingPattern {
    AntiDebug,
    ConsoleProtection,
    DomainLock,
    StackTraceCheck,
    ToStringCheck,
    DevToolsDetect,
}

#[derive(Debug, Clone)]
pub struct EnhancedDeobfuscationResult {
    pub code: String,
    pub original_code: String,
    pub success: bool,
    pub transformations: Vec<String>,
    pub stats: DeobfuscationStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeobfuscationStats {
    pub string_arrays_unpacked: usize,
    pub proxy_functions_removed: usize,
    pub control_flow_simplified: usize,
    pub constants_folded: usize,
    pub self_defending_removed: usize,
    pub string_decoders_evaluated: usize,
    pub size_reduction: i64,
    pub readability_improvement: f32,
}

pub struct EnhancedDeobfuscator {
    aggressive_mode: bool,
    max_iterations: usize,
    enable_vm_decoding: bool,
    stats: DeobfuscationStats,
}

impl EnhancedDeobfuscator {
    pub fn new() -> Self {
        Self {
            aggressive_mode: false,
            max_iterations: 5,
            enable_vm_decoding: false,
            stats: DeobfuscationStats::default(),
        }
    }

    pub fn new_aggressive() -> Self {
        Self {
            aggressive_mode: true,
            max_iterations: 10,
            enable_vm_decoding: false,
            stats: DeobfuscationStats::default(),
        }
    }

    pub fn enable_vm_decoding(&mut self) {
        self.enable_vm_decoding = true;
    }

    pub fn deobfuscate(&mut self, code: &str) -> Result<EnhancedDeobfuscationResult> {
        let original_code = code.to_string();
        let mut current_code = code.to_string();
        let mut transformations = Vec::new();

        self.stats = DeobfuscationStats::default();

        let iterations = if self.aggressive_mode {
            self.max_iterations * 2
        } else {
            self.max_iterations
        };

        for iteration in 0..iterations {
            let prev_code = current_code.clone();

            if let Some(string_array) = self.detect_string_array(&current_code) {
                current_code = self.unpack_string_array(&current_code, &string_array)?;
                self.stats.string_arrays_unpacked += 1;
                transformations.push(format!("Iter {}: Unpacked string array", iteration + 1));
            }

            let proxies = self.detect_proxy_functions(&current_code);
            if !proxies.is_empty() {
                current_code = self.remove_proxy_functions(&current_code, &proxies)?;
                self.stats.proxy_functions_removed += proxies.len();
                transformations.push(format!(
                    "Iter {}: Removed {} proxies",
                    iteration + 1,
                    proxies.len()
                ));
            }

            if self.detect_control_flow_flattening(&current_code) {
                current_code = self.unflatten_control_flow(&current_code)?;
                self.stats.control_flow_simplified += 1;
            }

            let folded = self.fold_constants(&current_code)?;
            if folded != current_code {
                self.stats.constants_folded += 1;
                current_code = folded;
            }

            let defending_patterns = self.detect_self_defending(&current_code);
            if !defending_patterns.is_empty() {
                current_code = self.remove_self_defending(&current_code, &defending_patterns)?;
                self.stats.self_defending_removed += defending_patterns.len();
            }

            current_code = self.simplify_member_expressions(&current_code)?;

            if current_code == prev_code {
                transformations.push(format!("Converged at iteration {}", iteration + 1));
                break;
            }
        }

        self.stats.size_reduction = original_code.len() as i64 - current_code.len() as i64;
        self.stats.readability_improvement =
            self.calculate_readability_improvement(&original_code, &current_code);

        Ok(EnhancedDeobfuscationResult {
            code: current_code,
            original_code,
            success: true,
            transformations,
            stats: self.stats.clone(),
        })
    }

    fn detect_string_array(&self, code: &str) -> Option<StringArray> {
        let array_pattern =
            Regex::new(r"(?:var|let|const)\s+(_0x[a-f0-9]{4,})\s*=\s*\[([^\]]+)\]").ok()?;
        if let Some(caps) = array_pattern.captures(code) {
            let name = caps.get(1)?.as_str().to_string();
            let array_content = caps.get(2)?.as_str();
            let string_regex = Regex::new(r#"['"]([^'"]*)['"]\s*,?"#).ok()?;
            let contents: Vec<String> = string_regex
                .captures_iter(array_content)
                .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
                .collect();
            if contents.len() >= 5 {
                Some(StringArray {
                    name,
                    contents,
                    is_rotated: false,
                    rotation_offset: 0,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    fn unpack_string_array(&self, code: &str, array: &StringArray) -> Result<String> {
        let mut result = code.to_string();
        for (index, value) in array.contents.iter().enumerate() {
            let escaped_value = value.replace('\\', "\\\\").replace('\'', "\\'");
            let patterns = vec![
                format!(r"\b{}\[{}\]", regex::escape(&array.name), index),
                format!(r"\b{}\[0x{:x}\]", regex::escape(&array.name), index),
            ];
            for pattern in patterns {
                if let Ok(re) = Regex::new(&pattern) {
                    result = re
                        .replace_all(&result, format!("'{}'", escaped_value))
                        .to_string();
                }
            }
        }
        let array_decl_pattern = format!(
            r"(?:var|let|const)\s+{}\s*=\s*\[[^\]]+\];?\s*",
            regex::escape(&array.name)
        );
        if let Ok(re) = Regex::new(&array_decl_pattern) {
            result = re.replace(&result, "").to_string();
        }
        Ok(result)
    }

    fn detect_proxy_functions(&self, code: &str) -> Vec<ProxyFunction> {
        let mut proxies = Vec::new();
        let simple_proxy =
            Regex::new(r"function\s+(\w+)\s*\([^)]*\)\s*\{\s*return\s+(\w+)\([^)]*\);\s*\}").ok();
        if let Some(re) = simple_proxy {
            for caps in re.captures_iter(code) {
                if let (Some(name), Some(target)) = (caps.get(1), caps.get(2)) {
                    proxies.push(ProxyFunction {
                        name: name.as_str().to_string(),
                        proxy_type: ProxyFunctionType::Simple,
                        target: target.as_str().to_string(),
                        can_inline: true,
                    });
                }
            }
        }
        proxies
    }

    fn remove_proxy_functions(&self, code: &str, proxies: &[ProxyFunction]) -> Result<String> {
        let mut result = code.to_string();
        for proxy in proxies {
            if proxy.proxy_type == ProxyFunctionType::Simple {
                let pattern = format!(r"\b{}\(", regex::escape(&proxy.name));
                let replacement = format!("{}(", proxy.target);
                if let Ok(re) = Regex::new(&pattern) {
                    result = re.replace_all(&result, replacement.as_str()).to_string();
                }
                let def_pattern = format!(
                    r"function\s+{}\s*\([^)]*\)\s*\{{[^}}]+\}}\s*",
                    regex::escape(&proxy.name)
                );
                if let Ok(re) = Regex::new(&def_pattern) {
                    result = re.replace(&result, "").to_string();
                }
            }
        }
        Ok(result)
    }

    fn detect_control_flow_flattening(&self, code: &str) -> bool {
        let switch_pattern = Regex::new(r"switch\s*\([^)]+\)\s*\{").ok();
        let case_pattern = Regex::new(r"case\s+").ok();
        if let (Some(switch_re), Some(case_re)) = (switch_pattern, case_pattern) {
            let switch_count = switch_re.find_iter(code).count();
            let case_count = case_re.find_iter(code).count();
            if switch_count > 0 {
                return case_count as f32 / switch_count as f32 > 10.0;
            }
        }
        false
    }

    fn unflatten_control_flow(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();
        result = result.replace("![]", "false");
        result = result.replace("!![]", "true");
        let if_true_pattern = Regex::new(r"if\s*\(\s*true\s*\)\s*\{([^}]*)\}")?;
        result = if_true_pattern.replace_all(&result, "$1").to_string();
        let if_false_pattern = Regex::new(r"if\s*\(\s*false\s*\)\s*\{[^}]*\}")?;
        result = if_false_pattern.replace_all(&result, "").to_string();
        Ok(result)
    }

    fn fold_constants(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();
        let hex_pattern = Regex::new(r"0x([0-9a-fA-F]+)")?;
        result = hex_pattern
            .replace_all(&result, |caps: &regex::Captures| {
                let hex = &caps[1];
                if let Ok(num) = i64::from_str_radix(hex, 16) {
                    if num < 1000 {
                        return num.to_string();
                    }
                }
                caps[0].to_string()
            })
            .to_string();
        let simple_add = Regex::new(r"(\d+)\s*\+\s*(\d+)")?;
        result = simple_add
            .replace_all(&result, |caps: &regex::Captures| {
                let a: i64 = caps[1].parse().unwrap_or(0);
                let b: i64 = caps[2].parse().unwrap_or(0);
                (a + b).to_string()
            })
            .to_string();
        Ok(result)
    }

    fn detect_self_defending(&self, code: &str) -> Vec<SelfDefendingPattern> {
        let mut patterns = Vec::new();
        if code.contains("debugger") {
            patterns.push(SelfDefendingPattern::AntiDebug);
        }
        if code.contains("console.log = function") {
            patterns.push(SelfDefendingPattern::ConsoleProtection);
        }
        if code.contains("window.outerHeight") && code.contains("window.innerHeight") {
            patterns.push(SelfDefendingPattern::DevToolsDetect);
        }
        patterns
    }

    fn remove_self_defending(
        &self,
        code: &str,
        patterns: &[SelfDefendingPattern],
    ) -> Result<String> {
        use std::sync::LazyLock;
        static CONSOLE_PATTERN: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"console\.\w+\s*=\s*function[^}]*\};").unwrap());

        let mut result = code.to_string();
        for pattern in patterns {
            match pattern {
                SelfDefendingPattern::AntiDebug => {
                    result = result.replace("debugger;", "");
                    result = result.replace("debugger", "");
                }
                SelfDefendingPattern::ConsoleProtection => {
                    result = CONSOLE_PATTERN.replace_all(&result, "").to_string();
                }
                _ => {}
            }
        }
        Ok(result)
    }

    fn simplify_member_expressions(&self, code: &str) -> Result<String> {
        let mut result = code.to_string();
        let bracket_pattern = Regex::new(r#"(\w+)\["(\w+)"\]"#)?;
        result = bracket_pattern.replace_all(&result, "$1.$2").to_string();
        Ok(result)
    }

    fn calculate_readability_improvement(&self, original: &str, deobfuscated: &str) -> f32 {
        let original_len = original.len();
        let deobfuscated_len = deobfuscated.len();
        if original_len == 0 {
            0.0
        } else {
            ((original_len - deobfuscated_len) as f32 / original_len as f32).min(1.0)
        }
    }
}

impl Default for EnhancedDeobfuscator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DeobfuscationStats {
    fn default() -> Self {
        Self {
            string_arrays_unpacked: 0,
            proxy_functions_removed: 0,
            control_flow_simplified: 0,
            constants_folded: 0,
            self_defending_removed: 0,
            string_decoders_evaluated: 0,
            size_reduction: 0,
            readability_improvement: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_array_detection() {
        let deob = EnhancedDeobfuscator::new();
        let code = r#"var _0xabcd = ['hello', 'world', 'test', 'foo', 'bar'];"#;
        let array = deob.detect_string_array(code);
        assert!(array.is_some());
        let array = array.unwrap();
        assert_eq!(array.name, "_0xabcd");
        assert_eq!(array.contents.len(), 5);
    }

    #[test]
    fn test_proxy_function_detection() {
        let deob = EnhancedDeobfuscator::new();
        let code = r#"function proxy(a, b) { return realFunc(a, b); }"#;
        let proxies = deob.detect_proxy_functions(code);
        assert!(!proxies.is_empty());
    }

    #[test]
    fn test_self_defending_detection() {
        let deob = EnhancedDeobfuscator::new();
        let code = r#"debugger; console.log = function() {};"#;
        let patterns = deob.detect_self_defending(code);
        assert!(patterns.contains(&SelfDefendingPattern::AntiDebug));
        assert!(patterns.contains(&SelfDefendingPattern::ConsoleProtection));
    }

    #[test]
    fn test_comprehensive_deobfuscation() {
        let mut deob = EnhancedDeobfuscator::new();
        let code = r#"var _0xabc = ['hello', 'world']; function proxy(x) { return console.log(x); } debugger; if(true){console.log('hi');}"#;
        let result = deob.deobfuscate(code).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_member_expression_simplification() {
        let deob = EnhancedDeobfuscator::new();
        let code = r#"obj["property"]"#;
        let result = deob.simplify_member_expressions(code).unwrap();
        assert!(result.contains("obj.property"));
    }
}
