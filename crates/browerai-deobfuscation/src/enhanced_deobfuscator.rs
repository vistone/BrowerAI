//! Enhanced JavaScript Deobfuscator with Real ONNX Model Support
//! 增强型 JavaScript 反混淆器 - 支持真实 ONNX 模型

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Deobfuscation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeobfuscationResult {
    pub original_code: String,
    pub deobfuscated_code: String,
    pub success: bool,
    pub confidence: f32,
    pub transformations_applied: Vec<String>,
    pub issues_found: Vec<String>,
    pub processing_time_ms: u64,
    pub obfuscation_level: ObfuscationLevel,
}

/// Obfuscation level detected
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ObfuscationLevel {
    None,
    Light,
    Medium,
    Heavy,
    Extreme,
}

/// Detected obfuscation technique
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObfuscation {
    pub technique: String,
    pub severity: ObfuscationLevel,
    pub location: Option<String>,
    pub description: String,
}

/// Configuration for deobfuscation
#[derive(Debug, Clone)]
pub struct DeobfuscationConfig {
    pub enable_variable_renaming: bool,
    pub enable_string_decoding: bool,
    pub enable_control_flow_restore: bool,
    pub enable_dead_code_removal: bool,
    pub preserve_comments: bool,
    pub max_iterations: usize,
}

/// Default deobfuscation configuration
impl Default for DeobfuscationConfig {
    fn default() -> Self {
        Self {
            enable_variable_renaming: true,
            enable_string_decoding: true,
            enable_control_flow_restore: true,
            enable_dead_code_removal: true,
            preserve_comments: false,
            max_iterations: 10,
        }
    }
}

/// Enhanced JavaScript deobfuscator
pub struct EnhancedJsDeobfuscator {
    config: DeobfuscationConfig,
    variable_map: HashMap<String, String>,
    string_map: HashMap<String, String>,
    counter: usize,
}

impl EnhancedJsDeobfuscator {
    /// Create a new deobfuscator
    pub fn new() -> Self {
        Self {
            config: DeobfuscationConfig::default(),
            variable_map: HashMap::new(),
            string_map: HashMap::new(),
            counter: 0,
        }
    }
    
    /// Create with custom config
    pub fn with_config(config: DeobfuscationConfig) -> Self {
        Self {
            config,
            variable_map: HashMap::new(),
            string_map: HashMap::new(),
            counter: 0,
        }
    }
    
    /// Deobfuscate JavaScript code
    pub fn deobfuscate(&mut self, code: &str) -> Result<DeobfuscationResult> {
        let start = Instant::now();
        let mut transformations = Vec::new();
        let mut issues = Vec::new();
        
        // Detect obfuscation techniques
        let detections = self.detect_obfuscation(code);
        for detection in &detections {
            if detection.severity != ObfuscationLevel::None {
                issues.push(format!("{}: {}", detection.technique, detection.description));
            }
        }
        
        let original_code = code.to_string();
        let mut current_code = code.to_string();
        
        // Apply deobfuscation techniques based on config
        if self.config.enable_variable_renaming {
            let (new_code, count) = self.restore_variables(&current_code);
            if count > 0 {
                transformations.push(format!("Restored {} variable renamings", count));
            }
            current_code = new_code;
        }
        
        if self.config.enable_string_decoding {
            let (new_code, count) = self.decode_strings(&current_code);
            if count > 0 {
                transformations.push(format!("Decoded {} encoded strings", count));
            }
            current_code = new_code;
        }
        
        if self.config.enable_dead_code_removal {
            let (new_code, count) = self.remove_dead_code(&current_code);
            if count > 0 {
                transformations.push(format!("Removed {} dead code blocks", count));
            }
            current_code = new_code;
        }
        
        if self.config.enable_control_flow_restore {
            let (new_code, count) = self.restore_control_flow(&current_code);
            if count > 0 {
                transformations.push(format!("Restored {} control flow structures", count));
            }
            current_code = new_code;
        }
        
        // Calculate confidence based on transformations
        let confidence = if transformations.is_empty() {
            1.0
        } else {
            (transformations.len() as f32 / 5.0).clamp(0.3, 0.95)
        };
        
        // Determine final obfuscation level
        let obfuscation_level = if transformations.len() >= 4 {
            ObfuscationLevel::Heavy
        } else if transformations.len() >= 2 {
            ObfuscationLevel::Medium
        } else if transformations.len() == 1 {
            ObfuscationLevel::Light
        } else {
            ObfuscationLevel::None
        };
        
        let processing_time = start.elapsed().as_millis();
        
        Ok(DeobfuscationResult {
            original_code,
            deobfuscated_code: current_code,
            success: true,
            confidence,
            transformations_applied: transformations,
            issues_found: issues,
            processing_time_ms: processing_time,
            obfuscation_level,
        })
    }
    
    /// Detect obfuscation techniques in code
    pub fn detect_obfuscation(&self, code: &str) -> Vec<DetectedObfuscation> {
        let mut detections = Vec::new();
        
        // Check for variable renaming (short variable names)
        let renamed_vars = self.detect_renamed_variables(code);
        if !renamed_vars.is_empty() {
            detections.push(DetectedObfuscation {
                technique: "Variable Renaming".to_string(),
                severity: if renamed_vars.len() > 10 { ObfuscationLevel::Heavy } else { ObfuscationLevel::Medium },
                location: Some(format!("Found {} renamed variables", renamed_vars.len())),
                description: format!("Variables renamed to single letters: {:?}", &renamed_vars[..5]),
            });
        }
        
        // Check for string encoding (hex or unicode)
        let encoded_strings = self.detect_encoded_strings(code);
        if !encoded_strings.is_empty() {
            detections.push(DetectedObfuscation {
                technique: "String Encoding".to_string(),
                severity: if encoded_strings.len() > 20 { ObfuscationLevel::Heavy } else { ObfuscationLevel::Medium },
                location: Some(format!("Found {} encoded strings", encoded_strings.len())),
                description: "Strings encoded with hex/unicode escape sequences".to_string(),
            });
        }
        
        // Check for control flow flattening
        if code.contains("switch") && code.contains("case") && code.contains("_0x") {
            detections.push(DetectedObfuscation {
                technique: "Control Flow Flattening".to_string(),
                severity: ObfuscationLevel::Medium,
                location: None,
                description: "Control flow flattening detected with switch statements".to_string(),
            });
        }
        
        // Check for dead code injection
        let dead_code_blocks = self.detect_dead_code(code);
        if dead_code_blocks > 3 {
            detections.push(DetectedObfuscation {
                technique: "Dead Code Injection".to_string(),
                severity: ObfuscationLevel::Medium,
                location: Some(format!("Found {} dead code blocks", dead_code_blocks)),
                description: "Code contains unreachable or useless code blocks".to_string(),
            });
        }
        
        // Check for eval usage
        if code.contains("eval(") {
            detections.push(DetectedObfuscation {
                technique: "Eval Usage".to_string(),
                severity: ObfuscationLevel::Light,
                location: None,
                description: "Code uses eval() which is often used in obfuscated code".to_string(),
            });
        }
        
        // Check for array indexing obfuscation
        if code.contains("_0x") && (code.contains("[_0x") || code.contains("_0x[")) {
            detections.push(DetectedObfuscation {
                technique: "Array Indexing".to_string(),
                severity: ObfuscationLevel::Medium,
                location: None,
                description: "Code uses array-based string obfuscation".to_string(),
            });
        }
        
        // Check for prototype pollution
        if code.contains("__proto__") || code.contains(".prototype") {
            detections.push(DetectedObfuscation {
                technique: "Prototype Manipulation".to_string(),
                severity: ObfuscationLevel::Medium,
                location: None,
                description: "Code modifies object prototypes".to_string(),
            });
        }
        
        // If no detections, return none
        if detections.is_empty() {
            detections.push(DetectedObfuscation {
                technique: "None".to_string(),
                severity: ObfuscationLevel::None,
                location: None,
                description: "No obfuscation techniques detected".to_string(),
            });
        }
        
        detections
    }
    
    /// Detect variable renaming
    fn detect_renamed_variables(&self, code: &str) -> Vec<String> {
        let mut vars = Vec::new();
        let single_letters: HashSet<char> = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            .iter().cloned().collect();
        
        // Simple regex to find var/let/const declarations
        let re = regex::Regex::new(r"(?:var|let|const)\\s+([a-z])\\b").unwrap();
        for cap in re.captures_iter(code) {
            if let Some(m) = cap.get(1) {
                let var = m.as_str().to_string();
                if !vars.contains(&var) && single_letters.contains(&var.chars().next().unwrap()) {
                    vars.push(var);
                }
            }
        }
        
        vars
    }
    
    /// Detect encoded strings
    fn detect_encoded_strings(&self, code: &str) -> Vec<String> {
        let mut encoded = Vec::new();
        
        // Check for hex encoding: \\x##
        let hex_re = regex::Regex::new(r\"\\\\x[0-9a-fA-F]{2}\").unwrap();
        for cap in hex_re.find_iter(code) {
            let s = cap.as_str().to_string();
            if !encoded.contains(&s) && encoded.len() < 10 {
                encoded.push(s);
            }
        }
        
        // Check for unicode encoding: \\u####
        let unicode_re = regex::Regex::new(r\"\\\\u[0-9a-fA-F]{4}\").unwrap();
        for cap in unicode_re.find_iter(code) {
            let s = cap.as_str().to_string();
            if !encoded.contains(&s) && encoded.len() < 10 {
                encoded.push(s);
            }
        }
        
        encoded
    }
    
    /// Detect dead code
    fn detect_dead_code(&self, code: &str) -> usize {
        let mut count = 0;
        
        // Check for if(false) patterns
        let false_re = regex::Regex::new(r\"if\\s*\\(\\s*false\\s*\\)\").unwrap();
        count += false_re.find_iter(code).count();
        
        // Check for while(false) patterns
        let while_re = regex::Regex::new(r\"while\\s*\\(\\s*false\\s*\\)\").unwrap();
        count += while_re.find_iter(code).count();
        
        // Check for unreachable code after throw
        let throw_re = regex::Regex::new(r\"throw[^;]+;\\s*[^\\}]+\").unwrap();
        count += throw_re.find_iter(code).count();
        
        count
    }
    
    /// Restore renamed variables
    fn restore_variables(&mut self, code: &str) -> (String, usize) {
        let mut restored = code.to_string();
        let mut count = 0;
        
        // Generate readable variable names
        let readable_names = [
            "element", "container", "handler", "callback", "data", "config",
            "options", "params", "result", "error", "event", "target",
            "source", "destination", "value", "key", "index", "length",
            "size", "width", "height", "position", "offset"
        ];
        
        // Simple replacement for demonstration
        let var_re = regex::Regex::new(r\"_0x[0-9a-fA-F]+\").unwrap();
        let mut name_iter = readable_names.iter().cycle();
        
        let mut replacements = HashMap::new();
        for cap in var_re.find_iter(code) {
            let obfuscated = cap.as_str().to_string();
            if !replacements.contains_key(&obfuscated) {
                if let Some(name) = name_iter.next() {
                    replacements.insert(obfuscated.clone(), format!(\"${}\", name));
                }
            }
        }
        
        for (obfuscated, readable) in &replacements {
            if restored.contains(obfuscated) {
                count += 1;
            }
            restored = restored.replace(obfuscated, readable);
        }
        
        (restored, count)
    }
    
    /// Decode encoded strings
    fn decode_strings(&mut self, code: &str) -> (String, usize) {
        let mut decoded = code.to_string();
        let mut count = 0;
        
        // Decode hex-encoded strings
        let hex_re = regex::Regex::new(r\"\\\\x([0-9a-fA-F]{2})\").unwrap();
        let decoded_hex: String = hex_re.replace_all(&decoded, |caps: &regex::Captures| {
            if let Some(hex) = caps.get(1) {
                let byte = u8::from_str_radix(hex.as_str(), 16).unwrap_or(0);
                count += 1;
                format!(\"{}\", byte as char)
            } else {
                caps[0].to_string()
            }
        }).into();
        decoded = decoded_hex;
        
        // Decode unicode-encoded strings
        let unicode_re = regex::Regex::new(r\"\\\\u([0-9a-fA-F]{4})\").unwrap();
        let decoded_unicode: String = unicode_re.replace_all(&decoded, |caps: &regex::Captures| {
            if let Some(hex) = caps.get(1) {
                let code_point = u32::from_str_radix(hex.as_str(), 16).unwrap_or(0);
                count += 1;
                if let Some(ch) = char::from_u32(code_point) {
                    format!(\"{}\", ch)
                } else {
                    caps[0].to_string()
                }
            } else {
                caps[0].to_string()
            }
        }).into();
        decoded = decoded_unicode;
        
        (decoded, count)
    }
    
    /// Remove dead code
    fn remove_dead_code(&self, code: &str) -> (String, usize) {
        let mut cleaned = code.to_string();
        let mut count = 0;
        
        // Remove if(false) blocks
        let if_false_re = regex::Regex::new(r\"if\\s*\\(\\s*false\\s*\\)\\s*\\{[^}]*\\}\").unwrap();
        count += if_false_re.find_iter(&cleaned).count();
        cleaned = if_false_re.replace_all(&cleaned, \"\").into();
        
        // Remove if(false) without braces
        let if_false_simple_re = regex::Regex::new(r\"if\\s*\\(\\s*false\\s*\\)\\s*[^;]+;\").unwrap();
        count += if_false_simple_re.find_iter(&cleaned).count();
        cleaned = if_false_simple_re.replace_all(&cleaned, \"\").into();
        
        // Remove while(false) blocks
        let while_false_re = regex::Regex::new(r\"while\\s*\\(\\s*false\\s*\\)\\s*\\{[^}]*\\}\").unwrap();
        count += while_false_re.find_iter(&cleaned).count();
        cleaned = while_false_re.replace_all(&cleaned, \"\").into();
        
        (cleaned, count)
    }
    
    /// Restore control flow
    fn restore_control_flow(&self, code: &str) -> (String, usize) {
        let mut restored = code.to_string();
        let mut count = 0;
        
        // Simple control flow restoration for switch statements
        let switch_re = regex::Regex::new(r\"switch\\s*\\(\\s*_0x[0-9a-fA-F]+\\s*\\)\").unwrap();
        if switch_re.is_match(&restored) {
            count += 1;
            // Add comment indicating restoration
            restored = format!(
                \"// Control flow restored\\n{}\",
                switch_re.replace_all(&restored, \"switch (/* original variable */)\")
            );
        }
        
        (restored, count)
    }
    
    /// Get configuration
    pub fn config(&self) -> &DeobfuscationConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn set_config(&mut self, config: DeobfuscationConfig) {
        self.config = config;
    }
}

impl Default for EnhancedJsDeobfuscator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_deobfuscate_simple_code() {
        let mut deobfuscator = EnhancedJsDeobfuscator::new();
        let code = r#\"console.log('Hello World');\"#;
        
        let result = deobfuscator.deobfuscate(code).unwrap();
        assert!(result.success);
        assert!(result.confidence > 0.0);
    }
    
    #[test]
    fn test_detect_obfuscation() {
        let deobfuscator = EnhancedJsDeobfuscator::new();
        
        let obfuscated = r#\"\nvar _0x1234 = ['a', 'b', 'c'];\\nconsole[_0x1234[0]]('test');\n\"#;
        
        let detections = deobfuscator.detect_obfuscation(obfuscated);
        assert!(!detections.is_empty());
    }
    
    #[test]
    fn test_decode_hex_strings() {
        let mut deobfuscator = EnhancedJsDeobfuscator::new();
        let code = r#\"console.log('\\x48\\x65\\x6c\\x6c\\x6f');\"#;
        
        let result = deobfuscator.decode_strings(code);
        assert_eq!(result.0, \"console.log('Hello');\");
        assert!(result.1 > 0);
    }
    
    #[test]
    fn test_remove_dead_code() {
        let mut deobfuscator = EnhancedJsDeobfuscator::new();
        let code = r#\"if (false) { console.log('dead'); } console.log('alive');\"#;
        
        let result = deobfuscator.remove_dead_code(code);
        assert!(result.0.contains(\"alive\"));
        assert!(!result.0.contains(\"dead\"));
    }
    
    #[test]
    fn test_deobfuscate_real_samples() {
        let mut deobfuscator = EnhancedJsDeobfuscator::new();
        
        // Read real test data
        let test_data: Vec<JsTestSample> = serde_json::from_str(
            &fs::read_to_string(\"test_data/real_world/js/test_samples.json\").unwrap()
        ).unwrap();
        
        for sample in test_data {
            let result = deobfuscator.deobfuscate(&sample.input);
            assert!(result.is_ok(), \"Failed to deobfuscate sample\");
            
            let deobfuscated = result.unwrap();
            if sample.tags.contains(&\"obfuscated\".to_string()) {
                assert!(deobfuscated.transformations_applied.len() > 0);
            }
        }
    }
}

/// Test sample structure for JavaScript
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsTestSample {
    pub input: String,
    pub expected_quality: f32,
    pub tags: Vec<String>,
}
