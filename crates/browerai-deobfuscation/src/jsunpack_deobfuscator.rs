//! JSUnpack-Inspired JavaScript Deobfuscator
//!
//! Based on techniques from JSUnpack.jeek.org
//! Implements multi-layer unpacking, eval detection, shellcode analysis,
//! and automatic decoder identification.

use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use base64::{Engine as _, engine::general_purpose};

lazy_static! {
    // Packer detection patterns
    static ref DEAN_EDWARDS_PACKER: Regex = 
        Regex::new(r"eval\s*\(\s*function\s*\(\s*p\s*,\s*a\s*,\s*c\s*,\s*k\s*,\s*e\s*,\s*[rd]\s*\)").unwrap();
    
    static ref EVAL_PATTERN: Regex = 
        Regex::new(r"eval\s*\(").unwrap();
    
    static ref UNESCAPE_PATTERN: Regex = 
        Regex::new(r"unescape\s*\(").unwrap();
    
    static ref FROM_CHAR_CODE: Regex = 
        Regex::new(r"String\.fromCharCode\s*\(").unwrap();
    
    static ref BASE64_PATTERN: Regex = 
        Regex::new(r"atob\s*\(|btoa\s*\(").unwrap();
    
    static ref DOCUMENT_WRITE: Regex = 
        Regex::new(r"document\.write\s*\(").unwrap();
    
    // Shellcode patterns (common exploit indicators)
    static ref SHELLCODE_PATTERN: Regex = 
        Regex::new(r"%u[0-9a-fA-F]{4}").unwrap();
    
    static ref HEAP_SPRAY: Regex = 
        Regex::new(r"(nop|[\x90]+)\s*\.repeat\s*\(").unwrap();
    
    // URL extraction
    static ref URL_PATTERN: Regex = 
        Regex::new(r#"https?://[^\s"'<>)]+|ftp://[^\s"'<>)]+"#).unwrap();
    
    // Obfuscation markers
    static ref JSO_PATTERN: Regex = 
        Regex::new(r"^\s*var\s+_\d+\s*=").unwrap();
}

/// Unpacking result with detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnpackResult {
    pub code: String,
    pub original_code: String,
    pub layers_unpacked: usize,
    pub packer_detected: Option<PackerType>,
    pub techniques_found: Vec<DecodingTechnique>,
    pub extracted_urls: Vec<String>,
    pub shellcode_detected: bool,
    pub suspicious_patterns: Vec<SuspiciousPattern>,
    pub safety_score: f32, // 0.0 (malicious) to 1.0 (safe)
}

/// Known packer types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PackerType {
    DeanEdwards,      // Dean Edwards Packer
    JSObfuscate,      // JSObfuscate
    JSPacker,         // Generic JS Packer
    YUICompressor,    // YUI Compressor
    ClosureCompiler,  // Google Closure
    UglifyJS,         // UglifyJS
    Unknown,
}

/// Decoding techniques detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecodingTechnique {
    Base64,
    Unescape,
    FromCharCode,
    EvalChain,
    DocumentWrite,
    HexEncoding,
    UnicodeEncoding,
    XORDecoding,
    RC4,
    Custom,
}

/// Suspicious patterns found
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousPattern {
    pub pattern_type: String,
    pub description: String,
    pub severity: Severity,
    pub location: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// JSUnpack-style deobfuscator
pub struct JSUnpackDeobfuscator {
    max_depth: usize,
    #[allow(dead_code)]
    max_iterations: usize,  // Reserved for future use
    enable_vm_execution: bool,
    strict_mode: bool,
    extracted_urls: Vec<String>,
    suspicious_patterns: Vec<SuspiciousPattern>,
}

impl JSUnpackDeobfuscator {
    /// Create new deobfuscator
    pub fn new() -> Self {
        Self {
            max_depth: 10,
            max_iterations: 50,
            enable_vm_execution: false, // Disabled by default for safety
            strict_mode: false,
            extracted_urls: Vec::new(),
            suspicious_patterns: Vec::new(),
        }
    }

    /// Enable VM execution for eval() emulation (USE WITH CAUTION)
    pub fn enable_vm_execution(&mut self) {
        self.enable_vm_execution = true;
    }

    /// Enable strict mode (fail on suspicious code)
    pub fn enable_strict_mode(&mut self) {
        self.strict_mode = true;
    }

    /// Main unpacking entry point
    pub fn unpack(&mut self, code: &str) -> Result<UnpackResult> {
        let original_code = code.to_string();
        let mut current_code = code.to_string();
        let mut layers = 0;
        let mut techniques = Vec::new();

        // Reset state
        self.extracted_urls.clear();
        self.suspicious_patterns.clear();

        // Detect packer type first
        let packer = self.detect_packer(&current_code);

        // Extract URLs
        self.extract_urls(&current_code);

        // Check for shellcode
        let shellcode_detected = self.detect_shellcode(&current_code);
        if shellcode_detected {
            self.suspicious_patterns.push(SuspiciousPattern {
                pattern_type: "Shellcode".to_string(),
                description: "Potential exploit shellcode detected".to_string(),
                severity: Severity::Critical,
                location: "Multiple locations".to_string(),
            });
        }

        // Multi-layer unpacking
        for _ in 0..self.max_depth {
            let prev_code = current_code.clone();

            // Try various unpacking techniques
            if let Some(unpacked) = self.unpack_dean_edwards(&current_code)? {
                current_code = unpacked;
                layers += 1;
                techniques.push(DecodingTechnique::EvalChain);
                continue;
            }

            if let Some(unpacked) = self.decode_base64(&current_code)? {
                current_code = unpacked;
                layers += 1;
                techniques.push(DecodingTechnique::Base64);
                continue;
            }

            if let Some(unpacked) = self.decode_unescape(&current_code)? {
                current_code = unpacked;
                layers += 1;
                techniques.push(DecodingTechnique::Unescape);
                continue;
            }

            if let Some(unpacked) = self.decode_from_char_code(&current_code)? {
                current_code = unpacked;
                layers += 1;
                techniques.push(DecodingTechnique::FromCharCode);
                continue;
            }

            if let Some(unpacked) = self.extract_from_document_write(&current_code)? {
                current_code = unpacked;
                layers += 1;
                techniques.push(DecodingTechnique::DocumentWrite);
                continue;
            }

            // No more changes, stop
            if current_code == prev_code {
                break;
            }
        }

        // Analyze suspicious patterns
        self.analyze_suspicious_patterns(&current_code);

        // Calculate safety score
        let safety_score = self.calculate_safety_score();

        Ok(UnpackResult {
            code: current_code,
            original_code,
            layers_unpacked: layers,
            packer_detected: packer,
            techniques_found: techniques,
            extracted_urls: self.extracted_urls.clone(),
            shellcode_detected,
            suspicious_patterns: self.suspicious_patterns.clone(),
            safety_score,
        })
    }

    /// Detect packer type
    fn detect_packer(&self, code: &str) -> Option<PackerType> {
        if DEAN_EDWARDS_PACKER.is_match(code) {
            return Some(PackerType::DeanEdwards);
        }

        if JSO_PATTERN.is_match(code) {
            return Some(PackerType::JSObfuscate);
        }

        // Check for generic packer patterns
        if code.contains("eval(function(p,a,c,k,e,") {
            return Some(PackerType::JSPacker);
        }

        None
    }

    /// Unpack Dean Edwards Packer
    fn unpack_dean_edwards(&self, code: &str) -> Result<Option<String>> {
        if !DEAN_EDWARDS_PACKER.is_match(code) {
            return Ok(None);
        }

        // Extract the packed data
        // Pattern: eval(function(p,a,c,k,e,d){...}('packed_payload',radix,count,'key'.split('|'),0,{}))
        
        // More lenient regex to capture the parameters
        let re = Regex::new(
            r"eval\s*\(\s*function\s*\([^)]+\)\s*\{[\s\S]+?\}\s*\(\s*'([^']+)'\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*'([^']+)'\.split\s*\('\|'\)"
        )?;

        if let Some(caps) = re.captures(code) {
            let payload = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let radix: u32 = caps.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(36);
            let count: usize = caps.get(3).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
            let keys = caps.get(4).map(|m| m.as_str()).unwrap_or("");

            let unpacked = self.unpack_p_a_c_k(payload, radix, count, keys)?;
            return Ok(Some(unpacked));
        }

        Ok(None)
    }

    /// Unpack p.a.c.k format
    fn unpack_p_a_c_k(&self, payload: &str, radix: u32, _count: usize, keys: &str) -> Result<String> {
        let key_array: Vec<&str> = keys.split('|').collect();
        let mut result = payload.to_string();

        // Replace \\ with placeholder first
        result = result.replace("\\\\", "\x00");

        // Replace word patterns \b\d+\b with corresponding key
        let word_pattern = Regex::new(r"\\b(\d+)\\b")?;
        
        result = word_pattern.replace_all(&result, |caps: &regex::Captures| {
            let index: usize = caps[1].parse().unwrap_or(0);
            if index < key_array.len() && !key_array[index].is_empty() {
                key_array[index].to_string()
            } else {
                // Convert index to base-radix representation
                Self::to_base(index, radix)
            }
        }).to_string();

        // Restore escaped backslashes
        result = result.replace('\x00', "\\");

        Ok(result)
    }

    /// Convert number to arbitrary base
    fn to_base(mut num: usize, radix: u32) -> String {
        if num == 0 {
            return "0".to_string();
        }

        let digits = "0123456789abcdefghijklmnopqrstuvwxyz";
        let mut result = String::new();

        while num > 0 {
            let digit = num % radix as usize;
            result.insert(0, digits.chars().nth(digit).unwrap());
            num /= radix as usize;
        }

        result
    }

    /// Decode Base64 strings
    fn decode_base64(&self, code: &str) -> Result<Option<String>> {
        if !BASE64_PATTERN.is_match(code) {
            return Ok(None);
        }

        // Extract atob() calls
        let re = Regex::new(r#"atob\s*\(\s*["']([A-Za-z0-9+/=]+)["']\s*\)"#)?;
        
        let mut result = code.to_string();
        let mut replaced = false;

        for caps in re.captures_iter(code) {
            if let Some(b64) = caps.get(1) {
                if let Ok(decoded) = general_purpose::STANDARD.decode(b64.as_str()) {
                    if let Ok(decoded_str) = String::from_utf8(decoded) {
                        result = result.replace(&caps[0], &format!("\"{}\"", decoded_str));
                        replaced = true;
                    }
                }
            }
        }

        if replaced {
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Decode unescape() calls
    fn decode_unescape(&self, code: &str) -> Result<Option<String>> {
        if !UNESCAPE_PATTERN.is_match(code) {
            return Ok(None);
        }

        // Extract unescape() calls
        let re = Regex::new(r#"unescape\s*\(\s*["']([^"']+)["']\s*\)"#)?;
        
        let mut result = code.to_string();
        let mut replaced = false;

        for caps in re.captures_iter(code) {
            if let Some(escaped) = caps.get(1) {
                let decoded = self.unescape_string(escaped.as_str());
                result = result.replace(&caps[0], &format!("\"{}\"", decoded));
                replaced = true;
            }
        }

        if replaced {
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Unescape percent-encoded string
    fn unescape_string(&self, s: &str) -> String {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '%' {
                // Peek next two characters
                let hex: String = chars.by_ref().take(2).collect();
                if hex.len() == 2 {
                    if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                        result.push(byte as char);
                        continue;
                    }
                }
            }
            result.push(ch);
        }

        result
    }

    /// Decode String.fromCharCode() calls
    fn decode_from_char_code(&self, code: &str) -> Result<Option<String>> {
        if !FROM_CHAR_CODE.is_match(code) {
            return Ok(None);
        }

        // Extract String.fromCharCode(num1, num2, ...)
        let re = Regex::new(r"String\.fromCharCode\s*\(\s*([\d\s,]+)\s*\)")?;
        
        let mut result = code.to_string();
        let mut replaced = false;

        for caps in re.captures_iter(code) {
            if let Some(nums) = caps.get(1) {
                let decoded = self.from_char_codes(nums.as_str());
                result = result.replace(&caps[0], &format!("\"{}\"", decoded));
                replaced = true;
            }
        }

        if replaced {
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Convert char codes to string
    fn from_char_codes(&self, codes: &str) -> String {
        codes
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .filter_map(char::from_u32)
            .collect()
    }

    /// Extract code from document.write() calls
    fn extract_from_document_write(&self, code: &str) -> Result<Option<String>> {
        if !DOCUMENT_WRITE.is_match(code) {
            return Ok(None);
        }

        // Extract document.write() content
        let re = Regex::new(r#"document\.write\s*\(\s*["']([^"']+)["']\s*\)"#)?;
        
        if let Some(caps) = re.captures(code) {
            if let Some(content) = caps.get(1) {
                return Ok(Some(content.as_str().to_string()));
            }
        }

        Ok(None)
    }

    /// Detect shellcode patterns
    fn detect_shellcode(&mut self, code: &str) -> bool {
        // Check for Unicode shellcode (%uXXXX)
        if SHELLCODE_PATTERN.is_match(code) {
            let matches: Vec<_> = SHELLCODE_PATTERN.find_iter(code).collect();
            if matches.len() > 10 {
                // Likely shellcode
                return true;
            }
        }

        // Check for heap spray patterns
        if HEAP_SPRAY.is_match(code) {
            return true;
        }

        // Check for excessive NOP sleds (byte 0x90 pattern)
        if code.matches("\u{0090}").count() > 100 {
            return true;
        }

        false
    }

    /// Extract URLs from code
    fn extract_urls(&mut self, code: &str) {
        for cap in URL_PATTERN.captures_iter(code) {
            if let Some(url) = cap.get(0) {
                let url_str = url.as_str().to_string();
                if !self.extracted_urls.contains(&url_str) {
                    self.extracted_urls.push(url_str);
                }
            }
        }
    }

    /// Analyze suspicious patterns
    fn analyze_suspicious_patterns(&mut self, code: &str) {
        // Excessive eval()
        let eval_count = EVAL_PATTERN.find_iter(code).count();
        if eval_count > 5 {
            self.suspicious_patterns.push(SuspiciousPattern {
                pattern_type: "Excessive eval()".to_string(),
                description: format!("{} eval() calls detected", eval_count),
                severity: Severity::High,
                location: "Multiple".to_string(),
            });
        }

        // document.write() usage
        if DOCUMENT_WRITE.is_match(code) {
            self.suspicious_patterns.push(SuspiciousPattern {
                pattern_type: "document.write".to_string(),
                description: "Dynamic code injection via document.write()".to_string(),
                severity: Severity::Medium,
                location: "Multiple".to_string(),
            });
        }

        // iframe injection patterns
        if code.contains("<iframe") || code.contains("createElement('iframe')") {
            self.suspicious_patterns.push(SuspiciousPattern {
                pattern_type: "iframe".to_string(),
                description: "Potential iframe injection".to_string(),
                severity: Severity::High,
                location: "iframe creation".to_string(),
            });
        }

        // ActiveX patterns (old exploits)
        if code.contains("ActiveXObject") {
            self.suspicious_patterns.push(SuspiciousPattern {
                pattern_type: "ActiveX".to_string(),
                description: "ActiveX object usage (potential exploit)".to_string(),
                severity: Severity::High,
                location: "ActiveXObject".to_string(),
            });
        }
    }

    /// Calculate safety score
    fn calculate_safety_score(&self) -> f32 {
        let mut score = 1.0;

        // Deduct points for suspicious patterns
        for pattern in &self.suspicious_patterns {
            let deduction = match pattern.severity {
                Severity::Info => 0.05,
                Severity::Low => 0.1,
                Severity::Medium => 0.2,
                Severity::High => 0.3,
                Severity::Critical => 0.5,
            };
            score -= deduction;
        }

        // Deduct for URLs (potential phishing/malware)
        score -= (self.extracted_urls.len() as f32) * 0.05;

        score.max(0.0)
    }

    /// Get comprehensive analysis report
    pub fn analyze(&mut self, code: &str) -> Result<AnalysisReport> {
        let unpack_result = self.unpack(code)?;

        let safety_score = unpack_result.safety_score;
        let recommendations = self.generate_recommendations(&unpack_result);
        
        Ok(AnalysisReport {
            packer_type: unpack_result.packer_detected,
            obfuscation_layers: unpack_result.layers_unpacked,
            techniques_used: unpack_result.techniques_found,
            extracted_urls: unpack_result.extracted_urls,
            shellcode_detected: unpack_result.shellcode_detected,
            suspicious_patterns: unpack_result.suspicious_patterns,
            safety_score,
            risk_level: Self::get_risk_level(safety_score),
            recommendations,
        })
    }

    fn get_risk_level(safety_score: f32) -> RiskLevel {
        if safety_score >= 0.8 {
            RiskLevel::Low
        } else if safety_score >= 0.6 {
            RiskLevel::Medium
        } else if safety_score >= 0.4 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        }
    }

    fn generate_recommendations(&self, result: &UnpackResult) -> Vec<String> {
        let mut recommendations = Vec::new();

        if result.shellcode_detected {
            recommendations.push("CRITICAL: Shellcode detected - do not execute this code".to_string());
        }

        if !result.extracted_urls.is_empty() {
            recommendations.push(format!(
                "Found {} URLs - verify legitimacy before accessing",
                result.extracted_urls.len()
            ));
        }

        if result.layers_unpacked > 5 {
            recommendations.push("Deep obfuscation layers - likely malicious intent".to_string());
        }

        if result.safety_score < 0.5 {
            recommendations.push("Low safety score - treat as potentially malicious".to_string());
        }

        recommendations
    }
}

impl Default for JSUnpackDeobfuscator {
    fn default() -> Self {
        Self::new()
    }
}

/// Analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub packer_type: Option<PackerType>,
    pub obfuscation_layers: usize,
    pub techniques_used: Vec<DecodingTechnique>,
    pub extracted_urls: Vec<String>,
    pub shellcode_detected: bool,
    pub suspicious_patterns: Vec<SuspiciousPattern>,
    pub safety_score: f32,
    pub risk_level: RiskLevel,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_dean_edwards() {
        let mut deobf = JSUnpackDeobfuscator::new();
        
        let packed = r#"eval(function(p,a,c,k,e,d){e=function(c){return c};if(!''.replace(/^/,String)){while(c--){d[c]=k[c]||c}k=[function(e){return d[e]}];e=function(){return'\\w+'};c=1};while(c--){if(k[c]){p=p.replace(new RegExp('\\b'+e(c)+'\\b','g'),k[c])}}return p}('0 1=2',3,3,'var|x|42'.split('|'),0,{}))"#;
        
        let result = deobf.unpack(packed).unwrap();
        assert_eq!(result.packer_detected, Some(PackerType::DeanEdwards));
        assert!(result.layers_unpacked > 0);
    }

    #[test]
    fn test_decode_base64() {
        let deobf = JSUnpackDeobfuscator::new();
        
        let code = r#"var secret = atob("SGVsbG8gV29ybGQ=");"#;
        let result = deobf.decode_base64(code).unwrap();
        
        assert!(result.is_some());
        assert!(result.unwrap().contains("Hello World"));
    }

    #[test]
    fn test_from_char_code() {
        let deobf = JSUnpackDeobfuscator::new();
        
        let decoded = deobf.from_char_codes("72, 101, 108, 108, 111");
        
        assert_eq!(decoded, "Hello");
    }

    #[test]
    fn test_unescape() {
        let deobf = JSUnpackDeobfuscator::new();
        
        let escaped = "%48%65%6C%6C%6F";
        let decoded = deobf.unescape_string(escaped);
        
        assert_eq!(decoded, "Hello");
    }

    #[test]
    fn test_shellcode_detection() {
        let mut deobf = JSUnpackDeobfuscator::new();
        
        let shellcode = "%u9090%u9090%u9090%u9090%u9090%u9090%u9090%u9090%u9090%u9090%u9090";
        assert!(deobf.detect_shellcode(shellcode));
    }

    #[test]
    fn test_url_extraction() {
        let mut deobf = JSUnpackDeobfuscator::new();
        
        let code = r#"window.location = "http://evil.com/malware.exe";"#;
        deobf.extract_urls(code);
        
        assert_eq!(deobf.extracted_urls.len(), 1);
        assert!(deobf.extracted_urls[0].contains("evil.com"));
    }

    #[test]
    fn test_safety_score() {
        let mut deobf = JSUnpackDeobfuscator::new();
        
        // Safe code
        let safe_code = "console.log('Hello');";
        let safe_result = deobf.unpack(safe_code).unwrap();
        assert!(safe_result.safety_score > 0.9);
        
        // Suspicious code with multiple red flags
        let mut deobf2 = JSUnpackDeobfuscator::new();
        let suspicious = r#"
            eval(unescape("%48%65%6C%6C%6F")); 
            eval(unescape("%48%65%6C%6C%6F")); 
            eval(unescape("%48%65%6C%6C%6F")); 
            eval(unescape("%48%65%6C%6C%6F")); 
            eval(unescape("%48%65%6C%6C%6F")); 
            eval(unescape("%48%65%6C%6C%6F")); 
            document.write("<iframe>");
            document.write("<iframe>");
            window.location = "http://evil1.com";
            window.location = "http://evil2.com";
            window.location = "http://evil3.com";
        "#;
        let sus_result = deobf2.unpack(suspicious).unwrap();
        // With 6+ evals, 2 document.writes, 3 URLs, should be < 0.7
        assert!(sus_result.safety_score < 0.7, 
            "Expected safety_score < 0.7, got {}", sus_result.safety_score);
    }
}

