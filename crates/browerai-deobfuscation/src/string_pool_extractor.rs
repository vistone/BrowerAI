use anyhow::Result;
use base64::{engine::general_purpose, Engine as _};
use regex::Regex;
/// 字符串池析取器 (String Pool Extractor)
///
/// 识别并提取所有字符串存储位置，支持多层编码解析，
/// 建立完整的字符串映射表。
use std::collections::HashMap;

/// 字符串来源类型
#[derive(Debug, Clone, PartialEq)]
pub enum StringSource {
    /// 字面量字符串
    Literal,
    /// 字符数组 (如 [65, 66, 67] = "ABC")
    CharArray,
    /// 编码字符串 (Base64, Hex, Unicode等)
    Encoded { encoding_type: EncodingType },
    /// 连接的字符串
    Concatenated,
    /// 模板字符串
    Template,
    /// 通过 String.fromCharCode() 生成
    FromCharCode,
    /// 通过 unescape() 解码
    Unescaped,
    /// 通过 atob() 解码
    AtobDecoded,
    /// 注入的字符串 (动态生成)
    Injected,
}

/// 编码类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodingType {
    Base64,
    Hex,
    Unicode,
    Octal,
    Custom,
}

/// 字符串项
#[derive(Debug, Clone)]
pub struct StringPoolEntry {
    /// 字符串的原始形式
    pub original: String,
    /// 解码后的值
    pub decoded: String,
    /// 来源类型
    pub source: StringSource,
    /// 出现的行号
    pub line_number: usize,
    /// 在代码中的位置
    pub column_start: usize,
    pub column_end: usize,
    /// 编码层数
    pub encoding_depth: usize,
    /// 置信度 (0.0 - 1.0)
    pub confidence: f32,
}

/// 字符串池
#[derive(Debug, Clone)]
pub struct StringPool {
    /// 所有提取的字符串
    pub entries: Vec<StringPoolEntry>,
    /// 字符串到其使用位置的映射
    pub string_usage_map: HashMap<String, Vec<usize>>,
    /// 解码的字符串统计
    pub decoding_stats: HashMap<EncodingType, usize>,
}

/// 字符串池分析器
pub struct StringPoolExtractor {
    pool: StringPool,
}

impl StringPoolExtractor {
    pub fn new() -> Self {
        Self {
            pool: StringPool {
                entries: Vec::new(),
                string_usage_map: HashMap::new(),
                decoding_stats: HashMap::new(),
            },
        }
    }

    /// 从代码中提取字符串池
    pub fn extract(&mut self, code: &str) -> Result<()> {
        for (line_num, line) in code.lines().enumerate() {
            self.extract_from_line(line, line_num);
        }

        // 构建使用映射
        self.build_usage_map();

        Ok(())
    }

    /// 从单行代码提取字符串
    fn extract_from_line(&mut self, line: &str, line_num: usize) {
        // 提取字面量字符串
        self.extract_literal_strings(line, line_num);

        // 提取字符数组
        self.extract_char_arrays(line, line_num);

        // 提取 Base64 编码的字符串
        self.extract_base64_strings(line, line_num);

        // 提取 Hex 编码的字符串
        self.extract_hex_strings(line, line_num);

        // 提取 Unicode 编码的字符串
        self.extract_unicode_strings(line, line_num);

        // 提取通过 String.fromCharCode() 生成的字符串
        self.extract_from_char_code(line, line_num);

        // 提取通过 unescape() 解码的字符串
        self.extract_unescape_strings(line, line_num);

        // 提取通过 atob() 解码的字符串
        self.extract_atob_strings(line, line_num);
    }

    /// 提取字面量字符串
    fn extract_literal_strings(&mut self, line: &str, line_num: usize) {
        // 匹配单引号或双引号的字符串
        let re = Regex::new(r#"['"]((?:\\.|[^'"])*?)['"]"#).unwrap();

        for caps in re.captures_iter(line) {
            if let Some(m) = caps.get(1) {
                let original = m.as_str();
                let decoded = self.unescape_string(original);

                self.add_entry(StringPoolEntry {
                    original: original.to_string(),
                    decoded,
                    source: StringSource::Literal,
                    line_number: line_num,
                    column_start: m.start(),
                    column_end: m.end(),
                    encoding_depth: 0,
                    confidence: 0.95,
                });
            }
        }
    }

    /// 提取字符数组 [65, 66, 67]
    fn extract_char_arrays(&mut self, line: &str, line_num: usize) {
        let re = Regex::new(r"\[(\d+(?:\s*,\s*\d+)*)\]").unwrap();

        for caps in re.captures_iter(line) {
            if let Some(m) = caps.get(1) {
                let numbers_str = m.as_str();

                // 检查是否都是有效的字符代码
                let numbers: Result<Vec<u8>, _> = numbers_str
                    .split(',')
                    .map(|s| s.trim().parse::<u8>())
                    .collect();

                if let Ok(num_vec) = numbers {
                    if num_vec.len() > 2 {
                        // 至少 3 个字符
                        if let Ok(decoded) = String::from_utf8(num_vec.to_vec()) {
                            self.add_entry(StringPoolEntry {
                                original: format!("[{}]", numbers_str),
                                decoded,
                                source: StringSource::CharArray,
                                line_number: line_num,
                                column_start: m.start(),
                                column_end: m.end(),
                                encoding_depth: 1,
                                confidence: 0.9,
                            });
                        }
                    }
                }
            }
        }
    }

    /// 提取 Base64 编码的字符串
    fn extract_base64_strings(&mut self, line: &str, line_num: usize) {
        // Base64 的特征：长的字母数字字符串，可能以 == 结尾
        let re = Regex::new(r#"['"]([A-Za-z0-9+/]{20,}={0,2})['"]"#).unwrap();

        for caps in re.captures_iter(line) {
            if let Some(m) = caps.get(1) {
                let b64_str = m.as_str();

                if let Ok(decoded_bytes) = general_purpose::STANDARD.decode(b64_str) {
                    if let Ok(decoded) = String::from_utf8(decoded_bytes) {
                        // 检查解码后是否是可打印字符
                        if decoded
                            .chars()
                            .all(|c| c.is_ascii() && (c.is_alphanumeric() || c.is_whitespace()))
                        {
                            self.add_entry(StringPoolEntry {
                                original: b64_str.to_string(),
                                decoded,
                                source: StringSource::Encoded {
                                    encoding_type: EncodingType::Base64,
                                },
                                line_number: line_num,
                                column_start: m.start(),
                                column_end: m.end(),
                                encoding_depth: 1,
                                confidence: 0.85,
                            });
                        }
                    }
                }
            }
        }
    }

    /// 提取 Hex 编码的字符串
    fn extract_hex_strings(&mut self, line: &str, line_num: usize) {
        // Hex 编码：\\x 或 0x 后跟两个十六进制数字
        let re = Regex::new(r"(?:\\x|0x)([0-9a-fA-F]{2})").unwrap();

        let mut hex_buffer = String::new();
        let mut last_match_end = 0;

        for caps in re.captures_iter(line) {
            if let Some(m) = caps.get(1) {
                let hex = m.as_str();
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    if (32..127).contains(&byte) {
                        // 可打印 ASCII
                        hex_buffer.push(byte as char);
                    } else if byte == 0 {
                        // 空字节，结束
                        if hex_buffer.len() > 2 {
                            self.add_entry(StringPoolEntry {
                                original: hex_buffer.clone(),
                                decoded: hex_buffer.clone(),
                                source: StringSource::Encoded {
                                    encoding_type: EncodingType::Hex,
                                },
                                line_number: line_num,
                                column_start: last_match_end,
                                column_end: m.end(),
                                encoding_depth: 1,
                                confidence: 0.85,
                            });
                        }
                        hex_buffer.clear();
                    }
                }
            }
            last_match_end = caps.get(0).unwrap().end();
        }

        if hex_buffer.len() > 2 {
            self.add_entry(StringPoolEntry {
                original: hex_buffer.clone(),
                decoded: hex_buffer.clone(),
                source: StringSource::Encoded {
                    encoding_type: EncodingType::Hex,
                },
                line_number: line_num,
                column_start: last_match_end,
                column_end: line.len(),
                encoding_depth: 1,
                confidence: 0.85,
            });
        }
    }

    /// 提取 Unicode 编码的字符串
    fn extract_unicode_strings(&mut self, line: &str, line_num: usize) {
        let re = Regex::new(r"\\u([0-9a-fA-F]{4})").unwrap();

        let mut unicode_buffer = String::new();
        let mut start_pos = 0;

        for (i, caps) in re.captures_iter(line).enumerate() {
            if i == 0 {
                start_pos = caps.get(0).unwrap().start();
            }

            if let Some(m) = caps.get(1) {
                let hex = m.as_str();
                if let Ok(code) = u32::from_str_radix(hex, 16) {
                    if let Some(c) = char::from_u32(code) {
                        unicode_buffer.push(c);
                    }
                }
            }
        }

        if unicode_buffer.len() > 2 {
            self.add_entry(StringPoolEntry {
                original: format!("\\u{{{}}}", unicode_buffer),
                decoded: unicode_buffer.clone(),
                source: StringSource::Encoded {
                    encoding_type: EncodingType::Unicode,
                },
                line_number: line_num,
                column_start: start_pos,
                column_end: line.len(),
                encoding_depth: 1,
                confidence: 0.9,
            });
        }
    }

    /// 提取通过 String.fromCharCode() 生成的字符串
    fn extract_from_char_code(&mut self, line: &str, line_num: usize) {
        let re = Regex::new(r"String\.fromCharCode\(([^)]+)\)").unwrap();

        for caps in re.captures_iter(line) {
            if let Some(m) = caps.get(1) {
                let args = m.as_str();
                let codes: Result<Vec<u8>, _> =
                    args.split(',').map(|s| s.trim().parse::<u8>()).collect();

                if let Ok(code_vec) = codes {
                    if let Ok(decoded) = String::from_utf8(code_vec) {
                        self.add_entry(StringPoolEntry {
                            original: format!("String.fromCharCode({})", args),
                            decoded,
                            source: StringSource::FromCharCode,
                            line_number: line_num,
                            column_start: m.start(),
                            column_end: m.end(),
                            encoding_depth: 1,
                            confidence: 0.95,
                        });
                    }
                }
            }
        }
    }

    /// 提取通过 unescape() 解码的字符串
    fn extract_unescape_strings(&mut self, line: &str, line_num: usize) {
        let re = Regex::new(r#"unescape\(['"]([^'"]+)['"]\)"#).unwrap();

        for caps in re.captures_iter(line) {
            if let Some(m) = caps.get(1) {
                let escaped = m.as_str();
                let decoded = self.unescape_string(escaped);

                self.add_entry(StringPoolEntry {
                    original: format!("unescape(\"{}\")", escaped),
                    decoded,
                    source: StringSource::Unescaped,
                    line_number: line_num,
                    column_start: m.start(),
                    column_end: m.end(),
                    encoding_depth: 1,
                    confidence: 0.95,
                });
            }
        }
    }

    /// 提取通过 atob() 解码的字符串
    fn extract_atob_strings(&mut self, line: &str, line_num: usize) {
        let re = Regex::new(r#"atob\(['"]([A-Za-z0-9+/]+={0,2})['"]\)"#).unwrap();

        for caps in re.captures_iter(line) {
            if let Some(m) = caps.get(1) {
                let b64_str = m.as_str();

                if let Ok(decoded_bytes) = general_purpose::STANDARD.decode(b64_str) {
                    if let Ok(decoded) = String::from_utf8(decoded_bytes) {
                        self.add_entry(StringPoolEntry {
                            original: format!("atob(\"{}\")", b64_str),
                            decoded,
                            source: StringSource::AtobDecoded,
                            line_number: line_num,
                            column_start: m.start(),
                            column_end: m.end(),
                            encoding_depth: 1,
                            confidence: 0.95,
                        });
                    }
                }
            }
        }
    }

    /// 添加字符串项到池
    fn add_entry(&mut self, entry: StringPoolEntry) {
        // 避免重复
        if !self.pool.entries.iter().any(|e| e.decoded == entry.decoded) {
            self.pool.entries.push(entry);
        }
    }

    /// 构建使用映射
    fn build_usage_map(&mut self) {
        for (index, entry) in self.pool.entries.iter().enumerate() {
            self.pool
                .string_usage_map
                .entry(entry.decoded.clone())
                .or_default()
                .push(index);
        }
    }

    /// 反转义字符串
    fn unescape_string(&self, s: &str) -> String {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(&next) = chars.peek() {
                    match next {
                        'n' => {
                            result.push('\n');
                            chars.next();
                        }
                        't' => {
                            result.push('\t');
                            chars.next();
                        }
                        'r' => {
                            result.push('\r');
                            chars.next();
                        }
                        '\\' => {
                            result.push('\\');
                            chars.next();
                        }
                        '"' => {
                            result.push('"');
                            chars.next();
                        }
                        '\'' => {
                            result.push('\'');
                            chars.next();
                        }
                        _ => result.push(c),
                    }
                } else {
                    result.push(c);
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    /// 获取字符串池
    pub fn get_pool(&self) -> &StringPool {
        &self.pool
    }

    /// 获取池统计
    pub fn get_statistics(&self) -> StringPoolStatistics {
        let mut decoding_stats = HashMap::new();
        for entry in &self.pool.entries {
            if let StringSource::Encoded { encoding_type } = entry.source {
                *decoding_stats.entry(encoding_type).or_insert(0) += 1;
            }
        }

        StringPoolStatistics {
            total_strings: self.pool.entries.len(),
            literal_strings: self
                .pool
                .entries
                .iter()
                .filter(|e| e.source == StringSource::Literal)
                .count(),
            encoded_strings: self
                .pool
                .entries
                .iter()
                .filter(|e| matches!(e.source, StringSource::Encoded { .. }))
                .count(),
            avg_encoding_depth: if self.pool.entries.is_empty() {
                0.0
            } else {
                self.pool
                    .entries
                    .iter()
                    .map(|e| e.encoding_depth as f32)
                    .sum::<f32>()
                    / self.pool.entries.len() as f32
            },
            decoding_stats,
            unique_strings: self.pool.string_usage_map.len(),
        }
    }

    /// 导出字符串映射表 (用于替换)
    pub fn export_mapping(&self) -> HashMap<String, String> {
        let mut mapping = HashMap::new();
        for entry in &self.pool.entries {
            mapping.insert(entry.original.clone(), entry.decoded.clone());
        }
        mapping
    }
}

#[derive(Debug, Clone)]
pub struct StringPoolStatistics {
    pub total_strings: usize,
    pub literal_strings: usize,
    pub encoded_strings: usize,
    pub avg_encoding_depth: f32,
    pub decoding_stats: HashMap<EncodingType, usize>,
    pub unique_strings: usize,
}

impl Default for StringPoolExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_string_extraction() {
        let mut extractor = StringPoolExtractor::new();
        let code = r#"let x = "hello"; let y = 'world';"#;
        assert!(extractor.extract(code).is_ok());

        let pool = extractor.get_pool();
        assert!(pool.entries.len() >= 2);
    }

    #[test]
    fn test_char_array_extraction() {
        let mut extractor = StringPoolExtractor::new();
        let code = "let x = [65, 66, 67];"; // ABC
        assert!(extractor.extract(code).is_ok());

        let pool = extractor.get_pool();
        assert!(!pool.entries.is_empty());
    }

    #[test]
    fn test_from_char_code_extraction() {
        let mut extractor = StringPoolExtractor::new();
        let code = "String.fromCharCode(72, 101, 108, 108, 111)"; // Hello
        assert!(extractor.extract(code).is_ok());

        let pool = extractor.get_pool();
        assert!(!pool.entries.is_empty());
    }

    #[test]
    fn test_statistics() {
        let mut extractor = StringPoolExtractor::new();
        let code = r#"let x = "test"; let y = "another";"#;
        assert!(extractor.extract(code).is_ok());

        let stats = extractor.get_statistics();
        assert!(stats.total_strings > 0);
    }

    #[test]
    fn test_export_mapping() {
        let mut extractor = StringPoolExtractor::new();
        let code = r#"let x = "hello";"#;
        assert!(extractor.extract(code).is_ok());

        let mapping = extractor.export_mapping();
        assert!(!mapping.is_empty());
    }
}
