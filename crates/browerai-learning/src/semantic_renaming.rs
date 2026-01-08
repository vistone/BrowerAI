//! 语义化重命名 - 基于函数行为推断有意义的变量名
//!
//! 通过分析代码的行为模式、类型和上下文来生成有意义的标识符

use std::collections::HashMap;

/// 语义重命名器
pub struct SemanticRenamer {
    /// 变量到语义名称的映射
    rename_map: HashMap<String, String>,

    /// 已使用的名称（避免冲突）
    used_names: HashMap<String, usize>,
}

impl SemanticRenamer {
    pub fn new() -> Self {
        Self {
            rename_map: HashMap::new(),
            used_names: HashMap::new(),
        }
    }

    /// 基于代码分析进行语义重命名
    pub fn analyze_and_rename(&mut self, code: &str) -> String {
        let mut result = code.to_string();

        // 分析常量值模式
        self.analyze_constants(&result);

        // 分析函数行为模式
        self.analyze_functions(&result);

        // 分析字符串字面量用途
        self.analyze_string_literals(&result);

        // 应用重命名
        result = self.apply_renames(&result);

        result
    }

    /// 分析常量值，推断语义
    fn analyze_constants(&mut self, code: &str) {
        // 时间相关常量
        let time_constants = vec![
            (r"var(\d+)=1e3", "MILLISECONDS_PER_SECOND"),
            (r"var(\d+)=6e4", "MILLISECONDS_PER_MINUTE"),
            (r"var(\d+)=36e5", "MILLISECONDS_PER_HOUR"),
            (r"var(\d+)=1000", "MILLISECONDS_PER_SECOND"),
            (r"var(\d+)=60", "SECONDS_OR_MINUTES"),
            (r"var(\d+)=24", "HOURS_PER_DAY"),
            (r"var(\d+)=7", "DAYS_PER_WEEK"),
            (r"var(\d+)=12", "MONTHS_PER_YEAR"),
        ];

        for (pattern, semantic_name) in time_constants {
            if let Some(captures) = regex::Regex::new(pattern)
                .ok()
                .and_then(|re| re.captures(code))
            {
                if let Some(var_num) = captures.get(1) {
                    let var_name = format!("var{}", var_num.as_str());
                    self.add_rename(&var_name, semantic_name);
                }
            }
        }

        // 字符串常量模式
        let string_constants = vec![
            (r#"var(\d+)="millisecond""#, "UNIT_MILLISECOND"),
            (r#"var(\d+)="second""#, "UNIT_SECOND"),
            (r#"var(\d+)="minute""#, "UNIT_MINUTE"),
            (r#"var(\d+)="hour""#, "UNIT_HOUR"),
            (r#"var(\d+)="day""#, "UNIT_DAY"),
            (r#"var(\d+)="week""#, "UNIT_WEEK"),
            (r#"var(\d+)="month""#, "UNIT_MONTH"),
            (r#"var(\d+)="year""#, "UNIT_YEAR"),
            (r#"var(\d+)="date""#, "UNIT_DATE"),
            (r#"var(\d+)="Invalid Date""#, "INVALID_DATE_MESSAGE"),
        ];

        for (pattern, semantic_name) in string_constants {
            if let Some(captures) = regex::Regex::new(pattern)
                .ok()
                .and_then(|re| re.captures(code))
            {
                if let Some(var_num) = captures.get(1) {
                    let var_name = format!("var{}", var_num.as_str());
                    self.add_rename(&var_name, semantic_name);
                }
            }
        }
    }

    /// 分析函数行为模式
    fn analyze_functions(&mut self, code: &str) {
        // 格式化函数模式
        let function_patterns = vec![
            (
                r"function\s+(\w+)\([^)]*\)\{[^}]*format[^}]*\}",
                "formatter",
            ),
            (r"function\s+(\w+)\([^)]*\)\{[^}]*parse[^}]*\}", "parser"),
            (
                r"function\s+(\w+)\([^)]*\)\{[^}]*validate[^}]*\}",
                "validator",
            ),
            (
                r"function\s+(\w+)\([^)]*\)\{[^}]*\.get[A-Z][^}]*\}",
                "getter",
            ),
            (
                r"function\s+(\w+)\([^)]*\)\{[^}]*\.set[A-Z][^}]*\}",
                "setter",
            ),
            (
                r"function\s+(\w+)\([^)]*\)\{[^}]*return[^}]*\+[^}]*\}",
                "calculator",
            ),
            (r"function\s+(\w+)\([^)]*\)\{[^}]*clone[^}]*\}", "cloner"),
            (
                r"function\s+(\w+)\([^)]*\)\{[^}]*new\s+Date[^}]*\}",
                "dateCreator",
            ),
        ];

        for (pattern, semantic_suffix) in function_patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for captures in re.captures_iter(code) {
                    if let Some(func_name) = captures.get(1) {
                        let name = func_name.as_str();
                        if name.starts_with("var") {
                            self.add_rename(name, semantic_suffix);
                        }
                    }
                }
            }
        }
    }

    /// 分析字符串字面量的用途
    fn analyze_string_literals(&mut self, _code: &str) {
        // 根据上下文推断字符串用途
        let literal_patterns = vec![
            (r#"weekdays:"([^"]+)""#, "WEEKDAYS_STRING"),
            (r#"months:"([^"]+)""#, "MONTHS_STRING"),
            (r#"name:"([^"]+)""#, "LOCALE_NAME"),
        ];

        for (pattern, _semantic_name) in literal_patterns {
            if let Ok(_re) = regex::Regex::new(pattern) {
                // 这里可以进一步分析
            }
        }
    }

    /// 添加重命名映射（避免冲突）
    fn add_rename(&mut self, old_name: &str, base_semantic_name: &str) {
        let count = self
            .used_names
            .entry(base_semantic_name.to_string())
            .or_insert(0);

        let new_name = if *count == 0 {
            base_semantic_name.to_string()
        } else {
            format!("{}_{}", base_semantic_name, count)
        };

        *count += 1;
        self.rename_map.insert(old_name.to_string(), new_name);
    }

    /// 应用所有重命名
    fn apply_renames(&self, code: &str) -> String {
        let mut result = code.to_string();

        // 按变量名长度降序排序（避免部分匹配问题）
        let mut renames: Vec<_> = self.rename_map.iter().collect();
        renames.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        for (old_name, new_name) in renames {
            // 使用词边界确保完整匹配
            let pattern = format!(r"\b{}\b", regex::escape(old_name));
            if let Ok(re) = regex::Regex::new(&pattern) {
                result = re.replace_all(&result, new_name.as_str()).to_string();
            }
        }

        result
    }

    /// 获取重命名映射表（用于调试）
    pub fn get_rename_map(&self) -> &HashMap<String, String> {
        &self.rename_map
    }
}

impl Default for SemanticRenamer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_detection() {
        let code = r#"
var var0=1e3;
var var1=6e4;
var var2="millisecond";
var var3="Invalid Date";
"#;

        let mut renamer = SemanticRenamer::new();
        let result = renamer.analyze_and_rename(code);

        println!("Rename map: {:?}", renamer.get_rename_map());
        println!("Result:\n{}", result);

        assert!(result.contains("MILLISECONDS_PER_SECOND"));
        assert!(result.contains("MILLISECONDS_PER_MINUTE"));
        assert!(result.contains("UNIT_MILLISECOND"));
        assert!(result.contains("INVALID_DATE_MESSAGE"));
    }

    #[test]
    fn test_function_pattern_detection() {
        let code = r#"
function var10(x) {
    return x.format();
}
function var11(x) {
    return new Date(x);
}
"#;

        let mut renamer = SemanticRenamer::new();
        let result = renamer.analyze_and_rename(code);

        println!("Result:\n{}", result);

        assert!(result.contains("formatter") || result.contains("dateCreator"));
    }

    #[test]
    fn test_no_conflicts() {
        let code = r#"
var var0=1e3;
var var1=1e3;
"#;

        let mut renamer = SemanticRenamer::new();
        let result = renamer.analyze_and_rename(code);

        println!("Rename map: {:?}", renamer.get_rename_map());

        // 应该有 MILLISECONDS_PER_SECOND 和 MILLISECONDS_PER_SECOND_1
        assert!(
            result.contains("MILLISECONDS_PER_SECOND")
                && (result.contains("MILLISECONDS_PER_SECOND_1")
                    || result.matches("MILLISECONDS_PER_SECOND").count() == 2)
        );
    }
}
