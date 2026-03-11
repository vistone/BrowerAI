//! CSS 样式表类型

use std::collections::HashMap;

/// CSS 样式表
#[derive(Debug, Clone, Default)]
pub struct Stylesheet {
    /// 规则列表
    pub rules: Vec<Rule>,
    /// 媒体查询
    pub media_rules: Vec<MediaRule>,
    /// 自定义属性（CSS 变量）
    pub variables: HashMap<String, String>,
}

impl Stylesheet {
    /// 创建新的样式表
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            media_rules: Vec::new(),
            variables: HashMap::new(),
        }
    }

    /// 添加规则
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// 查询匹配选择器的规则
    pub fn query_selector(&self, selector: &str) -> Vec<&Rule> {
        self.rules
            .iter()
            .filter(|r| r.selector.to_string() == selector)
            .collect()
    }

    /// 获取所有选择器
    pub fn all_selectors(&self) -> Vec<String> {
        self.rules.iter().map(|r| r.selector.to_string()).collect()
    }

    /// 合并另一个样式表
    pub fn merge(&mut self, other: &Stylesheet) {
        self.rules.extend(other.rules.clone());
        self.media_rules.extend(other.media_rules.clone());
        self.variables.extend(other.variables.clone());
    }

    /// 序列化为 CSS 字符串
    pub fn to_css(&self) -> String {
        let mut result = String::new();

        // 变量
        if !self.variables.is_empty() {
            result.push_str(":root {\n");
            for (key, value) in &self.variables {
                result.push_str(&format!("  {}: {};\n", key, value));
            }
            result.push_str("}\n\n");
        }

        // 规则
        for rule in &self.rules {
            result.push_str(&rule.to_css());
            result.push('\n');
        }

        result
    }
}

/// CSS 规则
#[derive(Debug, Clone, Default)]
pub struct Rule {
    /// 选择器
    pub selector: Selector,
    /// 声明列表
    pub declarations: Vec<Declaration>,
}

impl Rule {
    /// 创建新规则
    pub fn new() -> Self {
        Self {
            selector: Selector::new(),
            declarations: Vec::new(),
        }
    }

    /// 添加声明
    pub fn add_declaration(&mut self, property: impl Into<String>, value: impl Into<Value>) {
        self.declarations.push(Declaration {
            property: property.into(),
            value: value.into(),
            important: false,
        });
    }

    /// 获取声明值
    pub fn get_declaration(&self, property: &str) -> Option<&Value> {
        self.declarations
            .iter()
            .find(|d| d.property == property)
            .map(|d| &d.value)
    }

    /// 序列化为 CSS
    pub fn to_css(&self) -> String {
        let mut result = format!("{} {{\n", self.selector);

        for decl in &self.declarations {
            result.push_str(&format!("  {};\n", decl.to_css()));
        }

        result.push('}');
        result
    }
}

/// CSS 选择器
#[derive(Debug, Clone, Default)]
pub struct Selector {
    /// 选择器字符串
    pub raw: String,
    /// 选择器类型
    pub selector_type: SelectorType,
}

impl Selector {
    /// 创建新选择器
    pub fn new() -> Self {
        Self {
            raw: String::new(),
            selector_type: SelectorType::Universal,
        }
    }

    /// 从字符串创建
    pub fn from_string(s: impl Into<String>) -> Self {
        let raw = s.into();
        let selector_type = Self::parse_type(&raw);

        Self { raw, selector_type }
    }

    /// 解析选择器类型
    fn parse_type(raw: &str) -> SelectorType {
        let raw = raw.trim();

        if raw == "*" {
            SelectorType::Universal
        } else if let Some(stripped) = raw.strip_prefix('#') {
            SelectorType::Id(stripped.to_string())
        } else if let Some(stripped) = raw.strip_prefix('.') {
            SelectorType::Class(stripped.to_string())
        } else if raw.contains(':') {
            SelectorType::Pseudo(raw.to_string())
        } else if raw.contains('[') {
            SelectorType::Attribute(raw.to_string())
        } else {
            SelectorType::Element(raw.to_string())
        }
    }

    /// 获取选择器特异性（优先级）
    pub fn specificity(&self) -> (u32, u32, u32) {
        // (ID, Class/Attribute, Element)
        match &self.selector_type {
            SelectorType::Universal => (0, 0, 0),
            SelectorType::Element(_) => (0, 0, 1),
            SelectorType::Class(_) | SelectorType::Attribute(_) | SelectorType::Pseudo(_) => {
                (0, 1, 0)
            }
            SelectorType::Id(_) => (1, 0, 0),
        }
    }
}

impl std::fmt::Display for Selector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.raw)
    }
}

/// 选择器类型
#[derive(Debug, Clone, Default)]
pub enum SelectorType {
    /// 通用选择器 *
    #[default]
    Universal,
    /// 元素选择器
    Element(String),
    /// ID 选择器
    Id(String),
    /// 类选择器
    Class(String),
    /// 属性选择器
    Attribute(String),
    /// 伪类/伪元素选择器
    Pseudo(String),
}

/// CSS 声明
#[derive(Debug, Clone)]
pub struct Declaration {
    /// 属性名
    pub property: String,
    /// 属性值
    pub value: Value,
    /// 是否 !important
    pub important: bool,
}

impl Declaration {
    /// 序列化为 CSS
    pub fn to_css(&self) -> String {
        let important = if self.important { " !important" } else { "" };
        format!("{}: {}{}", self.property, self.value.to_css(), important)
    }
}

/// CSS 值
#[derive(Debug, Clone)]
pub enum Value {
    /// 字符串值
    String(String),
    /// 数字值
    Number(f64),
    /// 带单位的值
    Dimension(f64, Unit),
    /// 颜色
    Color(Color),
    /// 列表
    List(Vec<Value>, ListSeparator),
    /// 函数调用
    Function(String, Vec<Value>),
}

impl Value {
    /// 序列化为 CSS
    pub fn to_css(&self) -> String {
        match self {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Dimension(n, unit) => format!("{}{}", n, unit.to_css()),
            Value::Color(c) => c.to_css(),
            Value::List(values, sep) => {
                let sep_str = match sep {
                    ListSeparator::Comma => ", ",
                    ListSeparator::Space => " ",
                };
                values
                    .iter()
                    .map(|v| v.to_css())
                    .collect::<Vec<_>>()
                    .join(sep_str)
            }
            Value::Function(name, args) => {
                let args_str = args
                    .iter()
                    .map(|v| v.to_css())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", name, args_str)
            }
        }
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Number(n)
    }
}

/// CSS 单位
#[derive(Debug, Clone, Copy)]
pub enum Unit {
    /// 像素
    Px,
    /// 百分比
    Percent,
    /// em
    Em,
    /// rem
    Rem,
    /// vw
    Vw,
    /// vh
    Vh,
    /// 秒
    S,
    /// 毫秒
    Ms,
    /// 无单位
    None,
}

impl Unit {
    /// 序列化为 CSS
    pub fn to_css(&self) -> &'static str {
        match self {
            Unit::Px => "px",
            Unit::Percent => "%",
            Unit::Em => "em",
            Unit::Rem => "rem",
            Unit::Vw => "vw",
            Unit::Vh => "vh",
            Unit::S => "s",
            Unit::Ms => "ms",
            Unit::None => "",
        }
    }
}

/// 列表分隔符
#[derive(Debug, Clone, Copy)]
pub enum ListSeparator {
    /// 逗号分隔
    Comma,
    /// 空格分隔
    Space,
}

/// CSS 颜色
#[derive(Debug, Clone, Copy)]
pub struct Color {
    /// 红色通道
    pub r: u8,
    /// 绿色通道
    pub g: u8,
    /// 蓝色通道
    pub b: u8,
    /// 透明度
    pub a: f64,
}

impl Color {
    /// 创建 RGB 颜色
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// 创建 RGBA 颜色
    pub fn rgba(r: u8, g: u8, b: u8, a: f64) -> Self {
        Self { r, g, b, a }
    }

    /// 序列化为 CSS
    pub fn to_css(&self) -> String {
        if self.a == 1.0 {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            format!("rgba({}, {}, {}, {})", self.r, self.g, self.b, self.a)
        }
    }

    /// 从十六进制解析
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');

        match hex.len() {
            3 => {
                let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?;
                let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?;
                let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?;
                Some(Self::rgb(r, g, b))
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some(Self::rgb(r, g, b))
            }
            _ => None,
        }
    }
}

/// 媒体查询规则
#[derive(Debug, Clone)]
pub struct MediaRule {
    /// 媒体查询条件
    pub condition: String,
    /// 规则列表
    pub rules: Vec<Rule>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stylesheet_to_css() {
        let mut stylesheet = Stylesheet::new();
        let mut rule = Rule::new();
        rule.selector = Selector::from_string("body");
        rule.add_declaration("color", "red");
        stylesheet.add_rule(rule);

        let css = stylesheet.to_css();
        assert!(css.contains("body"));
        assert!(css.contains("color: red"));
    }

    #[test]
    fn test_selector_specificity() {
        let universal = Selector::from_string("*");
        let element = Selector::from_string("div");
        let class = Selector::from_string(".class");
        let id = Selector::from_string("#id");

        assert_eq!(universal.specificity(), (0, 0, 0));
        assert_eq!(element.specificity(), (0, 0, 1));
        assert_eq!(class.specificity(), (0, 1, 0));
        assert_eq!(id.specificity(), (1, 0, 0));
    }

    #[test]
    fn test_color_from_hex() {
        let color = Color::from_hex("#ff0000").unwrap();
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);

        let short = Color::from_hex("#f00").unwrap();
        assert_eq!(short.r, 255);
        assert_eq!(short.g, 0);
        assert_eq!(short.b, 0);
    }

    #[test]
    fn test_color_to_css() {
        let color = Color::rgb(255, 0, 0);
        assert_eq!(color.to_css(), "#ff0000");

        let rgba = Color::rgba(255, 0, 0, 0.5);
        assert_eq!(rgba.to_css(), "rgba(255, 0, 0, 0.5)");
    }

    #[test]
    fn test_value_to_css() {
        assert_eq!(Value::String("red".to_string()).to_css(), "red");
        assert_eq!(Value::Number(42.0).to_css(), "42");
        assert_eq!(Value::Dimension(16.0, Unit::Px).to_css(), "16px");
    }
}
