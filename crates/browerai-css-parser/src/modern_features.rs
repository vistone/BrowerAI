/// Modern CSS Features for 2026 Standards
///
/// Implements cutting-edge CSS features including:
/// - Container Queries
/// - :has() pseudo-class selector
/// - CSS Nesting
/// - CSS Custom Properties (CSS Variables)
/// - CSS Subgrid
use std::collections::HashMap;

/// Container Query conditions and rules
#[derive(Debug, Clone, PartialEq)]
pub struct ContainerQuery {
    /// Container name (optional)
    pub container_name: Option<String>,
    /// Query condition (e.g., "min-width: 400px")
    pub condition: String,
    /// Rules that apply when condition is met
    pub rules: Vec<CssRule>,
}

impl ContainerQuery {
    /// Create a new container query
    pub fn new(condition: String) -> Self {
        Self {
            container_name: None,
            condition,
            rules: Vec::new(),
        }
    }

    /// Set container name
    pub fn with_name(mut self, name: String) -> Self {
        self.container_name = Some(name);
        self
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: CssRule) {
        self.rules.push(rule);
    }

    /// Check if container query condition is met
    pub fn evaluate(&self, container_width: f32, container_height: f32) -> bool {
        // Parse simple conditions like "min-width: 400px" or "max-width: 800px"
        let condition = self.condition.trim();

        if condition.starts_with("min-width:") {
            if let Some(value_str) = condition.strip_prefix("min-width:") {
                if let Ok(min_width) = parse_css_dimension(value_str.trim()) {
                    return container_width >= min_width;
                }
            }
        } else if condition.starts_with("max-width:") {
            if let Some(value_str) = condition.strip_prefix("max-width:") {
                if let Ok(max_width) = parse_css_dimension(value_str.trim()) {
                    return container_width <= max_width;
                }
            }
        } else if condition.starts_with("min-height:") {
            if let Some(value_str) = condition.strip_prefix("min-height:") {
                if let Ok(min_height) = parse_css_dimension(value_str.trim()) {
                    return container_height >= min_height;
                }
            }
        } else if condition.starts_with("max-height:") {
            if let Some(value_str) = condition.strip_prefix("max-height:") {
                if let Ok(max_height) = parse_css_dimension(value_str.trim()) {
                    return container_height <= max_height;
                }
            }
        }

        false
    }

    /// Parse CSS container query from string
    pub fn parse(query_str: &str) -> Option<Self> {
        // Example: "@container (min-width: 400px) { ... }"
        if !query_str.trim().starts_with("@container") {
            return None;
        }

        let content = query_str.trim().strip_prefix("@container")?.trim();

        // Extract container name if present
        let (name, condition_part) = if content.starts_with('(') {
            (None, content)
        } else {
            let parts: Vec<&str> = content.splitn(2, '(').collect();
            if parts.len() == 2 {
                (Some(parts[0].trim().to_string()), parts[1])
            } else {
                (None, content)
            }
        };

        // Extract condition from parentheses
        let condition = condition_part
            .trim_start_matches('(')
            .split(')')
            .next()?
            .trim()
            .to_string();

        let mut query = ContainerQuery::new(condition);
        if let Some(n) = name {
            query.container_name = Some(n);
        }

        Some(query)
    }
}

/// :has() pseudo-class selector implementation
#[derive(Debug, Clone, PartialEq)]
pub struct HasSelector {
    /// Parent selector
    pub parent_selector: String,
    /// Child selector to check for
    pub child_selector: String,
}

impl HasSelector {
    /// Create a new :has() selector
    pub fn new(parent_selector: String, child_selector: String) -> Self {
        Self {
            parent_selector,
            child_selector,
        }
    }

    /// Parse :has() selector from string
    pub fn parse(selector: &str) -> Option<Self> {
        // Example: "section:has(.active)"
        if !selector.contains(":has(") {
            return None;
        }

        let parts: Vec<&str> = selector.split(":has(").collect();
        if parts.len() != 2 {
            return None;
        }

        let parent = parts[0].trim().to_string();
        let child = parts[1].trim_end_matches(')').trim().to_string();

        Some(HasSelector::new(parent, child))
    }

    /// Convert to CSS selector string
    pub fn to_css(&self) -> String {
        format!("{}:has({})", self.parent_selector, self.child_selector)
    }
}

/// CSS Nesting structure
#[derive(Debug, Clone, PartialEq)]
pub struct NestedCssRule {
    /// Selector for this rule
    pub selector: String,
    /// Properties at this level
    pub properties: Vec<CssProperty>,
    /// Nested child rules
    pub nested_rules: Vec<NestedCssRule>,
}

impl NestedCssRule {
    /// Create a new nested rule
    pub fn new(selector: String) -> Self {
        Self {
            selector,
            properties: Vec::new(),
            nested_rules: Vec::new(),
        }
    }

    /// Add a property
    pub fn add_property(&mut self, property: CssProperty) {
        self.properties.push(property);
    }

    /// Add a nested rule
    pub fn add_nested_rule(&mut self, rule: NestedCssRule) {
        self.nested_rules.push(rule);
    }

    /// Flatten nested rules to regular CSS rules
    pub fn flatten(&self, parent_selector: &str) -> Vec<CssRule> {
        let mut rules = Vec::new();

        // Current rule
        let full_selector = if parent_selector.is_empty() {
            self.selector.clone()
        } else if self.selector.starts_with('&') {
            // & refers to parent selector
            self.selector.replace('&', parent_selector)
        } else {
            format!("{} {}", parent_selector, self.selector)
        };

        if !self.properties.is_empty() {
            rules.push(CssRule {
                selector: full_selector.clone(),
                properties: self.properties.clone(),
            });
        }

        // Nested rules
        for nested in &self.nested_rules {
            rules.extend(nested.flatten(&full_selector));
        }

        rules
    }

    /// Parse nested CSS from string
    pub fn parse(css: &str) -> Option<Self> {
        // Simplified parser for nested CSS
        // Real implementation would use a proper CSS parser
        let trimmed = css.trim();
        let selector_end = trimmed.find('{')?;
        let selector = trimmed[..selector_end].trim().to_string();

        Some(NestedCssRule::new(selector))
    }
}

/// CSS Custom Properties (CSS Variables) manager
#[derive(Debug, Clone)]
pub struct CssCustomProperties {
    /// Custom properties storage
    properties: HashMap<String, String>,
    /// Parent scope (for inheritance)
    parent: Option<Box<CssCustomProperties>>,
}

impl CssCustomProperties {
    /// Create a new custom properties scope
    pub fn new() -> Self {
        Self {
            properties: HashMap::new(),
            parent: None,
        }
    }

    /// Create a child scope
    pub fn with_parent(parent: CssCustomProperties) -> Self {
        Self {
            properties: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    /// Set a custom property
    pub fn set_property(&mut self, name: String, value: String) {
        // Custom properties start with --
        if name.starts_with("--") {
            self.properties.insert(name, value);
        }
    }

    /// Get a custom property value (with fallback chain)
    pub fn get_property(&self, name: &str) -> Option<String> {
        // Check local scope first
        if let Some(value) = self.properties.get(name) {
            return Some(value.clone());
        }

        // Check parent scope
        if let Some(ref parent) = self.parent {
            return parent.get_property(name);
        }

        None
    }

    /// Resolve var() references in a CSS value
    pub fn resolve_var(&self, value: &str) -> String {
        let mut result = value.to_string();

        // Find all var() references
        while let Some(start) = result.find("var(") {
            let end = result[start..].find(')').unwrap_or(result.len() - start) + start;
            let var_content = &result[start + 4..end];

            // Parse var(--name, fallback)
            let parts: Vec<&str> = var_content.split(',').collect();
            let var_name = parts[0].trim();
            let fallback = parts.get(1).map(|s| s.trim());

            // Resolve the variable
            let resolved = if let Some(var_value) = self.get_property(var_name) {
                var_value
            } else if let Some(fb) = fallback {
                fb.to_string()
            } else {
                // Invalid var reference, keep as is
                result[start..=end].to_string()
            };

            // Replace var() with resolved value
            result.replace_range(start..=end, &resolved);
        }

        result
    }

    /// Parse and set custom properties from CSS declaration
    pub fn parse_declaration(&mut self, property: &str, value: &str) {
        if property.starts_with("--") {
            self.set_property(property.to_string(), value.to_string());
        }
    }

    /// Get all custom properties in this scope
    pub fn all_properties(&self) -> &HashMap<String, String> {
        &self.properties
    }
}

impl Default for CssCustomProperties {
    fn default() -> Self {
        Self::new()
    }
}

/// CSS Subgrid support
#[derive(Debug, Clone, PartialEq)]
pub struct SubgridDefinition {
    /// Grid axis (rows or columns)
    pub axis: SubgridAxis,
    /// Parent grid track lines to use
    pub track_lines: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SubgridAxis {
    Rows,
    Columns,
    Both,
}

impl SubgridDefinition {
    /// Create a new subgrid
    pub fn new(axis: SubgridAxis) -> Self {
        Self {
            axis,
            track_lines: Vec::new(),
        }
    }

    /// Parse subgrid from CSS value
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();

        if value == "subgrid" {
            Some(SubgridDefinition::new(SubgridAxis::Both))
        } else if value.starts_with("subgrid") {
            // Parse axis specification
            if value.contains("rows") {
                Some(SubgridDefinition::new(SubgridAxis::Rows))
            } else if value.contains("columns") {
                Some(SubgridDefinition::new(SubgridAxis::Columns))
            } else {
                Some(SubgridDefinition::new(SubgridAxis::Both))
            }
        } else {
            None
        }
    }

    /// Check if this is a subgrid
    pub fn is_subgrid(&self) -> bool {
        true
    }

    /// Get CSS value string
    pub fn to_css(&self) -> String {
        match self.axis {
            SubgridAxis::Rows => "subgrid [rows]".to_string(),
            SubgridAxis::Columns => "subgrid [columns]".to_string(),
            SubgridAxis::Both => "subgrid".to_string(),
        }
    }
}

/// Basic CSS Rule structure
#[derive(Debug, Clone, PartialEq)]
pub struct CssRule {
    pub selector: String,
    pub properties: Vec<CssProperty>,
}

impl CssRule {
    pub fn new(selector: String) -> Self {
        Self {
            selector,
            properties: Vec::new(),
        }
    }

    pub fn add_property(&mut self, property: CssProperty) {
        self.properties.push(property);
    }
}

/// CSS Property
#[derive(Debug, Clone, PartialEq)]
pub struct CssProperty {
    pub name: String,
    pub value: String,
}

impl CssProperty {
    pub fn new(name: String, value: String) -> Self {
        Self { name, value }
    }
}

/// Parse CSS dimension value (e.g., "400px" -> 400.0)
fn parse_css_dimension(value: &str) -> Result<f32, String> {
    let value = value.trim();
    
    // Remove unit suffix (px, em, rem, etc.)
    let numeric_part = value
        .trim_end_matches("px")
        .trim_end_matches("em")
        .trim_end_matches("rem")
        .trim_end_matches("%")
        .trim_end_matches("vh")
        .trim_end_matches("vw");

    numeric_part
        .parse::<f32>()
        .map_err(|_| format!("Invalid dimension value: {}", value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_query_parse() {
        let query = ContainerQuery::parse("@container (min-width: 400px)");
        assert!(query.is_some());

        let q = query.unwrap();
        assert_eq!(q.condition, "min-width: 400px");
        assert!(q.container_name.is_none());
    }

    #[test]
    fn test_container_query_with_name() {
        let query = ContainerQuery::parse("@container sidebar (min-width: 300px)");
        assert!(query.is_some());

        let q = query.unwrap();
        assert_eq!(q.condition, "min-width: 300px");
        assert_eq!(q.container_name, Some("sidebar".to_string()));
    }

    #[test]
    fn test_container_query_evaluate() {
        let mut query = ContainerQuery::new("min-width: 400px".to_string());
        
        assert!(query.evaluate(500.0, 300.0));  // Width >= 400
        assert!(!query.evaluate(300.0, 300.0)); // Width < 400

        query.condition = "max-width: 800px".to_string();
        assert!(query.evaluate(600.0, 300.0));  // Width <= 800
        assert!(!query.evaluate(900.0, 300.0)); // Width > 800
    }

    #[test]
    fn test_has_selector_parse() {
        let selector = HasSelector::parse("section:has(.active)");
        assert!(selector.is_some());

        let sel = selector.unwrap();
        assert_eq!(sel.parent_selector, "section");
        assert_eq!(sel.child_selector, ".active");
        assert_eq!(sel.to_css(), "section:has(.active)");
    }

    #[test]
    fn test_has_selector_complex() {
        let selector = HasSelector::parse("div.container:has(> .important)");
        assert!(selector.is_some());

        let sel = selector.unwrap();
        assert_eq!(sel.parent_selector, "div.container");
        assert_eq!(sel.child_selector, "> .important");
    }

    #[test]
    fn test_nested_css_flatten() {
        let mut root = NestedCssRule::new(".card".to_string());
        root.add_property(CssProperty::new("padding".to_string(), "1rem".to_string()));

        let mut nested = NestedCssRule::new(".title".to_string());
        nested.add_property(CssProperty::new("font-size".to_string(), "1.5rem".to_string()));

        root.add_nested_rule(nested);

        let flattened = root.flatten("");
        assert_eq!(flattened.len(), 2);
        assert_eq!(flattened[0].selector, ".card");
        assert_eq!(flattened[1].selector, ".card .title");
    }

    #[test]
    fn test_nested_css_with_ampersand() {
        let mut root = NestedCssRule::new(".button".to_string());
        root.add_property(CssProperty::new("color".to_string(), "blue".to_string()));

        let mut hover = NestedCssRule::new("&:hover".to_string());
        hover.add_property(CssProperty::new("color".to_string(), "red".to_string()));

        root.add_nested_rule(hover);

        let flattened = root.flatten("");
        assert_eq!(flattened[1].selector, ".button:hover");
    }

    #[test]
    fn test_css_custom_properties() {
        let mut props = CssCustomProperties::new();
        
        props.set_property("--primary-color".to_string(), "blue".to_string());
        assert_eq!(props.get_property("--primary-color"), Some("blue".to_string()));

        // Non-custom property should not be stored
        props.set_property("color".to_string(), "red".to_string());
        assert!(props.get_property("color").is_none());
    }

    #[test]
    fn test_css_var_resolution() {
        let mut props = CssCustomProperties::new();
        props.set_property("--main-bg".to_string(), "#ffffff".to_string());
        props.set_property("--accent".to_string(), "blue".to_string());

        assert_eq!(
            props.resolve_var("background: var(--main-bg);"),
            "background: #ffffff;"
        );

        assert_eq!(
            props.resolve_var("color: var(--accent);"),
            "color: blue;"
        );
    }

    #[test]
    fn test_css_var_with_fallback() {
        let props = CssCustomProperties::new();

        assert_eq!(
            props.resolve_var("color: var(--undefined, red);"),
            "color: red;"
        );
    }

    #[test]
    fn test_css_var_inheritance() {
        let mut parent = CssCustomProperties::new();
        parent.set_property("--primary".to_string(), "blue".to_string());

        let mut child = CssCustomProperties::with_parent(parent);
        child.set_property("--secondary".to_string(), "green".to_string());

        assert_eq!(child.get_property("--primary"), Some("blue".to_string()));
        assert_eq!(child.get_property("--secondary"), Some("green".to_string()));
    }

    #[test]
    fn test_subgrid_parse() {
        let subgrid = SubgridDefinition::parse("subgrid");
        assert!(subgrid.is_some());
        assert_eq!(subgrid.unwrap().axis, SubgridAxis::Both);

        let rows = SubgridDefinition::parse("subgrid rows");
        assert!(rows.is_some());
        assert_eq!(rows.unwrap().axis, SubgridAxis::Rows);
    }

    #[test]
    fn test_subgrid_to_css() {
        let both = SubgridDefinition::new(SubgridAxis::Both);
        assert_eq!(both.to_css(), "subgrid");

        let rows = SubgridDefinition::new(SubgridAxis::Rows);
        assert_eq!(rows.to_css(), "subgrid [rows]");

        let cols = SubgridDefinition::new(SubgridAxis::Columns);
        assert_eq!(cols.to_css(), "subgrid [columns]");
    }

    #[test]
    fn test_parse_css_dimension() {
        assert_eq!(parse_css_dimension("400px"), Ok(400.0));
        assert_eq!(parse_css_dimension("50%"), Ok(50.0));
        assert_eq!(parse_css_dimension("2.5em"), Ok(2.5));
        assert!(parse_css_dimension("invalid").is_err());
    }
}
