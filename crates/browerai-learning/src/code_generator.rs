/// AI-powered code generation module
///
/// Provides capabilities to generate simple HTML/CSS/JS based on learned patterns
use anyhow::Result;
use browerai_core::CodeType;
use handlebars::Handlebars;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

/// Code generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Maximum code length to generate
    pub max_length: usize,
    /// Temperature for sampling (higher = more creative)
    pub temperature: f32,
    /// Use learned patterns
    pub use_patterns: bool,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            max_length: 500,
            temperature: 0.7,
            use_patterns: true,
        }
    }
}

/// Code generation request
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Type of code to generate
    pub code_type: CodeType,
    /// Context or description
    pub description: String,
    /// Additional constraints
    pub constraints: HashMap<String, String>,
}

/// Generated code result
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    /// Generated code
    pub code: String,
    /// Code type
    pub code_type: CodeType,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Generation metadata
    pub metadata: GenerationMetadata,
}

/// Metadata about code generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    /// Pattern templates used
    pub patterns_used: Vec<String>,
    /// Generation time in milliseconds
    pub generation_time_ms: f64,
    /// Number of tokens generated
    pub token_count: usize,
}

/// AI-powered code generator
#[allow(dead_code)]
pub struct CodeGenerator {
    config: GeneratorConfig,
    template_dir: Option<PathBuf>,
    html_patterns: Vec<HtmlPattern>,
    css_patterns: Vec<CssPattern>,
    js_patterns: Vec<JsPattern>,
}

impl CodeGenerator {
    /// Create a new code generator
    pub fn new(config: GeneratorConfig) -> Self {
        let template_dir = std::env::var("BROWERAI_TEMPLATE_DIR")
            .ok()
            .map(PathBuf::from);

        Self {
            config,
            template_dir: template_dir.clone(),
            html_patterns: Self::load_html_patterns(template_dir.as_ref()),
            css_patterns: Self::load_css_patterns(template_dir.as_ref()),
            js_patterns: Self::load_js_patterns(template_dir.as_ref()),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(GeneratorConfig::default())
    }

    /// Generate code based on request
    pub fn generate(&self, request: &GenerationRequest) -> Result<GeneratedCode> {
        let start = std::time::Instant::now();

        let code = match request.code_type {
            CodeType::Html => self.generate_html(request)?,
            CodeType::Css => self.generate_css(request)?,
            CodeType::JavaScript => self.generate_js(request)?,
            _ => anyhow::bail!("Unsupported code type: {:?}", request.code_type),
        };

        let generation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        let token_count = self.estimate_tokens(&code);

        Ok(GeneratedCode {
            code,
            code_type: request.code_type.clone(),
            confidence: 0.85, // Placeholder - would be from model
            metadata: GenerationMetadata {
                patterns_used: vec!["basic_structure".to_string()],
                generation_time_ms,
                token_count,
            },
        })
    }

    /// Generate HTML from description
    fn generate_html(&self, request: &GenerationRequest) -> Result<String> {
        // Extract keywords from description
        let desc_lower = request.description.to_lowercase();

        // Select appropriate pattern
        let pattern = if desc_lower.contains("form") {
            self.html_patterns.iter().find(|p| p.name == "form")
        } else if desc_lower.contains("table") {
            self.html_patterns.iter().find(|p| p.name == "table")
        } else if desc_lower.contains("list") {
            self.html_patterns.iter().find(|p| p.name == "list")
        } else {
            self.html_patterns.iter().find(|p| p.name == "basic")
        };

        if let Some(pattern) = pattern {
            let defaults: HashMap<String, String> = [
                ("title".to_string(), "Generated Page".to_string()),
                ("heading".to_string(), "Welcome".to_string()),
                ("content".to_string(), "Generated content".to_string()),
                ("action".to_string(), "/submit".to_string()),
                ("method".to_string(), "POST".to_string()),
                ("field".to_string(), "input".to_string()),
                ("label".to_string(), "Input".to_string()),
                ("header1".to_string(), "Column 1".to_string()),
                ("header2".to_string(), "Column 2".to_string()),
                ("data1".to_string(), "Data 1".to_string()),
                ("data2".to_string(), "Data 2".to_string()),
                ("item1".to_string(), "Item 1".to_string()),
                ("item2".to_string(), "Item 2".to_string()),
                ("item3".to_string(), "Item 3".to_string()),
            ]
            .into_iter()
            .collect();

            let rendered = Self::render_with_handlebars(
                &pattern.template,
                &defaults,
                &request.constraints,
            )
            .unwrap_or_else(|e| {
                log::warn!(
                    "Handlebars render failed for HTML pattern {}: {}. Falling back to manual replace.",
                    pattern.name,
                    e
                );
                Self::manual_apply(&pattern.template, &defaults, &request.constraints)
            });

            Ok(rendered)
        } else {
            // Fallback to basic HTML structure
            Ok(format!(
                "<!DOCTYPE html>\n<html>\n<head>\n  <title>{}</title>\n</head>\n<body>\n  <h1>{}</h1>\n  <p>Generated content</p>\n</body>\n</html>",
                request.description,
                request.description
            ))
        }
    }

    /// Generate CSS from description
    fn generate_css(&self, request: &GenerationRequest) -> Result<String> {
        let desc_lower = request.description.to_lowercase();

        let pattern = if desc_lower.contains("button") {
            self.css_patterns.iter().find(|p| p.name == "button")
        } else if desc_lower.contains("card") {
            self.css_patterns.iter().find(|p| p.name == "card")
        } else if desc_lower.contains("layout") {
            self.css_patterns.iter().find(|p| p.name == "layout")
        } else {
            self.css_patterns.iter().find(|p| p.name == "basic")
        };

        if let Some(pattern) = pattern {
            let defaults: HashMap<String, String> = [
                ("font".to_string(), "Arial, sans-serif".to_string()),
                ("padding".to_string(), "20px".to_string()),
                ("background".to_string(), "#ffffff".to_string()),
                ("bg_color".to_string(), "#007bff".to_string()),
                ("text_color".to_string(), "#ffffff".to_string()),
                ("radius".to_string(), "4px".to_string()),
                (
                    "shadow".to_string(),
                    "0 2px 4px rgba(0,0,0,0.1)".to_string(),
                ),
                ("margin".to_string(), "10px".to_string()),
                ("width".to_string(), "1200px".to_string()),
                ("gap".to_string(), "16px".to_string()),
            ]
            .into_iter()
            .collect();

            let rendered = Self::render_with_handlebars(
                &pattern.template,
                &defaults,
                &request.constraints,
            )
            .unwrap_or_else(|e| {
                log::warn!(
                    "Handlebars render failed for CSS pattern {}: {}. Falling back to manual replace.",
                    pattern.name,
                    e
                );
                Self::manual_apply(&pattern.template, &defaults, &request.constraints)
            });

            Ok(rendered)
        } else {
            // Fallback to basic CSS
            Ok("body {\n  font-family: Arial, sans-serif;\n  margin: 0;\n  padding: 20px;\n}\n\nh1 {\n  color: #333;\n}".to_string())
        }
    }

    /// Generate JavaScript from description
    fn generate_js(&self, request: &GenerationRequest) -> Result<String> {
        let desc_lower = request.description.to_lowercase();

        let pattern = if desc_lower.contains("function") {
            self.js_patterns.iter().find(|p| p.name == "function")
        } else if desc_lower.contains("event") || desc_lower.contains("click") {
            self.js_patterns.iter().find(|p| p.name == "event_handler")
        } else if desc_lower.contains("async") || desc_lower.contains("fetch") {
            self.js_patterns.iter().find(|p| p.name == "async")
        } else {
            self.js_patterns.iter().find(|p| p.name == "basic")
        };

        if let Some(pattern) = pattern {
            let defaults: HashMap<String, String> = [
                ("name".to_string(), "generatedFunction".to_string()),
                ("params".to_string(), "data".to_string()),
                (
                    "body".to_string(),
                    "// Process data\n  console.log('Processing:', data);\n  return data;"
                        .to_string(),
                ),
                ("description".to_string(), "Generated function".to_string()),
                ("return_value".to_string(), "data".to_string()),
                ("element_id".to_string(), "element".to_string()),
                ("event".to_string(), "click".to_string()),
                (
                    "handler_body".to_string(),
                    "// Handle event\n  console.log('Event triggered');\n  return true;"
                        .to_string(),
                ),
                ("url".to_string(), "/api/data".to_string()),
                (
                    "process_data".to_string(),
                    "console.log(data); return data;".to_string(),
                ),
            ]
            .into_iter()
            .collect();

            let rendered = Self::render_with_handlebars(
                &pattern.template,
                &defaults,
                &request.constraints,
            )
            .unwrap_or_else(|e| {
                log::warn!(
                    "Handlebars render failed for JS pattern {}: {}. Falling back to manual replace.",
                    pattern.name,
                    e
                );
                Self::manual_apply(&pattern.template, &defaults, &request.constraints)
            });

            Ok(rendered)
        } else {
            // Fallback to basic JS
            Ok("// Generated JavaScript\nfunction main() {\n  console.log('Generated code');\n}\n\nmain();".to_string())
        }
    }

    /// Estimate token count (simple word-based estimation)
    fn estimate_tokens(&self, code: &str) -> usize {
        code.split_whitespace().count()
    }

    fn build_context(
        defaults: &HashMap<String, String>,
        overrides: &HashMap<String, String>,
    ) -> Value {
        let mut map = serde_json::Map::new();

        for (k, v) in defaults {
            map.insert(k.clone(), Value::String(v.clone()));
        }

        for (k, v) in overrides {
            map.insert(k.clone(), Value::String(v.clone()));
        }

        Value::Object(map)
    }

    fn render_with_handlebars(
        template: &str,
        defaults: &HashMap<String, String>,
        overrides: &HashMap<String, String>,
    ) -> Result<String> {
        let context = Self::build_context(defaults, overrides);
        let mut handlebars = Handlebars::new();
        handlebars.set_strict_mode(false);
        handlebars
            .render_template(template, &context)
            .map_err(|e| anyhow::anyhow!("Failed to render template: {}", e))
    }

    fn manual_apply(
        template: &str,
        defaults: &HashMap<String, String>,
        overrides: &HashMap<String, String>,
    ) -> String {
        let mut code = template.to_string();

        for (key, value) in overrides {
            let placeholder = format!("{{{{{}}}}}", key);
            code = code.replace(&placeholder, value);
        }

        for (key, value) in defaults {
            let placeholder = format!("{{{{{}}}}}", key);
            if code.contains(&placeholder) {
                code = code.replace(&placeholder, value);
            }
        }

        code
    }

    fn load_template(
        template_dir: Option<&PathBuf>,
        category: &str,
        name: &str,
        inline: &str,
    ) -> String {
        if let Some(dir) = template_dir {
            let path = dir.join(category).join(format!("{}.hbs", name));
            match std::fs::read_to_string(&path) {
                Ok(content) => return content,
                Err(e) => {
                    log::debug!(
                        "Failed to read template {:?}: {}. Falling back to inline template.",
                        path,
                        e
                    );
                }
            }
        }

        inline.to_string()
    }

    /// Load HTML patterns
    fn load_html_patterns(template_dir: Option<&PathBuf>) -> Vec<HtmlPattern> {
        vec![
            HtmlPattern {
                name: "basic".to_string(),
                template: Self::load_template(
                    template_dir,
                    "html",
                    "basic",
                    "<!DOCTYPE html>\n<html>\n<head>\n  <title>{{title}}</title>\n</head>\n<body>\n  <h1>{{heading}}</h1>\n  <p>{{content}}</p>\n</body>\n</html>",
                ),
            },
            HtmlPattern {
                name: "form".to_string(),
                template: Self::load_template(
                    template_dir,
                    "html",
                    "form",
                    "<form action=\"{{action}}\" method=\"{{method}}\">\n  <label for=\"{{field}}\">{{label}}:</label>\n  <input type=\"text\" id=\"{{field}}\" name=\"{{field}}\">\n  <button type=\"submit\">Submit</button>\n</form>",
                ),
            },
            HtmlPattern {
                name: "table".to_string(),
                template: Self::load_template(
                    template_dir,
                    "html",
                    "table",
                    "<table>\n  <thead>\n    <tr>\n      <th>{{header1}}</th>\n      <th>{{header2}}</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>{{data1}}</td>\n      <td>{{data2}}</td>\n    </tr>\n  </tbody>\n</table>",
                ),
            },
            HtmlPattern {
                name: "list".to_string(),
                template: Self::load_template(
                    template_dir,
                    "html",
                    "list",
                    "<ul>\n  <li>{{item1}}</li>\n  <li>{{item2}}</li>\n  <li>{{item3}}</li>\n</ul>",
                ),
            },
        ]
    }

    /// Load CSS patterns
    fn load_css_patterns(template_dir: Option<&PathBuf>) -> Vec<CssPattern> {
        vec![
            CssPattern {
                name: "basic".to_string(),
                template: Self::load_template(
                    template_dir,
                    "css",
                    "basic",
                    "body {\n  font-family: {{font}};\n  margin: 0;\n  padding: {{padding}};\n  background: {{background}};\n}",
                ),
            },
            CssPattern {
                name: "button".to_string(),
                template: Self::load_template(
                    template_dir,
                    "css",
                    "button",
                    ".button {\n  background: {{bg_color}};\n  color: {{text_color}};\n  padding: {{padding}};\n  border: none;\n  border-radius: {{radius}};\n  cursor: pointer;\n}\n\n.button:hover {\n  opacity: 0.8;\n}",
                ),
            },
            CssPattern {
                name: "card".to_string(),
                template: Self::load_template(
                    template_dir,
                    "css",
                    "card",
                    ".card {\n  background: white;\n  border-radius: {{radius}};\n  box-shadow: {{shadow}};\n  padding: {{padding}};\n  margin: {{margin}};\n}",
                ),
            },
            CssPattern {
                name: "layout".to_string(),
                template: Self::load_template(
                    template_dir,
                    "css",
                    "layout",
                    ".container {\n  max-width: {{width}};\n  margin: 0 auto;\n  padding: {{padding}};\n}\n\n.flex {\n  display: flex;\n  gap: {{gap}};\n}",
                ),
            },
        ]
    }

    /// Load JS patterns
    fn load_js_patterns(template_dir: Option<&PathBuf>) -> Vec<JsPattern> {
        vec![
            JsPattern {
                name: "basic".to_string(),
                template: Self::load_template(
                    template_dir,
                    "js",
                    "basic",
                    "function {{name}}({{params}}) {\n  {{body}}\n}",
                ),
            },
            JsPattern {
                name: "function".to_string(),
                template: Self::load_template(
                    template_dir,
                    "js",
                    "function",
                    "function {{name}}({{params}}) {\n  // {{description}}\n  {{body}}\n  return {{return_value}};\n}",
                ),
            },
            JsPattern {
                name: "event_handler".to_string(),
                template: Self::load_template(
                    template_dir,
                    "js",
                    "event_handler",
                    "document.getElementById('{{element_id}}').addEventListener('{{event}}', function(e) {\n  e.preventDefault();\n  {{handler_body}}\n});",
                ),
            },
            JsPattern {
                name: "async".to_string(),
                template: Self::load_template(
                    template_dir,
                    "js",
                    "async",
                    "async function {{name}}() {\n  try {\n    const response = await fetch('{{url}}');\n    const data = await response.json();\n    {{process_data}}\n  } catch (error) {\n    console.error('Error:', error);\n  }\n}",
                ),
            },
        ]
    }

    /// Update configuration
    pub fn set_config(&mut self, config: GeneratorConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &GeneratorConfig {
        &self.config
    }
}

/// HTML generation pattern
#[derive(Debug, Clone)]
struct HtmlPattern {
    name: String,
    template: String,
}

/// CSS generation pattern
#[derive(Debug, Clone)]
struct CssPattern {
    name: String,
    template: String,
}

/// JS generation pattern
#[derive(Debug, Clone)]
struct JsPattern {
    name: String,
    template: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_generator_creation() {
        let generator = CodeGenerator::with_defaults();
        assert_eq!(generator.config.max_length, 500);
    }

    #[test]
    fn test_generate_html() {
        let generator = CodeGenerator::with_defaults();
        let mut constraints = HashMap::new();
        constraints.insert("title".to_string(), "Test Page".to_string());
        constraints.insert("heading".to_string(), "Welcome".to_string());
        constraints.insert("content".to_string(), "Hello World".to_string());

        let request = GenerationRequest {
            code_type: CodeType::Html,
            description: "basic page".to_string(),
            constraints,
        };

        let result = generator.generate(&request);
        assert!(result.is_ok());
        let generated = result.unwrap();
        assert!(generated.code.contains("Test Page"));
        assert!(generated.code.contains("Welcome"));
    }

    #[test]
    fn test_generate_css() {
        let generator = CodeGenerator::with_defaults();
        let mut constraints = HashMap::new();
        constraints.insert("bg_color".to_string(), "#007bff".to_string());
        constraints.insert("text_color".to_string(), "white".to_string());
        constraints.insert("padding".to_string(), "10px 20px".to_string());
        constraints.insert("radius".to_string(), "5px".to_string());

        let request = GenerationRequest {
            code_type: CodeType::Css,
            description: "button style".to_string(),
            constraints,
        };

        let result = generator.generate(&request);
        assert!(result.is_ok());
        let generated = result.unwrap();
        assert!(generated.code.contains("#007bff"));
        assert!(generated.code.contains("white"));
    }

    #[test]
    fn test_generate_javascript() {
        let generator = CodeGenerator::with_defaults();
        let mut constraints = HashMap::new();
        constraints.insert("name".to_string(), "calculate".to_string());
        constraints.insert("params".to_string(), "x, y".to_string());
        constraints.insert("body".to_string(), "const result = x + y;".to_string());
        constraints.insert("return_value".to_string(), "result".to_string());

        let request = GenerationRequest {
            code_type: CodeType::JavaScript,
            description: "function to calculate".to_string(),
            constraints,
        };

        let result = generator.generate(&request);
        assert!(result.is_ok());
        let generated = result.unwrap();
        assert!(generated.code.contains("calculate"));
        assert!(generated.code.contains("x, y"));
    }

    #[test]
    fn test_generation_metadata() {
        let generator = CodeGenerator::with_defaults();
        let request = GenerationRequest {
            code_type: CodeType::Html,
            description: "test page".to_string(),
            constraints: HashMap::new(),
        };

        let result = generator.generate(&request).unwrap();
        assert!(result.metadata.generation_time_ms >= 0.0);
        assert!(result.metadata.token_count > 0);
    }
}
