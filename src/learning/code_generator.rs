/// AI-powered code generation module
/// 
/// Provides capabilities to generate simple HTML/CSS/JS based on learned patterns

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

/// Type of code to generate
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeType {
    Html,
    Css,
    JavaScript,
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
pub struct CodeGenerator {
    config: GeneratorConfig,
    html_patterns: Vec<HtmlPattern>,
    css_patterns: Vec<CssPattern>,
    js_patterns: Vec<JsPattern>,
}

impl CodeGenerator {
    /// Create a new code generator
    pub fn new(config: GeneratorConfig) -> Self {
        Self {
            config,
            html_patterns: Self::load_html_patterns(),
            css_patterns: Self::load_css_patterns(),
            js_patterns: Self::load_js_patterns(),
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
            // Apply constraints to customize the pattern
            let mut code = pattern.template.clone();
            
            // Replace placeholders
            for (key, value) in &request.constraints {
                let placeholder = format!("{{{{{}}}}}", key);
                code = code.replace(&placeholder, value);
            }
            
            Ok(code)
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
            let mut code = pattern.template.clone();
            
            // Replace placeholders
            for (key, value) in &request.constraints {
                let placeholder = format!("{{{{{}}}}}", key);
                code = code.replace(&placeholder, value);
            }
            
            Ok(code)
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
            let mut code = pattern.template.clone();
            
            // Replace placeholders
            for (key, value) in &request.constraints {
                let placeholder = format!("{{{{{}}}}}", key);
                code = code.replace(&placeholder, value);
            }
            
            Ok(code)
        } else {
            // Fallback to basic JS
            Ok("// Generated JavaScript\nfunction main() {\n  console.log('Generated code');\n}\n\nmain();".to_string())
        }
    }

    /// Estimate token count (simple word-based estimation)
    fn estimate_tokens(&self, code: &str) -> usize {
        code.split_whitespace().count()
    }

    /// Load HTML patterns
    fn load_html_patterns() -> Vec<HtmlPattern> {
        vec![
            HtmlPattern {
                name: "basic".to_string(),
                template: "<!DOCTYPE html>\n<html>\n<head>\n  <title>{{title}}</title>\n</head>\n<body>\n  <h1>{{heading}}</h1>\n  <p>{{content}}</p>\n</body>\n</html>".to_string(),
            },
            HtmlPattern {
                name: "form".to_string(),
                template: "<form action=\"{{action}}\" method=\"{{method}}\">\n  <label for=\"{{field}}\">{{label}}:</label>\n  <input type=\"text\" id=\"{{field}}\" name=\"{{field}}\">\n  <button type=\"submit\">Submit</button>\n</form>".to_string(),
            },
            HtmlPattern {
                name: "table".to_string(),
                template: "<table>\n  <thead>\n    <tr>\n      <th>{{header1}}</th>\n      <th>{{header2}}</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>{{data1}}</td>\n      <td>{{data2}}</td>\n    </tr>\n  </tbody>\n</table>".to_string(),
            },
            HtmlPattern {
                name: "list".to_string(),
                template: "<ul>\n  <li>{{item1}}</li>\n  <li>{{item2}}</li>\n  <li>{{item3}}</li>\n</ul>".to_string(),
            },
        ]
    }

    /// Load CSS patterns
    fn load_css_patterns() -> Vec<CssPattern> {
        vec![
            CssPattern {
                name: "basic".to_string(),
                template: "body {\n  font-family: {{font}};\n  margin: 0;\n  padding: {{padding}};\n  background: {{background}};\n}".to_string(),
            },
            CssPattern {
                name: "button".to_string(),
                template: ".button {\n  background: {{bg_color}};\n  color: {{text_color}};\n  padding: {{padding}};\n  border: none;\n  border-radius: {{radius}};\n  cursor: pointer;\n}\n\n.button:hover {\n  opacity: 0.8;\n}".to_string(),
            },
            CssPattern {
                name: "card".to_string(),
                template: ".card {\n  background: white;\n  border-radius: {{radius}};\n  box-shadow: {{shadow}};\n  padding: {{padding}};\n  margin: {{margin}};\n}".to_string(),
            },
            CssPattern {
                name: "layout".to_string(),
                template: ".container {\n  max-width: {{width}};\n  margin: 0 auto;\n  padding: {{padding}};\n}\n\n.flex {\n  display: flex;\n  gap: {{gap}};\n}".to_string(),
            },
        ]
    }

    /// Load JS patterns
    fn load_js_patterns() -> Vec<JsPattern> {
        vec![
            JsPattern {
                name: "basic".to_string(),
                template: "function {{name}}({{params}}) {\n  {{body}}\n}".to_string(),
            },
            JsPattern {
                name: "function".to_string(),
                template: "function {{name}}({{params}}) {\n  // {{description}}\n  {{body}}\n  return {{return_value}};\n}".to_string(),
            },
            JsPattern {
                name: "event_handler".to_string(),
                template: "document.getElementById('{{element_id}}').addEventListener('{{event}}', function(e) {\n  e.preventDefault();\n  {{handler_body}}\n});".to_string(),
            },
            JsPattern {
                name: "async".to_string(),
                template: "async function {{name}}() {\n  try {\n    const response = await fetch('{{url}}');\n    const data = await response.json();\n    {{process_data}}\n  } catch (error) {\n    console.error('Error:', error);\n  }\n}".to_string(),
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
