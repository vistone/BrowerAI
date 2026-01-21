//! Output Formatter
//!
//! Formats and writes the generated website to files.

use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::pipeline::{FileType, GeneratedWebsite, OutputBundle, OutputFile, UserPreferences};

/// Output formatter configuration
#[derive(Debug, Clone)]
pub struct OutputFormatterConfig {
    /// Output directory
    pub output_dir: String,

    /// Create timestamped directory
    pub timestamp_dir: bool,

    /// Include manifest file
    pub include_manifest: bool,
}

impl Default for OutputFormatterConfig {
    fn default() -> Self {
        Self {
            output_dir: "./output".to_string(),
            timestamp_dir: true,
            include_manifest: true,
        }
    }
}

/// Output Formatter
///
/// Formats and writes the generated website to files.
#[derive(Debug, Clone)]
pub struct OutputFormatter {
    config: OutputFormatterConfig,
}

impl OutputFormatter {
    /// Create a new output formatter
    pub fn new(output_dir: &str) -> Result<Self> {
        let config = OutputFormatterConfig {
            output_dir: output_dir.to_string(),
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create with custom configuration
    pub fn with_config(config: OutputFormatterConfig) -> Result<Self> {
        // Ensure output directory exists
        fs::create_dir_all(&config.output_dir).context("Failed to create output directory")?;
        Ok(Self { config })
    }

    /// Format and write the generated website
    pub async fn format(
        &self,
        website: GeneratedWebsite,
        preferences: &Option<UserPreferences>,
    ) -> Result<OutputBundle> {
        // Determine output directory
        let output_dir = if self.config.timestamp_dir {
            let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            PathBuf::from(&self.config.output_dir).join(format!("generated_{}", timestamp))
        } else {
            PathBuf::from(&self.config.output_dir)
        };

        // Create directory structure
        fs::create_dir_all(&output_dir).context("Failed to create output directory")?;

        let css_dir = output_dir.join("css");
        let js_dir = output_dir.join("js");
        let assets_dir = output_dir.join("assets");

        fs::create_dir_all(&css_dir).context("Failed to create CSS directory")?;
        fs::create_dir_all(&js_dir).context("Failed to create JS directory")?;
        fs::create_dir_all(&assets_dir).context("Failed to create assets directory")?;

        // Apply preferences if provided
        let (html, css) = if let Some(prefs) = preferences {
            self.apply_preferences(&website, prefs)
        } else {
            (website.html.clone(), website.css.clone())
        };

        // Write HTML
        let html_path = output_dir.join("index.html");
        fs::write(&html_path, &html).context("Failed to write HTML file")?;

        // Write CSS
        let css_path = css_dir.join("main.css");
        fs::write(&css_path, &css).context("Failed to write CSS file")?;

        // Write JS
        let js_path = js_dir.join("main.js");
        fs::write(&js_path, &website.js).context("Failed to write JS file")?;

        // Write manifest if enabled
        let manifest_path = if self.config.include_manifest {
            let manifest = self.generate_manifest(&website);
            let path = output_dir.join("manifest.json");
            fs::write(&path, &manifest).context("Failed to write manifest file")?;
            Some(path)
        } else {
            None
        };

        // Build output bundle
        let mut files = Vec::new();

        files.push(OutputFile {
            name: "index.html".to_string(),
            file_type: FileType::Html,
            content: html,
        });

        files.push(OutputFile {
            name: "main.css".to_string(),
            file_type: FileType::Css,
            content: css,
        });

        files.push(OutputFile {
            name: "main.js".to_string(),
            file_type: FileType::JavaScript,
            content: website.js,
        });

        if let Some(path) = manifest_path {
            let content = fs::read_to_string(&path).context("Failed to read manifest")?;
            files.push(OutputFile {
                name: "manifest.json".to_string(),
                file_type: FileType::Json,
                content,
            });
        }

        Ok(OutputBundle {
            directory: output_dir.to_string_lossy().to_string(),
            files,
        })
    }

    /// Apply user preferences to styles
    fn apply_preferences(
        &self,
        website: &GeneratedWebsite,
        preferences: &UserPreferences,
    ) -> (String, String) {
        let mut html = website.html.clone();
        let mut css = website.css.clone();

        // Apply color scheme
        if let Some(color_scheme) = &preferences.color_scheme {
            css = css.replace(
                "--primary-color: #007bff;",
                &format!("--primary-color: {};", color_scheme),
            );
        }

        // Apply font preference
        if let Some(font) = &preferences.font_preference {
            css = css.replace(
                "--font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;",
                &format!("--font-family: {};", font),
            );
        }

        // Apply layout density
        if let Some(density) = preferences.layout_density {
            let padding = format!("{}rem", density * 2.0);
            css = css.replace("padding: 0 20px;", &format!("padding: 0 {};", padding));
        }

        // Apply dark mode
        if preferences.dark_mode {
            css = css.replace("background-color: #ffffff;", "background-color: #1a1a1a;");
            css = css.replace("color: #333333;", "color: #e0e0e0;");
            // Update header and footer
            html = html.replace(
                "<header class=\"header\">",
                "<header class=\"header\" data-dark=\"true\">",
            );
        }

        (html, css)
    }

    /// Generate manifest.json
    fn generate_manifest(&self, website: &GeneratedWebsite) -> String {
        let manifest = Manifest {
            name: format!("Generated Website - {}", website.html.len()),
            short_name: "Generated Site".to_string(),
            version: "1.0.0".to_string(),
            description: "A website generated by BrowerAI".to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            features: Vec::new(),
            styling: StylingInfo {
                colors: None,
                fonts: None,
            },
            files: FilesInfo {
                html: true,
                css: true,
                js: true,
                assets: !website.assets.is_empty(),
            },
        };

        serde_json::to_string_pretty(&manifest).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Manifest structure
#[derive(Debug, Serialize)]
struct Manifest {
    name: String,
    short_name: String,
    version: String,
    description: String,
    generated_at: String,
    features: Vec<String>,
    styling: StylingInfo,
    files: FilesInfo,
}

#[derive(Debug, Serialize)]
struct StylingInfo {
    colors: Option<HashMap<String, String>>,
    fonts: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct FilesInfo {
    html: bool,
    css: bool,
    js: bool,
    assets: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_output_formatter_creation() {
        let temp_dir = TempDir::new().unwrap();
        let formatter = OutputFormatter::new(temp_dir.path().to_str().unwrap()).unwrap();
        assert!(formatter.config.include_manifest);
    }

    #[test]
    fn test_format_output() {
        let temp_dir = TempDir::new().unwrap();
        let formatter = OutputFormatter::new(temp_dir.path().to_str().unwrap()).unwrap();

        let website = GeneratedWebsite {
            html: "<!DOCTYPE html><html><body>Test</body></html>".to_string(),
            css: "body { margin: 0; }".to_string(),
            js: "console.log('test');".to_string(),
            assets: Vec::new(),
        };

        let output = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(formatter.format(website, &None))
            .unwrap();

        assert!(output.directory.contains("generated_"));
        assert_eq!(output.files.len(), 4); // HTML, CSS, JS, manifest

        // Check files exist
        let html_path = PathBuf::from(&output.directory).join("index.html");
        let css_path = PathBuf::from(&output.directory).join("css/main.css");
        let js_path = PathBuf::from(&output.directory).join("js/main.js");

        assert!(html_path.exists());
        assert!(css_path.exists());
        assert!(js_path.exists());
    }

    #[test]
    fn test_dark_mode_preference() {
        let temp_dir = TempDir::new().unwrap();
        let formatter = OutputFormatter::new(temp_dir.path().to_str().unwrap()).unwrap();

        let website = GeneratedWebsite {
            html: "<header class=\"header\">Test</header>".to_string(),
            css: "body { background-color: #ffffff; color: #333333; }".to_string(),
            js: "".to_string(),
            assets: Vec::new(),
        };

        let prefs = UserPreferences {
            color_scheme: None,
            font_preference: None,
            layout_density: None,
            dark_mode: true,
        };

        let (html, css) = formatter.apply_preferences(&website, &prefs);

        assert!(html.contains("data-dark=\"true\""));
        assert!(css.contains("background-color: #1a1a1a;"));
        assert!(css.contains("color: #e0e0e0;"));
    }
}
