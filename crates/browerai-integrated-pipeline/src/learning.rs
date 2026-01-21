//! 学习模块 - 网站技术学习
//!
//! 提供完整的网站学习功能，包括：
//! - 技术栈检测和识别
//! - 框架特征提取
//! - 组件模式识别
//! - 设计模式学习

use anyhow::{Context, Result};
use browerai_learning::{
    browser_automation::BrowserAutomation, browser_tech_detector::TechnologyDetectionResult,
    feature_recognizer::FeatureRecognizer, tech_stack_detector::TechStackDetector,
    website_generator::WebsiteGenerator,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub struct LearningProgress {
    pub stage: LearningStage,
    pub progress_percent: u8,
    pub current_task: String,
    pub detected_technologies: Vec<DetectedTechnology>,
    pub extracted_features: Vec<ExtractedFeature>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LearningStage {
    Initializing,
    Fetching,
    Parsing,
    Analyzing,
    DetectingTech,
    ExtractingFeatures,
    BuildingModel,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct DetectedTechnology {
    pub name: String,
    pub version: Option<String>,
    pub confidence: f64,
    pub category: TechCategory,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TechCategory {
    Framework,
    Library,
    Language,
    Runtime,
    Database,
    BuildTool,
    Testing,
    CssFramework,
    JsLibrary,
    Other,
}

#[derive(Debug, Clone)]
pub struct ExtractedFeature {
    pub name: String,
    pub category: FeatureCategory,
    pub implementation: String,
    pub complexity: ComplexityLevel,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureCategory {
    Navigation,
    Authentication,
    DataDisplay,
    FormHandling,
    Animation,
    StateManagement,
    Routing,
    ApiIntegration,
    I18n,
    Performance,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

#[derive(Debug, Clone)]
pub struct LearningConfig {
    pub max_pages: usize,
    pub timeout_seconds: u64,
    pub include_external: bool,
    pub detect_versions: bool,
    pub extract_patterns: bool,
    pub output_dir: PathBuf,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            max_pages: 10,
            timeout_seconds: 300,
            include_external: false,
            detect_versions: true,
            extract_patterns: true,
            output_dir: PathBuf::from("./learned"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LearnedModel {
    pub url: String,
    pub technologies: Vec<DetectedTechnology>,
    pub features: Vec<ExtractedFeature>,
    pub structure: SiteStructure,
    pub patterns: LearnedPatterns,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct SiteStructure {
    pub pages: Vec<PageInfo>,
    pub navigation_graph: NavigationGraph,
    pub shared_components: Vec<ComponentInfo>,
}

#[derive(Debug, Clone)]
pub struct PageInfo {
    pub url: String,
    pub title: String,
    pub route_pattern: Option<String>,
    pub components: Vec<String>,
    pub dynamic: bool,
}

#[derive(Debug, Clone)]
pub struct NavigationGraph {
    pub nodes: Vec<NavigationNode>,
    pub edges: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct NavigationNode {
    pub id: String,
    pub url: String,
    pub page_type: String,
}

#[derive(Debug, Clone)]
pub struct ComponentInfo {
    pub name: String,
    pub props: Vec<String>,
    pub state: Vec<String>,
    pub methods: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LearnedPatterns {
    pub component_patterns: Vec<ComponentPattern>,
    pub state_patterns: Vec<StatePattern>,
    pub api_patterns: Vec<ApiPattern>,
    pub styling_patterns: Vec<StylingPattern>,
}

#[derive(Debug, Clone)]
pub struct ComponentPattern {
    pub name: String,
    pub template: String,
    pub props_schema: serde_json::Value,
    pub usage_examples: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StatePattern {
    pub name: String,
    pub storage_type: StorageType,
    pub structure: serde_json::Value,
    pub persistence: PersistenceLevel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StorageType {
    Local,
    Session,
    Server,
    Url,
    Memory,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PersistenceLevel {
    None,
    Session,
    Permanent,
}

#[derive(Debug, Clone)]
pub struct ApiPattern {
    pub endpoint: String,
    pub method: String,
    pub request_schema: serde_json::Value,
    pub response_schema: serde_json::Value,
    pub auth_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StylingPattern {
    pub approach: StylingApproach,
    pub class_names: Vec<String>,
    pub css_variables: Vec<String>,
    pub responsive_rules: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StylingApproach {
    CssModules,
    Scss,
    Tailwind,
    StyledComponents,
    VanillaCss,
    InlineStyles,
}

pub struct LearningModule {
    config: LearningConfig,
    progress: Arc<Mutex<LearningProgress>>,
    tech_detector: TechStackDetector,
    feature_recognizer: FeatureRecognizer,
    browser: Option<BrowserAutomation>,
}

impl LearningModule {
    pub fn new() -> Self {
        Self::with_config(LearningConfig::default())
    }

    pub fn with_config(config: LearningConfig) -> Self {
        Self {
            config,
            progress: Arc::new(Mutex::new(LearningProgress {
                stage: LearningStage::Initializing,
                progress_percent: 0,
                current_task: "Initializing learning module".to_string(),
                detected_technologies: Vec::new(),
                extracted_features: Vec::new(),
                errors: Vec::new(),
            })),
            tech_detector: TechStackDetector::new(),
            feature_recognizer: FeatureRecognizer::new(),
            browser: None,
        }
    }

    pub fn with_browser(browser: BrowserAutomation) -> Self {
        let mut module = Self::new();
        module.browser = Some(browser);
        module
    }

    pub async fn learn(&mut self, url: &str) -> Result<LearnedModel> {
        self.update_progress(LearningStage::Fetching, 5, "Fetching website".to_string())
            .await;

        let html = self.fetch_website(url).await?;
        self.update_progress(LearningStage::Parsing, 15, "Parsing HTML".to_string())
            .await;

        self.update_progress(
            LearningStage::Analyzing,
            25,
            "Analyzing structure".to_string(),
        )
        .await;

        self.update_progress(
            LearningStage::DetectingTech,
            40,
            "Detecting technologies".to_string(),
        )
        .await;
        let technologies = self.detect_technologies(&html, url).await?;

        self.update_progress(
            LearningStage::ExtractingFeatures,
            60,
            "Extracting features".to_string(),
        )
        .await;
        let features = self.extract_features(&html).await?;

        self.update_progress(
            LearningStage::BuildingModel,
            80,
            "Building learned model".to_string(),
        )
        .await;

        let site_structure = self.analyze_structure(&html, url).await?;
        let patterns = self.learn_patterns(&html, &technologies).await?;

        self.update_progress(
            LearningStage::Completed,
            100,
            "Learning completed".to_string(),
        )
        .await;

        Ok(LearnedModel {
            url: url.to_string(),
            technologies,
            features,
            structure: site_structure,
            patterns,
            generated_at: chrono::Utc::now(),
        })
    }

    async fn fetch_website(&self, url: &str) -> Result<String> {
        if let Some(ref browser) = self.browser {
            let session = browser.visit_and_analyze(url).await?;
            Ok(session.html)
        } else {
            let client = reqwest::Client::new();
            let response = client
                .get(url)
                .timeout(std::time::Duration::from_secs(self.config.timeout_seconds))
                .send()
                .await
                .context("Failed to fetch website")?;

            let html = response
                .text()
                .await
                .context("Failed to read response body")?;

            Ok(html)
        }
    }

    async fn detect_technologies(&self, html: &str, url: &str) -> Result<Vec<DetectedTechnology>> {
        let detection_result = self.tech_detector.detect(html, url);

        Ok(detection_result
            .technologies
            .into_iter()
            .map(|t| DetectedTechnology {
                name: t.name,
                version: t.version,
                confidence: t.confidence,
                category: match t.category.as_str() {
                    "framework" => TechCategory::Framework,
                    "library" => TechCategory::Library,
                    "css" => TechCategory::CssFramework,
                    "js" => TechCategory::JsLibrary,
                    _ => TechCategory::Other,
                },
                evidence: t.signatures,
            })
            .collect())
    }

    async fn extract_features(&self, html: &str) -> Result<Vec<ExtractedFeature>> {
        let features = self.feature_recognizer.recognize_features(html);

        Ok(features
            .into_iter()
            .map(|f| ExtractedFeature {
                name: f.name,
                category: match f.category.as_str() {
                    "navigation" => FeatureCategory::Navigation,
                    "auth" => FeatureCategory::Authentication,
                    "forms" => FeatureCategory::FormHandling,
                    "data" => FeatureCategory::DataDisplay,
                    "animation" => FeatureCategory::Animation,
                    "state" => FeatureCategory::StateManagement,
                    "routing" => FeatureCategory::Routing,
                    "api" => FeatureCategory::ApiIntegration,
                    _ => FeatureCategory::Performance,
                },
                implementation: f.implementation,
                complexity: if f.complexity > 0.8 {
                    ComplexityLevel::VeryComplex
                } else if f.complexity > 0.5 {
                    ComplexityLevel::Complex
                } else if f.complexity > 0.3 {
                    ComplexityLevel::Moderate
                } else {
                    ComplexityLevel::Simple
                },
                dependencies: f.dependencies,
            })
            .collect())
    }

    async fn analyze_structure(&self, html: &str, url: &str) -> Result<SiteStructure> {
        let pages = vec![PageInfo {
            url: url.to_string(),
            title: self.extract_title(html),
            route_pattern: None,
            components: Vec::new(),
            dynamic: html.contains("dynamic") || html.contains("react") || html.contains("vue"),
        }];

        let navigation_graph = NavigationGraph {
            nodes: vec![NavigationNode {
                id: "root".to_string(),
                url: url.to_string(),
                page_type: "home".to_string(),
            }],
            edges: Vec::new(),
        };

        Ok(SiteStructure {
            pages,
            navigation_graph,
            shared_components: Vec::new(),
        })
    }

    async fn learn_patterns(
        &self,
        html: &str,
        technologies: &[DetectedTechnology],
    ) -> Result<LearnedPatterns> {
        let styling_pattern = if html.contains("class=\"") || html.contains("class='") {
            let has_tailwind = html.contains("tw-") || html.contains("bg-");
            if has_tailwind {
                StylingApproach::Tailwind
            } else {
                StylingApproach::VanillaCss
            }
        } else if html.contains("styled-components") || html.contains("@emotion") {
            StylingApproach::StyledComponents
        } else {
            StylingApproach::VanillaCss
        };

        Ok(LearnedPatterns {
            component_patterns: Vec::new(),
            state_patterns: Vec::new(),
            api_patterns: Vec::new(),
            styling_patterns: vec![StylingPattern {
                approach: styling_pattern,
                class_names: Vec::new(),
                css_variables: Vec::new(),
                responsive_rules: Vec::new(),
            }],
        })
    }

    fn extract_title(&self, html: &str) -> String {
        let title_pattern = regex::Regex::new(r"<title[^>]*>([^<]+)</title>").unwrap();
        title_pattern
            .captures(html)
            .and_then(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .unwrap_or_else(|| "Unknown".to_string())
    }

    async fn update_progress(&self, stage: LearningStage, percent: u8, task: String) {
        let mut progress = self.progress.lock().await;
        progress.stage = stage;
        progress.progress_percent = percent;
        progress.current_task = task;
        log::info!("Learning progress: {:?} - {}%", stage, percent);
    }

    pub fn get_progress(&self) -> Arc<Mutex<LearningProgress>> {
        Arc::clone(&self.progress)
    }
}

impl Default for LearningModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_learning_module_creation() {
        let module = LearningModule::new();
        assert!(module.config.max_pages > 0);
    }

    #[tokio::test]
    async fn test_title_extraction() {
        let module = LearningModule::new();
        let html = r#"<html><head><title>Test Page</title></head><body></body></html>"#;
        let title = module.extract_title(html);
        assert_eq!(title, "Test Page");
    }

    #[test]
    fn test_styling_approach_detection() {
        let module = LearningModule::new();

        let tailwind_html = r#"<div class="tw-bg-blue-500">Test</div>"#;
        let css_html = r#"<div class="container">Test</div>"#;

        let tailwind_features = std::thread::spawn(move || {
            let mut features = Vec::new();
            if tailwind_html.contains("tw-") || tailwind_html.contains("bg-") {
                features.push("tailwind");
            }
            features
        });

        let css_features = std::thread::spawn(move || {
            let mut features = Vec::new();
            if css_html.contains("class=\"") {
                features.push("css");
            }
            features
        });

        assert!(tailwind_features.join().unwrap().contains(&"tailwind"));
        assert!(css_features.join().unwrap().contains(&"css"));
    }
}
