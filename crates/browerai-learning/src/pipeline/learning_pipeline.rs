//! Learning Pipeline
//!
//! Orchestrates the complete learning flow from URL input to website generation.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::data_models::{BehaviorRecord, PageContent};
use crate::generators::{CssGenerator, HtmlGenerator, JsGenerator, OutputFormatter};
use crate::learning_sandbox::{IntentAnalyzer, WebsiteIntent};
use crate::safe_sandbox::{BehaviorRecorder, PageFetcher};
use crate::validation::WebsiteValidator;

/// Progress event callback type
pub type ProgressCallback = dyn Fn(ProgressEvent) + Send + Sync;

/// Learning input
#[derive(Debug, Clone)]
pub struct LearningInput {
    /// Target URL to learn from
    pub url: String,

    /// Optional user preferences for personalization
    pub preferences: Option<UserPreferences>,

    /// Whether to record behavior
    pub record_behavior: bool,

    /// Whether to enable validation
    pub enable_validation: bool,
}

/// User preferences for personalization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Color scheme preference
    pub color_scheme: Option<String>,

    /// Font preference
    pub font_preference: Option<String>,

    /// Layout density
    pub layout_density: Option<f32>,

    /// Dark mode
    pub dark_mode: bool,
}

/// Learning output
#[derive(Debug, Clone)]
pub struct LearningOutput {
    /// Whether learning was successful
    pub success: bool,

    /// Output bundle
    pub output: OutputBundle,

    /// Intent analysis result
    pub intent: WebsiteIntent,

    /// Behavior record (if recorded)
    pub behavior_record: Option<BehaviorRecord>,

    /// Validation report (if validation enabled)
    pub validation_report: Option<ValidationReport>,

    /// Processing metadata
    pub metadata: LearningMetadata,
}

/// Output bundle containing generated files
#[derive(Debug, Clone)]
pub struct OutputBundle {
    /// Output directory path
    pub directory: String,

    /// Generated files
    pub files: Vec<OutputFile>,
}

/// Output file information
#[derive(Debug, Clone)]
pub struct OutputFile {
    /// File name
    pub name: String,

    /// File type
    pub file_type: FileType,

    /// Content
    pub content: String,
}

#[derive(Debug, Clone)]
pub enum FileType {
    Html,
    Css,
    JavaScript,
    Json,
}

/// Validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Overall pass/fail
    pub passed: bool,

    /// Overall score (0.0 - 1.0)
    pub score: f32,

    /// Individual checks
    pub checks: Vec<ValidationCheck>,
}

/// Validation check result
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub message: String,
    pub score: f32,
}

/// Processing metadata
#[derive(Debug, Clone)]
pub struct LearningMetadata {
    /// Source URL
    pub source_url: String,

    /// Processing start time
    pub start_time: chrono::DateTime<chrono::Utc>,

    /// Processing end time
    pub end_time: chrono::DateTime<chrono::Utc>,

    /// Total duration in milliseconds
    pub duration_ms: u64,
}

/// Progress event
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    StageStarted(String),
    StageProgress(String, f32),
    StageCompleted(String),
    Error(String),
    Info(String),
}

/// Learning pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Page fetcher configuration
    pub fetch_timeout_seconds: u64,

    /// Maximum retries on failure
    pub max_retries: usize,

    /// Whether to use incremental learning
    pub use_incremental: bool,

    /// Output directory
    pub output_directory: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            fetch_timeout_seconds: 30,
            max_retries: 3,
            use_incremental: false,
            output_directory: "./output".to_string(),
        }
    }
}

/// Learning Pipeline
///
/// Orchestrates the complete learning flow.
#[derive(Clone)]
#[allow(dead_code)]
pub struct LearningPipeline {
    config: PipelineConfig,

    // Core components
    page_fetcher: Arc<PageFetcher>,
    behavior_recorder: Arc<BehaviorRecorder>,
    intent_analyzer: Arc<IntentAnalyzer>,
    html_generator: Arc<HtmlGenerator>,
    css_generator: Arc<CssGenerator>,
    js_generator: Arc<JsGenerator>,
    validator: Arc<WebsiteValidator>,
    output_formatter: Arc<OutputFormatter>,

    // State (not included in Debug)
    #[doc(hidden)]
    progress_callback: Option<Arc<ProgressCallback>>,
}

impl LearningPipeline {
    /// Create a new learning pipeline with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(PipelineConfig::default())
    }

    /// Create a learning pipeline with custom configuration
    pub fn with_config(config: PipelineConfig) -> Result<Self> {
        let page_fetcher = Arc::new(PageFetcher::new()?);
        let behavior_recorder = Arc::new(BehaviorRecorder::new());
        let intent_analyzer = Arc::new(IntentAnalyzer::new());
        let html_generator = Arc::new(HtmlGenerator::new());
        let css_generator = Arc::new(CssGenerator::new());
        let js_generator = Arc::new(JsGenerator::new());
        let validator = Arc::new(WebsiteValidator::new());
        let output_formatter = Arc::new(OutputFormatter::new(&config.output_directory)?);

        Ok(Self {
            config,
            page_fetcher,
            behavior_recorder,
            intent_analyzer,
            html_generator,
            css_generator,
            js_generator,
            validator,
            output_formatter,
            progress_callback: None,
        })
    }

    /// Set progress callback
    pub fn set_progress_callback(&mut self, callback: Arc<ProgressCallback>) {
        self.progress_callback = Some(callback);
    }

    /// Emit progress event
    fn emit_progress(&self, event: ProgressEvent) {
        if let Some(ref callback) = self.progress_callback {
            callback(event);
        }
    }

    /// Run the complete learning pipeline
    ///
    /// # Arguments
    /// * `input` - Learning input containing URL and options
    ///
    /// # Returns
    /// * `LearningOutput` containing the results
    pub async fn run(&self, input: &LearningInput) -> Result<LearningOutput> {
        let start_time = chrono::Utc::now();
        self.emit_progress(ProgressEvent::StageStarted("fetching".to_string()));

        // Stage 1: Fetch page content
        let page_content = self
            .fetch_page(&input.url)
            .await
            .context("Failed to fetch page")?;

        self.emit_progress(ProgressEvent::StageCompleted("fetching".to_string()));
        self.emit_progress(ProgressEvent::StageStarted("analyzing".to_string()));

        // Stage 2: Record behavior (if enabled)
        let behavior_record = if input.record_behavior {
            self.record_behavior(&page_content).await
        } else {
            None
        };

        // Stage 3: Analyze intent
        let intent = self.intent_analyzer.analyze(&page_content);

        self.emit_progress(ProgressEvent::StageCompleted("analyzing".to_string()));
        self.emit_progress(ProgressEvent::StageStarted("generating".to_string()));

        // Stage 4: Generate code
        let generated = self.generate_code(&intent, &behavior_record).await?;

        self.emit_progress(ProgressEvent::StageCompleted("generating".to_string()));

        // Stage 5: Validate (if enabled)
        let validation_report = if input.enable_validation {
            self.validate(&page_content, &generated).await
        } else {
            None
        };

        // Stage 6: Format output
        self.emit_progress(ProgressEvent::StageStarted("formatting".to_string()));
        let output = self.format_output(generated, &input.preferences).await?;
        self.emit_progress(ProgressEvent::StageCompleted("formatting".to_string()));

        let end_time = chrono::Utc::now();
        let duration_ms = (end_time - start_time).num_milliseconds() as u64;

        let metadata = LearningMetadata {
            source_url: input.url.clone(),
            start_time,
            end_time,
            duration_ms,
        };

        Ok(LearningOutput {
            success: true,
            output,
            intent,
            behavior_record,
            validation_report,
            metadata,
        })
    }

    /// Fetch page content
    async fn fetch_page(&self, url: &str) -> Result<PageContent> {
        let mut retries = 0;
        loop {
            match self.page_fetcher.fetch(url).await {
                Ok(content) => return Ok(content),
                Err(e) => {
                    if retries < self.config.max_retries {
                        retries += 1;
                        self.emit_progress(ProgressEvent::Info(format!(
                            "Retry {}/{} after error: {}",
                            retries, self.config.max_retries, e
                        )));
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// Record behavior
    async fn record_behavior(&self, page: &PageContent) -> Option<BehaviorRecord> {
        // In a full implementation, this would:
        // 1. Inject instrumentation script into the page
        // 2. Execute the page in a controlled environment (V8)
        // 3. Capture all API calls, state changes, events

        // For MVP, we'll create an empty behavior record
        let mut recorder = BehaviorRecorder::new();
        recorder.start_recording(&page.url).await;
        Some(recorder.stop_recording().await)
    }

    /// Generate code based on intent
    async fn generate_code(
        &self,
        intent: &WebsiteIntent,
        _behavior: &Option<BehaviorRecord>,
    ) -> Result<GeneratedWebsite> {
        // Generate HTML
        let html = self.html_generator.generate(intent).await?;

        // Generate CSS
        let css = self.css_generator.generate(intent).await?;

        // Generate JavaScript
        let js = self.js_generator.generate(intent).await?;

        Ok(GeneratedWebsite {
            html,
            css,
            js,
            assets: Vec::new(),
        })
    }

    /// Validate generated output
    async fn validate(
        &self,
        original: &PageContent,
        generated: &GeneratedWebsite,
    ) -> Option<ValidationReport> {
        self.validator.validate(original, generated).await
    }

    /// Format output
    async fn format_output(
        &self,
        website: GeneratedWebsite,
        preferences: &Option<UserPreferences>,
    ) -> Result<OutputBundle> {
        self.output_formatter.format(website, preferences).await
    }
}

impl Default for LearningPipeline {
    fn default() -> Self {
        LearningPipeline::new().unwrap()
    }
}

/// Placeholder for generated website
#[derive(Debug, Clone)]
pub struct GeneratedWebsite {
    pub html: String,
    pub css: String,
    pub js: String,
    pub assets: Vec<GeneratedAsset>,
}

/// Placeholder for generated asset
#[derive(Debug, Clone)]
pub struct GeneratedAsset {
    pub name: String,
    pub content: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = LearningPipeline::new().unwrap();
        assert!(pipeline.config.max_retries > 0);
    }

    #[test]
    fn test_pipeline_config() {
        let config = PipelineConfig::default();
        assert_eq!(config.fetch_timeout_seconds, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_learning_input() {
        let input = LearningInput {
            url: "https://example.com".to_string(),
            preferences: None,
            record_behavior: true,
            enable_validation: true,
        };
        assert_eq!(input.url, "https://example.com");
    }
}
