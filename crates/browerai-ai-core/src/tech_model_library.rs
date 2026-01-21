//! Technical Model Library (browser standards, frameworks, obfuscation/deobfuscation)
//! Provides foundational traits and registry for Learn -> Infer -> Generate pipeline.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf};

use crate::inference::InferenceEngine;

/// Tasks this library organizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskKind {
    ParseHtml,
    ParseCss,
    AnalyzeJs,
    DeobfuscateJs,
    LayoutRegenerate,
    StyleSynthesis,
}

/// A registered model specification for a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub version: String,
    pub path: PathBuf,
    pub task: TaskKind,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub seq_len: Option<usize>,
    #[serde(default)]
    pub extras: HashMap<String, String>,
}

/// Registry loaded from models/model_config.toml (or built programmatically)
#[derive(Default, Debug, Clone)]
pub struct ModelRegistry {
    specs: Vec<ModelSpec>,
}

impl ModelRegistry {
    /// Load registry from a TOML configuration file
    pub fn load_from(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let txt = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read model config: {}", path.display()))?;
        // Expect a TOML array of [[models]] with required fields
        #[derive(Deserialize)]
        struct FileModel {
            name: String,
            version: String,
            path: String,
            task: String,
            vocab_size: Option<usize>,
            seq_len: Option<usize>,
            #[serde(default)]
            extras: HashMap<String, String>,
        }
        #[derive(Deserialize)]
        struct Root {
            #[serde(default)]
            models: Vec<FileModel>,
        }
        let root: Root = toml::from_str(&txt).context("Failed to parse model_config.toml")?;
        let mut specs = Vec::new();
        for m in root.models {
            let task = match m.task.to_lowercase().as_str() {
                "parsehtml" | "html" => TaskKind::ParseHtml,
                "parsecss" | "css" => TaskKind::ParseCss,
                "analyzejs" | "js-analyze" => TaskKind::AnalyzeJs,
                "deobfuscatejs" | "js-deobf" | "deobf" => TaskKind::DeobfuscateJs,
                "layoutregenerate" | "layout" => TaskKind::LayoutRegenerate,
                "stylesynthesis" | "style" => TaskKind::StyleSynthesis,
                other => return Err(anyhow::anyhow!(format!("Unknown task kind: {}", other))),
            };
            specs.push(ModelSpec {
                name: m.name,
                version: m.version,
                path: PathBuf::from(m.path),
                task,
                vocab_size: m.vocab_size,
                seq_len: m.seq_len,
                extras: m.extras,
            });
        }
        Ok(Self { specs })
    }

    /// List all model specs
    pub fn list(&self) -> &[ModelSpec] {
        &self.specs
    }

    /// Find models by task kind
    pub fn by_task(&self, task: TaskKind) -> Vec<&ModelSpec> {
        self.specs.iter().filter(|s| s.task == task).collect()
    }

    /// Find preferred model by task (first match)
    pub fn preferred(&self, task: TaskKind) -> Option<&ModelSpec> {
        self.specs.iter().find(|s| s.task == task)
    }
}

/// Learning interface (data preparation / training orchestration) – placeholder
pub trait LearnTask {
    fn task_kind(&self) -> TaskKind;
    fn describe(&self) -> &'static str;
    /// Prepare training data or metadata (no-op placeholder)
    fn prepare_data(&self, _inputs: &HashMap<String, String>) -> Result<()> {
        Ok(())
    }
}

/// Inference interface – runs a model against input content
pub trait InferTask {
    fn task_kind(&self) -> TaskKind;
    fn name(&self) -> &'static str;
    fn infer(&self, engine: &InferenceEngine, input: &str) -> Result<String>;
}

/// Generation interface – compose structure/style with original text
pub trait GenerateTask {
    fn task_kind(&self) -> TaskKind;
    fn name(&self) -> &'static str;
    fn generate_layout(&self, original_text: &str, structure_hint: Option<&str>) -> String;
}

/// Example: JS deobfuscation inference adapter (placeholder)
pub struct JsDeobfuscationInfer;
impl InferTask for JsDeobfuscationInfer {
    fn task_kind(&self) -> TaskKind {
        TaskKind::DeobfuscateJs
    }
    fn name(&self) -> &'static str {
        "js-deobfuscation"
    }
    fn infer(&self, _engine: &InferenceEngine, input: &str) -> Result<String> {
        // This is a placeholder. Real implementation should call ONNX session with proper tokenization
        // and decode policies; here we simply echo a sanitized/trimmed version for scaffolding.
        let trimmed = input.trim();
        Ok(trimmed.to_string())
    }
}

/// Example: Layout regeneration (text passthrough + style wrapper)
pub struct LayoutRegeneration;
impl GenerateTask for LayoutRegeneration {
    fn task_kind(&self) -> TaskKind {
        TaskKind::LayoutRegenerate
    }
    fn name(&self) -> &'static str {
        "layout-regeneration"
    }
    fn generate_layout(&self, original_text: &str, structure_hint: Option<&str>) -> String {
        let text = html_escape(original_text);
        let style = structure_hint
            .map(html_escape)
            .unwrap_or_else(|| "/* default layout */".to_string());
        format!(
            r#"
<section class=\"ai-layout\">
  <style>
    /* model style hint */
    {style}
    .ai-layout {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      color: #1f2933;
      background: #f8fafc;
      padding: 24px; border-radius: 16px;
      box-shadow: 0 12px 40px rgba(15,23,42,.12);
    }}
    .ai-layout .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; }}
    .ai-layout pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; }}
  </style>
  <div class=\"card\"><pre>{text}</pre></div>
</section>
"#
        )
    }
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}
