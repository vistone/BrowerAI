use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod data;
pub mod types;
pub use data::*;
pub use types::*;

pub const FRAMEWORK_KNOWLEDGE_BASE_VERSION: &str = "1.0.0";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub framework_id: String,
    pub framework_name: String,
    pub confidence: f32,
    pub matched_signatures: Vec<String>,
    pub matched_patterns: Vec<String>,
    pub detected_techniques: Vec<ObfuscationTechnique>,
    pub recommended_strategies: Vec<DeobfuscationStrategy>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KnowledgeBaseStats {
    pub total_frameworks: usize,
    pub total_signatures: usize,
    pub total_patterns: usize,
    pub total_strategies: usize,
    pub categories: HashMap<FrameworkCategory, usize>,
    pub last_updated: String,
}

impl FrameworkKnowledge {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn category(&self) -> &FrameworkCategory {
        &self.category
    }
}

pub trait FrameworkKnowledgeProvider {
    fn get_all_frameworks(&self) -> Vec<&FrameworkKnowledge>;
    fn get_frameworks_by_category(&self, category: FrameworkCategory) -> Vec<&FrameworkKnowledge>;
    fn get_framework(&self, id: &str) -> Option<&FrameworkKnowledge>;
    fn detect_framework(&self, code: &str) -> Vec<DetectionResult>;
    fn get_knowledge_base_stats(&self) -> KnowledgeBaseStats;
}

fn create_react_signatures() -> Vec<ObfuscationSignature> {
    vec![
        ObfuscationSignature {
            name: "React imports".to_string(),
            pattern_type: SignatureType::ImportStatement,
            pattern: "import\\s+.*\\s+from\\s+[\"']?(react|@?react/)[\"']?".to_string(),
            weight: 0.9,
            required: true,
            context: "import statements".to_string(),
        },
        ObfuscationSignature {
            name: "JSX syntax".to_string(),
            pattern_type: SignatureType::ASTPattern,
            pattern: r"<[A-Z][a-zA-Z0-9]*[^>]*>.*</[A-Z][a-zA-Z0-9]*>".to_string(),
            weight: 0.8,
            required: false,
            context: "JSX tags".to_string(),
        },
    ]
}

fn create_vue_signatures() -> Vec<ObfuscationSignature> {
    vec![
        ObfuscationSignature {
            name: "Vue imports".to_string(),
            pattern_type: SignatureType::ImportStatement,
            pattern: "import\\s+.*\\s+from\\s+[\"']?(vue|@?vue/)[\"']?".to_string(),
            weight: 0.9,
            required: false,
            context: "import statements".to_string(),
        },
        ObfuscationSignature {
            name: "Vue global object".to_string(),
            pattern_type: SignatureType::GlobalObject,
            pattern: r"\{\s*(createApp|ref|reactive|computed|watch|onMounted|setup)\s*\}\s*=\s*Vue"
                .to_string(),
            weight: 0.85,
            required: false,
            context: "Vue global object destructuring".to_string(),
        },
        ObfuscationSignature {
            name: "Vue template".to_string(),
            pattern_type: SignatureType::ASTPattern,
            pattern: r"(v-if|v-for|v-bind|v-on|@click|:)[^>]*=".to_string(),
            weight: 0.85,
            required: false,
            context: "Vue directives".to_string(),
        },
        ObfuscationSignature {
            name: "Vue internal functions".to_string(),
            pattern_type: SignatureType::ASTPattern,
            pattern: r"(_createElementVNode|_toDisplayString|_openBlock|_createVNode)".to_string(),
            weight: 0.7,
            required: false,
            context: "Vue compiled template functions".to_string(),
        },
    ]
}

pub fn create_knowledge_base() -> Vec<FrameworkKnowledge> {
    vec![
        FrameworkKnowledge {
            id: "react".to_string(),
            name: "React".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "USA".to_string(),
            maintainer: "Meta".to_string(),
            signatures: create_react_signatures(),
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "React production build".to_string(),
                technique: ObfuscationTechnique::Minification,
                example_obfuscated: "var n=e.exports,r=t.exports,o=require(\"react\");".to_string(),
                example_deobfuscated: "import React, { useState } from 'react';".to_string(),
                complexity: 3,
                prevalence: 0.7,
                detection_hints: vec![
                    "Short variable names".to_string(),
                    "CommonJS require".to_string(),
                ],
            }],
            strategies: vec![DeobfuscationStrategy {
                name: "React import recovery".to_string(),
                target: ObfuscationTechnique::StringEncoding,
                approach: "Identify React imports and recover variable names from context"
                    .to_string(),
                success_rate: 0.85,
                priority: 5,
                requirements: vec!["Bundler knowledge".to_string()],
                limitations: vec!["Minified variable names may be ambiguous".to_string()],
            }],
            confidence_weights: ConfidenceWeights::default(),
            related_frameworks: vec!["nextjs".to_string(), "gatsby".to_string()],
            last_updated: "2024-01-01".to_string(),
        },
        FrameworkKnowledge {
            id: "vue".to_string(),
            name: "Vue.js".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "China".to_string(),
            maintainer: "Evan You".to_string(),
            signatures: create_vue_signatures(),
            obfuscation_patterns: vec![ObfuscationPattern {
                name: "Vue SFC compilation".to_string(),
                technique: ObfuscationTechnique::TemplateCompilation,
                example_obfuscated: "_vm._v(\"Hello\")".to_string(),
                example_deobfuscated: "<div>Hello</div>".to_string(),
                complexity: 4,
                prevalence: 0.6,
                detection_hints: vec!["_vm prefix".to_string(), "_v function calls".to_string()],
            }],
            strategies: vec![],
            confidence_weights: ConfidenceWeights::default(),
            related_frameworks: vec!["nuxtjs".to_string(), "vite".to_string()],
            last_updated: "2024-01-01".to_string(),
        },
    ]
}

pub struct FrameworkKnowledgeBase {
    knowledge_base: Vec<FrameworkKnowledge>,
}

impl FrameworkKnowledgeBase {
    pub fn new() -> Self {
        Self {
            knowledge_base: create_knowledge_base(),
        }
    }

    pub fn get_frameworks(&self) -> &[FrameworkKnowledge] {
        &self.knowledge_base
    }

    pub fn get_framework(&self, id: &str) -> Option<&FrameworkKnowledge> {
        self.knowledge_base.iter().find(|f| f.id == id)
    }

    pub fn framework_count(&self) -> usize {
        self.knowledge_base.len()
    }

    pub fn get_knowledge_base_stats(&self) -> KnowledgeBaseStats {
        let mut categories = HashMap::new();
        for framework in &self.knowledge_base {
            *categories.entry(framework.category.clone()).or_insert(0) += 1;
        }
        KnowledgeBaseStats {
            total_frameworks: self.knowledge_base.len(),
            total_signatures: self.knowledge_base.iter().map(|f| f.signatures.len()).sum(),
            total_patterns: self
                .knowledge_base
                .iter()
                .map(|f| f.obfuscation_patterns.len())
                .sum(),
            total_strategies: self.knowledge_base.iter().map(|f| f.strategies.len()).sum(),
            categories,
            last_updated: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn get_frameworks_by_category(
        &self,
        category: &FrameworkCategory,
    ) -> Vec<&FrameworkKnowledge> {
        self.knowledge_base
            .iter()
            .filter(|f| &f.category == category)
            .collect()
    }

    pub fn analyze_code(&self, code: &str) -> Vec<DetectionResult> {
        self.detect(code)
    }

    pub fn detect(&self, code: &str) -> Vec<DetectionResult> {
        let mut results = Vec::new();

        for framework in &self.knowledge_base {
            let mut matched_signatures = Vec::new();
            let mut confidence = 0.0f32;

            for sig in &framework.signatures {
                if let Ok(re) = regex::Regex::new(&sig.pattern) {
                    if re.is_match(code) {
                        matched_signatures.push(sig.name.clone());
                        confidence += sig.weight;
                    }
                } else if code.contains(&sig.pattern) {
                    matched_signatures.push(sig.name.clone());
                    confidence += sig.weight;
                }
            }

            if !matched_signatures.is_empty() {
                confidence = (confidence / framework.signatures.len() as f32).min(1.0);
                results.push(DetectionResult {
                    framework_id: framework.id.clone(),
                    framework_name: framework.name.clone(),
                    confidence,
                    matched_signatures,
                    matched_patterns: Vec::new(),
                    detected_techniques: Vec::new(),
                    recommended_strategies: framework.strategies.clone(),
                    metadata: HashMap::new(),
                });
            }
        }

        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

impl Default for FrameworkKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_knowledge_creation() {
        let framework = FrameworkKnowledge {
            id: "test".to_string(),
            name: "Test Framework".to_string(),
            category: FrameworkCategory::FrontendFramework,
            origin: "Test".to_string(),
            maintainer: "Test".to_string(),
            signatures: vec![],
            obfuscation_patterns: vec![],
            strategies: vec![],
            confidence_weights: ConfidenceWeights::default(),
            related_frameworks: vec![],
            last_updated: "2024-01-01".to_string(),
        };
        assert_eq!(framework.id, "test");
    }
}
