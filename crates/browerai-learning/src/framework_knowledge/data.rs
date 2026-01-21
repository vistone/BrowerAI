use super::types::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObfuscationSignature {
    pub name: String,
    pub pattern_type: SignatureType,
    pub pattern: String,
    pub weight: f32,
    pub required: bool,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObfuscationPattern {
    pub name: String,
    pub technique: ObfuscationTechnique,
    pub example_obfuscated: String,
    pub example_deobfuscated: String,
    pub complexity: u8,
    pub prevalence: f32,
    pub detection_hints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeobfuscationStrategy {
    pub name: String,
    pub target: ObfuscationTechnique,
    pub approach: String,
    pub success_rate: f32,
    pub priority: u8,
    pub requirements: Vec<String>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceWeights {
    pub signature_match: f32,
    pub pattern_match: f32,
    pub import_match: f32,
    pub structure_match: f32,
    pub heuristic_match: f32,
}

impl Default for ConfidenceWeights {
    fn default() -> Self {
        Self {
            signature_match: 0.3,
            pattern_match: 0.25,
            import_match: 0.2,
            structure_match: 0.15,
            heuristic_match: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkKnowledge {
    pub id: String,
    pub name: String,
    pub category: FrameworkCategory,
    pub origin: String,
    pub maintainer: String,
    pub signatures: Vec<ObfuscationSignature>,
    pub obfuscation_patterns: Vec<ObfuscationPattern>,
    pub strategies: Vec<DeobfuscationStrategy>,
    pub confidence_weights: ConfidenceWeights,
    pub related_frameworks: Vec<String>,
    pub last_updated: String,
}
