//! 树形视图模式

use crate::*;
use anyhow::Result;

pub struct TreeViewPattern;

impl TreeViewPattern {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TreeViewPattern {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternImplementation for TreeViewPattern {
    fn pattern_type(&self) -> ComplexPatternType {
        ComplexPatternType::TreeView
    }

    fn recognize(
        &self,
        _observations: &[auto_observer::Observation],
    ) -> Option<InteractionPattern> {
        None // TODO
    }

    fn generate_code(
        &self,
        _pattern: &InteractionPattern,
        _language: CodeLanguage,
    ) -> Result<GeneratedCode> {
        anyhow::bail!("Not implemented")
    }

    fn get_template(&self) -> &str {
        ""
    }
}
