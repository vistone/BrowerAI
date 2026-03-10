//! 数据表格模式

use crate::*;
use anyhow::Result;

pub struct DataTablePattern;

impl DataTablePattern {
    pub fn new() -> Self {
        Self
    }
}

impl PatternImplementation for DataTablePattern {
    fn pattern_type(&self) -> ComplexPatternType {
        ComplexPatternType::DataTable
    }

    fn recognize(&self, _observations: &[auto_observer::Observation]) -> Option<InteractionPattern> {
        None // TODO
    }

    fn generate_code(&self, _pattern: &InteractionPattern, _language: CodeLanguage) -> Result<GeneratedCode> {
        anyhow::bail!("Not implemented")
    }

    fn get_template(&self) -> &str {
        ""
    }
}
