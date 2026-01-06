use crate::renderer::layout::{BoxType, Dimensions, LayoutBox, Rect};
/// Layout validation and baseline assertions
/// Implements M4 milestone for AI-Centric Execution Refresh
use anyhow::{Context, Result};

/// Validates that a layout conforms to CSS box model rules
#[derive(Debug, Clone)]
pub struct LayoutValidator {
    /// Enable strict validation
    strict: bool,
}

impl Default for LayoutValidator {
    fn default() -> Self {
        Self { strict: false }
    }
}

impl LayoutValidator {
    /// Create a new layout validator
    pub fn new(strict: bool) -> Self {
        Self { strict }
    }

    /// Validate a complete layout tree
    pub fn validate_tree(&self, root: &LayoutBox) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();
        self.validate_box_recursive(root, &mut report)?;
        Ok(report)
    }

    /// Validate a single layout box recursively
    fn validate_box_recursive(
        &self,
        layout_box: &LayoutBox,
        report: &mut ValidationReport,
    ) -> Result<()> {
        // Validate dimensions
        self.validate_dimensions(&layout_box.dimensions, report)?;

        // Validate box type consistency
        self.validate_box_type(&layout_box.box_type, &layout_box.element_type, report)?;

        // Validate children
        for child in &layout_box.children {
            self.validate_box_recursive(child, report)?;
        }

        report.boxes_validated += 1;
        Ok(())
    }

    /// Validate box model dimensions
    fn validate_dimensions(&self, dims: &Dimensions, report: &mut ValidationReport) -> Result<()> {
        // Content box cannot have negative dimensions
        if dims.content.width < 0.0 || dims.content.height < 0.0 {
            report.errors.push(ValidationError {
                kind: ErrorKind::NegativeDimensions,
                message: format!(
                    "Content box has negative dimensions: {}x{}",
                    dims.content.width, dims.content.height
                ),
            });
            if self.strict {
                anyhow::bail!("Negative dimensions found");
            }
        }

        // Padding/border/margin cannot be negative
        if self.has_negative_edges(&dims.padding) {
            report
                .warnings
                .push("Negative padding detected".to_string());
        }
        if self.has_negative_edges(&dims.border) {
            report.warnings.push("Negative border detected".to_string());
        }
        if self.has_negative_edges(&dims.margin) {
            // Negative margins are technically allowed in CSS
            // but we warn about them
            report
                .warnings
                .push("Negative margin detected (allowed but unusual)".to_string());
        }

        // Check box model arithmetic
        let padding_box = dims.padding_box();
        let border_box = dims.border_box();

        // Padding box should contain content box
        if !self.contains_or_equal(&padding_box, &dims.content) {
            report.errors.push(ValidationError {
                kind: ErrorKind::BoxModelViolation,
                message: "Padding box doesn't contain content box".to_string(),
            });
        }

        // Border box should contain padding box
        if !self.contains_or_equal(&border_box, &padding_box) {
            report.errors.push(ValidationError {
                kind: ErrorKind::BoxModelViolation,
                message: "Border box doesn't contain padding box".to_string(),
            });
        }

        Ok(())
    }

    /// Check if any edge size is negative
    fn has_negative_edges(&self, edges: &crate::renderer::layout::EdgeSizes) -> bool {
        edges.top < 0.0 || edges.right < 0.0 || edges.bottom < 0.0 || edges.left < 0.0
    }

    /// Check if outer rect contains or equals inner rect
    fn contains_or_equal(&self, outer: &Rect, inner: &Rect) -> bool {
        outer.x <= inner.x
            && outer.y <= inner.y
            && (outer.x + outer.width) >= (inner.x + inner.width)
            && (outer.y + outer.height) >= (inner.y + inner.height)
    }

    /// Validate box type consistency
    fn validate_box_type(
        &self,
        box_type: &BoxType,
        element_type: &str,
        report: &mut ValidationReport,
    ) -> Result<()> {
        // Check common inconsistencies
        match box_type {
            BoxType::Inline => {
                // Inline elements shouldn't be certain types
                if element_type == "div" || element_type == "section" || element_type == "article" {
                    report.warnings.push(format!(
                        "Block-level element '{}' rendered as inline",
                        element_type
                    ));
                }
            }
            BoxType::Block => {
                // Block rendering is generally fine for most elements
            }
            _ => {
                // Other types are more specialized
            }
        }

        Ok(())
    }
}

/// AI layout hints for optimization
#[derive(Debug, Clone)]
pub struct AiLayoutHint {
    /// Hint type
    pub hint_type: LayoutHintType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Description
    pub description: String,
}

/// Types of AI layout hints
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutHintType {
    /// Suggest using flexbox layout
    UseFlexbox,
    /// Suggest using grid layout
    UseGrid,
    /// Suggest collapsing margins
    CollapseMargins,
    /// Suggest absolute positioning
    UseAbsolutePositioning,
    /// Suggest fixed positioning
    UseFixedPositioning,
    /// Custom hint
    Custom(String),
}

impl AiLayoutHint {
    /// Create a new layout hint
    pub fn new(hint_type: LayoutHintType, confidence: f32, description: String) -> Self {
        Self {
            hint_type,
            confidence,
            description,
        }
    }

    /// Stub for future AI model integration
    pub fn generate_hints_for_element(_element_type: &str) -> Vec<Self> {
        // Future: Use AI model to suggest layout optimizations
        // For now, return empty - stub for future implementation
        Vec::new()
    }
}

/// Validation report
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// Number of boxes validated
    pub boxes_validated: usize,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get total issue count
    pub fn issue_count(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub kind: ErrorKind,
    pub message: String,
}

/// Error kinds
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    NegativeDimensions,
    BoxModelViolation,
    InvalidBoxType,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::layout::EdgeSizes;

    #[test]
    fn test_validator_creation() {
        let validator = LayoutValidator::default();
        assert!(!validator.strict);

        let strict_validator = LayoutValidator::new(true);
        assert!(strict_validator.strict);
    }

    #[test]
    fn test_validate_valid_box() {
        let validator = LayoutValidator::default();
        let layout_box = LayoutBox {
            box_type: BoxType::Block,
            dimensions: Dimensions {
                content: Rect::new(0.0, 0.0, 100.0, 50.0),
                padding: EdgeSizes {
                    top: 10.0,
                    right: 10.0,
                    bottom: 10.0,
                    left: 10.0,
                },
                border: EdgeSizes::default(),
                margin: EdgeSizes::default(),
            },
            children: Vec::new(),
            element_type: "div".to_string(),
        };

        let report = validator.validate_tree(&layout_box).unwrap();
        assert!(report.is_valid());
        assert_eq!(report.boxes_validated, 1);
    }

    #[test]
    fn test_validate_negative_dimensions() {
        let validator = LayoutValidator::default();
        let layout_box = LayoutBox {
            box_type: BoxType::Block,
            dimensions: Dimensions {
                content: Rect::new(0.0, 0.0, -100.0, 50.0),
                padding: EdgeSizes::default(),
                border: EdgeSizes::default(),
                margin: EdgeSizes::default(),
            },
            children: Vec::new(),
            element_type: "div".to_string(),
        };

        let report = validator.validate_tree(&layout_box).unwrap();
        assert!(!report.is_valid());
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].kind, ErrorKind::NegativeDimensions);
    }

    #[test]
    fn test_validate_negative_padding() {
        let validator = LayoutValidator::default();
        let layout_box = LayoutBox {
            box_type: BoxType::Block,
            dimensions: Dimensions {
                content: Rect::new(0.0, 0.0, 100.0, 50.0),
                padding: EdgeSizes {
                    top: -10.0,
                    right: 10.0,
                    bottom: 10.0,
                    left: 10.0,
                },
                border: EdgeSizes::default(),
                margin: EdgeSizes::default(),
            },
            children: Vec::new(),
            element_type: "div".to_string(),
        };

        let report = validator.validate_tree(&layout_box).unwrap();
        assert!(report.warnings.len() > 0);
    }

    #[test]
    fn test_validate_nested_boxes() {
        let validator = LayoutValidator::default();
        let child_box = LayoutBox {
            box_type: BoxType::Inline,
            dimensions: Dimensions {
                content: Rect::new(10.0, 10.0, 50.0, 20.0),
                padding: EdgeSizes::default(),
                border: EdgeSizes::default(),
                margin: EdgeSizes::default(),
            },
            children: Vec::new(),
            element_type: "span".to_string(),
        };

        let parent_box = LayoutBox {
            box_type: BoxType::Block,
            dimensions: Dimensions {
                content: Rect::new(0.0, 0.0, 100.0, 50.0),
                padding: EdgeSizes::default(),
                border: EdgeSizes::default(),
                margin: EdgeSizes::default(),
            },
            children: vec![child_box],
            element_type: "div".to_string(),
        };

        let report = validator.validate_tree(&parent_box).unwrap();
        assert_eq!(report.boxes_validated, 2);
    }

    #[test]
    fn test_ai_layout_hint_creation() {
        let hint = AiLayoutHint::new(
            LayoutHintType::UseFlexbox,
            0.85,
            "Container would benefit from flexbox layout".to_string(),
        );
        assert_eq!(hint.confidence, 0.85);
        assert_eq!(hint.hint_type, LayoutHintType::UseFlexbox);
    }

    #[test]
    fn test_ai_layout_hint_stub() {
        let hints = AiLayoutHint::generate_hints_for_element("div");
        // Currently returns empty as it's a stub
        assert_eq!(hints.len(), 0);
    }

    #[test]
    fn test_validation_report_is_valid() {
        let mut report = ValidationReport::default();
        assert!(report.is_valid());

        report.errors.push(ValidationError {
            kind: ErrorKind::NegativeDimensions,
            message: "Test error".to_string(),
        });
        assert!(!report.is_valid());
    }

    #[test]
    fn test_validation_report_issue_count() {
        let mut report = ValidationReport::default();
        assert_eq!(report.issue_count(), 0);

        report.errors.push(ValidationError {
            kind: ErrorKind::NegativeDimensions,
            message: "Test error".to_string(),
        });
        report.warnings.push("Test warning".to_string());
        assert_eq!(report.issue_count(), 2);
    }
}
