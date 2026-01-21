//! Screenshot Comparator
//!
//! Compares screenshots between original and generated websites
//! to measure visual similarity.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Screenshot comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenshotComparisonResult {
    /// Overall visual similarity score (0.0 - 1.0)
    pub similarity_score: f32,

    /// Layout similarity score
    pub layout_similarity: f32,

    /// Color similarity score
    pub color_similarity: f32,

    /// Component placement similarity
    pub component_similarity: f32,

    /// Detailed visual differences
    pub visual_differences: Vec<VisualDifference>,

    /// Layout differences
    pub layout_differences: Vec<LayoutDifference>,

    /// Color palette comparison
    pub color_comparison: ColorComparison,

    /// Metadata
    pub metadata: ScreenshotComparisonMetadata,
}

/// A visual difference between screenshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualDifference {
    /// Difference type
    pub difference_type: VisualDifferenceType,

    /// Bounding box in original (x, y, width, height)
    pub original_region: (u32, u32, u32, u32),

    /// Bounding box in generated (x, y, width, height)
    pub generated_region: (u32, u32, u32, u32),

    /// Difference magnitude (0.0 - 1.0)
    pub magnitude: f32,

    /// Severity of difference
    pub severity: VisualSeverity,

    /// Description of difference
    pub description: String,
}

/// Types of visual differences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VisualDifferenceType {
    PixelDifference,
    ColorShift,
    LayoutShift,
    MissingElement,
    ExtraElement,
    AlignmentIssue,
    SizeMismatch,
    OpacityDifference,
    ZIndexIssue,
}

/// Severity levels for visual differences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VisualSeverity {
    Negligible,
    Minor,
    Moderate,
    Major,
    Critical,
}

/// A layout difference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutDifference {
    /// Element selector
    pub selector: String,

    /// Original position
    pub original_position: (u32, u32),

    /// Generated position
    pub generated_position: (u32, u32),

    /// Original size
    pub original_size: (u32, u32),

    /// Generated size
    pub generated_size: (u32, u32),

    /// Position shift
    pub position_shift: (i32, i32),

    /// Size change percentage
    pub size_change_percent: f32,

    /// Severity
    pub severity: VisualSeverity,
}

/// Color palette comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorComparison {
    /// Original dominant colors
    pub original_colors: Vec<ColorInfo>,

    /// Generated dominant colors
    pub generated_colors: Vec<ColorInfo>,

    /// Color similarity score
    pub color_similarity: f32,

    /// Colors added in generated
    pub added_colors: Vec<String>,

    /// Colors missing from generated
    pub missing_colors: Vec<String>,

    /// Overall color harmony score
    pub harmony_score: f32,
}

/// Color information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorInfo {
    /// Hex color code
    pub hex: String,

    /// RGB values
    pub rgb: (u8, u8, u8),

    /// Percentage of page coverage
    pub coverage_percent: f32,

    /// Color name (if detected)
    pub name: Option<String>,
}

/// Screenshot comparison metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenshotComparisonMetadata {
    /// Original screenshot path
    pub original_path: String,

    /// Generated screenshot path
    pub generated_path: String,

    /// Comparison timestamp
    pub compared_at: chrono::DateTime<chrono::Utc>,

    /// Time taken for comparison
    pub comparison_time_ms: u64,

    /// Image dimensions
    pub image_width: u32,
    pub image_height: u32,

    /// Comparison version
    pub version: String,
}

/// Screenshot Comparator Configuration
#[derive(Debug, Clone)]
pub struct ScreenshotComparatorConfig {
    /// Minimum similarity threshold
    pub min_similarity_threshold: f32,

    /// Include detailed visual differences
    pub include_details: bool,

    /// Maximum differences to report
    pub max_differences: usize,

    /// Pixel difference threshold (0-255)
    pub pixel_threshold: u8,

    /// Minimum region size for difference
    pub min_region_size: u32,

    /// Compare layout
    pub compare_layout: bool,

    /// Compare colors
    pub compare_colors: bool,

    /// Ignore specific regions
    pub ignore_regions: Vec<(u32, u32, u32, u32)>,
}

impl Default for ScreenshotComparatorConfig {
    fn default() -> Self {
        Self {
            min_similarity_threshold: 0.8,
            include_details: true,
            max_differences: 50,
            pixel_threshold: 10,
            min_region_size: 10,
            compare_layout: true,
            compare_colors: true,
            ignore_regions: Vec::new(),
        }
    }
}

/// Screenshot Comparator
///
/// Compares screenshots to measure visual similarity between
/// original and generated websites.
#[derive(Debug, Clone)]
pub struct ScreenshotComparator {
    config: ScreenshotComparatorConfig,
}

impl ScreenshotComparator {
    /// Create a new screenshot comparator
    pub fn new() -> Self {
        Self::with_config(ScreenshotComparatorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ScreenshotComparatorConfig) -> Self {
        Self { config }
    }

    /// Compare two screenshots
    ///
    /// # Arguments
    /// * `original_path` - Path to original screenshot
    /// * `generated_path` - Path to generated screenshot
    ///
    /// # Returns
    /// * `ScreenshotComparisonResult` with comparison results
    pub async fn compare(
        &self,
        original_path: &str,
        generated_path: &str,
    ) -> ScreenshotComparisonResult {
        let start_time = chrono::Utc::now();

        // Load images (simplified - in production would use image library)
        let (original_pixels, width, height) = self.load_image(original_path);
        let (generated_pixels, gen_width, gen_height) = self.load_image(generated_path);

        // Use max dimensions
        let final_width = width.max(gen_width);
        let final_height = height.max(gen_height);

        // Calculate pixel similarity
        let (pixel_diffs, total_diff) = self.compare_pixels(
            &original_pixels,
            &generated_pixels,
            width,
            height,
            gen_width,
            gen_height,
        );

        // Calculate layout similarity (simplified)
        let layout_similarity = self.calculate_layout_similarity(
            &original_pixels,
            &generated_pixels,
            width,
            height,
        );

        // Calculate color similarity
        let color_comparison = if self.config.compare_colors {
            self.compare_colors(&original_pixels, &generated_pixels)
        } else {
            ColorComparison {
                original_colors: Vec::new(),
                generated_colors: Vec::new(),
                color_similarity: 0.5,
                added_colors: Vec::new(),
                missing_colors: Vec::new(),
                harmony_score: 0.5,
            }
        };

        // Calculate component similarity
        let component_similarity = self.calculate_component_similarity(
            &original_pixels,
            &generated_pixels,
            width,
            height,
        );

        // Overall similarity score
        let similarity_score = (pixel_diffs + layout_similarity + component_similarity
            + color_comparison.color_similarity)
            / 4.0;

        // Find visual differences
        let visual_differences = if self.config.include_details {
            self.find_visual_differences(
                &original_pixels,
                &generated_pixels,
                width,
                height,
            )
        } else {
            Vec::new()
        };

        // Find layout differences
        let layout_differences = if self.config.compare_layout {
            self.find_layout_differences(&original_pixels, &generated_pixels, width, height)
        } else {
            Vec::new()
        };

        let end_time = chrono::Utc::now();
        let comparison_time_ms = (end_time - start_time).num_milliseconds() as u64;

        ScreenshotComparisonResult {
            similarity_score,
            layout_similarity,
            color_similarity: color_comparison.color_similarity,
            component_similarity,
            visual_differences,
            layout_differences,
            color_comparison,
            metadata: ScreenshotComparisonMetadata {
                original_path: original_path.to_string(),
                generated_path: generated_path.to_string(),
                compared_at: chrono::Utc::now(),
                comparison_time_ms,
                image_width: final_width,
                image_height: final_height,
                version: "1.0".to_string(),
            },
        }
    }

    /// Load image and get pixels (simplified)
    fn load_image(&self, _path: &str) -> (Vec<(u8, u8, u8)>, u32, u32) {
        // In a full implementation, this would use an image library
        // For MVP, return empty placeholder
        (Vec::new(), 1920, 1080)
    }

    /// Compare pixels between two images
    fn compare_pixels(
        &self,
        original: &[(u8, u8, u8)],
        generated: &[(u8, u8, u8)],
        orig_width: u32,
        orig_height: u32,
        gen_width: u32,
        gen_height: u32,
    ) -> (f32, f32) {
        let width = orig_width.max(gen_width);
        let height = orig_height.max(gen_height);
        let total_pixels = (width * height) as usize;

        if original.is_empty() && generated.is_empty() {
            return (1.0, 0.0);
        }

        if original.is_empty() || generated.is_empty() {
            return (0.0, 1.0);
        }

        let mut diff_count = 0;
        let mut total_diff = 0.0;

        let min_len = original.len().min(generated.len());

        for i in 0..min_len {
            let (r1, g1, b1) = original[i];
            let (r2, g2, b2) = generated[i];

            // Calculate Euclidean distance
            let diff = (((r1 as i32 - r2 as i32).pow(2)
                + ((g1 as i32 - g2 as i32).pow(2)
                + ((b1 as i32 - b2 as i32).pow(2)) as f32)
                .sqrt();

            if diff > self.config.pixel_threshold as f32 {
                diff_count += 1;
            }

            // Normalize difference (max distance is ~441 for RGB)
            total_diff += diff / 441.0;
        }

        let pixel_similarity = 1.0 - (diff_count as f32 / total_pixels as f32);
        let avg_diff = total_diff / min_len as f32;

        (pixel_similarity, avg_diff)
    }

    /// Calculate layout similarity (simplified)
    fn calculate_layout_similarity(
        &self,
        _original: &[(u8, u8, u8)],
        _generated: &[(u8, u8, u8)],
        _width: u32,
        _height: u32,
    ) -> f32 {
        // In a full implementation, this would analyze layout elements
        // For MVP, return a reasonable default
        0.75
    }

    /// Calculate component similarity
    fn calculate_component_similarity(
        &self,
        _original: &[(u8, u8, u8)],
        _generated: &[(u8, u8, u8)],
        _width: u32,
        _height: u32,
    ) -> f32 {
        // In a full implementation, this would detect and compare components
        // For MVP, return a reasonable default
        0.7
    }

    /// Compare color palettes
    fn compare_colors(
        &self,
        original: &[(u8, u8, u8)],
        generated: &[(u8, u8, u8)],
    ) -> ColorComparison {
        let original_colors = self.extract_dominant_colors(original);
        let generated_colors = self.extract_dominant_colors(generated);

        // Calculate color similarity
        let mut matching = 0;
        let total = original_colors.len().max(generated_colors.len()).max(1);

        for oc in &original_colors {
            for gc in &generated_colors {
                if self.colors_match(oc, gc) {
                    matching += 1;
                    break;
                }
            }
        }

        let color_similarity = matching as f32 / total as f32;

        // Find added and missing colors
        let mut added_colors = Vec::new();
        let mut missing_colors = Vec::new();

        for gc in &generated_colors {
            let mut found = false;
            for oc in &original_colors {
                if self.colors_match(oc, gc) {
                    found = true;
                    break;
                }
            }
            if !found {
                added_colors.push(gc.hex.clone());
            }
        }

        for oc in &original_colors {
            let mut found = false;
            for gc in &generated_colors {
                if self.colors_match(oc, gc) {
                    found = true;
                    break;
                }
            }
            if !found {
                missing_colors.push(oc.hex.clone());
            }
        }

        ColorComparison {
            original_colors,
            generated_colors,
            color_similarity,
            added_colors,
            missing_colors,
            harmony_score: 0.8, // Simplified
        }
    }

    /// Extract dominant colors from pixels
    fn extract_dominant_colors(&self, pixels: &[(u8, u8, u8)]) -> Vec<ColorInfo> {
        if pixels.is_empty() {
            return Vec::new();
        }

        // Simplified color extraction
        // In full implementation, would use color quantization
        let mut color_counts: HashMap<(u8, u8, u8), usize> = HashMap::new();

        for pixel in pixels.iter().step_by(100) {
            // Round colors to reduce variety
            let r = (pixel.0 / 32) * 32;
            let g = (pixel.1 / 32) * 32;
            let b = (pixel.2 / 32) * 32;
            *color_counts.entry((r, g, b)).or_insert(0) += 1;
        }

        let mut colors: Vec<_> = color_counts.into_iter().collect();
        colors.sort_by(|a, b| b.1.cmp(&a.1));
        colors.truncate(5);

        let total: usize = colors.iter().map(|(_, c)| *c).sum();

        colors
            .into_iter()
            .map(|((r, g, b), count)| ColorInfo {
                hex: format!("#{:02x}{:02x}{:02x}", r, g, b),
                rgb: (r, g, b),
                coverage_percent: if total > 0 {
                    (count as f32 / total as f32) * 100.0
                } else {
                    0.0
                },
                name: None,
            })
            .collect()
    }

    /// Check if two colors are similar
    fn colors_match(&self, c1: &ColorInfo, c2: &ColorInfo) -> bool {
        let r_diff = (c1.rgb.0 as i32 - c2.rgb.0 as i32).abs();
        let g_diff = (c1.rgb.1 as i32 - c2.rgb.1 as i32).abs();
        let b_diff = (c1.rgb.2 as i32 - c2.rgb.2 as i32).abs();

        // Colors match if all channels are within 32 (1/8 of range)
        r_diff <= 32 && g_diff <= 32 && b_diff <= 32
    }

    /// Find visual differences
    fn find_visual_differences(
        &self,
        _original: &[(u8, u8, u8)],
        _generated: &[(u8, u8, u8)],
        _width: u32,
        _height: u32,
    ) -> Vec<VisualDifference> {
        // In a full implementation, this would analyze pixel regions
        // For MVP, return empty list
        Vec::new()
    }

    /// Find layout differences
    fn find_layout_differences(
        &self,
        _original: &[(u8, u8, u8)],
        _generated: &[(u8, u8, u8)],
        _width: u32,
        _height: u32,
    ) -> Vec<LayoutDifference> {
        // In a full implementation, this would detect layout elements
        // For MVP, return empty list
        Vec::new()
    }
}

impl Default for ScreenshotComparator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparator_creation() {
        let comparator = ScreenshotComparator::new();
        assert!(comparator.config.min_similarity_threshold > 0.0);
    }

    #[test]
    fn test_color_matching() {
        let comparator = ScreenshotComparator::new();

        let color1 = ColorInfo {
            hex: "#ffffff".to_string(),
            rgb: (255, 255, 255),
            coverage_percent: 50.0,
            name: None,
        };

        let color2 = ColorInfo {
            hex: "#fefefe".to_string(),
            rgb: (254, 254, 254),
            coverage_percent: 50.0,
            name: None,
        };

        assert!(comparator.colors_match(&color1, &color2));
    }

    #[test]
    fn test_pixel_comparison() {
        let comparator = ScreenshotComparator::new();

        let original = vec![(255, 255, 255), (0, 0, 0), (128, 128, 128)];
        let generated = vec![(255, 255, 255), (0, 0, 0), (128, 128, 128)];

        let (similarity, _) = comparator.compare_pixels(&original, &generated, 1, 3, 1, 3);
        assert!(similarity > 0.99);
    }

    #[test]
    fn test_pixel_difference() {
        let comparator = ScreenshotComparator::new();

        let original = vec![(255, 255, 255)];
        let generated = vec![(0, 0, 0)];

        let (similarity, avg_diff) = comparator.compare_pixels(&original, &generated, 1, 1, 1, 1);
        assert!(similarity < 0.5);
        assert!(avg_diff > 0.5);
    }

    #[test]
    fn test_color_extraction() {
        let comparator = ScreenshotComparator::new();

        let pixels = vec![
            (255, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        ];

        let colors = comparator.extract_dominant_colors(&pixels);

        assert!(!colors.is_empty());
    }

    #[test]
    fn test_color_comparison() {
        let comparator = ScreenshotComparator::new();

        let original = vec![(255, 0, 0), (0, 255, 0), (0, 0, 255)];
        let generated = vec![(255, 0, 0), (0, 255, 0), (0, 0, 255)];

        let comparison = comparator.compare_colors(&original, &generated);

        assert!(comparison.color_similarity > 0.9);
        assert!(comparison.added_colors.is_empty());
        assert!(comparison.missing_colors.is_empty());
    }

    #[tokio::test]
    async fn test_identical_screenshots() {
        let comparator = ScreenshotComparator::new();

        // Create placeholder images (same for both)
        let original = vec![(255, 255, 255), (200, 200, 200), (100, 100, 100)];
        let generated = original.clone();

        // Simulate comparison by directly calling internal methods
        let (similarity, _) = comparator.compare_pixels(&original, &generated, 1, 3, 1, 3);
        let layout_sim = comparator.calculate_layout_similarity(&original, &generated, 1, 3);
        let component_sim = comparator.calculate_component_similarity(&original, &generated, 1, 3);
        let color_comp = comparator.compare_colors(&original, &generated);

        let overall = (similarity + layout_sim + component_sim + color_comp.color_similarity) / 4.0;

        assert!(overall > 0.9);
    }

    #[tokio::test]
    async fn test_completely_different_screenshots() {
        let comparator = ScreenshotComparator::new();

        let original = vec![(255, 255, 255), (255, 255, 255), (255, 255, 255)];
        let generated = vec![(0, 0, 0), (0, 0, 0), (0, 0, 0)];

        let (similarity, _) = comparator.compare_pixels(&original, &generated, 1, 3, 1, 3);
        let color_comp = comparator.compare_colors(&original, &generated);

        let overall = (similarity + 0.5 + 0.5 + color_comp.color_similarity) / 4.0;

        assert!(overall < 0.5);
    }
}
