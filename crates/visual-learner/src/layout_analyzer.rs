//! 布局分析器
//! 分析页面的布局结构

use crate::*;
use anyhow::Result;

/// 布局分析器
pub struct LayoutAnalyzer {
    config: VisualLearningConfig,
}

impl LayoutAnalyzer {
    pub fn new(config: &VisualLearningConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// 获取布局分析器配置
    pub fn config(&self) -> &VisualLearningConfig {
        &self.config
    }

    /// 分析布局
    pub fn analyze_layout(&self, image: &image::DynamicImage, components: &[VisualComponent]) -> Result<LayoutInfo> {
        let width = image.width();
        let height = image.height();

        // 1. 检测布局类型
        let layout_type = self.detect_layout_type(components, width, height);

        // 2. 识别布局区域
        let sections = self.identify_sections(components, width, height);

        // 3. 分析网格
        let (grid_columns, grid_rows, gap) = self.analyze_grid(components);

        Ok(LayoutInfo {
            layout_type,
            sections,
            grid_columns,
            grid_rows,
            gap,
        })
    }

    /// 检测布局类型
    fn detect_layout_type(&self, components: &[VisualComponent], width: u32, _height: u32) -> LayoutType {
        if components.len() < 3 {
            return LayoutType::SingleColumn;
        }

        // 分析组件的X坐标分布
        let mut x_positions: Vec<u32> = components.iter()
            .map(|c| c.bounding_box.x)
            .collect();
        x_positions.sort();

        // 计算X坐标的聚类
        let clusters = self.cluster_positions(&x_positions, width / 10);

        match clusters.len() {
            1 => LayoutType::SingleColumn,
            2 => LayoutType::TwoColumn,
            3 => LayoutType::ThreeColumn,
            _ => {
                // 检查是否为网格布局
                if self.is_grid_layout(components) {
                    LayoutType::Grid
                } else if self.is_flex_layout(components) {
                    LayoutType::FlexRow
                } else {
                    LayoutType::Complex
                }
            }
        }
    }

    /// 位置聚类
    fn cluster_positions(&self, positions: &[u32], threshold: u32) -> Vec<Vec<u32>> {
        let mut clusters: Vec<Vec<u32>> = Vec::new();

        for &pos in positions {
            let mut found_cluster = false;

            for cluster in &mut clusters {
                let avg = cluster.iter().sum::<u32>() / cluster.len() as u32;
                if pos.abs_diff(avg) <= threshold {
                    cluster.push(pos);
                    found_cluster = true;
                    break;
                }
            }

            if !found_cluster {
                clusters.push(vec![pos]);
            }
        }

        clusters
    }

    /// 判断是否为网格布局
    fn is_grid_layout(&self, components: &[VisualComponent]) -> bool {
        if components.len() < 4 {
            return false;
        }

        // 检查是否有规律的行列排列
        let rows = self.group_by_rows(components);
        let cols = self.group_by_columns(components);

        // 如果行数和列数都大于2，可能是网格
        rows.len() >= 2 && cols.len() >= 2
    }

    /// 判断是否为弹性布局
    fn is_flex_layout(&self, components: &[VisualComponent]) -> bool {
        if components.len() < 2 {
            return false;
        }

        // 检查组件是否在同一行或列上
        let y_variance = self.calculate_variance(
            &components.iter().map(|c| c.bounding_box.y).collect::<Vec<_>>()
        );

        y_variance < 50.0
    }

    /// 按行分组
    fn group_by_rows<'a>(&self, components: &'a [VisualComponent]) -> Vec<Vec<&'a VisualComponent>> {
        let mut rows: Vec<Vec<&VisualComponent>> = Vec::new();
        let threshold = 30;

        for component in components {
            let y = component.bounding_box.y;
            let mut found_row = false;

            for row in &mut rows {
                let avg_y = row.iter().map(|c| c.bounding_box.y).sum::<u32>() / row.len() as u32;
                if y.abs_diff(avg_y) <= threshold {
                    row.push(component);
                    found_row = true;
                    break;
                }
            }

            if !found_row {
                rows.push(vec![component]);
            }
        }

        rows
    }

    /// 按列分组
    fn group_by_columns<'a>(&self, components: &'a [VisualComponent]) -> Vec<Vec<&'a VisualComponent>> {
        let mut cols: Vec<Vec<&VisualComponent>> = Vec::new();
        let threshold = 30;

        for component in components {
            let x = component.bounding_box.x;
            let mut found_col = false;

            for col in &mut cols {
                let avg_x = col.iter().map(|c| c.bounding_box.x).sum::<u32>() / col.len() as u32;
                if x.abs_diff(avg_x) <= threshold {
                    col.push(component);
                    found_col = true;
                    break;
                }
            }

            if !found_col {
                cols.push(vec![component]);
            }
        }

        cols
    }

    /// 计算方差
    fn calculate_variance(&self, values: &[u32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<u32>() as f32 / values.len() as f32;
        let variance = values.iter()
            .map(|&v| {
                let diff = v as f32 - mean;
                diff * diff
            })
            .sum::<f32>() / values.len() as f32;

        variance
    }

    /// 识别布局区域
    fn identify_sections(&self, components: &[VisualComponent], _width: u32, height: u32) -> Vec<LayoutSection> {
        let mut sections = Vec::new();

        // 识别头部区域（顶部 0-15%）
        let header_components: Vec<_> = components.iter()
            .filter(|c| {
                let center_y = c.bounding_box.y + c.bounding_box.height / 2;
                center_y < height / 7
            })
            .collect();

        if !header_components.is_empty() {
            let bbox = self.calculate_bounding_box(&header_components);
            sections.push(LayoutSection {
                name: "Header".to_string(),
                bounding_box: bbox,
                section_type: SectionType::Header,
                components: header_components.iter().map(|c| c.id.clone()).collect(),
            });
        }

        // 识别导航区域
        let nav_components: Vec<_> = components.iter()
            .filter(|c| c.component_type == ComponentType::Navigation)
            .collect();

        if !nav_components.is_empty() {
            let bbox = self.calculate_bounding_box(&nav_components);
            sections.push(LayoutSection {
                name: "Navigation".to_string(),
                bounding_box: bbox,
                section_type: SectionType::Navigation,
                components: nav_components.iter().map(|c| c.id.clone()).collect(),
            });
        }

        // 识别主要内容区域（中间 15-85%）
        let content_components: Vec<_> = components.iter()
            .filter(|c| {
                let center_y = c.bounding_box.y + c.bounding_box.height / 2;
                center_y >= height / 7 && center_y <= height * 6 / 7
            })
            .collect();

        if !content_components.is_empty() {
            let bbox = self.calculate_bounding_box(&content_components);
            sections.push(LayoutSection {
                name: "Content".to_string(),
                bounding_box: bbox,
                section_type: SectionType::Content,
                components: content_components.iter().map(|c| c.id.clone()).collect(),
            });
        }

        // 识别底部区域（底部 85-100%）
        let footer_components: Vec<_> = components.iter()
            .filter(|c| {
                let center_y = c.bounding_box.y + c.bounding_box.height / 2;
                center_y > height * 6 / 7
            })
            .collect();

        if !footer_components.is_empty() {
            let bbox = self.calculate_bounding_box(&footer_components);
            sections.push(LayoutSection {
                name: "Footer".to_string(),
                bounding_box: bbox,
                section_type: SectionType::Footer,
                components: footer_components.iter().map(|c| c.id.clone()).collect(),
            });
        }

        sections
    }

    /// 计算边界框
    fn calculate_bounding_box(&self, components: &[&VisualComponent]) -> BoundingBox {
        let min_x = components.iter().map(|c| c.bounding_box.x).min().unwrap_or(0);
        let min_y = components.iter().map(|c| c.bounding_box.y).min().unwrap_or(0);
        let max_x = components.iter()
            .map(|c| c.bounding_box.x + c.bounding_box.width)
            .max()
            .unwrap_or(0);
        let max_y = components.iter()
            .map(|c| c.bounding_box.y + c.bounding_box.height)
            .max()
            .unwrap_or(0);

        BoundingBox {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }

    /// 分析网格
    fn analyze_grid(&self, components: &[VisualComponent]) -> (Option<u8>, Option<u8>, u32) {
        let rows = self.group_by_rows(components);
        let cols = self.group_by_columns(components);

        let grid_columns = if cols.len() >= 2 {
            Some(cols.len() as u8)
        } else {
            None
        };

        let grid_rows = if rows.len() >= 2 {
            Some(rows.len() as u8)
        } else {
            None
        };

        // 计算平均间距
        let gap = self.calculate_average_gap(components);

        (grid_columns, grid_rows, gap)
    }

    /// 计算平均间距
    fn calculate_average_gap(&self, components: &[VisualComponent]) -> u32 {
        if components.len() < 2 {
            return 0;
        }

        let mut gaps = Vec::new();

        // 水平间距
        let mut sorted_by_x: Vec<_> = components.iter().collect();
        sorted_by_x.sort_by(|a, b| a.bounding_box.x.cmp(&b.bounding_box.x));

        for i in 1..sorted_by_x.len() {
            let gap = sorted_by_x[i].bounding_box.x
                - (sorted_by_x[i-1].bounding_box.x + sorted_by_x[i-1].bounding_box.width);
            if gap > 0 {
                gaps.push(gap);
            }
        }

        // 垂直间距
        let mut sorted_by_y: Vec<_> = components.iter().collect();
        sorted_by_y.sort_by(|a, b| a.bounding_box.y.cmp(&b.bounding_box.y));

        for i in 1..sorted_by_y.len() {
            let gap = sorted_by_y[i].bounding_box.y
                - (sorted_by_y[i-1].bounding_box.y + sorted_by_y[i-1].bounding_box.height);
            if gap > 0 {
                gaps.push(gap);
            }
        }

        if gaps.is_empty() {
            0
        } else {
            gaps.iter().sum::<u32>() / gaps.len() as u32
        }
    }

    /// 生成布局CSS
    pub fn generate_layout_css(&self, layout: &LayoutInfo) -> String {
        let mut css = String::new();

        // 容器布局
        css.push_str(".container {\n");
        
        match layout.layout_type {
            LayoutType::Grid => {
                if let Some(cols) = layout.grid_columns {
                    css.push_str("  display: grid;\n");
                    css.push_str(&format!("  grid-template-columns: repeat({}, 1fr);\n", cols));
                    css.push_str(&format!("  gap: {}px;\n", layout.gap));
                }
            }
            LayoutType::FlexRow => {
                css.push_str("  display: flex;\n");
                css.push_str("  flex-direction: row;\n");
                css.push_str(&format!("  gap: {}px;\n", layout.gap));
            }
            LayoutType::FlexColumn => {
                css.push_str("  display: flex;\n");
                css.push_str("  flex-direction: column;\n");
                css.push_str(&format!("  gap: {}px;\n", layout.gap));
            }
            _ => {
                css.push_str("  display: block;\n");
            }
        }

        css.push_str("}\n\n");

        // 区域样式
        for section in &layout.sections {
            let class_name = section.name.to_lowercase();
            css.push_str(&format!(".{} {{\n", class_name));
            css.push_str("  position: absolute;\n");
            css.push_str(&format!("  left: {}px;\n", section.bounding_box.x));
            css.push_str(&format!("  top: {}px;\n", section.bounding_box.y));
            css.push_str(&format!("  width: {}px;\n", section.bounding_box.width));
            css.push_str(&format!("  height: {}px;\n", section.bounding_box.height));
            css.push_str("}\n\n");
        }

        css
    }
}
