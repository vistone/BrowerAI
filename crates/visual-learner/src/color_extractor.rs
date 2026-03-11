//! 颜色提取器
//! 提取和分析图像的配色方案

use crate::*;
use anyhow::Result;
use image::DynamicImage;

use kdtree::distance::squared_euclidean;
use std::collections::HashMap;

/// 颜色提取器
pub struct ColorExtractor {
    config: VisualLearningConfig,
}

impl ColorExtractor {
    pub fn new(config: &VisualLearningConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// 提取图像的颜色方案
    pub fn extract_colors(&self, image: &DynamicImage) -> Result<ColorScheme> {
        let rgba = image.to_rgba8();

        // 1. 提取所有颜色
        let all_colors = self.extract_all_colors(&rgba);

        // 2. 聚类颜色
        let clustered = self.cluster_colors(&all_colors, self.config.color_cluster_count);

        // 3. 识别颜色角色
        let color_scheme = self.identify_color_roles(&clustered, image)?;

        Ok(color_scheme)
    }

    /// 提取所有颜色
    fn extract_all_colors(&self, rgba: &image::RgbaImage) -> Vec<(Color, u32)> {
        let mut color_counts: HashMap<(u8, u8, u8, u8), u32> = HashMap::new();

        // 采样（提高性能）
        let step = ((rgba.width() * rgba.height()) / 10000).max(1);

        for (idx, pixel) in rgba.pixels().enumerate() {
            if !(idx as u32).is_multiple_of(step) {
                continue;
            }

            let key = (pixel[0], pixel[1], pixel[2], pixel[3]);
            *color_counts.entry(key).or_insert(0) += 1;
        }

        color_counts
            .iter()
            .map(|((r, g, b, a), count)| {
                (
                    Color {
                        r: *r,
                        g: *g,
                        b: *b,
                        a: *a,
                    },
                    *count,
                )
            })
            .collect()
    }

    /// K-means 聚类颜色
    fn cluster_colors(&self, colors: &[(Color, u32)], k: usize) -> Vec<(Color, f32)> {
        if colors.len() <= k {
            let total: u32 = colors.iter().map(|(_, count)| *count).sum();
            return colors
                .iter()
                .map(|(c, count)| (c.clone(), *count as f32 / total as f32))
                .collect();
        }

        // 初始化聚类中心
        let mut centroids: Vec<[f32; 3]> = colors
            .iter()
            .take(k)
            .map(|(c, _)| [c.r as f32, c.g as f32, c.b as f32])
            .collect();

        let mut assignments: Vec<usize> = vec![0; colors.len()];
        let max_iterations = 20;

        for _ in 0..max_iterations {
            let mut changed = false;

            // 分配每个颜色到最近的中心
            for (i, (color, _)) in colors.iter().enumerate() {
                let point = [color.r as f32, color.g as f32, color.b as f32];
                let nearest = self.find_nearest_centroid(&point, &centroids);

                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // 更新聚类中心
            let mut new_centroids: Vec<[f32; 3]> = vec![[0.0; 3]; k];
            let mut counts: Vec<f32> = vec![0.0; k];

            for (i, (color, weight)) in colors.iter().enumerate() {
                let cluster = assignments[i];
                let w = *weight as f32;

                new_centroids[cluster][0] += color.r as f32 * w;
                new_centroids[cluster][1] += color.g as f32 * w;
                new_centroids[cluster][2] += color.b as f32 * w;
                counts[cluster] += w;
            }

            for i in 0..k {
                if counts[i] > 0.0 {
                    centroids[i][0] = new_centroids[i][0] / counts[i];
                    centroids[i][1] = new_centroids[i][1] / counts[i];
                    centroids[i][2] = new_centroids[i][2] / counts[i];
                }
            }
        }

        // 计算每个聚类的权重
        let mut cluster_weights: Vec<f32> = vec![0.0; k];
        let total_weight: f32 = colors.iter().map(|(_, w)| *w as f32).sum();

        for (i, (_, weight)) in colors.iter().enumerate() {
            cluster_weights[assignments[i]] += *weight as f32;
        }

        // 转换为结果
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let color = Color {
                    r: c[0] as u8,
                    g: c[1] as u8,
                    b: c[2] as u8,
                    a: 255,
                };
                (color, cluster_weights[i] / total_weight)
            })
            .collect()
    }

    /// 找到最近的聚类中心
    fn find_nearest_centroid(&self, point: &[f32; 3], centroids: &[[f32; 3]]) -> usize {
        let mut min_dist = f32::MAX;
        let mut nearest = 0;

        for (i, centroid) in centroids.iter().enumerate() {
            let dist = squared_euclidean(point, centroid) as f32;
            if dist < min_dist {
                min_dist = dist;
                nearest = i;
            }
        }

        nearest
    }

    /// 识别颜色角色
    fn identify_color_roles(
        &self,
        colors: &[(Color, f32)],
        _image: &DynamicImage,
    ) -> Result<ColorScheme> {
        let mut scheme = ColorScheme {
            all_colors: colors.to_vec(),
            ..Default::default()
        };

        if colors.is_empty() {
            return Ok(scheme);
        }

        // 按占比排序
        let mut sorted = colors.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 背景色通常是占比最大的
        scheme.background = Some(sorted[0].0.clone());

        // 文本颜色
        if let Some(bg) = &scheme.background {
            scheme.text_primary = Some(self.find_contrasting_color(bg, &sorted, 0.7));
            scheme.text_secondary = Some(self.find_contrasting_color(bg, &sorted, 0.5));
        }

        // 主色调（通常是饱和度较高的颜色）
        scheme.primary = sorted
            .iter()
            .find(|(c, _)| self.is_accent_color(c))
            .map(|(c, _)| c.clone())
            .or_else(|| sorted.get(1).map(|(c, _)| c.clone()));

        // 次要色
        scheme.secondary = sorted
            .iter()
            .skip(1)
            .find(|(c, _)| self.is_accent_color(c))
            .map(|(c, _)| c.clone())
            .or_else(|| sorted.get(2).map(|(c, _)| c.clone()));

        // 表面色（比背景稍亮或稍暗）
        if let Some(bg) = &scheme.background {
            scheme.surface = Some(self.adjust_brightness(bg, if bg.is_dark() { 20 } else { -20 }));
        }

        // 语义颜色
        scheme.accent = scheme.primary.clone();
        scheme.error = Some(self.find_or_create_semantic_color(&sorted, "error"));
        scheme.warning = Some(self.find_or_create_semantic_color(&sorted, "warning"));
        scheme.success = Some(self.find_or_create_semantic_color(&sorted, "success"));

        Ok(scheme)
    }

    /// 判断是否为强调色
    fn is_accent_color(&self, color: &Color) -> bool {
        let saturation = self.calculate_saturation(color);
        saturation > 0.3
    }

    /// 计算饱和度
    fn calculate_saturation(&self, color: &Color) -> f32 {
        let r = color.r as f32 / 255.0;
        let g = color.g as f32 / 255.0;
        let b = color.b as f32 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);

        if max == 0.0 {
            0.0
        } else {
            (max - min) / max
        }
    }

    /// 查找对比色
    fn find_contrasting_color(
        &self,
        base: &Color,
        candidates: &[(Color, f32)],
        min_contrast: f32,
    ) -> Color {
        let base_lum = base.luminance();

        candidates
            .iter()
            .find(|(c, _)| {
                let contrast = if base_lum > c.luminance() {
                    (base_lum + 0.05) / (c.luminance() + 0.05)
                } else {
                    (c.luminance() + 0.05) / (base_lum + 0.05)
                };
                contrast >= min_contrast
            })
            .map(|(c, _)| c.clone())
            .unwrap_or_else(|| {
                // 如果没有找到，生成一个
                if base.is_dark() {
                    Color {
                        r: 255,
                        g: 255,
                        b: 255,
                        a: 255,
                    }
                } else {
                    Color {
                        r: 0,
                        g: 0,
                        b: 0,
                        a: 255,
                    }
                }
            })
    }

    /// 调整亮度
    fn adjust_brightness(&self, color: &Color, amount: i16) -> Color {
        let adjust = |v: u8, a: i16| -> u8 {
            let result = v as i16 + a;
            result.clamp(0, 255) as u8
        };

        Color {
            r: adjust(color.r, amount),
            g: adjust(color.g, amount),
            b: adjust(color.b, amount),
            a: color.a,
        }
    }

    /// 查找或创建语义颜色
    fn find_or_create_semantic_color(&self, colors: &[(Color, f32)], semantic_type: &str) -> Color {
        match semantic_type {
            "error" => {
                // 查找红色系
                colors
                    .iter()
                    .find(|(c, _)| c.r > 150 && c.g < 100 && c.b < 100)
                    .map(|(c, _)| c.clone())
                    .unwrap_or(Color {
                        r: 220,
                        g: 53,
                        b: 69,
                        a: 255,
                    })
            }
            "warning" => {
                // 查找黄色/橙色系
                colors
                    .iter()
                    .find(|(c, _)| c.r > 150 && c.g > 100 && c.b < 100)
                    .map(|(c, _)| c.clone())
                    .unwrap_or(Color {
                        r: 255,
                        g: 193,
                        b: 7,
                        a: 255,
                    })
            }
            "success" => {
                // 查找绿色系
                colors
                    .iter()
                    .find(|(c, _)| c.r < 100 && c.g > 150 && c.b < 100)
                    .map(|(c, _)| c.clone())
                    .unwrap_or(Color {
                        r: 40,
                        g: 167,
                        b: 69,
                        a: 255,
                    })
            }
            _ => Color {
                r: 128,
                g: 128,
                b: 128,
                a: 255,
            },
        }
    }

    /// 生成颜色变量代码
    pub fn generate_css_variables(&self, scheme: &ColorScheme) -> String {
        let mut css = String::from(":root {\n");

        if let Some(c) = &scheme.primary {
            css.push_str(&format!("  --color-primary: {};\n", c.to_hex()));
        }
        if let Some(c) = &scheme.secondary {
            css.push_str(&format!("  --color-secondary: {};\n", c.to_hex()));
        }
        if let Some(c) = &scheme.background {
            css.push_str(&format!("  --color-background: {};\n", c.to_hex()));
        }
        if let Some(c) = &scheme.surface {
            css.push_str(&format!("  --color-surface: {};\n", c.to_hex()));
        }
        if let Some(c) = &scheme.text_primary {
            css.push_str(&format!("  --color-text-primary: {};\n", c.to_hex()));
        }
        if let Some(c) = &scheme.text_secondary {
            css.push_str(&format!("  --color-text-secondary: {};\n", c.to_hex()));
        }
        if let Some(c) = &scheme.accent {
            css.push_str(&format!("  --color-accent: {};\n", c.to_hex()));
        }
        if let Some(c) = &scheme.error {
            css.push_str(&format!("  --color-error: {};\n", c.to_hex()));
        }
        if let Some(c) = &scheme.warning {
            css.push_str(&format!("  --color-warning: {};\n", c.to_hex()));
        }
        if let Some(c) = &scheme.success {
            css.push_str(&format!("  --color-success: {};\n", c.to_hex()));
        }

        css.push_str("}\n");
        css
    }
}
