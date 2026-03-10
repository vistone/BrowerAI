//! 组件检测器
//! 使用图像特征识别页面组件

use crate::*;
use anyhow::Result;
use image::{DynamicImage, GrayImage, Luma};
use imageproc::contours::find_contours;
use imageproc::contrast::{threshold, ThresholdType};
use imageproc::filter::gaussian_blur_f32;

/// 组件检测器
pub struct ComponentDetector {
    config: VisualLearningConfig,
    templates: Vec<ComponentTemplate>,
}

/// 组件模板
#[derive(Debug, Clone)]
struct ComponentTemplate {
    component_type: ComponentType,
    features: ComponentFeatures,
    confidence_weight: f64,
}

/// 组件特征
#[derive(Debug, Clone)]
struct ComponentFeatures {
    aspect_ratio_range: (f32, f32), // (min, max)
    min_area: u32,
    max_area: u32,
    has_text: bool,
    border_radius_range: (u8, u8),
    _typical_colors: Vec<Color>,
}

impl ComponentDetector {
    pub fn new(config: &VisualLearningConfig) -> Self {
        let templates = Self::initialize_templates();
        
        Self {
            config: config.clone(),
            templates,
        }
    }

    /// 初始化组件模板
    fn initialize_templates() -> Vec<ComponentTemplate> {
        vec![
            // 按钮模板
            ComponentTemplate {
                component_type: ComponentType::Button,
                features: ComponentFeatures {
                    aspect_ratio_range: (1.5, 6.0),
                    min_area: 2000,
                    max_area: 50000,
                    has_text: true,
                    border_radius_range: (2, 12),
                    _typical_colors: vec![],
                },
                confidence_weight: 1.0,
            },
            // 输入框模板
            ComponentTemplate {
                component_type: ComponentType::Input,
                features: ComponentFeatures {
                    aspect_ratio_range: (4.0, 15.0),
                    min_area: 3000,
                    max_area: 100000,
                    has_text: false,
                    border_radius_range: (2, 8),
                    _typical_colors: vec![],
                },
                confidence_weight: 1.0,
            },
            // 卡片模板
            ComponentTemplate {
                component_type: ComponentType::Card,
                features: ComponentFeatures {
                    aspect_ratio_range: (0.5, 2.0),
                    min_area: 50000,
                    max_area: 500000,
                    has_text: true,
                    border_radius_range: (4, 16),
                    _typical_colors: vec![],
                },
                confidence_weight: 0.9,
            },
            // 导航模板
            ComponentTemplate {
                component_type: ComponentType::Navigation,
                features: ComponentFeatures {
                    aspect_ratio_range: (5.0, 50.0),
                    min_area: 5000,
                    max_area: 200000,
                    has_text: true,
                    border_radius_range: (0, 4),
                    _typical_colors: vec![],
                },
                confidence_weight: 0.85,
            },
            // 头像模板
            ComponentTemplate {
                component_type: ComponentType::Avatar,
                features: ComponentFeatures {
                    aspect_ratio_range: (0.9, 1.1),
                    min_area: 400,
                    max_area: 10000,
                    has_text: false,
                    border_radius_range: (50, 50), // 圆形
                    _typical_colors: vec![],
                },
                confidence_weight: 0.9,
            },
            // 图标模板
            ComponentTemplate {
                component_type: ComponentType::Icon,
                features: ComponentFeatures {
                    aspect_ratio_range: (0.8, 1.2),
                    min_area: 100,
                    max_area: 2500,
                    has_text: false,
                    border_radius_range: (0, 0),
                    _typical_colors: vec![],
                },
                confidence_weight: 0.8,
            },
        ]
    }

    /// 检测图像中的所有组件
    pub fn detect_components(&self, image: &DynamicImage) -> Result<Vec<VisualComponent>> {
        let mut components = Vec::new();
        
        // 1. 边缘检测找到潜在组件区域
        let regions = self.detect_regions(image)?;
        
        // 2. 对每个区域进行分类
        for (idx, region) in regions.iter().enumerate() {
            if let Some(component) = self.classify_region(image, region, idx) {
                if component.confidence >= self.config.component_confidence_threshold {
                    components.push(component);
                }
            }
        }
        
        // 3. 合并重叠的组件
        let merged = self.merge_overlapping_components(components);
        
        // 4. 推断组件层级关系
        let hierarchical = self.infer_hierarchy(merged);
        
        Ok(hierarchical)
    }

    /// 检测图像区域
    fn detect_regions(&self, image: &DynamicImage) -> Result<Vec<BoundingBox>> {
        let gray = image.to_luma8();
        
        // 高斯模糊减少噪声
        let blurred = gaussian_blur_f32(&gray, 2.0);
        
        // 边缘检测
        let edges = self.detect_edges(&blurred);
        
        // 二值化
        let binary = threshold(&edges, 30, ThresholdType::Binary);
        
        // 查找轮廓
        let contours = find_contours::<i32>(&binary);
        
        // 将轮廓转换为边界框
        let mut regions = Vec::new();
        for contour in contours {
            if contour.points.len() < 4 {
                continue;
            }
            
            // 计算边界框
            let min_x = contour.points.iter().map(|p| p.x).min().unwrap_or(0) as u32;
            let max_x = contour.points.iter().map(|p| p.x).max().unwrap_or(0) as u32;
            let min_y = contour.points.iter().map(|p| p.y).min().unwrap_or(0) as u32;
            let max_y = contour.points.iter().map(|p| p.y).max().unwrap_or(0) as u32;
            
            let width = max_x - min_x;
            let height = max_y - min_y;
            
            // 过滤太小的区域
            if width >= 20 && height >= 20 {
                regions.push(BoundingBox {
                    x: min_x,
                    y: min_y,
                    width,
                    height,
                });
            }
        }
        
        Ok(regions)
    }

    /// 边缘检测
    fn detect_edges(&self, image: &GrayImage) -> GrayImage {
        let (width, height) = image.dimensions();
        let mut edges = GrayImage::new(width, height);
        
        // Sobel 算子
        let sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
        let sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
        
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut gx = 0i32;
                let mut gy = 0i32;
                
                for ky in 0..3 {
                    for kx in 0..3 {
                        let pixel = image.get_pixel(x + kx - 1, y + ky - 1).0[0] as i32;
                        gx += pixel * sobel_x[ky as usize][kx as usize];
                        gy += pixel * sobel_y[ky as usize][kx as usize];
                    }
                }
                
                let magnitude = ((gx * gx + gy * gy) as f32).sqrt() as u8;
                edges.put_pixel(x, y, Luma([magnitude]));
            }
        }
        
        edges
    }

    /// 分类区域
    fn classify_region(&self, image: &DynamicImage, region: &BoundingBox, idx: usize) -> Option<VisualComponent> {
        let area = region.area();
        let aspect_ratio = region.width as f32 / region.height as f32;
        
        // 提取区域图像
        let region_image = image.crop_imm(region.x, region.y, region.width, region.height);
        
        // 分析视觉特征
        let border_radius = self.estimate_border_radius(&region_image);
        let has_text = self.detect_text(&region_image);
        let colors = self.extract_dominant_colors(&region_image);
        
        // 匹配最佳模板
        let mut best_match: Option<(ComponentType, f64)> = None;
        
        for template in &self.templates {
            let mut score = 0.0;
            let mut checks = 0;
            
            // 检查面积
            if area >= template.features.min_area && area <= template.features.max_area {
                score += 1.0;
            }
            checks += 1;
            
            // 检查宽高比
            let (min_ar, max_ar) = template.features.aspect_ratio_range;
            if aspect_ratio >= min_ar && aspect_ratio <= max_ar {
                score += 1.0;
            }
            checks += 1;
            
            // 检查圆角
            let (min_br, max_br) = template.features.border_radius_range;
            if border_radius >= min_br && border_radius <= max_br {
                score += 1.0;
            }
            checks += 1;
            
            // 检查文本
            if template.features.has_text == has_text {
                score += 0.5;
            }
            checks += 1;
            
            let confidence = (score / checks as f64) * template.confidence_weight;
            
            if confidence > best_match.as_ref().map(|(_, c)| *c).unwrap_or(0.0) {
                best_match = Some((template.component_type.clone(), confidence));
            }
        }
        
        best_match.map(|(component_type, confidence)| {
            VisualComponent {
                id: format!("comp_{}", idx),
                component_type,
                bounding_box: region.clone(),
                confidence,
                visual_style: VisualStyle {
                    background_color: colors.first().map(|(c, _)| c.clone()),
                    text_color: None,
                    border_color: None,
                    border_width: 0,
                    border_radius,
                    shadow: None,
                    opacity: 1.0,
                },
                semantic_label: None,
                children: Vec::new(),
                parent: None,
            }
        })
    }

    /// 估计圆角半径
    fn estimate_border_radius(&self, image: &DynamicImage) -> u8 {
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        
        if width < 10 || height < 10 {
            return 0;
        }
        
        // 检查四个角的透明度/颜色变化
        let corner_size = (width.min(height) / 4).min(20) as usize;
        
        let mut corner_transparency = 0u32;
        
        // 左上角
        for y in 0..corner_size {
            for x in 0..corner_size {
                if x * x + y * y < corner_size * corner_size {
                    let pixel = rgba.get_pixel(x as u32, y as u32);
                    if pixel[3] < 128 {
                        corner_transparency += 1;
                    }
                }
            }
        }
        
        // 根据透明度比例估算圆角
        let transparency_ratio = corner_transparency as f32 / (corner_size * corner_size) as f32;
        
        if transparency_ratio > 0.7 {
            (corner_size as f32 * 0.8) as u8
        } else if transparency_ratio > 0.3 {
            (corner_size as f32 * 0.4) as u8
        } else {
            0
        }
    }

    /// 检测文本
    fn detect_text(&self, image: &DynamicImage) -> bool {
        let gray = image.to_luma8();
        
        // 简单的文本检测：高对比度区域
        let mut high_contrast_pixels = 0;
        let total_pixels = gray.width() * gray.height();
        
        for y in 1..gray.height() - 1 {
            for x in 1..gray.width() - 1 {
                let _center = gray.get_pixel(x, y).0[0] as i32;
                let left = gray.get_pixel(x - 1, y).0[0] as i32;
                let right = gray.get_pixel(x + 1, y).0[0] as i32;
                
                let horizontal_diff = (left - right).abs();
                
                if horizontal_diff > 50 {
                    high_contrast_pixels += 1;
                }
            }
        }
        
        // 如果高对比度像素比例适中，可能是文本
        let ratio = high_contrast_pixels as f32 / total_pixels as f32;
        ratio > 0.05 && ratio < 0.4
    }

    /// 提取主导颜色
    fn extract_dominant_colors(&self, image: &DynamicImage) -> Vec<(Color, f32)> {
        let rgba = image.to_rgba8();
        let mut color_counts: HashMap<(u8, u8, u8), usize> = HashMap::new();
        
        // 采样像素
        let step = (rgba.width() * rgba.height() / 1000).max(1);
        
        for (idx, pixel) in rgba.pixels().enumerate() {
            if !(idx as u32).is_multiple_of(step) {
                continue;
            }
            
            // 量化颜色（减少颜色数量）
            let r = (pixel[0] / 16) * 16;
            let g = (pixel[1] / 16) * 16;
            let b = (pixel[2] / 16) * 16;
            
            *color_counts.entry((r, g, b)).or_insert(0) += 1;
        }
        
        // 排序并返回前5个颜色
        let mut sorted: Vec<_> = color_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        
        let total: usize = sorted.iter().map(|(_, count)| *count).sum();
        
        sorted.iter()
            .take(5)
            .map(|((r, g, b), count)| {
                (Color { r: *r, g: *g, b: *b, a: 255 }, **count as f32 / total as f32)
            })
            .collect()
    }

    /// 合并重叠的组件
    fn merge_overlapping_components(&self, components: Vec<VisualComponent>) -> Vec<VisualComponent> {
        let mut merged = Vec::new();
        let mut processed = vec![false; components.len()];
        
        for i in 0..components.len() {
            if processed[i] {
                continue;
            }
            
            let mut current = components[i].clone();
            processed[i] = true;
            
            // 查找重叠的组件
            for j in (i + 1)..components.len() {
                if processed[j] {
                    continue;
                }
                
                if current.bounding_box.intersects(&components[j].bounding_box) {
                    // 如果重叠，保留置信度更高的
                    if components[j].confidence > current.confidence {
                        current = components[j].clone();
                    }
                    processed[j] = true;
                }
            }
            
            merged.push(current);
        }
        
        merged
    }

    /// 推断组件层级关系
    fn infer_hierarchy(&self, mut components: Vec<VisualComponent>) -> Vec<VisualComponent> {
        // 按面积排序（从大到小）
        components.sort_by(|a, b| b.bounding_box.area().cmp(&a.bounding_box.area()));
        
        // 建立父子关系
        for i in 0..components.len() {
            let parent_box = components[i].bounding_box.clone();
            
            for j in (i + 1)..components.len() {
                if parent_box.contains(&components[j].bounding_box) {
                    // j 是 i 的子组件
                    let child_id = components[j].id.clone();
                    components[i].children.push(child_id);
                    components[j].parent = Some(components[i].id.clone());
                }
            }
        }
        
        components
    }
}
