//! Compositor - 合成器
//!
//! 管理渲染图层，包括：
//! - 图层创建和管理
//! - 图层合成
//! - 脏区域追踪
//! - 增量更新

use browerai_core::Result;
use crate::paint::PaintRecord;
use crate::Viewport;
use std::collections::HashMap;

/// 合成器
#[derive(Debug, Clone)]
pub struct Compositor {
    /// 配置
    config: CompositorConfig,
    /// 图层映射
    layers: HashMap<LayerId, CompositedLayer>,
    /// 下一图层ID
    next_layer_id: u64,
}

impl Compositor {
    /// 创建新的合成器
    pub fn new(config: CompositorConfig) -> Self {
        Self {
            config,
            layers: HashMap::new(),
            next_layer_id: 1,
        }
    }

    /// 合成图层
    pub fn composite(&mut self, paint_records: &[PaintRecord], viewport: &Viewport) -> Result<Vec<CompositedLayer>> {
        // 简化实现：将绘制记录分组为图层
        let mut layers = Vec::new();
        
        // 创建背景图层
        let background_layer = CompositedLayer {
            id: self.allocate_layer_id(),
            rect: crate::Rect::new(0.0, 0.0, viewport.width as f32, viewport.height as f32),
            paint_records: paint_records.to_vec(),
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
        };
        
        layers.push(background_layer);
        
        // 存储图层
        for layer in &layers {
            self.layers.insert(layer.id, layer.clone());
        }
        
        Ok(layers)
    }

    /// 分配图层ID
    fn allocate_layer_id(&mut self) -> LayerId {
        let id = LayerId(self.next_layer_id);
        self.next_layer_id += 1;
        id
    }

    /// 获取图层
    pub fn get_layer(&self, id: LayerId) -> Option<&CompositedLayer> {
        self.layers.get(&id)
    }

    /// 获取图层数量
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// 移除图层
    pub fn remove_layer(&mut self, id: LayerId) -> Option<CompositedLayer> {
        self.layers.remove(&id)
    }

    /// 清空所有图层
    pub fn clear_layers(&mut self) {
        self.layers.clear();
    }

    /// 清空缓存
    pub fn clear_cache(&mut self) {
        // 简化实现
        self.layers.clear();
    }

    /// 获取配置
    pub fn config(&self) -> &CompositorConfig {
        &self.config
    }
}

impl Default for Compositor {
    fn default() -> Self {
        Self::new(CompositorConfig::default())
    }
}

/// 图层ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerId(pub u64);

/// 合成后的图层
#[derive(Debug, Clone)]
pub struct CompositedLayer {
    /// 图层ID
    pub id: LayerId,
    /// 矩形区域
    pub rect: crate::Rect,
    /// 绘制记录
    pub paint_records: Vec<PaintRecord>,
    /// 不透明度
    pub opacity: f32,
    /// 混合模式
    pub blend_mode: BlendMode,
}

impl CompositedLayer {
    /// 创建新图层
    pub fn new(id: LayerId, rect: crate::Rect) -> Self {
        Self {
            id,
            rect,
            paint_records: Vec::new(),
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
        }
    }

    /// 设置不透明度
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// 设置混合模式
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// 添加绘制记录
    pub fn add_paint_record(&mut self, record: PaintRecord) {
        self.paint_records.push(record);
    }

    /// 检查点是否在图层内
    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        self.rect.contains(x, y)
    }

    /// 检查是否与区域相交
    pub fn intersects_rect(&self, rect: &crate::Rect) -> bool {
        self.rect.intersects(rect)
    }
}

/// 混合模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// 正常
    Normal,
    /// 正片叠底
    Multiply,
    /// 屏幕
    Screen,
    /// 叠加
    Overlay,
    /// 变暗
    Darken,
    /// 变亮
    Lighten,
    /// 颜色减淡
    ColorDodge,
    /// 颜色加深
    ColorBurn,
    /// 差值
    Difference,
    /// 排除
    Exclusion,
}

/// 脏区域追踪器
#[derive(Debug, Clone, Default)]
pub struct DirtyRegionTracker {
    /// 脏区域列表
    regions: Vec<crate::Rect>,
}

impl DirtyRegionTracker {
    /// 创建新的追踪器
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加脏区域
    pub fn add_region(&mut self, rect: crate::Rect) {
        self.regions.push(rect);
    }

    /// 获取所有脏区域
    pub fn regions(&self) -> &[crate::Rect] {
        &self.regions
    }

    /// 清除所有脏区域
    pub fn clear(&mut self) {
        self.regions.clear();
    }

    /// 是否有脏区域
    pub fn has_dirty_regions(&self) -> bool {
        !self.regions.is_empty()
    }

    /// 合并所有脏区域为一个包围盒
    pub fn bounding_box(&self) -> Option<crate::Rect> {
        if self.regions.is_empty() {
            return None;
        }

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for rect in &self.regions {
            min_x = min_x.min(rect.x);
            min_y = min_y.min(rect.y);
            max_x = max_x.max(rect.x + rect.width);
            max_y = max_y.max(rect.y + rect.height);
        }

        Some(crate::Rect::new(min_x, min_y, max_x - min_x, max_y - min_y))
    }
}

/// 合成器配置
#[derive(Debug, Clone)]
pub struct CompositorConfig {
    /// 最大图层数
    pub max_layers: usize,
    /// 启用硬件加速
    pub hardware_accelerated: bool,
    /// 启用增量合成
    pub incremental_compositing: bool,
    /// 脏区域合并阈值
    pub dirty_region_merge_threshold: f32,
}

impl Default for CompositorConfig {
    fn default() -> Self {
        Self {
            max_layers: 100,
            hardware_accelerated: false,
            incremental_compositing: true,
            dirty_region_merge_threshold: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compositor_creation() {
        let compositor = Compositor::new(CompositorConfig::default());
        assert_eq!(compositor.layer_count(), 0);
    }

    #[test]
    fn test_composited_layer() {
        let layer = CompositedLayer::new(LayerId(1), crate::Rect::new(0.0, 0.0, 100.0, 100.0))
            .with_opacity(0.5)
            .with_blend_mode(BlendMode::Multiply);
        
        assert_eq!(layer.id.0, 1);
        assert_eq!(layer.opacity, 0.5);
        assert_eq!(layer.blend_mode, BlendMode::Multiply);
        
        assert!(layer.contains_point(50.0, 50.0));
        assert!(!layer.contains_point(150.0, 50.0));
    }

    #[test]
    fn test_dirty_region_tracker() {
        let mut tracker = DirtyRegionTracker::new();
        
        assert!(!tracker.has_dirty_regions());
        
        tracker.add_region(crate::Rect::new(0.0, 0.0, 100.0, 100.0));
        tracker.add_region(crate::Rect::new(50.0, 50.0, 100.0, 100.0));
        
        assert!(tracker.has_dirty_regions());
        assert_eq!(tracker.regions().len(), 2);
        
        let bbox = tracker.bounding_box().unwrap();
        assert_eq!(bbox.x, 0.0);
        assert_eq!(bbox.y, 0.0);
        assert_eq!(bbox.width, 150.0);
        assert_eq!(bbox.height, 150.0);
    }

    #[test]
    fn test_layer_operations() {
        let mut compositor = Compositor::new(CompositorConfig::default());
        
        let layer = CompositedLayer::new(LayerId(1), crate::Rect::new(0.0, 0.0, 100.0, 100.0));
        compositor.layers.insert(LayerId(1), layer);
        
        assert_eq!(compositor.layer_count(), 1);
        
        let retrieved = compositor.get_layer(LayerId(1));
        assert!(retrieved.is_some());
        
        compositor.remove_layer(LayerId(1));
        assert_eq!(compositor.layer_count(), 0);
    }
}
