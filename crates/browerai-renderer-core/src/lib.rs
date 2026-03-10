//! BrowerAI Renderer Core
//!
//! 渲染引擎核心，提供：
//! - 布局引擎 (Layout Engine)
//! - 绘制系统 (Paint System)
//! - 合成器 (Compositor)
//! - 资源管理 (Resource Management)
//! - AI增强渲染
//!
//! # 架构
//! ```text
//! DOM Tree → Layout Tree → Paint Records → Composited Layers → Output
//! ```
//!
//! # 示例
//! ```
//! use browerai_renderer_core::{Renderer, RenderConfig, Viewport};
//!
//! let renderer = Renderer::new(RenderConfig::default());
//! let viewport = Viewport::new(1920, 1080);
//! // 渲染内容...
//! ```

#![warn(missing_docs)]

use browerai_core::Result;
use browerai_css_parser::Stylesheet;
use browerai_dom::Document;

pub mod layout;
pub mod paint;
pub mod compositing;
pub mod resources;

pub use layout::{LayoutEngine, LayoutTree, LayoutNode, BoxModel};
pub use paint::{PaintEngine, PaintRecord, PaintCommand};
pub use compositing::{Compositor, CompositedLayer, LayerId};
pub use resources::{ResourceManager, ResourceType, CachedImage, CachedFont, CachedStylesheet};

/// 渲染器
///
/// BrowerAI的主渲染引擎，整合布局、绘制和合成功能
#[derive(Debug)]
pub struct Renderer {
    /// 配置
    config: RenderConfig,
    /// 布局引擎
    layout_engine: LayoutEngine,
    /// 绘制引擎
    paint_engine: PaintEngine,
    /// 合成器
    compositor: Compositor,
    /// 资源管理器
    resource_manager: ResourceManager,
}

impl Renderer {
    /// 创建新的渲染器
    ///
    /// # 示例
    /// ```
    /// use browerai_renderer_core::{Renderer, RenderConfig};
    ///
    /// let renderer = Renderer::new(RenderConfig::default());
    /// ```
    pub fn new(config: RenderConfig) -> Self {
        Self {
            config: config.clone(),
            layout_engine: LayoutEngine::new(config.layout_config),
            paint_engine: PaintEngine::new(config.paint_config),
            compositor: Compositor::new(config.compositor_config),
            resource_manager: ResourceManager::new(config.resource_config),
        }
    }

    /// 渲染文档
    ///
    /// # 流程
    /// 1. 构建布局树
    /// 2. 计算布局
    /// 3. 生成绘制记录
    /// 4. 合成图层
    /// 5. 输出渲染结果
    pub fn render(&mut self, document: &Document, stylesheet: &Stylesheet, viewport: &Viewport) -> Result<RenderOutput> {
        // 阶段1: 构建布局树
        let layout_tree = self.layout_engine.build_tree(document, stylesheet)?;
        
        // 阶段2: 计算布局
        let computed_layout = self.layout_engine.compute_layout(&layout_tree, viewport)?;
        
        // 阶段3: 生成绘制记录
        let paint_records = self.paint_engine.generate_paints(&computed_layout)?;
        
        // 阶段4: 合成图层
        let layers = self.compositor.composite(&paint_records, viewport)?;
        
        // 阶段5: 生成输出
        let output = RenderOutput {
            layers,
            viewport: viewport.clone(),
            metadata: RenderMetadata {
                layer_count: self.compositor.layer_count(),
                paint_count: paint_records.len(),
                node_count: computed_layout.node_count(),
            },
        };
        
        Ok(output)
    }

    /// 增量渲染（仅更新变化部分）
    pub fn render_incremental(&mut self, changes: &[DomChange], viewport: &Viewport) -> Result<RenderOutput> {
        // 简化实现：实际应该只更新受影响的区域
        log::info!("Incremental rendering {} changes", changes.len());
        
        // 返回空输出作为占位
        Ok(RenderOutput {
            layers: Vec::new(),
            viewport: viewport.clone(),
            metadata: RenderMetadata {
                layer_count: 0,
                paint_count: 0,
                node_count: 0,
            },
        })
    }

    /// 获取布局引擎
    pub fn layout_engine(&self) -> &LayoutEngine {
        &self.layout_engine
    }

    /// 获取绘制引擎
    pub fn paint_engine(&self) -> &PaintEngine {
        &self.paint_engine
    }

    /// 获取合成器
    pub fn compositor(&self) -> &Compositor {
        &self.compositor
    }

    /// 获取资源管理器
    pub fn resource_manager(&self) -> &ResourceManager {
        &self.resource_manager
    }

    /// 清除缓存
    pub fn clear_cache(&mut self) {
        self.resource_manager.clear_cache();
        self.compositor.clear_cache();
    }

    /// 获取渲染统计
    pub fn stats(&self) -> RenderStats {
        RenderStats {
            cached_resources: self.resource_manager.cache_size(),
            layer_count: self.compositor.layer_count(),
            memory_usage: self.estimate_memory_usage(),
        }
    }

    /// 估算内存使用
    fn estimate_memory_usage(&self) -> usize {
        // 简化实现
        self.resource_manager.cache_size() * 1024
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new(RenderConfig::default())
    }
}

/// 渲染配置
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// 布局配置
    pub layout_config: layout::LayoutConfig,
    /// 绘制配置
    pub paint_config: paint::PaintConfig,
    /// 合成器配置
    pub compositor_config: compositing::CompositorConfig,
    /// 资源配置
    pub resource_config: resources::ResourceConfig,
    /// 启用AI增强
    pub ai_enhanced: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            layout_config: layout::LayoutConfig::default(),
            paint_config: paint::PaintConfig::default(),
            compositor_config: compositing::CompositorConfig::default(),
            resource_config: resources::ResourceConfig::default(),
            ai_enhanced: false,
        }
    }
}

/// 视口
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// 宽度
    pub width: u32,
    /// 高度
    pub height: u32,
    /// 设备像素比
    pub device_pixel_ratio: f32,
    /// 滚动位置X
    pub scroll_x: f32,
    /// 滚动位置Y
    pub scroll_y: f32,
}

impl Viewport {
    /// 创建新的视口
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            device_pixel_ratio: 1.0,
            scroll_x: 0.0,
            scroll_y: 0.0,
        }
    }

    /// 设置设备像素比
    pub fn with_dpr(mut self, dpr: f32) -> Self {
        self.device_pixel_ratio = dpr;
        self
    }

    /// 设置滚动位置
    pub fn with_scroll(mut self, x: f32, y: f32) -> Self {
        self.scroll_x = x;
        self.scroll_y = y;
        self
    }

    /// 获取物理宽度
    pub fn physical_width(&self) -> u32 {
        (self.width as f32 * self.device_pixel_ratio) as u32
    }

    /// 获取物理高度
    pub fn physical_height(&self) -> u32 {
        (self.height as f32 * self.device_pixel_ratio) as u32
    }

    /// 获取可见区域
    pub fn visible_rect(&self) -> Rect {
        Rect {
            x: self.scroll_x,
            y: self.scroll_y,
            width: self.width as f32,
            height: self.height as f32,
        }
    }
}

/// 矩形
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// X坐标
    pub x: f32,
    /// Y坐标
    pub y: f32,
    /// 宽度
    pub width: f32,
    /// 高度
    pub height: f32,
}

impl Rect {
    /// 创建新矩形
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// 检查点是否在矩形内
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width &&
        py >= self.y && py <= self.y + self.height
    }

    /// 检查是否与另一个矩形相交
    pub fn intersects(&self, other: &Rect) -> bool {
        !(self.x + self.width < other.x ||
          other.x + other.width < self.x ||
          self.y + self.height < other.y ||
          other.y + other.height < self.y)
    }
}

/// 渲染输出
#[derive(Debug, Clone)]
pub struct RenderOutput {
    /// 合成图层
    pub layers: Vec<CompositedLayer>,
    /// 视口
    pub viewport: Viewport,
    /// 元数据
    pub metadata: RenderMetadata,
}

/// 渲染元数据
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderMetadata {
    /// 图层数量
    pub layer_count: usize,
    /// 绘制命令数量
    pub paint_count: usize,
    /// 布局节点数量
    pub node_count: usize,
}

/// 渲染统计
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStats {
    /// 缓存资源数
    pub cached_resources: usize,
    /// 图层数量
    pub layer_count: usize,
    /// 内存使用（字节）
    pub memory_usage: usize,
}

/// DOM变更
#[derive(Debug, Clone)]
pub struct DomChange {
    /// 变更类型
    pub change_type: ChangeType,
    /// 目标节点ID
    pub target_id: String,
    /// 变更内容
    pub content: Option<String>,
}

/// 变更类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// 插入节点
    Insert,
    /// 删除节点
    Remove,
    /// 修改属性
    Attribute,
    /// 修改文本
    Text,
    /// 修改样式
    Style,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = Renderer::new(RenderConfig::default());
        let stats = renderer.stats();
        assert_eq!(stats.layer_count, 0);
    }

    #[test]
    fn test_viewport() {
        let vp = Viewport::new(1920, 1080)
            .with_dpr(2.0)
            .with_scroll(100.0, 200.0);
        
        assert_eq!(vp.physical_width(), 3840);
        assert_eq!(vp.scroll_x, 100.0);
    }

    #[test]
    fn test_rect() {
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
        
        assert!(rect.contains(50.0, 50.0));
        assert!(!rect.contains(150.0, 50.0));
        
        let other = Rect::new(50.0, 50.0, 100.0, 100.0);
        assert!(rect.intersects(&other));
    }

    #[test]
    fn test_render_config_default() {
        let config = RenderConfig::default();
        assert!(!config.ai_enhanced);
    }
}
