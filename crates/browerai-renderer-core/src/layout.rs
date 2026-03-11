//! Layout Engine - 布局引擎
//!
//! 实现CSS布局算法，包括：
//! - 盒模型计算
//! - 流式布局
//! - Flexbox布局
//! - 定位（Static/Relative/Absolute/Fixed）

use crate::{Rect, Viewport};
use browerai_core::Result;
use browerai_css_parser::Stylesheet;
use browerai_dom::Document;
use std::collections::HashMap;

/// 布局引擎
#[derive(Debug, Clone)]
pub struct LayoutEngine {
    /// 配置
    config: LayoutConfig,
}

impl LayoutEngine {
    /// 创建新的布局引擎
    pub fn new(config: LayoutConfig) -> Self {
        Self { config }
    }

    /// 构建布局树
    pub fn build_tree(&self, _document: &Document, _stylesheet: &Stylesheet) -> Result<LayoutTree> {
        let mut tree = LayoutTree::new();

        // 简化实现：创建根节点
        let root = LayoutNode {
            id: "root".to_string(),
            tag_name: "html".to_string(),
            box_model: BoxModel::default(),
            children: Vec::new(),
            style_properties: HashMap::new(),
            positioning: Positioning::Static,
        };

        tree.root = Some(root);
        tree.node_count = 1;

        Ok(tree)
    }

    /// 计算布局
    pub fn compute_layout(&self, tree: &LayoutTree, viewport: &Viewport) -> Result<ComputedLayout> {
        // 简化实现
        let mut computed = ComputedLayout {
            nodes: HashMap::new(),
            viewport: *viewport,
        };

        if let Some(ref root) = tree.root {
            computed.nodes.insert(
                root.id.clone(),
                ComputedBox {
                    rect: Rect::new(0.0, 0.0, viewport.width as f32, viewport.height as f32),
                    box_model: root.box_model,
                },
            );
        }

        Ok(computed)
    }

    /// 获取配置
    pub fn config(&self) -> &LayoutConfig {
        &self.config
    }
}

impl Default for LayoutEngine {
    fn default() -> Self {
        Self::new(LayoutConfig::default())
    }
}

/// 布局树
#[derive(Debug, Clone)]
pub struct LayoutTree {
    /// 根节点
    pub root: Option<LayoutNode>,
    /// 节点数量
    pub node_count: usize,
}

impl LayoutTree {
    /// 创建新的布局树
    pub fn new() -> Self {
        Self {
            root: None,
            node_count: 0,
        }
    }

    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.node_count
    }
}

impl Default for LayoutTree {
    fn default() -> Self {
        Self::new()
    }
}

/// 布局节点
#[derive(Debug, Clone)]
pub struct LayoutNode {
    /// 节点ID
    pub id: String,
    /// 标签名
    pub tag_name: String,
    /// 盒模型
    pub box_model: BoxModel,
    /// 子节点
    pub children: Vec<LayoutNode>,
    /// 样式属性
    pub style_properties: HashMap<String, String>,
    /// 定位方式
    pub positioning: Positioning,
}

/// 盒模型
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxModel {
    /// 内容宽度
    pub content_width: f32,
    /// 内容高度
    pub content_height: f32,
    /// 内边距
    pub padding: EdgeInsets,
    /// 边框
    pub border: EdgeInsets,
    /// 外边距
    pub margin: EdgeInsets,
    /// 显示类型
    pub display: DisplayType,
}

impl Default for BoxModel {
    fn default() -> Self {
        Self {
            content_width: 0.0,
            content_height: 0.0,
            padding: EdgeInsets::zero(),
            border: EdgeInsets::zero(),
            margin: EdgeInsets::zero(),
            display: DisplayType::Block,
        }
    }
}

impl BoxModel {
    /// 获取总宽度（内容 + 内边距 + 边框 + 外边距）
    pub fn total_width(&self) -> f32 {
        self.margin.left
            + self.border.left
            + self.padding.left
            + self.content_width
            + self.padding.right
            + self.border.right
            + self.margin.right
    }

    /// 获取总高度
    pub fn total_height(&self) -> f32 {
        self.margin.top
            + self.border.top
            + self.padding.top
            + self.content_height
            + self.padding.bottom
            + self.border.bottom
            + self.margin.bottom
    }

    /// 获取内容区域
    pub fn content_rect(&self, x: f32, y: f32) -> Rect {
        Rect::new(
            x + self.margin.left + self.border.left + self.padding.left,
            y + self.margin.top + self.border.top + self.padding.top,
            self.content_width,
            self.content_height,
        )
    }
}

/// 边距
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeInsets {
    /// 上
    pub top: f32,
    /// 右
    pub right: f32,
    /// 下
    pub bottom: f32,
    /// 左
    pub left: f32,
}

impl EdgeInsets {
    /// 创建零边距
    pub fn zero() -> Self {
        Self {
            top: 0.0,
            right: 0.0,
            bottom: 0.0,
            left: 0.0,
        }
    }

    /// 创建统一边距
    pub fn all(value: f32) -> Self {
        Self {
            top: value,
            right: value,
            bottom: value,
            left: value,
        }
    }

    /// 创建水平/垂直边距
    pub fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Self {
            top: vertical,
            right: horizontal,
            bottom: vertical,
            left: horizontal,
        }
    }
}

/// 显示类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayType {
    /// 块级
    Block,
    /// 行内
    Inline,
    /// 行内块
    InlineBlock,
    /// Flex容器
    Flex,
    /// Grid容器
    Grid,
    /// 不显示
    None,
}

/// 定位方式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Positioning {
    /// 静态（默认）
    Static,
    /// 相对定位
    Relative,
    /// 绝对定位
    Absolute,
    /// 固定定位
    Fixed,
    /// 粘性定位
    Sticky,
}

/// 计算后的布局
#[derive(Debug, Clone)]
pub struct ComputedLayout {
    /// 节点布局映射
    pub nodes: HashMap<String, ComputedBox>,
    /// 视口
    pub viewport: Viewport,
}

impl ComputedLayout {
    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 获取节点的计算盒子
    pub fn get_box(&self, node_id: &str) -> Option<&ComputedBox> {
        self.nodes.get(node_id)
    }
}

/// 计算后的盒子
#[derive(Debug, Clone)]
pub struct ComputedBox {
    /// 矩形区域
    pub rect: Rect,
    /// 盒模型
    pub box_model: BoxModel,
}

/// 布局配置
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// 默认字体大小
    pub default_font_size: f32,
    /// 默认行高
    pub default_line_height: f32,
    /// 最大布局宽度
    pub max_width: Option<f32>,
    /// 启用Flex布局
    pub enable_flex: bool,
    /// 启用Grid布局
    pub enable_grid: bool,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            default_font_size: 16.0,
            default_line_height: 1.5,
            max_width: None,
            enable_flex: true,
            enable_grid: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_engine_creation() {
        let engine = LayoutEngine::new(LayoutConfig::default());
        assert_eq!(engine.config().default_font_size, 16.0);
    }

    #[test]
    fn test_box_model() {
        let box_model = BoxModel {
            content_width: 100.0,
            content_height: 50.0,
            padding: EdgeInsets::all(10.0),
            border: EdgeInsets::all(5.0),
            margin: EdgeInsets::all(15.0),
            display: DisplayType::Block,
        };

        // 总宽度 = 15 + 5 + 10 + 100 + 10 + 5 + 15 = 160
        assert_eq!(box_model.total_width(), 160.0);

        // 总高度 = 15 + 5 + 10 + 50 + 10 + 5 + 15 = 110
        assert_eq!(box_model.total_height(), 110.0);
    }

    #[test]
    fn test_edge_insets() {
        let zero = EdgeInsets::zero();
        assert_eq!(zero.top, 0.0);

        let all = EdgeInsets::all(10.0);
        assert_eq!(all.left, 10.0);

        let sym = EdgeInsets::symmetric(20.0, 10.0);
        assert_eq!(sym.left, 20.0);
        assert_eq!(sym.top, 10.0);
    }

    #[test]
    fn test_computed_layout() {
        let mut layout = ComputedLayout {
            nodes: HashMap::new(),
            viewport: Viewport::new(800, 600),
        };

        layout.nodes.insert(
            "test".to_string(),
            ComputedBox {
                rect: Rect::new(0.0, 0.0, 100.0, 100.0),
                box_model: BoxModel::default(),
            },
        );

        assert_eq!(layout.node_count(), 1);
        assert!(layout.get_box("test").is_some());
    }
}
