// browerai-renderer: 布局与绘制引擎
// 输入: DOM (from HTML parser) + ComputedStyle (from CSS)
// 输出: LayoutBox 树 + PaintCommand 序列

pub mod dom;
pub mod error;
pub mod layout;
pub mod paint;

pub use dom::*;
pub use error::*;
pub use layout::*;
pub use paint::*;

use browerai_cache::CacheStore;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

/// 渲染管线入口: DOM + Style → LayoutBox → PaintCommand
/// 集成了缓存层以提高性能
pub struct RenderingPipeline {
    /// 布局缓存 (缓存 LayoutBox 树)
    layout_cache: Arc<CacheStore<LayoutBox>>,
    /// 绘制缓存 (缓存 PaintCommand 序列)
    paint_cache: Arc<CacheStore<Vec<PaintCommand>>>,
}

impl RenderingPipeline {
    /// 创建新的渲染管线
    pub fn new() -> Self {
        Self {
            layout_cache: Arc::new(CacheStore::new()),
            paint_cache: Arc::new(CacheStore::new()),
        }
    }

    /// 标准渲染流程（不使用缓存）
    pub fn render(
        dom: &DomNode,
        styles: &std::collections::HashMap<String, ComputedStyle>,
    ) -> anyhow::Result<(LayoutBox, Vec<PaintCommand>)> {
        // 步骤 1: 布局
        let layout_box = layout::LayoutEngine::compute(dom, styles)?;

        // 步骤 2: 绘制
        let paint_commands = paint::PaintEngine::generate(&layout_box)?;

        Ok((layout_box, paint_commands))
    }

    /// 带缓存的渲染流程（异步）
    pub async fn render_with_cache(
        &self,
        dom: &DomNode,
        styles: &std::collections::HashMap<String, ComputedStyle>,
    ) -> anyhow::Result<(LayoutBox, Vec<PaintCommand>)> {
        // 生成缓存键 (基于 DOM 内容的哈希)
        let cache_key = format!("layout_{:x}", self.hash_dom(dom));

        // 检查布局缓存
        if let Some(cached_layout) = self.layout_cache.get(&cache_key).await.ok().flatten() {
            // 缓存命中，直接生成绘制命令
            let paint_commands = paint::PaintEngine::generate(&cached_layout)?;
            return Ok((cached_layout, paint_commands));
        }

        // 缓存未命中，执行完整渲染
        let layout_box = layout::LayoutEngine::compute(dom, styles)?;

        // 缓存布局结果 (TTL: 5 分钟)
        let _ = self
            .layout_cache
            .set(&cache_key, layout_box.clone(), Duration::from_secs(300))
            .await;

        // 生成绘制命令
        let paint_commands = paint::PaintEngine::generate(&layout_box)?;

        Ok((layout_box, paint_commands))
    }

    /// 获取缓存指标
    pub fn get_cache_metrics(&self) -> String {
        format!(
            "缓存指标:\nLayout Cache:\n{}\n\nPaint Cache:\n{}",
            self.layout_cache.export_prometheus_metrics(),
            self.paint_cache.export_prometheus_metrics()
        )
    }

    /// 内部方法: 计算 DOM 的哈希值
    fn hash_dom(&self, dom: &DomNode) -> u64 {
        let mut hasher = DefaultHasher::new();
        format!("{:?}", dom).hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for RenderingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rendering_pipeline() {
        // 简单页面: <div>Hello</div>
        let dom = DomNode::Element {
            tag: "div".to_string(),
            attributes: Default::default(),
            children: vec![DomNode::Text("Hello".to_string())],
        };

        let mut styles = std::collections::HashMap::new();
        styles.insert(
            "div".to_string(),
            ComputedStyle {
                display: Display::Block,
                width: Some(100.0),
                height: Some(50.0),
                ..Default::default()
            },
        );

        let result = RenderingPipeline::render(&dom, &styles);
        assert!(result.is_ok(), "Rendering pipeline should succeed");

        let (layout, commands) = result.unwrap();
        assert!(!commands.is_empty(), "Should generate paint commands");
    }
}
