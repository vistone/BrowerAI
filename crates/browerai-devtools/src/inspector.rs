//! DOM Inspector - DOM检查器
//!
//! 提供DOM树的查看、搜索和编辑功能：
//! - 节点遍历
//! - 属性查看和修改
//! - CSS样式检查
//! - 搜索和过滤

use browerai_core::Result;
use browerai_dom::Document;
use std::collections::HashMap;

/// DOM检查器
#[derive(Debug, Clone)]
pub struct DomInspector {
    /// 配置
    config: InspectorConfig,
    /// 选中的节点ID
    selected_node: Option<String>,
}

impl DomInspector {
    /// 创建新的DOM检查器
    pub fn new() -> Self {
        Self {
            config: InspectorConfig::default(),
            selected_node: None,
        }
    }

    /// 使用配置创建检查器
    pub fn with_config(config: InspectorConfig) -> Self {
        Self {
            config,
            selected_node: None,
        }
    }

    /// 检查文档
    pub fn inspect(&self, _document: &Document) -> Result<InspectionResult> {
        // 简化实现：Document API可能需要适配
        let node_count = 0;
        let max_depth = 0;

        Ok(InspectionResult {
            node_count,
            max_depth,
            document_type: "HTML".to_string(),
            selected_node: self.selected_node.clone(),
        })
    }

    /// 获取节点信息
    pub fn get_node_info(&self, _document: &Document, node_id: &str) -> Option<NodeInfo> {
        // 简化实现
        Some(NodeInfo {
            id: node_id.to_string(),
            tag_name: "div".to_string(),
            attributes: HashMap::new(),
            styles: HashMap::new(),
            child_count: 0,
        })
    }

    /// 搜索节点
    pub fn search_nodes(&self, _document: &Document, query: &str) -> Vec<String> {
        // 简化实现：返回空列表
        log::info!("Searching for nodes matching: {}", query);
        Vec::new()
    }

    /// 选择节点
    pub fn select_node(&mut self, node_id: &str) {
        self.selected_node = Some(node_id.to_string());
        log::info!("Selected node: {}", node_id);
    }

    /// 获取选中的节点
    pub fn selected_node(&self) -> Option<&str> {
        self.selected_node.as_deref()
    }

    /// 获取检查器配置
    pub fn config(&self) -> &InspectorConfig {
        &self.config
    }

    /// 清除选择
    pub fn clear_selection(&mut self) {
        self.selected_node = None;
    }

    /// 高亮节点（简化实现）
    pub fn highlight_node(&self, node_id: &str) {
        log::info!("Highlighting node: {}", node_id);
    }
}

impl Default for DomInspector {
    fn default() -> Self {
        Self::new()
    }
}

/// 检查结果
#[derive(Debug, Clone)]
pub struct InspectionResult {
    /// 节点数量
    pub node_count: usize,
    /// 最大深度
    pub max_depth: usize,
    /// 文档类型
    pub document_type: String,
    /// 选中的节点
    pub selected_node: Option<String>,
}

/// 节点信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeInfo {
    /// 节点ID
    pub id: String,
    /// 标签名
    pub tag_name: String,
    /// 属性
    pub attributes: HashMap<String, String>,
    /// 计算样式
    pub styles: HashMap<String, String>,
    /// 子节点数量
    pub child_count: usize,
}

impl NodeInfo {
    /// 创建新的节点信息
    pub fn new(id: impl Into<String>, tag_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tag_name: tag_name.into(),
            attributes: HashMap::new(),
            styles: HashMap::new(),
            child_count: 0,
        }
    }

    /// 添加属性
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// 添加样式
    pub fn with_style(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.styles.insert(key.into(), value.into());
        self
    }
}

/// 检查器配置
#[derive(Debug, Clone)]
pub struct InspectorConfig {
    /// 显示注释节点
    pub show_comments: bool,
    /// 显示文本节点
    pub show_text_nodes: bool,
    /// 高亮选中节点
    pub highlight_selected: bool,
    /// 最大搜索深度
    pub max_search_depth: usize,
}

impl Default for InspectorConfig {
    fn default() -> Self {
        Self {
            show_comments: false,
            show_text_nodes: true,
            highlight_selected: true,
            max_search_depth: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspector_creation() {
        let inspector = DomInspector::new();
        assert!(inspector.selected_node().is_none());
    }

    #[test]
    fn test_node_selection() {
        let mut inspector = DomInspector::new();

        inspector.select_node("node-1");
        assert_eq!(inspector.selected_node(), Some("node-1"));

        inspector.clear_selection();
        assert!(inspector.selected_node().is_none());
    }

    #[test]
    fn test_node_info() {
        let info = NodeInfo::new("test-id", "div")
            .with_attribute("class", "container")
            .with_style("color", "red");

        assert_eq!(info.id, "test-id");
        assert_eq!(info.tag_name, "div");
        assert!(info.attributes.contains_key("class"));
        assert!(info.styles.contains_key("color"));
    }

    #[test]
    fn test_inspector_config() {
        let config = InspectorConfig::default();
        assert!(config.show_text_nodes);
        assert!(!config.show_comments);
    }
}
