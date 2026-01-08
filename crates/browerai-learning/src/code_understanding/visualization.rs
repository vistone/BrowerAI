//! 可视化生成器 - 生成架构图、依赖图等

use crate::code_understanding::{ModuleInfo, UnderstandingReport};
use anyhow::Result;

/// 可视化格式
#[derive(Debug, Clone, Copy)]
pub enum VisualizationFormat {
    /// DOT 格式（Graphviz）
    Dot,
    /// Mermaid 格式
    Mermaid,
    /// JSON 格式
    Json,
}

/// 图形可视化生成器
pub struct GraphVisualization {
    name: String,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

#[derive(Debug, Clone)]
struct Node {
    id: String,
    label: String,
    node_type: NodeType,
}

#[derive(Debug, Clone, Copy)]
enum NodeType {
    Module,
    Function,
    Data,
}

#[derive(Debug, Clone)]
struct Edge {
    from: String,
    to: String,
    label: String,
    edge_type: EdgeType,
}

#[derive(Debug, Clone, Copy)]
enum EdgeType {
    Dependency,
    DataFlow,
    EventFlow,
}

impl GraphVisualization {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_module_node(&mut self, module: &ModuleInfo) {
        self.nodes.push(Node {
            id: module.name.clone(),
            label: format!("{}\n({} lines)", module.name, module.size),
            node_type: NodeType::Module,
        });
    }

    pub fn add_dependency_edge(&mut self, from: &str, to: &str) {
        self.edges.push(Edge {
            from: from.to_string(),
            to: to.to_string(),
            label: "depends".to_string(),
            edge_type: EdgeType::Dependency,
        });
    }

    pub fn add_dataflow(&mut self, from: &str, to: &str, label: &str) {
        self.edges.push(Edge {
            from: from.to_string(),
            to: to.to_string(),
            label: label.to_string(),
            edge_type: EdgeType::DataFlow,
        });
    }

    pub fn render(&self, format: VisualizationFormat) -> Result<String> {
        match format {
            VisualizationFormat::Dot => self.render_dot(),
            VisualizationFormat::Mermaid => self.render_mermaid(),
            VisualizationFormat::Json => self.render_json(),
        }
    }

    fn render_dot(&self) -> Result<String> {
        let mut output = String::new();
        output.push_str("digraph ");
        output.push_str(&self.name.replace(' ', "_"));
        output.push_str(" {\n");

        // 图属性
        output.push_str("  rankdir=LR;\n");
        output.push_str("  node [shape=box, style=rounded];\n\n");

        // 节点
        for node in &self.nodes {
            let shape = match node.node_type {
                NodeType::Module => "box",
                NodeType::Function => "ellipse",
                NodeType::Data => "cylinder",
            };
            output.push_str(&format!(
                "  \"{}\" [label=\"{}\", shape={}];\n",
                node.id, node.label, shape
            ));
        }

        output.push_str("\n");

        // 边
        for edge in &self.edges {
            let color = match edge.edge_type {
                EdgeType::Dependency => "blue",
                EdgeType::DataFlow => "red",
                EdgeType::EventFlow => "green",
            };
            output.push_str(&format!(
                "  \"{}\" -> \"{}\" [label=\"{}\", color={}];\n",
                edge.from, edge.to, edge.label, color
            ));
        }

        output.push_str("}\n");
        Ok(output)
    }

    fn render_mermaid(&self) -> Result<String> {
        let mut output = String::new();
        output.push_str("graph LR\n");

        // 节点
        for node in &self.nodes {
            output.push_str(&format!("    {}[\"{}\"];\n", node.id, node.label));
        }

        output.push_str("\n");

        // 边
        for edge in &self.edges {
            let arrow = match edge.edge_type {
                EdgeType::Dependency => "-->|depends|",
                EdgeType::DataFlow => "-->|flow|",
                EdgeType::EventFlow => "-->|event|",
            };
            output.push_str(&format!("    {} {} {};\n", edge.from, arrow, edge.to));
        }

        Ok(output)
    }

    fn render_json(&self) -> Result<String> {
        let json = serde_json::json!({
            "name": self.name,
            "nodes": self.nodes.iter().map(|n| {
                serde_json::json!({
                    "id": n.id,
                    "label": n.label,
                    "type": format!("{:?}", n.node_type),
                })
            }).collect::<Vec<_>>(),
            "edges": self.edges.iter().map(|e| {
                serde_json::json!({
                    "from": e.from,
                    "to": e.to,
                    "label": e.label,
                    "type": format!("{:?}", e.edge_type),
                })
            }).collect::<Vec<_>>(),
        });

        Ok(serde_json::to_string_pretty(&json)?)
    }
}

/// 生成 Markdown 可视化报告
pub fn generate_markdown_visualization(report: &UnderstandingReport) -> String {
    let mut output = String::new();

    // 模块拓扑图（文字）
    output.push_str("## 模块拓扑\n\n");
    output.push_str("```\n");

    for (i, module) in report.modules.iter().enumerate() {
        output.push_str(&format!("┌─ {}\n", module.name));
        if !module.dependencies.is_empty() {
            output.push_str(&format!(
                "│  └─ depends: {}\n",
                module.dependencies.join(", ")
            ));
        }
        if !module.exports.is_empty() {
            output.push_str(&format!("│  └─ exports: {}\n", module.exports.join(", ")));
        }
        if i < report.modules.len() - 1 {
            output.push_str("│\n");
        }
    }

    output.push_str("```\n\n");

    // 依赖关系表
    output.push_str("## 依赖关系\n\n");
    output.push_str("| 模块 | 依赖于 | 导出 |\n");
    output.push_str("|------|--------|------|\n");

    for module in &report.modules {
        output.push_str(&format!(
            "| {} | {} | {} |\n",
            module.name,
            module.dependencies.join(", "),
            module.exports.join(", ")
        ));
    }

    output.push_str("\n");

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_rendering() {
        let mut viz = GraphVisualization::new("test");
        let module = ModuleInfo {
            name: "core".to_string(),
            responsibility: "test".to_string(),
            exports: vec!["run".to_string()],
            dependencies: vec![],
            functions: vec![],
            variables: vec![],
            size: 100,
        };
        viz.add_module_node(&module);

        let result = viz.render(VisualizationFormat::Dot).unwrap();
        assert!(result.contains("digraph"));
    }

    #[test]
    fn test_mermaid_rendering() {
        let mut viz = GraphVisualization::new("test");
        let module = ModuleInfo {
            name: "core".to_string(),
            responsibility: "test".to_string(),
            exports: vec!["run".to_string()],
            dependencies: vec![],
            functions: vec![],
            variables: vec![],
            size: 100,
        };
        viz.add_module_node(&module);

        let result = viz.render(VisualizationFormat::Mermaid).unwrap();
        assert!(result.contains("graph LR"));
    }
}
