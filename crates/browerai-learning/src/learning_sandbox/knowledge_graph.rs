//! Knowledge Graph Builder
//!
//! Builds a knowledge graph representing the relationships between
//! page elements, features, and content.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Knowledge graph representing the analyzed page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// All nodes in the graph
    pub nodes: Vec<GraphNode>,

    /// All edges in the graph
    pub edges: Vec<GraphEdge>,

    /// Node index for fast lookup
    #[serde(skip)]
    pub node_index: HashMap<String, usize>,

    /// Graph statistics
    pub statistics: GraphStatistics,

    /// Metadata
    pub metadata: GraphMetadata,
}

/// A node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: String,

    /// Node type
    pub node_type: NodeType,

    /// Node label
    pub label: String,

    /// Node properties
    pub properties: HashMap<String, String>,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,

    /// Source location
    pub source: Option<String>,
}

/// Types of nodes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Page,
    Section,
    Component,
    Feature,
    Element,
    Link,
    Form,
    Script,
    Style,
    Media,
    Text,
    Interactive,
    Navigation,
    Metadata,
    Unknown,
}

/// An edge connecting two nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub source: String,

    /// Target node ID
    pub target: String,

    /// Edge type
    pub edge_type: EdgeType,

    /// Edge weight
    pub weight: f32,

    /// Edge properties
    pub properties: HashMap<String, String>,
}

/// Types of edges
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EdgeType {
    Contains,
    ParentOf,
    ChildOf,
    RelatedTo,
    LinksTo,
    DependsOn,
    Uses,
    Calls,
    References,
    Styles,
    ContainsText,
    HasFeature,
    NavigatesTo,
    SubmitsTo,
    Invalidates,
}

/// Statistics about the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of nodes
    pub node_count: usize,

    /// Total number of edges
    pub edge_count: usize,

    /// Number of nodes by type
    pub nodes_by_type: HashMap<String, usize>,

    /// Number of edges by type
    pub edges_by_type: HashMap<String, usize>,

    /// Graph density
    pub density: f32,

    /// Average degree
    pub average_degree: f32,

    /// Connected components
    pub connected_components: usize,

    /// Maximum depth
    pub max_depth: usize,
}

/// Metadata about the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// URL the graph was built from
    pub source_url: String,

    /// Build timestamp
    pub built_at: chrono::DateTime<chrono::Utc>,

    /// Time taken to build
    pub build_time_ms: u64,

    /// Number of elements processed
    pub elements_processed: usize,

    /// Graph version
    pub version: String,
}

/// Knowledge Graph Builder Configuration
#[derive(Debug, Clone)]
pub struct KnowledgeGraphBuilderConfig {
    /// Minimum confidence threshold for nodes
    pub min_node_confidence: f32,

    /// Minimum weight for edges
    pub min_edge_weight: f32,

    /// Include all elements (not just semantic ones)
    pub include_all_elements: bool,

    /// Maximum nodes to include
    pub max_nodes: usize,

    /// Enable edge inference
    pub infer_edges: bool,
}

impl Default for KnowledgeGraphBuilderConfig {
    fn default() -> Self {
        Self {
            min_node_confidence: 0.3,
            min_edge_weight: 0.1,
            include_all_elements: false,
            max_nodes: 1000,
            infer_edges: true,
        }
    }
}

/// Knowledge Graph Builder
///
/// Builds a knowledge graph from analyzed page content.
#[derive(Debug, Clone)]
pub struct KnowledgeGraphBuilder {
    config: KnowledgeGraphBuilderConfig,
}

impl KnowledgeGraphBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::with_config(KnowledgeGraphBuilderConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: KnowledgeGraphBuilderConfig) -> Self {
        Self { config }
    }

    /// Build a knowledge graph from analysis results
    ///
    /// # Arguments
    /// * `url` - Source URL
    /// * `structure` - Page structure analysis
    /// * `features` - Recognized features
    /// * `tech_stack` - Detected tech stack
    ///
    /// # Returns
    /// * `KnowledgeGraph` with nodes and edges
    pub async fn build(
        &self,
        url: &str,
        structure: &crate::learning_sandbox::structure_analyzer::PageStructure,
        features: &crate::learning_sandbox::feature_recognizer::FeatureMap,
        tech_stack: Option<&crate::learning_sandbox::tech_stack_detector::TechStackReport>,
    ) -> KnowledgeGraph {
        let start_time = chrono::Utc::now();

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_index = HashMap::new();

        // Create page node
        let page_node_id = self.create_page_node(url, &mut nodes, &mut node_index);

        // Add structure nodes
        self.add_structure_nodes(
            structure,
            page_node_id,
            &mut nodes,
            &mut edges,
            &mut node_index,
        );

        // Add feature nodes
        self.add_feature_nodes(
            features,
            page_node_id,
            &mut nodes,
            &mut edges,
            &mut node_index,
        );

        // Add tech stack nodes
        if let Some(ts) = tech_stack {
            self.add_tech_stack_nodes(ts, page_node_id, &mut nodes, &mut edges, &mut node_index);
        }

        // Infer additional edges
        if self.config.infer_edges {
            self.infer_edges(&mut nodes, &mut edges, &mut node_index);
        }

        // Calculate statistics
        let statistics = self.calculate_statistics(&nodes, &edges);

        let end_time = chrono::Utc::now();
        let build_time_ms = (end_time - start_time).num_milliseconds() as u64;

        let mut graph = KnowledgeGraph {
            nodes,
            edges,
            node_index,
            statistics,
            metadata: GraphMetadata {
                source_url: url.to_string(),
                built_at: chrono::Utc::now(),
                build_time_ms,
                elements_processed: structure.dom_info.total_elements,
                version: "1.0".to_string(),
            },
        };

        // Recalculate statistics with the complete graph
        graph.statistics = graph.calculate_statistics();

        graph
    }

    /// Create the main page node
    fn create_page_node(
        &self,
        url: &str,
        nodes: &mut Vec<GraphNode>,
        node_index: &mut HashMap<String, usize>,
    ) -> String {
        let node_id = "page".to_string();

        let node = GraphNode {
            id: node_id.clone(),
            node_type: NodeType::Page,
            label: url.to_string(),
            properties: HashMap::from([
                ("url".to_string(), url.to_string()),
                ("layout".to_string(), format!("{:?}", structure.layout_type)),
            ]),
            confidence: 1.0,
            source: None,
        };

        self.add_node(node, nodes, node_index);
        node_id
    }

    /// Add structure-related nodes
    fn add_structure_nodes(
        &self,
        structure: &crate::learning_sandbox::structure_analyzer::PageStructure,
        parent_id: String,
        nodes: &mut Vec<GraphNode>,
        edges: &mut Vec<GraphEdge>,
        node_index: &mut HashMap<String, usize>,
    ) {
        // Add layout node
        let layout_node_id = format!("layout-{}", structure.layout_type.as_str());
        let layout_node = GraphNode {
            id: layout_node_id.clone(),
            node_type: NodeType::Component,
            label: format!("{:?} Layout", structure.layout_type),
            properties: HashMap::from([(
                "confidence".to_string(),
                structure.layout_confidence.to_string(),
            )]),
            confidence: structure.layout_confidence,
            source: None,
        };
        self.add_node(layout_node, nodes, node_index);

        edges.push(GraphEdge {
            source: parent_id.clone(),
            target: layout_node_id,
            edge_type: EdgeType::Contains,
            weight: 1.0,
            properties: HashMap::new(),
        });

        // Add semantic structure nodes
        let sem = &structure.semantic_structure;
        let mut position = 0;

        if sem.has_header {
            let header_id = "semantic-header".to_string();
            self.add_node(
                GraphNode {
                    id: header_id.clone(),
                    node_type: NodeType::Section,
                    label: "Header Section".to_string(),
                    properties: HashMap::new(),
                    confidence: 0.9,
                    source: Some("<header>".to_string()),
                },
                nodes,
                node_index,
            );
            edges.push(GraphEdge {
                source: parent_id.clone(),
                target: header_id,
                edge_type: EdgeType::Contains,
                weight: 1.0,
                properties: HashMap::new(),
            });
            position += 1;
        }

        if sem.has_navigation {
            let nav_id = "semantic-nav".to_string();
            self.add_node(
                GraphNode {
                    id: nav_id.clone(),
                    node_type: NodeType::Navigation,
                    label: "Navigation Section".to_string(),
                    properties: HashMap::new(),
                    confidence: 0.9,
                    source: Some("<nav>".to_string()),
                },
                nodes,
                node_index,
            );
            edges.push(GraphEdge {
                source: parent_id.clone(),
                target: nav_id,
                edge_type: EdgeType::Contains,
                weight: 1.0,
                properties: HashMap::new(),
            });
            position += 1;
        }

        if sem.has_main {
            let main_id = "semantic-main".to_string();
            self.add_node(
                GraphNode {
                    id: main_id.clone(),
                    node_type: NodeType::Section,
                    label: "Main Content".to_string(),
                    properties: HashMap::from([(
                        "heading_levels".to_string(),
                        format!("{:?}", sem.heading_levels),
                    )]),
                    confidence: 0.95,
                    source: Some("<main>".to_string()),
                },
                nodes,
                node_index,
            );
            edges.push(GraphEdge {
                source: parent_id.clone(),
                target: main_id,
                edge_type: EdgeType::Contains,
                weight: 1.0,
                properties: HashMap::new(),
            });

            // Add heading info
            if !sem.heading_levels.is_empty() {
                let heading_info_id = "heading-info".to_string();
                self.add_node(
                    GraphNode {
                        id: heading_info_id.clone(),
                        node_type: NodeType::Metadata,
                        label: "Heading Hierarchy".to_string(),
                        properties: HashMap::from([(
                            "levels".to_string(),
                            format!("{:?}", sem.heading_levels),
                        )]),
                        confidence: 1.0,
                        source: None,
                    },
                    nodes,
                    node_index,
                );
                edges.push(GraphEdge {
                    source: main_id,
                    target: heading_info_id,
                    edge_type: EdgeType::RelatedTo,
                    weight: 0.5,
                    properties: HashMap::new(),
                });
            }
            position += 1;
        }

        if sem.has_footer {
            let footer_id = "semantic-footer".to_string();
            self.add_node(
                GraphNode {
                    id: footer_id.clone(),
                    node_type: NodeType::Section,
                    label: "Footer Section".to_string(),
                    properties: HashMap::new(),
                    confidence: 0.9,
                    source: Some("<footer>".to_string()),
                },
                nodes,
                node_index,
            );
            edges.push(GraphEdge {
                source: parent_id.clone(),
                target: footer_id,
                edge_type: EdgeType::Contains,
                weight: 1.0,
                properties: HashMap::new(),
            });
        }
    }

    /// Add feature nodes
    fn add_feature_nodes(
        &self,
        features: &crate::learning_sandbox::feature_recognizer::FeatureMap,
        parent_id: String,
        nodes: &mut Vec<GraphNode>,
        edges: &mut Vec<GraphEdge>,
        node_index: &mut HashMap<String, usize>,
    ) {
        // Add top-level features
        for feature in &features.features {
            if feature.confidence >= self.config.min_node_confidence {
                let feature_id =
                    format!("feature-{}", feature.name.to_lowercase().replace(' ', "-"));

                self.add_node(
                    GraphNode {
                        id: feature_id.clone(),
                        node_type: self.feature_category_to_node_type(&feature.category),
                        label: feature.name.clone(),
                        properties: feature.metadata.clone(),
                        confidence: feature.confidence,
                        source: Some(feature.location.clone()),
                    },
                    nodes,
                    node_index,
                );

                edges.push(GraphEdge {
                    source: parent_id.clone(),
                    target: feature_id.clone(),
                    edge_type: EdgeType::HasFeature,
                    weight: feature.confidence,
                    properties: HashMap::from([(
                        "category".to_string(),
                        format!("{:?}", feature.category),
                    )]),
                });

                // Add navigation-related features
                if feature.category
                    == crate::learning_sandbox::feature_recognizer::FeatureCategory::Navigation
                {
                    if let Some(nav_items) = features.navigation_items.first() {
                        let nav_id = format!("{}-nav", feature_id);
                        self.add_node(
                            GraphNode {
                                id: nav_id.clone(),
                                node_type: NodeType::Navigation,
                                label: "Navigation Items".to_string(),
                                properties: HashMap::from([(
                                    "count".to_string(),
                                    features.navigation_items.len().to_string(),
                                )]),
                                confidence: feature.confidence,
                                source: Some(nav_items.selector.clone()),
                            },
                            nodes,
                            node_index,
                        );
                        edges.push(GraphEdge {
                            source: feature_id,
                            target: nav_id,
                            edge_type: EdgeType::Contains,
                            weight: feature.confidence,
                            properties: HashMap::new(),
                        });
                    }
                }
            }
        }

        // Add form node if forms exist
        if !features.form_fields.is_empty() {
            let form_id = "feature-form".to_string();
            self.add_node(
                GraphNode {
                    id: form_id.clone(),
                    node_type: NodeType::Form,
                    label: "Form".to_string(),
                    properties: HashMap::from([(
                        "field_count".to_string(),
                        features.form_fields.len().to_string(),
                    )]),
                    confidence: 0.9,
                    source: Some("form".to_string()),
                },
                nodes,
                node_index,
            );
            edges.push(GraphEdge {
                source: parent_id,
                target: form_id,
                edge_type: EdgeType::HasFeature,
                weight: 0.9,
                properties: HashMap::new(),
            });
        }
    }

    /// Add tech stack nodes
    fn add_tech_stack_nodes(
        &self,
        tech_stack: &crate::learning_sandbox::tech_stack_detector::TechStackReport,
        parent_id: String,
        nodes: &mut Vec<GraphNode>,
        edges: &mut Vec<GraphEdge>,
        node_index: &mut HashMap<String, usize>,
    ) {
        // Add JavaScript frameworks
        for framework in &tech_stack.js_frameworks {
            let framework_id = format!("js-{}", framework.name.to_lowercase().replace(' ', "-"));

            self.add_node(
                GraphNode {
                    id: framework_id.clone(),
                    node_type: NodeType::Script,
                    label: framework.name.clone(),
                    properties: HashMap::from([
                        ("category".to_string(), framework.category.clone()),
                        (
                            "version".to_string(),
                            framework
                                .version
                                .clone()
                                .unwrap_or_else(|| "unknown".to_string()),
                        ),
                    ]),
                    confidence: framework.confidence,
                    source: framework.source.clone(),
                },
                nodes,
                node_index,
            );

            edges.push(GraphEdge {
                source: parent_id.clone(),
                target: framework_id,
                edge_type: EdgeType::Uses,
                weight: framework.confidence,
                properties: HashMap::from([("type".to_string(), "JavaScript".to_string())]),
            });
        }

        // Add CSS frameworks
        for framework in &tech_stack.css_frameworks {
            let framework_id = format!("css-{}", framework.name.to_lowercase().replace(' ', "-"));

            self.add_node(
                GraphNode {
                    id: framework_id.clone(),
                    node_type: NodeType::Style,
                    label: framework.name.clone(),
                    properties: HashMap::from([
                        ("category".to_string(), framework.category.clone()),
                        (
                            "version".to_string(),
                            framework
                                .version
                                .clone()
                                .unwrap_or_else(|| "unknown".to_string()),
                        ),
                    ]),
                    confidence: framework.confidence,
                    source: framework.source.clone(),
                },
                nodes,
                node_index,
            );

            edges.push(GraphEdge {
                source: parent_id.clone(),
                target: framework_id,
                edge_type: EdgeType::Styles,
                weight: framework.confidence,
                properties: HashMap::from([("type".to_string(), "CSS".to_string())]),
            });
        }

        // Add CMS/platform if detected
        if let Some(cms) = &tech_stack.cms_platform {
            let cms_id = "cms-platform".to_string();
            self.add_node(
                GraphNode {
                    id: cms_id.clone(),
                    node_type: NodeType::Metadata,
                    label: cms.name.clone(),
                    properties: HashMap::from([(
                        "version".to_string(),
                        cms.version.clone().unwrap_or_else(|| "unknown".to_string()),
                    )]),
                    confidence: cms.confidence,
                    source: None,
                },
                nodes,
                node_index,
            );
            edges.push(GraphEdge {
                source: parent_id,
                target: cms_id,
                edge_type: EdgeType::Uses,
                weight: cms.confidence,
                properties: HashMap::new(),
            });
        }
    }

    /// Infer additional edges based on patterns
    fn infer_edges(
        &self,
        nodes: &mut Vec<GraphNode>,
        edges: &mut Vec<GraphEdge>,
        node_index: &mut HashMap<String, usize>,
    ) {
        // Find navigation nodes and link them to related sections
        let nav_nodes: Vec<_> = nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Navigation)
            .collect();

        for nav_node in nav_nodes {
            let section_nodes: Vec<_> = nodes
                .iter()
                .filter(|n| n.node_type == NodeType::Section)
                .collect();

            for section_node in section_nodes {
                // Check if we already have an edge
                let has_edge = edges
                    .iter()
                    .any(|e| e.source == nav_node.id && e.target == section_node.id);

                if !has_edge {
                    edges.push(GraphEdge {
                        source: nav_node.id.clone(),
                        target: section_node.id.clone(),
                        edge_type: EdgeType::NavigatesTo,
                        weight: 0.5,
                        properties: HashMap::from([("inferred".to_string(), "true".to_string())]),
                    });
                }
            }
        }
    }

    /// Add a node to the graph
    fn add_node(
        &self,
        node: GraphNode,
        nodes: &mut Vec<GraphNode>,
        node_index: &mut HashMap<String, usize>,
    ) {
        if !node_index.contains_key(&node.id) && nodes.len() < self.config.max_nodes {
            let idx = nodes.len();
            nodes.push(node);
            node_index.insert(nodes[idx].id.clone(), idx);
        }
    }

    /// Convert feature category to node type
    fn feature_category_to_node_type(
        &self,
        category: &crate::learning_sandbox::feature_recognizer::FeatureCategory,
    ) -> NodeType {
        match category {
            crate::learning_sandbox::feature_recognizer::FeatureCategory::Navigation => {
                NodeType::Navigation
            }
            crate::learning_sandbox::feature_recognizer::FeatureCategory::Form => NodeType::Form,
            crate::learning_sandbox::feature_recognizer::FeatureCategory::Interactive => {
                NodeType::Interactive
            }
            crate::learning_sandbox::feature_recognizer::FeatureCategory::Media => NodeType::Media,
            crate::learning_sandbox::feature_recognizer::FeatureCategory::Search => {
                NodeType::Element
            }
            crate::learning_sandbox::feature_recognizer::FeatureCategory::Content => NodeType::Text,
            _ => NodeType::Feature,
        }
    }

    /// Calculate graph statistics
    fn calculate_statistics(&self, nodes: &[GraphNode], edges: &[GraphEdge]) -> GraphStatistics {
        let node_count = nodes.len();
        let edge_count = edges.len();

        // Count nodes by type
        let mut nodes_by_type = HashMap::new();
        for node in nodes {
            let type_name = format!("{:?}", node.node_type);
            *nodes_by_type.entry(type_name).or_insert(0) += 1;
        }

        // Count edges by type
        let mut edges_by_type = HashMap::new();
        for edge in edges {
            let type_name = format!("{:?}", edge.edge_type);
            *edges_by_type.entry(type_name).or_insert(0) += 1;
        }

        // Calculate average degree
        let total_degree: usize = edges
            .iter()
            .map(|e| if node_count > 0 { 2 } else { 0 })
            .sum();
        let average_degree = if node_count > 0 {
            total_degree as f32 / node_count as f32
        } else {
            0.0
        };

        // Calculate density (maximum possible edges / actual edges)
        let max_edges = if node_count > 1 {
            node_count * (node_count - 1) / 2
        } else {
            0
        };
        let density = if max_edges > 0 {
            edge_count as f32 / max_edges as f32
        } else {
            0.0
        };

        // Find connected components using simple BFS
        let connected_components = self.count_connected_components(nodes, edges);

        // Find maximum depth (longest path from root)
        let max_depth = self.find_max_depth(nodes, edges);

        GraphStatistics {
            node_count,
            edge_count,
            nodes_by_type,
            edges_by_type,
            density,
            average_degree,
            connected_components,
            max_depth,
        }
    }

    /// Count connected components
    fn count_connected_components(&self, nodes: &[GraphNode], edges: &[GraphEdge]) -> usize {
        if nodes.is_empty() {
            return 0;
        }

        let mut visited = HashSet::new();
        let mut components = 0;

        let node_ids: Vec<_> = nodes.iter().map(|n| n.id.clone()).collect();

        for node_id in &node_ids {
            if !visited.contains(node_id) {
                components += 1;
                self.bfs_visit(node_id, &mut visited, edges, &node_ids);
            }
        }

        components
    }

    /// BFS traversal for component detection
    fn bfs_visit(
        &self,
        start: &str,
        visited: &mut HashSet<String>,
        edges: &[GraphEdge],
        node_ids: &[String],
    ) {
        let mut queue = vec![start.to_string()];
        visited.insert(start.to_string());

        while let Some(current) = queue.pop() {
            // Find all neighbors
            for edge in edges {
                let neighbor = if edge.source == current {
                    Some(&edge.target)
                } else if edge.target == current {
                    Some(&edge.source)
                } else {
                    None
                };

                if let Some(n) = neighbor {
                    if !visited.contains(*n) {
                        visited.insert(*n.clone());
                        queue.push(*n.clone());
                    }
                }
            }
        }
    }

    /// Find maximum depth in the graph
    fn find_max_depth(&self, nodes: &[GraphNode], edges: &[GraphEdge]) -> usize {
        // Find the page node
        let page_node = nodes.iter().find(|n| n.node_type == NodeType::Page);

        if let Some(page) = page_node {
            self.dfs_depth(&page.id, edges, &mut HashSet::new(), 0)
        } else {
            0
        }
    }

    /// DFS to find depth
    fn dfs_depth(
        &self,
        node_id: &str,
        edges: &[GraphEdge],
        visited: &mut HashSet<String>,
        current_depth: usize,
    ) -> usize {
        if visited.contains(node_id) {
            return current_depth;
        }

        visited.insert(node_id.to_string());

        let mut max_depth = current_depth;

        for edge in edges {
            if edge.source == node_id && edge.weight >= self.config.min_edge_weight {
                let depth = self.dfs_depth(&edge.target, edges, visited, current_depth + 1);
                if depth > max_depth {
                    max_depth = depth;
                }
            }
        }

        max_depth
    }
}

impl KnowledgeGraph {
    /// Calculate statistics for the graph
    pub fn calculate_statistics(&mut self) -> GraphStatistics {
        self.statistics = self.calculate_statistics();
        self.statistics.clone()
    }

    /// Calculate statistics (internal)
    fn calculate_statistics(&self) -> GraphStatistics {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();

        // Count nodes by type
        let mut nodes_by_type = HashMap::new();
        for node in &self.nodes {
            let type_name = format!("{:?}", node.node_type);
            *nodes_by_type.entry(type_name).or_insert(0) += 1;
        }

        // Count edges by type
        let mut edges_by_type = HashMap::new();
        for edge in &self.edges {
            let type_name = format!("{:?}", edge.edge_type);
            *edges_by_type.entry(type_name).or_insert(0) += 1;
        }

        // Calculate density
        let max_edges = if node_count > 1 {
            node_count * (node_count - 1) / 2
        } else {
            0
        };
        let density = if max_edges > 0 {
            edge_count as f32 / max_edges as f32
        } else {
            0.0
        };

        // Average degree
        let total_degree = edge_count * 2;
        let average_degree = if node_count > 0 {
            total_degree as f32 / node_count as f32
        } else {
            0.0
        };

        GraphStatistics {
            node_count,
            edge_count,
            nodes_by_type,
            edges_by_type,
            density,
            average_degree,
            connected_components: 1,
            max_depth: 0,
        }
    }
}

impl Default for KnowledgeGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = KnowledgeGraphBuilder::new();
        assert!(builder.config.min_node_confidence > 0.0);
    }

    #[tokio::test]
    async fn test_graph_building() {
        let builder = KnowledgeGraphBuilder::new();

        let structure = crate::learning_sandbox::structure_analyzer::PageStructure {
            layout_type: crate::learning_sandbox::structure_analyzer::LayoutType::SingleColumn,
            layout_confidence: 0.8,
            dom_info: crate::learning_sandbox::structure_analyzer::DomInfo {
                max_depth: 5,
                total_elements: 50,
                text_nodes: 10,
                average_depth: 2.5,
                max_siblings: 10,
                depth_distribution: HashMap::new(),
            },
            element_distribution:
                crate::learning_sandbox::structure_analyzer::ElementDistribution {
                    semantic_elements: HashMap::new(),
                    interactive_elements: HashMap::new(),
                    media_elements: HashMap::new(),
                    form_elements: HashMap::new(),
                    top_tags: vec![("div".to_string(), 20)],
                    total_count: 50,
                },
            patterns: Vec::new(),
            semantic_structure: crate::learning_sandbox::structure_analyzer::SemanticStructure {
                has_header: true,
                has_navigation: true,
                has_main: true,
                has_footer: true,
                has_sidebar: false,
                has_article: true,
                has_section: true,
                has_aside: false,
                heading_levels: HashMap::from([(1, 1), (2, 3)]),
                organization_score: 0.8,
            },
            complexity_metrics: crate::learning_sandbox::structure_analyzer::ComplexityMetrics {
                structural_complexity: 0.3,
                nesting_complexity: 0.2,
                element_variety: 0.4,
                complexity_rating:
                    crate::learning_sandbox::structure_analyzer::ComplexityLevel::Simple,
                estimated_render_time: 5.0,
            },
            metadata: crate::learning_sandbox::structure_analyzer::StructureMetadata {
                url: "https://example.com".to_string(),
                analyzed_at: chrono::Utc::now(),
                analysis_time_ms: 100,
                html_size: 1000,
                characters_processed: 1000,
            },
        };

        let features = crate::learning_sandbox::feature_recognizer::FeatureMap {
            features: vec![crate::learning_sandbox::feature_recognizer::Feature {
                name: "Navigation".to_string(),
                category: crate::learning_sandbox::feature_recognizer::FeatureCategory::Navigation,
                location: "nav".to_string(),
                confidence: 0.9,
                metadata: HashMap::new(),
            }],
            form_fields: Vec::new(),
            navigation_items: Vec::new(),
            interactive_elements: Vec::new(),
            content_sections: Vec::new(),
            metadata: crate::learning_sandbox::feature_recognizer::FeatureMetadata {
                url: "https://example.com".to_string(),
                extracted_at: chrono::Utc::now(),
                extraction_time_ms: 50,
                elements_scanned: 100,
                features_extracted: 1,
            },
        };

        let graph = builder
            .build("https://example.com", &structure, &features, None)
            .await;

        assert!(graph.nodes.len() > 0);
        assert!(graph.edges.len() > 0);
        assert!(graph.statistics.node_count > 0);
    }

    #[tokio::test]
    async fn test_graph_statistics() {
        let builder = KnowledgeGraphBuilder::new();

        let structure = crate::learning_sandbox::structure_analyzer::PageStructure::default();
        let features = crate::learning_sandbox::feature_recognizer::FeatureMap::default();

        let graph = builder
            .build("https://example.com", &structure, &features, None)
            .await;

        let stats = &graph.statistics;
        assert!(stats.node_count > 0);
        assert!(stats.density >= 0.0 && stats.density <= 1.0);
    }

    #[tokio::test]
    async fn test_node_lookup() {
        let builder = KnowledgeGraphBuilder::new();

        let structure = crate::learning_sandbox::structure_analyzer::PageStructure::default();
        let features = crate::learning_sandbox::feature_recognizer::FeatureMap::default();

        let graph = builder
            .build("https://example.com", &structure, &features, None)
            .await;

        // Test node lookup
        let page_node = graph.node_index.get("page");
        assert!(page_node.is_some());

        if let Some(&idx) = page_node {
            assert_eq!(graph.nodes[idx].id, "page");
        }
    }
}
