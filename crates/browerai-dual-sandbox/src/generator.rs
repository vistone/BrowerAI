//! 生成器 - 基于学习到的组件和功能意图，生成全新网站
//!
//! 核心：保持功能等价，但全新UI体验

use crate::common::StyleSystem;
use crate::component_extractor::ComponentLibrary;
use crate::function_generator::{generate_js_file, FunctionGenerator, TargetFramework};
use crate::js_understander::FunctionIntents;
use crate::style_transform::TransformType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 网站生成器
pub struct WebsiteGenerator {
    /// 组件库
    components: ComponentLibrary,
    /// 功能意图
    intents: FunctionIntents,
    /// 样式系统
    styles: StyleSystem,
}

/// 生成的网站
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedWebsite {
    /// HTML结构（全新生成，非复制）
    pub html: String,
    /// CSS样式（基于学习到的模式）
    pub css: String,
    /// JS功能（基于意图重新实现）
    pub js: String,
    /// 组件树
    pub component_tree: ComponentTree,
    /// 元数据
    pub metadata: WebsiteMetadata,
}

/// 组件树
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentTree {
    /// 根节点
    pub root: ComponentNode,
    /// 所有组件
    pub components: Vec<ComponentNode>,
}

/// 组件节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentNode {
    /// 组件ID
    pub id: String,
    /// 组件类型
    pub component_type: String,
    /// 子组件
    pub children: Vec<ComponentNode>,
    /// 属性
    pub props: HashMap<String, String>,
    /// 样式类
    pub classes: Vec<String>,
}

/// 网站元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebsiteMetadata {
    pub title: String,
    pub description: String,
    pub components_used: Vec<String>,
    pub features_implemented: Vec<String>,
}

impl WebsiteGenerator {
    pub fn new(
        components: ComponentLibrary,
        intents: FunctionIntents,
        styles: StyleSystem,
    ) -> Self {
        Self {
            components,
            intents,
            styles,
        }
    }

    /// 生成全新网站
    pub fn generate(&self, config: GenerationConfig) -> GeneratedWebsite {
        // 1. 构建组件树（基于意图，而非复制）
        let component_tree = self.build_component_tree(&config);

        // 2. 生成HTML
        let html = self.generate_html(&component_tree);

        // 3. 生成CSS（基于学习到的样式模式）
        let css = self.generate_css(&component_tree);

        // 4. 生成JS（基于功能意图重新实现，保持功能等价）
        let js = self.generate_functional_js();

        GeneratedWebsite {
            html,
            css,
            js,
            component_tree,
            metadata: WebsiteMetadata {
                title: config.title,
                description: config.description,
                components_used: self.list_components_used(),
                features_implemented: self.list_features(),
            },
        }
    }

    /// 构建组件树
    fn build_component_tree(&self, config: &GenerationConfig) -> ComponentTree {
        // 根据网站类型和意图，智能组合组件
        let root = match config.website_type {
            WebsiteType::LandingPage => self.build_landing_page_layout(),
            WebsiteType::Dashboard => self.build_dashboard_layout(),
            WebsiteType::Blog => self.build_blog_layout(),
            WebsiteType::Ecommerce => self.build_ecommerce_layout(),
            _ => self.build_generic_layout(),
        };

        ComponentTree {
            root: root.clone(),
            components: self.flatten_tree(&root),
        }
    }

    /// 构建落地页布局
    fn build_landing_page_layout(&self) -> ComponentNode {
        ComponentNode {
            id: "root".to_string(),
            component_type: "div".to_string(),
            children: vec![
                // Header
                ComponentNode {
                    id: "header".to_string(),
                    component_type: "header".to_string(),
                    children: vec![self.build_nav_component()],
                    props: HashMap::new(),
                    classes: vec!["site-header".to_string()],
                },
                // Hero Section
                ComponentNode {
                    id: "hero".to_string(),
                    component_type: "section".to_string(),
                    children: vec![self.build_hero_component()],
                    props: HashMap::new(),
                    classes: vec!["hero-section".to_string()],
                },
                // Features Section
                ComponentNode {
                    id: "features".to_string(),
                    component_type: "section".to_string(),
                    children: self.build_feature_cards(),
                    props: HashMap::new(),
                    classes: vec!["features-section".to_string()],
                },
                // CTA Section
                ComponentNode {
                    id: "cta".to_string(),
                    component_type: "section".to_string(),
                    children: vec![self.build_cta_component()],
                    props: HashMap::new(),
                    classes: vec!["cta-section".to_string()],
                },
                // Footer
                ComponentNode {
                    id: "footer".to_string(),
                    component_type: "footer".to_string(),
                    children: vec![],
                    props: HashMap::new(),
                    classes: vec!["site-footer".to_string()],
                },
            ],
            props: HashMap::new(),
            classes: vec!["landing-page".to_string()],
        }
    }

    /// 构建仪表盘布局
    fn build_dashboard_layout(&self) -> ComponentNode {
        ComponentNode {
            id: "root".to_string(),
            component_type: "div".to_string(),
            children: vec![
                ComponentNode {
                    id: "sidebar".to_string(),
                    component_type: "aside".to_string(),
                    children: vec![self.build_nav_component()],
                    props: HashMap::new(),
                    classes: vec!["sidebar".to_string()],
                },
                ComponentNode {
                    id: "main".to_string(),
                    component_type: "main".to_string(),
                    children: vec![
                        ComponentNode {
                            id: "header".to_string(),
                            component_type: "header".to_string(),
                            children: vec![],
                            props: HashMap::new(),
                            classes: vec!["main-header".to_string()],
                        },
                        ComponentNode {
                            id: "content".to_string(),
                            component_type: "div".to_string(),
                            children: self.build_dashboard_widgets(),
                            props: HashMap::new(),
                            classes: vec!["dashboard-content".to_string()],
                        },
                    ],
                    props: HashMap::new(),
                    classes: vec!["main-area".to_string()],
                },
            ],
            props: HashMap::new(),
            classes: vec!["dashboard-layout".to_string()],
        }
    }

    /// 构建博客布局
    fn build_blog_layout(&self) -> ComponentNode {
        ComponentNode {
            id: "root".to_string(),
            component_type: "div".to_string(),
            children: vec![
                self.build_header(),
                ComponentNode {
                    id: "content".to_string(),
                    component_type: "div".to_string(),
                    children: vec![
                        ComponentNode {
                            id: "articles".to_string(),
                            component_type: "main".to_string(),
                            children: self.build_article_list(),
                            props: HashMap::new(),
                            classes: vec!["articles".to_string()],
                        },
                        ComponentNode {
                            id: "sidebar".to_string(),
                            component_type: "aside".to_string(),
                            children: vec![],
                            props: HashMap::new(),
                            classes: vec!["sidebar".to_string()],
                        },
                    ],
                    props: HashMap::new(),
                    classes: vec!["content-wrapper".to_string()],
                },
                self.build_footer(),
            ],
            props: HashMap::new(),
            classes: vec!["blog-layout".to_string()],
        }
    }

    /// 构建电商布局
    fn build_ecommerce_layout(&self) -> ComponentNode {
        ComponentNode {
            id: "root".to_string(),
            component_type: "div".to_string(),
            children: vec![
                self.build_header(),
                ComponentNode {
                    id: "product-grid".to_string(),
                    component_type: "div".to_string(),
                    children: self.build_product_cards(),
                    props: HashMap::new(),
                    classes: vec!["product-grid".to_string()],
                },
                self.build_footer(),
            ],
            props: HashMap::new(),
            classes: vec!["ecommerce-layout".to_string()],
        }
    }

    fn build_generic_layout(&self) -> ComponentNode {
        ComponentNode {
            id: "root".to_string(),
            component_type: "div".to_string(),
            children: vec![
                self.build_header(),
                ComponentNode {
                    id: "main".to_string(),
                    component_type: "main".to_string(),
                    children: vec![],
                    props: HashMap::new(),
                    classes: vec!["main-content".to_string()],
                },
                self.build_footer(),
            ],
            props: HashMap::new(),
            classes: vec!["generic-layout".to_string()],
        }
    }

    // 辅助构建方法
    fn build_header(&self) -> ComponentNode {
        ComponentNode {
            id: "header".to_string(),
            component_type: "header".to_string(),
            children: vec![self.build_nav_component()],
            props: HashMap::new(),
            classes: vec!["site-header".to_string()],
        }
    }

    fn build_footer(&self) -> ComponentNode {
        ComponentNode {
            id: "footer".to_string(),
            component_type: "footer".to_string(),
            children: vec![],
            props: HashMap::new(),
            classes: vec!["site-footer".to_string()],
        }
    }

    fn build_nav_component(&self) -> ComponentNode {
        ComponentNode {
            id: "nav".to_string(),
            component_type: "nav".to_string(),
            children: vec![ComponentNode {
                id: "logo".to_string(),
                component_type: "a".to_string(),
                children: vec![],
                props: {
                    let mut p = HashMap::new();
                    p.insert("href".to_string(), "/".to_string());
                    p
                },
                classes: vec!["logo".to_string()],
            }],
            props: HashMap::new(),
            classes: vec!["main-nav".to_string()],
        }
    }

    fn build_hero_component(&self) -> ComponentNode {
        ComponentNode {
            id: "hero-content".to_string(),
            component_type: "div".to_string(),
            children: vec![
                ComponentNode {
                    id: "hero-title".to_string(),
                    component_type: "h1".to_string(),
                    children: vec![],
                    props: HashMap::new(),
                    classes: vec!["hero-title".to_string()],
                },
                ComponentNode {
                    id: "hero-cta".to_string(),
                    component_type: "button".to_string(),
                    children: vec![],
                    props: HashMap::new(),
                    classes: vec!["btn".to_string(), "btn-primary".to_string()],
                },
            ],
            props: HashMap::new(),
            classes: vec!["hero-content".to_string()],
        }
    }

    fn build_cta_component(&self) -> ComponentNode {
        ComponentNode {
            id: "cta-content".to_string(),
            component_type: "div".to_string(),
            children: vec![ComponentNode {
                id: "cta-button".to_string(),
                component_type: "button".to_string(),
                children: vec![],
                props: HashMap::new(),
                classes: vec!["btn".to_string(), "btn-large".to_string()],
            }],
            props: HashMap::new(),
            classes: vec!["cta-content".to_string()],
        }
    }

    fn build_feature_cards(&self) -> Vec<ComponentNode> {
        (0..3)
            .map(|i| ComponentNode {
                id: format!("feature-{}", i),
                component_type: "div".to_string(),
                children: vec![],
                props: HashMap::new(),
                classes: vec!["feature-card".to_string()],
            })
            .collect()
    }

    fn build_dashboard_widgets(&self) -> Vec<ComponentNode> {
        (0..4)
            .map(|i| ComponentNode {
                id: format!("widget-{}", i),
                component_type: "div".to_string(),
                children: vec![],
                props: HashMap::new(),
                classes: vec!["dashboard-widget".to_string()],
            })
            .collect()
    }

    fn build_article_list(&self) -> Vec<ComponentNode> {
        (0..5)
            .map(|i| ComponentNode {
                id: format!("article-{}", i),
                component_type: "article".to_string(),
                children: vec![],
                props: HashMap::new(),
                classes: vec!["article-card".to_string()],
            })
            .collect()
    }

    fn build_product_cards(&self) -> Vec<ComponentNode> {
        (0..6)
            .map(|i| ComponentNode {
                id: format!("product-{}", i),
                component_type: "div".to_string(),
                children: vec![],
                props: HashMap::new(),
                classes: vec!["product-card".to_string()],
            })
            .collect()
    }

    fn flatten_tree(&self, root: &ComponentNode) -> Vec<ComponentNode> {
        let mut result = vec![root.clone()];
        for child in &root.children {
            result.extend(self.flatten_tree(child));
        }
        result
    }

    /// 生成HTML
    fn generate_html(&self, tree: &ComponentTree) -> String {
        self.render_node(&tree.root)
    }

    fn render_node(&self, node: &ComponentNode) -> String {
        let class_attr = if node.classes.is_empty() {
            String::new()
        } else {
            format!(" class=\"{}\"", node.classes.join(" "))
        };

        let props_attr = node
            .props
            .iter()
            .map(|(k, v)| format!(" {}=\"{}\"", k, v))
            .collect::<String>();

        let children_html = node
            .children
            .iter()
            .map(|c| self.render_node(c))
            .collect::<String>();

        format!(
            "<{}{}{}>{}</{ }>",
            node.component_type, class_attr, props_attr, children_html, node.component_type
        )
    }

    /// 生成CSS
    fn generate_css(&self, tree: &ComponentTree) -> String {
        let mut css = String::new();

        // 生成CSS变量
        css.push_str(":root {\n");
        for (i, color) in self.styles.colors.primary_colors.iter().enumerate() {
            css.push_str(&format!("  --color-primary-{}: {};\n", i, color.hex));
        }
        css.push_str("}\n\n");

        // 为每个组件生成样式
        for component in &tree.components {
            let component_css = self.generate_component_css(component);
            css.push_str(&component_css);
        }

        css
    }

    fn generate_component_css(&self, node: &ComponentNode) -> String {
        let selector = format!(".{}", node.classes.join("."));
        format!(
            "{} {{\n  /* Component: {} */\n}}\n\n",
            selector, node.component_type
        )
    }

    /// 生成JS（基于意图重新实现）
    /// 生成功能等价的JS代码（使用功能生成器）
    fn generate_functional_js(&self) -> String {
        // 使用功能生成器，基于学习到的意图重新实现
        let function_generator =
            FunctionGenerator::new(self.intents.clone(), self.components.clone());

        let functions = function_generator.generate(TargetFramework::VanillaJs);

        generate_js_file(&functions)
    }

    fn list_components_used(&self) -> Vec<String> {
        vec![
            "Header".to_string(),
            "Navigation".to_string(),
            "Hero".to_string(),
            "Card".to_string(),
            "Button".to_string(),
            "Footer".to_string(),
        ]
    }

    fn list_features(&self) -> Vec<String> {
        vec![
            "Responsive Navigation".to_string(),
            "Interactive Buttons".to_string(),
            "Form Validation".to_string(),
        ]
    }
}

/// 生成配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub title: String,
    pub description: String,
    pub website_type: WebsiteType,
    pub theme: TransformType,
    pub primary_color: String,
    pub secondary_color: String,
}

/// 网站类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebsiteType {
    LandingPage,
    Dashboard,
    Blog,
    Ecommerce,
    Documentation,
    Portfolio,
    Generic,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            title: "Generated Website".to_string(),
            description: "A website generated by AI".to_string(),
            website_type: WebsiteType::LandingPage,
            theme: TransformType::Original,
            primary_color: "#3498db".to_string(),
            secondary_color: "#2ecc71".to_string(),
        }
    }
}
