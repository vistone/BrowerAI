use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebsiteSample {
    pub url: String,
    pub html: String,
    pub css_files: Vec<CSSFile>,
    pub js_files: Vec<JSFile>,
    pub category: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CSSFile {
    pub path: String,
    pub content: String,
    pub order: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSFile {
    pub path: String,
    pub content: String,
    pub order: u32,
}

#[derive(Debug)]
pub struct WebsiteClassifier {
    // 电商特征
    ecommerce_patterns: Vec<Regex>,
    // 商业特征
    business_patterns: Vec<Regex>,
    // 新闻特征
    news_patterns: Vec<Regex>,
    // 教育特征
    education_patterns: Vec<Regex>,
    // 娱乐特征
    entertainment_patterns: Vec<Regex>,
    // 社交特征
    social_patterns: Vec<Regex>,
    // 政府特征
    government_patterns: Vec<Regex>,
    // 文档特征
    documentation_patterns: Vec<Regex>,
    // 工具特征
    tools_patterns: Vec<Regex>,
}

impl Default for WebsiteClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl WebsiteClassifier {
    pub fn new() -> Self {
        Self {
            ecommerce_patterns: vec![
                Regex::new(r"product[-_]?(grid|card|list|item)").unwrap(),
                Regex::new(r"addToCart|add_to_cart|cart\.|shopping[-_]?cart").unwrap(),
                Regex::new(r"price|price[-_]?(tag|format)").unwrap(),
                Regex::new(r"sku|product[-_]?id|inventory|stock").unwrap(),
                Regex::new(r"checkout|order[-_]?status|payment").unwrap(),
            ],
            business_patterns: vec![
                Regex::new(r"hero\s*\{|hero[-_]?section").unwrap(),
                Regex::new(r"cta[-_]?button|call[-_]?to[-_]?action").unwrap(),
                Regex::new(r"features?\s*\{|features?[-_]?section").unwrap(),
                Regex::new(r"about[-_]?us|our[-_]?team|company[-_]?info").unwrap(),
                Regex::new(r"testimonial|client[-_]?logo|partner").unwrap(),
            ],
            news_patterns: vec![
                Regex::new(r"article\s*\{|article[-_]?content").unwrap(),
                Regex::new(r"headline|title[-_]?large").unwrap(),
                Regex::new(r"author|published[-_]?at|updated[-_]?at").unwrap(),
                Regex::new(r"category.*news|section.*news").unwrap(),
                Regex::new(r"read[-_]?more|share.*article").unwrap(),
            ],
            education_patterns: vec![
                Regex::new(r"course[-_]?(card|list|detail)").unwrap(),
                Regex::new(r"instructor|teacher|tutor").unwrap(),
                Regex::new(r"enroll|registration|syllabus").unwrap(),
                Regex::new(r"lesson|module|chapter").unwrap(),
                Regex::new(r"certificate|degree|diploma").unwrap(),
            ],
            entertainment_patterns: vec![
                Regex::new(r"media[-_]?(card|item|grid)").unwrap(),
                Regex::new(r"video[-_]?player|player.*video").unwrap(),
                Regex::new(r"trailer|preview|episode").unwrap(),
                Regex::new(r"rating|star.*rating|★").unwrap(),
                Regex::new(r"genre|category.*movie|category.*show").unwrap(),
            ],
            social_patterns: vec![
                Regex::new(r"post\s*\{|feed\s*\{").unwrap(),
                Regex::new(r"user.*profile|profile[-_]?page").unwrap(),
                Regex::new(r"follow.*er|follower").unwrap(),
                Regex::new(r"comment|reply|share.*post").unwrap(),
                Regex::new(r"avatar|profile[-_]?pic").unwrap(),
            ],
            government_patterns: vec![
                Regex::new(r"\.gov\.|government").unwrap(),
                Regex::new(r"official.*website|public.*service").unwrap(),
                Regex::new(r"application.*form|form.*submit").unwrap(),
                Regex::new(r"department|agency|bureau").unwrap(),
                Regex::new(r"policy|regulation|law").unwrap(),
            ],
            documentation_patterns: vec![
                Regex::new(r"sidebar\s*\{|navigation[-_]?sidebar").unwrap(),
                Regex::new(r"api[-_]?reference|doc[-_]?section").unwrap(),
                Regex::new(r"code[-_]?block|pre\s*code").unwrap(),
                Regex::new(r"getting[-_]?started|tutorial").unwrap(),
                Regex::new(r"version|changelog").unwrap(),
            ],
            tools_patterns: vec![
                Regex::new(r"dashboard\s*\{|dashboard[-_]?page").unwrap(),
                Regex::new(r"api[-_]?key|integration").unwrap(),
                Regex::new(r"pricing.*table|pricing[-_]?card").unwrap(),
                Regex::new(r"setting|configuration|config").unwrap(),
                Regex::new(r"analytics|metric|stat").unwrap(),
                Regex::new(r"deploy|build.*tool").unwrap(),
                Regex::new(r"repo|repository|git").unwrap(),
            ],
        }
    }

    fn calculate_pattern_score(&self, text: &str, patterns: &[Regex]) -> f64 {
        let mut score = 0.0;
        for pattern in patterns {
            if pattern.is_match(text) {
                score += 1.0;
            }
        }
        score
    }

    fn count_css_classes(&self, text: &str, class_names: &[&str]) -> f64 {
        let re = Regex::new(r"\.([a-zA-Z0-9_-]+)").unwrap();
        let counts: HashMap<String, usize> = re
            .find_iter(text)
            .map(|m| (m.as_str().trim_start_matches('.').to_string(), 1))
            .fold(HashMap::new(), |mut acc, (k, v)| {
                *acc.entry(k).or_insert(0) += v;
                acc
            });

        class_names
            .iter()
            .map(|c| counts.get(*c).unwrap_or(&0))
            .sum::<usize>() as f64
    }

    pub fn classify(&self, html: &str) -> (String, HashMap<String, f64>) {
        let combined = html.to_lowercase();

        // 计算各类别分数
        let ecommerce_score = self.calculate_pattern_score(&combined, &self.ecommerce_patterns);
        let business_score = self.calculate_pattern_score(&combined, &self.business_patterns);
        let news_score = self.calculate_pattern_score(&combined, &self.news_patterns);
        let education_score = self.calculate_pattern_score(&combined, &self.education_patterns);
        let entertainment_score =
            self.calculate_pattern_score(&combined, &self.entertainment_patterns);
        let social_score = self.calculate_pattern_score(&combined, &self.social_patterns);
        let government_score = self.calculate_pattern_score(&combined, &self.government_patterns);
        let documentation_score =
            self.calculate_pattern_score(&combined, &self.documentation_patterns);
        let tools_score = self.calculate_pattern_score(&combined, &self.tools_patterns);

        // CSS类名分析
        let ecommerce_classes = self.count_css_classes(
            &combined,
            &[
                "product",
                "cart",
                "price",
                "checkout",
                "sku",
                "inventory",
                "add-to-cart",
                "addToCart",
                "product-card",
                "product-grid",
                "shopping-cart",
                "order-summary",
                "payment-form",
                "billing-address",
                "shipping-method",
                "promo-code",
                "discount",
                "coupon",
                "wishlist",
                "compare-products",
                "review-stars",
                "rating-stars",
                "breadcrumb",
                "filter-sidebar",
                "sort-dropdown",
                "product-details",
                "variant-selector",
                "size-guide",
                "color-swatch",
                "quantity-selector",
            ],
        );
        let news_classes =
            self.count_css_classes(&combined, &["article", "headline", "author", "meta"]);
        let news_classes_extended = self.count_css_classes(
            &combined,
            &[
                "breaking-news",
                "featured-story",
                "latest-update",
                "trending-topic",
                "headline-text",
                "byline-author",
                "timestamp-date",
                "editorial-content",
                "press-release",
            ],
        );
        let social_classes =
            self.count_css_classes(&combined, &["post", "feed", "profile", "avatar", "comment"]);
        let tool_classes = self.count_css_classes(
            &combined,
            &[
                "dashboard",
                "setting",
                "config",
                "analytics",
                "metric",
                "repo",
                "repository",
                "deploy",
                "preview",
                "production",
                "terminal",
                "command",
                "domain",
                "hosting",
                "build",
                "stat",
                "widget",
                "chart",
                "graph",
                "timeline",
                "api-key",
                "integration",
                "pipeline",
                "workflow",
                "monitor",
                "debug",
                "log",
                "server",
                "database",
                "cluster",
            ],
        );
        let doc_classes =
            self.count_css_classes(&combined, &["sidebar", "code-block", "api", "example"]);
        let government_classes = self.count_css_classes(
            &combined,
            &[
                "header-official",
                "gov-banner",
                "service-card",
                "form-application",
                "permit-license",
                "tax-info",
            ],
        );

        // 语义特征
        let has_shop = combined.contains("shop")
            || combined.contains("buy")
            || combined.contains("store")
            || combined.contains("product")
            || combined.contains("cart")
            || combined.contains("checkout")
            || combined.contains("payment")
            || combined.contains("order");
        let has_news =
            combined.contains("news") || combined.contains("article") || combined.contains("blog");
        let has_education = combined.contains("course")
            || combined.contains("learn")
            || combined.contains("education");
        let has_entertainment =
            combined.contains("video") || combined.contains("movie") || combined.contains("show");
        let has_social = combined.contains("login")
            || combined.contains("sign")
            || combined.contains("register");
        let has_social_network = combined.contains("follow")
            || combined.contains("share")
            || combined.contains("like")
            || combined.contains("friend")
            || combined.contains("community")
            || combined.contains("notification")
            || combined.contains("message")
            || combined.contains("chat")
            || combined.contains("connection");
        let has_news_extended = combined.contains("headline")
            || combined.contains("byline")
            || combined.contains("published")
            || combined.contains("updated")
            || combined.contains("editorial")
            || combined.contains("press")
            || combined.contains("journalist")
            || combined.contains("breaking")
            || combined.contains("trending");
        let has_government = combined.contains("tax")
            || combined.contains("permit")
            || combined.contains("license")
            || combined.contains("citizen")
            || combined.contains("service")
            || combined.contains("government")
            || combined.contains("official")
            || combined.contains("agency")
            || combined.contains("department");
        let has_contact =
            combined.contains("contact") || combined.contains("about") || combined.contains("team");
        let has_api =
            combined.contains("api") || combined.contains("doc") || combined.contains("reference");
        let has_pricing = combined.contains("pricing")
            || combined.contains("plan")
            || combined.contains("subscription");
        let has_deploy = combined.contains("deploy")
            || combined.contains("build")
            || combined.contains("hosting");
        let has_repo =
            combined.contains("repo") || combined.contains("git") || combined.contains("commit");

        // 综合分数
        let mut category_scores: HashMap<String, f64> = HashMap::new();

        // 电商检测 - 增强权重
        category_scores.insert(
            "ecommerce".to_string(),
            ecommerce_score * 1.5 + ecommerce_classes * 0.5 + if has_shop { 0.5 } else { 0.0 },
        );
        // 商业网站检测
        category_scores.insert(
            "business".to_string(),
            business_score + if has_contact { 0.3 } else { 0.0 },
        );
        // 新闻检测 - 增强权重
        category_scores.insert(
            "news".to_string(),
            news_score * 1.5
                + news_classes * 0.5
                + news_classes_extended * 0.5
                + if has_news { 0.5 } else { 0.0 }
                + if has_news_extended { 0.5 } else { 0.0 },
        );
        category_scores.insert(
            "education".to_string(),
            education_score + if has_education { 0.5 } else { 0.0 },
        );
        category_scores.insert(
            "entertainment".to_string(),
            entertainment_score + if has_entertainment { 0.5 } else { 0.0 },
        );
        category_scores.insert(
            "social".to_string(),
            social_score * 1.5
                + social_classes * 0.5
                + if has_social { 0.5 } else { 0.0 }
                + if has_social_network { 0.5 } else { 0.0 },
        );
        category_scores.insert(
            "government".to_string(),
            government_score * 1.5
                + government_classes * 0.5
                + if has_government { 0.5 } else { 0.0 },
        );
        category_scores.insert(
            "documentation".to_string(),
            documentation_score * 1.5 + doc_classes * 0.5 + if has_api { 0.3 } else { 0.0 },
        );
        category_scores.insert(
            "tools".to_string(),
            tools_score * 1.5
                + tool_classes * 0.5
                + if has_pricing { 0.2 } else { 0.0 }
                + if has_deploy { 0.2 } else { 0.0 }
                + if has_repo { 0.2 } else { 0.0 },
        );

        // 归一化分数
        let total: f64 = category_scores.values().sum();
        let normalized: HashMap<String, f64> = if total > 0.0 {
            category_scores
                .iter()
                .map(|(k, v)| (k.clone(), v / total))
                .collect()
        } else {
            category_scores.clone()
        };

        // 选择最高分数的类别
        let best_category = category_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "business".to_string());

        (best_category, normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_ecommerce() {
        let classifier = WebsiteClassifier::new();

        let html = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <div class="product-card">
                <h3>Product Name</h3>
                <p class="price">$99.99</p>
                <button onclick="addToCart()">Add to Cart</button>
            </div>
        </body>
        </html>
        "#;

        let (category, scores) = classifier.classify(html);
        assert_eq!(category, "ecommerce");
        assert!(scores.get("ecommerce").unwrap() > &0.0);
    }

    #[test]
    fn test_classify_news() {
        let classifier = WebsiteClassifier::new();

        let html = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <article class="article-content">
                <h2 class="headline">Breaking News</h2>
                <p class="meta">By Author | Published today</p>
            </article>
        </body>
        </html>
        "#;

        let (category, scores) = classifier.classify(html);
        assert_eq!(category, "news");
        assert!(scores.get("news").unwrap() > &0.0);
    }

    #[test]
    fn test_classify_tools() {
        let classifier = WebsiteClassifier::new();

        let html = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <div class="dashboard">
                <div class="analytics-chart"></div>
                <div class="repo-list"></div>
            </div>
            <button class="deploy-button">Deploy</button>
        </body>
        </html>
        "#;

        let (category, scores) = classifier.classify(html);
        assert_eq!(category, "tools");
        assert!(scores.get("tools").unwrap() > &0.0);
    }

    #[test]
    fn test_classify_social() {
        let classifier = WebsiteClassifier::new();

        let html = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <div class="user-profile">
                <div class="avatar"></div>
                <div class="post-feed">
                    <div class="post-card">
                        <p class="post-content">Hello world!</p>
                        <button class="like-button">Like</button>
                        <button class="share-button">Share</button>
                    </div>
                </div>
            </div>
            <button class="follow-button">Follow</button>
        </body>
        </html>
        "#;

        let (category, scores) = classifier.classify(html);
        assert_eq!(category, "social");
        assert!(scores.get("social").unwrap() > &0.0);
    }

    #[test]
    fn test_classify_government() {
        let classifier = WebsiteClassifier::new();

        let html = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <div class="header-official">
                <div class="gov-banner">Official Government Website</div>
            </div>
            <div class="service-card">
                <div class="form-application">
                    <h2>Apply for License</h2>
                    <p>Citizen Services Portal</p>
                </div>
            </div>
            <div class="tax-info">
                <h3>Tax Information</h3>
            </div>
        </body>
        </html>
        "#;

        let (category, scores) = classifier.classify(html);
        assert_eq!(category, "government");
        assert!(scores.get("government").unwrap() > &0.0);
    }

    #[test]
    fn test_classify_documentation() {
        let classifier = WebsiteClassifier::new();

        let html = r##"
        <!DOCTYPE html>
        <html>
        <body>
            <nav class="sidebar">
                <h2>Documentation</h2>
                <a href="#getting-started">Getting Started</a>
                <a href="#api">API Reference</a>
                <a href="#tutorials">Tutorials</a>
            </nav>
            <main class="main-content">
                <div class="doc-section">
                    <h2>Getting Started</h2>
                    <pre><code>npm install package</code></pre>
                </div>
            </main>
        </body>
        </html>
        "##;

        let (category, scores) = classifier.classify(html);
        assert_eq!(category, "documentation");
        assert!(scores.get("documentation").unwrap() > &0.0);
    }

    #[test]
    fn test_classify_education() {
        let classifier = WebsiteClassifier::new();

        let html = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <div class="course-card">
                <h3>Introduction to Programming</h3>
                <p class="instructor">Dr. Smith</p>
                <button class="enroll-button">Enroll Now</button>
            </div>
            <div class="syllabus">
                <h4>Course Syllabus</h4>
                <ul>
                    <li>Week 1: Variables</li>
                    <li>Week 2: Functions</li>
                </ul>
            </div>
        </body>
        </html>
        "#;

        let (category, scores) = classifier.classify(html);
        assert_eq!(category, "education");
        assert!(scores.get("education").unwrap() > &0.0);
    }

    #[test]
    fn test_classify_entertainment() {
        let classifier = WebsiteClassifier::new();

        let html = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <div class="video-player">
                <video controls>
                    <source src="movie.mp4" type="video/mp4">
                </video>
            </div>
            <div class="episode-list">
                <div class="episode-card">
                    <h4>Episode 1</h4>
                    <p class="rating">★ 4.5</p>
                </div>
            </div>
        </body>
        </html>
        "#;

        let (category, scores) = classifier.classify(html);
        assert_eq!(category, "entertainment");
        assert!(scores.get("entertainment").unwrap() > &0.0);
    }

    #[test]
    fn test_classify_business() {
        let classifier = WebsiteClassifier::new();

        let html = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <header class="hero-section">
                <h1>Welcome to Our Company</h1>
                <p class="cta-text">Get Started Today</p>
                <button class="cta-button">Contact Us</button>
            </header>
            <section class="features-section">
                <div class="feature-card">
                    <h3>Feature 1</h3>
                    <p>Amazing feature description</p>
                </div>
            </section>
            <footer class="about-footer">
                <p>About Our Team</p>
            </footer>
        </body>
        </html>
        "#;

        let (category, scores) = classifier.classify(html);
        assert_eq!(category, "business");
        assert!(scores.get("business").unwrap() > &0.0);
    }

    #[test]
    fn test_empty_html() {
        let classifier = WebsiteClassifier::new();

        let html = r#""#;

        let (category, scores) = classifier.classify(html);
        // Should return some category even for empty HTML
        assert!(scores.values().len() > 0);
    }

    #[test]
    fn test_minimal_html() {
        let classifier = WebsiteClassifier::new();

        let html = r#"<html><body></body></html>"#;

        let (category, scores) = classifier.classify(html);
        // Should return some category even for minimal HTML
        assert!(scores.values().len() > 0);
    }

    #[test]
    fn test_scores_normalize_correctly() {
        let classifier = WebsiteClassifier::new();

        let html = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <div class="product-card">
                <h3>Product</h3>
                <p class="price">$99</p>
                <button>Add to Cart</button>
            </div>
        </body>
        </html>
        "#;

        let (_, scores) = classifier.classify(html);
        let total: f64 = scores.values().sum();
        // Scores should approximately normalize to 1.0
        assert!(
            (total - 1.0).abs() < 0.001,
            "Scores should normalize to 1.0, got {}",
            total
        );
    }
}
