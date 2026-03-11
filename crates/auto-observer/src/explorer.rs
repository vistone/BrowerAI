//! 自动探索器
//! 使用无头浏览器自动探索网站

use crate::*;
use anyhow::Result;
use playwright::api::Viewport;
use playwright::Playwright;
use std::collections::{HashMap, HashSet, VecDeque};
use tokio::time::{timeout, Duration};

/// 自动探索器
pub struct AutoExplorer {
    config: ExplorationConfig,
    playwright: Option<Playwright>,
    visited_urls: HashSet<String>,
    url_queue: VecDeque<(String, usize)>, // (url, depth)
    observations: Vec<Observation>,
    pages: Vec<PageExploration>,
    errors: Vec<ExplorationError>,
    start_time: Option<DateTime<Utc>>,
}

impl AutoExplorer {
    pub fn new(config: ExplorationConfig) -> Self {
        Self {
            config,
            playwright: None,
            visited_urls: HashSet::new(),
            url_queue: VecDeque::new(),
            observations: Vec::new(),
            pages: Vec::new(),
            errors: Vec::new(),
            start_time: None,
        }
    }

    /// 开始探索
    pub async fn explore(&mut self, start_url: &str) -> Result<ExplorationReport> {
        log::info!("Starting exploration of: {}", start_url);

        self.start_time = Some(Utc::now());
        self.url_queue.push_back((start_url.to_string(), 0));

        // 初始化 Playwright
        let playwright = Playwright::initialize().await?;
        self.playwright = Some(playwright);

        // 启动浏览器
        let browser = self
            .playwright
            .as_ref()
            .unwrap()
            .chromium()
            .launcher()
            .headless(true)
            .launch()
            .await?;

        // 创建浏览器上下文
        let context = browser
            .context_builder()
            .viewport(Some(Viewport {
                width: self.config.viewport.width as i32,
                height: self.config.viewport.height as i32,
            }))
            .build()
            .await?;

        // 主探索循环
        while let Some((url, depth)) = self.url_queue.pop_front() {
            // 检查限制条件
            if self.should_stop() {
                log::info!("Stopping exploration due to limits");
                break;
            }

            if depth > self.config.max_depth {
                continue;
            }

            if self.visited_urls.contains(&url) {
                continue;
            }

            // 探索页面
            match self.explore_page(&context, &url, depth).await {
                Ok(page_exploration) => {
                    self.visited_urls.insert(url.clone());
                    self.pages.push(page_exploration);
                }
                Err(e) => {
                    log::error!("Failed to explore page {}: {}", url, e);
                    self.errors.push(ExplorationError {
                        timestamp: Utc::now(),
                        url: url.clone(),
                        action: "explore_page".to_string(),
                        error_message: e.to_string(),
                        recoverable: true,
                    });
                }
            }
        }

        // 关闭浏览器
        browser.close().await?;

        // 生成报告
        let report = self.generate_report(start_url);

        log::info!(
            "Exploration completed. Pages: {}, Observations: {}",
            report.pages_explored.len(),
            report.total_observations
        );

        Ok(report)
    }

    /// 探索单个页面
    async fn explore_page(
        &mut self,
        context: &playwright::api::BrowserContext,
        url: &str,
        depth: usize,
    ) -> Result<PageExploration> {
        log::info!("Exploring page: {} (depth: {})", url, depth);

        let page = context.new_page().await?;

        // 导航到页面
        page.goto_builder(url).goto().await?;

        // 获取页面基本信息
        let title = page.title().await?;
        let current_url = page.url()?;

        // 收集页面元素
        let elements = self.collect_elements(&page).await?;

        let mut page_exploration = PageExploration {
            url: current_url.clone(),
            title,
            visit_count: 1,
            interactions: Vec::new(),
            elements_found: elements.clone(),
            explored_elements: Vec::new(),
        };

        // 按优先级排序交互元素
        let interactive_elements = self.prioritize_elements(&elements);

        // 探索每个交互元素
        for element_info in interactive_elements {
            if self.should_stop() {
                break;
            }

            match self.interact_with_element(&page, &element_info).await {
                Ok(interaction) => {
                    page_exploration.interactions.push(interaction.clone());
                    page_exploration
                        .explored_elements
                        .push(element_info.selector.clone());

                    // 记录观察
                    let observation = Observation {
                        timestamp: Utc::now(),
                        event_type: format!("{:?}", interaction.action),
                        target: element_info.clone(),
                        page_url: current_url.clone(),
                        details: HashMap::new(),
                        before_state: None,
                        after_state: None,
                    };
                    self.observations.push(observation);

                    // 检查是否有新链接
                    self.discover_new_links(&page, depth).await?;
                }
                Err(e) => {
                    log::warn!("Interaction failed for {}: {}", element_info.selector, e);
                }
            }
        }

        page.close(None).await?;

        Ok(page_exploration)
    }

    /// 收集页面元素
    async fn collect_elements(&self, page: &playwright::api::Page) -> Result<Vec<ElementInfo>> {
        let selectors = vec![
            "button",
            "a",
            "input",
            "select",
            "textarea",
            "[role='button']",
            "[role='link']",
            "[role='tab']",
            "[role='menuitem']",
            "[data-testid]",
            "[class*='btn']",
            "[class*='button']",
            "[class*='link']",
            "[onclick]",
        ];

        let mut all_elements = Vec::new();

        for selector in selectors {
            let elements = page.query_selector_all(selector).await?;

            for (idx, element) in elements.iter().enumerate() {
                if let Ok(info) = self
                    .extract_element_info(page, element, &format!("{}[{}]", selector, idx))
                    .await
                {
                    if info.is_visible && info.is_interactive {
                        all_elements.push(info);
                    }
                }
            }
        }

        // 去重
        let mut seen = HashSet::new();
        all_elements.retain(|e| seen.insert(e.selector.clone()));

        Ok(all_elements)
    }

    /// 提取元素信息
    async fn extract_element_info(
        &self,
        _page: &playwright::api::Page,
        element: &playwright::api::ElementHandle,
        selector: &str,
    ) -> Result<ElementInfo> {
        // 获取基本属性
        let tag = element
            .get_attribute("tagName")
            .await?
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        let id = element.get_attribute("id").await?;
        let class_attr = element.get_attribute("class").await?;
        let classes = class_attr
            .as_ref()
            .map(|c| c.split_whitespace().map(|s| s.to_string()).collect())
            .unwrap_or_default();

        // 获取所有属性 - 使用 evaluate_on_selector_all 替代
        let attributes: HashMap<String, String> = HashMap::new();

        // 获取文本内容
        let text_content = element.text_content().await?;

        // 获取边界框
        let bounding_box = element.bounding_box().await?.map(|bb| BoundingBox {
            x: bb.x,
            y: bb.y,
            width: bb.width,
            height: bb.height,
        });

        // 检查可见性和交互性
        let is_visible = element.is_visible().await.unwrap_or(false);
        let is_interactive = element.is_enabled().await.unwrap_or(false);

        Ok(ElementInfo {
            tag,
            id,
            classes,
            attributes,
            text_content,
            selector: selector.to_string(),
            bounding_box,
            is_visible,
            is_interactive,
        })
    }

    /// 优先级排序元素
    fn prioritize_elements(&self, elements: &[ElementInfo]) -> Vec<ElementInfo> {
        let mut scored: Vec<(i32, ElementInfo)> = elements
            .iter()
            .map(|e| (self.calculate_priority(e), e.clone()))
            .collect();

        // 按分数降序排序
        scored.sort_by(|a, b| b.0.cmp(&a.0));

        scored.into_iter().map(|(_, e)| e).collect()
    }

    /// 计算元素优先级
    fn calculate_priority(&self, element: &ElementInfo) -> i32 {
        let mut score = 0;

        // 基于标签类型的优先级
        score += match element.tag.as_str() {
            "button" => 100,
            "a" => 90,
            "input" => 80,
            "select" => 70,
            "textarea" => 60,
            _ => 50,
        };

        // 基于角色的优先级
        if let Some(role) = element.attributes.get("role") {
            score += match role.as_str() {
                "button" => 20,
                "link" => 15,
                "tab" => 15,
                "menuitem" => 10,
                _ => 5,
            };
        }

        // 基于位置的优先级（页面顶部的元素更重要）
        if let Some(ref bbox) = element.bounding_box {
            if bbox.y < 200.0 {
                score += 10;
            }
        }

        // 基于文本内容的优先级
        if let Some(ref text) = element.text_content {
            let lower = text.to_lowercase();
            if lower.contains("submit") || lower.contains("save") {
                score += 15;
            } else if lower.contains("cancel") || lower.contains("close") {
                score += 5;
            }
        }

        score
    }

    /// 与元素交互
    async fn interact_with_element(
        &mut self,
        page: &playwright::api::Page,
        element: &ElementInfo,
    ) -> Result<InteractionRecord> {
        let start = std::time::Instant::now();

        // 记录交互前的状态
        let before_url = page.url()?;

        // 根据元素类型选择交互方式
        let action = match element.tag.as_str() {
            "input" | "textarea" => {
                // 输入测试数据
                let test_value = self.generate_test_input(element);
                page.fill_builder(&element.selector, &test_value)
                    .fill()
                    .await?;
                InteractionAction::Input { value: test_value }
            }
            "select" => {
                // 选择第一个选项 - 使用 select_option_builder
                page.select_option_builder(&element.selector)
                    .select_option()
                    .await?;
                InteractionAction::Click
            }
            _ => {
                // 点击
                page.click_builder(&element.selector).click().await?;
                InteractionAction::Click
            }
        };

        // 等待页面稳定
        tokio::time::sleep(Duration::from_millis(self.config.wait_after_action_ms)).await;

        // 记录交互后的状态
        let after_url = page.url()?;
        let navigation_occurred = before_url != after_url;

        // 等待导航完成
        if navigation_occurred {
            let _ = timeout(
                Duration::from_millis(self.config.wait_for_navigation_ms),
                tokio::time::sleep(Duration::from_millis(100)),
            )
            .await;
        }

        let duration = start.elapsed().as_millis() as u64;

        Ok(InteractionRecord {
            timestamp: Utc::now(),
            action,
            target: element.clone(),
            result: InteractionResult {
                success: true,
                state_changed: true,
                navigation_occurred,
                new_url: if navigation_occurred {
                    Some(after_url)
                } else {
                    None
                },
                errors: vec![],
            },
            duration_ms: duration,
        })
    }

    /// 生成测试输入
    fn generate_test_input(&self, element: &ElementInfo) -> String {
        let input_type = element
            .attributes
            .get("type")
            .map(|s| s.as_str())
            .unwrap_or("text");

        match input_type {
            "email" => "test@example.com".to_string(),
            "password" => "TestPassword123!".to_string(),
            "number" => "42".to_string(),
            "tel" => "+1234567890".to_string(),
            "url" => "https://example.com".to_string(),
            "search" => "test search query".to_string(),
            _ => "test input".to_string(),
        }
    }

    /// 发现新链接
    async fn discover_new_links(
        &mut self,
        page: &playwright::api::Page,
        current_depth: usize,
    ) -> Result<()> {
        let links: Result<serde_json::Value, _> = page
            .evaluate_on_selector_all(
                "a[href]",
                r#"
            (elements) => elements
                .map(el => el.href)
                .filter(href => href && href.startsWith('http'))
        "#,
                None::<()>,
            )
            .await;

        if let Ok(urls_value) = links {
            if let Some(urls) = urls_value.as_array() {
                for url_value in urls {
                    if let Some(url) = url_value.as_str() {
                        // 检查是否应该添加
                        if self.should_explore_url(url) {
                            self.url_queue
                                .push_back((url.to_string(), current_depth + 1));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// 检查是否应该探索URL
    fn should_explore_url(&self, url: &str) -> bool {
        // 检查是否已访问
        if self.visited_urls.contains(url) {
            return false;
        }

        // 检查是否在队列中
        if self.url_queue.iter().any(|(u, _)| u == url) {
            return false;
        }

        // 检查域名限制
        if !self.config.allowed_domains.is_empty() {
            // 解析URL检查域名
            // 简化处理，实际需要更完善的URL解析
        }

        // 检查是否被阻止
        for pattern in &self.config.blocked_urls {
            if pattern.is_match(url) {
                return false;
            }
        }

        true
    }

    /// 检查是否应该停止探索
    fn should_stop(&self) -> bool {
        // 检查页面数量限制
        if self.visited_urls.len() >= self.config.max_pages {
            return true;
        }

        // 检查时间限制
        if let Some(start) = self.start_time {
            let elapsed = Utc::now().signed_duration_since(start).num_seconds() as u64;
            if elapsed >= self.config.max_time_seconds {
                return true;
            }
        }

        false
    }

    /// 生成探索报告
    fn generate_report(&self, target_url: &str) -> ExplorationReport {
        let end_time = Utc::now();

        // 计算覆盖率
        let total_elements: usize = self.pages.iter().map(|p| p.elements_found.len()).sum();

        let explored_elements: usize = self.pages.iter().map(|p| p.explored_elements.len()).sum();

        let coverage_percentage = if total_elements > 0 {
            (explored_elements as f64 / total_elements as f64) * 100.0
        } else {
            0.0
        };

        // 按类型统计
        let mut by_type: HashMap<String, TypeCoverage> = HashMap::new();
        for page in &self.pages {
            for element in &page.elements_found {
                let entry = by_type.entry(element.tag.clone()).or_insert(TypeCoverage {
                    total: 0,
                    explored: 0,
                    percentage: 0.0,
                });
                entry.total += 1;
                if page.explored_elements.contains(&element.selector) {
                    entry.explored += 1;
                }
            }
        }

        // 计算百分比
        for coverage in by_type.values_mut() {
            coverage.percentage = if coverage.total > 0 {
                (coverage.explored as f64 / coverage.total as f64) * 100.0
            } else {
                0.0
            };
        }

        // 识别行为模式
        let behaviors = self.identify_behavior_patterns();

        ExplorationReport {
            start_time: self.start_time.unwrap_or(end_time),
            end_time,
            target_url: target_url.to_string(),
            pages_explored: self.pages.clone(),
            total_observations: self.observations.len(),
            unique_behaviors: behaviors,
            coverage: CoverageReport {
                total_elements,
                explored_elements,
                coverage_percentage,
                by_type,
                unexplored_elements: vec![], // 可以计算
            },
            errors: self.errors.clone(),
        }
    }

    /// 识别行为模式
    fn identify_behavior_patterns(&self) -> Vec<BehaviorPattern> {
        let mut patterns: HashMap<String, Vec<&InteractionRecord>> = HashMap::new();

        // 按动作类型分组
        for page in &self.pages {
            for interaction in &page.interactions {
                let key = format!("{:?}", interaction.action);
                patterns.entry(key).or_default().push(interaction);
            }
        }

        patterns
            .iter()
            .map(|(action_type, interactions)| {
                let pattern_type = self.infer_pattern_type(action_type, interactions);

                BehaviorPattern {
                    pattern_id: format!("pattern_{}", action_type),
                    pattern_type,
                    trigger: interactions[0].action.clone(),
                    typical_targets: interactions
                        .iter()
                        .map(|i| i.target.selector.clone())
                        .collect::<HashSet<_>>()
                        .into_iter()
                        .collect(),
                    effects: vec![], // 可以分析
                    frequency: interactions.len(),
                    confidence: 0.8,
                }
            })
            .collect()
    }

    /// 推断模式类型
    fn infer_pattern_type(
        &self,
        action_type: &str,
        interactions: &[&InteractionRecord],
    ) -> PatternType {
        let navigation_count = interactions
            .iter()
            .filter(|i| i.result.navigation_occurred)
            .count();

        if navigation_count > interactions.len() / 2 {
            return PatternType::ClickToNavigate;
        }

        match action_type {
            "Input" => PatternType::InputWithDebounce,
            _ => PatternType::Custom(action_type.to_string()),
        }
    }
}
