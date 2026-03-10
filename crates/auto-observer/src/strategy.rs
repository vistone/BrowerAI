//! 探索策略
//! 定义不同的页面探索策略

use crate::*;
use std::collections::HashSet;

/// 探索策略 trait
#[async_trait::async_trait]
pub trait ExplorationStrategy: Send + Sync {
    /// 获取下一个要探索的元素
    async fn next_element(&mut self, elements: &[ElementInfo]) -> Option<ElementInfo>;
    
    /// 记录交互结果
    fn record_interaction(&mut self, interaction: &InteractionRecord);
    
    /// 获取策略名称
    fn name(&self) -> &str;
}

/// 优先级策略 - 基于元素重要性排序
pub struct PriorityStrategy {
    explored_selectors: HashSet<String>,
    priority_weights: HashMap<String, i32>,
}

impl PriorityStrategy {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        
        // 定义元素类型权重
        weights.insert("button".to_string(), 100);
        weights.insert("a".to_string(), 90);
        weights.insert("input".to_string(), 80);
        weights.insert("select".to_string(), 70);
        weights.insert("textarea".to_string(), 60);
        
        // 定义角色权重
        weights.insert("role:button".to_string(), 95);
        weights.insert("role:link".to_string(), 85);
        weights.insert("role:tab".to_string(), 75);
        weights.insert("role:menuitem".to_string(), 65);
        
        Self {
            explored_selectors: HashSet::new(),
            priority_weights: weights,
        }
    }

    /// 计算元素分数
    fn calculate_score(&self, element: &ElementInfo) -> i32 {
        let mut score = 0;

        // 基础类型分数
        score += self.priority_weights.get(&element.tag)
            .copied()
            .unwrap_or(50);

        // 角色分数
        if let Some(role) = element.attributes.get("role") {
            let role_key = format!("role:{}", role);
            score += self.priority_weights.get(&role_key)
                .copied()
                .unwrap_or(10);
        }

        // 位置分数（页面顶部更重要）
        if let Some(ref bbox) = element.bounding_box {
            if bbox.y < 100.0 {
                score += 20;
            } else if bbox.y < 300.0 {
                score += 10;
            }
            
            // 可见区域优先
            if bbox.y < 600.0 {
                score += 15;
            }
        }

        // 文本内容分析
        if let Some(ref text) = element.text_content {
            let lower = text.to_lowercase();
            
            // 重要操作关键词
            if lower.contains("submit") || lower.contains("save") || lower.contains("confirm") {
                score += 25;
            } else if lower.contains("login") || lower.contains("sign in") {
                score += 20;
            } else if lower.contains("get started") || lower.contains("try") {
                score += 15;
            }
            
            // 避免危险操作
            if lower.contains("delete") || lower.contains("remove") {
                score -= 30;
            }
        }

        // 数据测试ID存在通常表示重要元素
        if element.attributes.contains_key("data-testid") {
            score += 10;
        }

        score
    }
}

#[async_trait::async_trait]
impl ExplorationStrategy for PriorityStrategy {
    async fn next_element(&mut self, elements: &[ElementInfo]) -> Option<ElementInfo> {
        let mut candidates: Vec<(i32, &ElementInfo)> = elements.iter()
            .filter(|e| !self.explored_selectors.contains(&e.selector))
            .map(|e| (self.calculate_score(e), e))
            .collect();

        // 按分数降序排序
        candidates.sort_by(|a, b| b.0.cmp(&a.0));

        candidates.first().map(|(_, e)| {
            self.explored_selectors.insert(e.selector.clone());
            (*e).clone()
        })
    }

    fn record_interaction(&mut self, interaction: &InteractionRecord) {
        // 可以基于交互结果调整策略
        if !interaction.result.success {
            // 如果交互失败，可以降低类似元素的优先级
        }
    }

    fn name(&self) -> &str {
        "PriorityStrategy"
    }
}

/// 随机策略 - 随机选择元素
pub struct RandomStrategy {
    explored_selectors: HashSet<String>,
}

impl RandomStrategy {
    pub fn new() -> Self {
        Self {
            explored_selectors: HashSet::new(),
        }
    }
}

use rand::seq::SliceRandom;

#[async_trait::async_trait]
impl ExplorationStrategy for RandomStrategy {
    async fn next_element(&mut self, elements: &[ElementInfo]) -> Option<ElementInfo> {
        let candidates: Vec<&ElementInfo> = elements.iter()
            .filter(|e| !self.explored_selectors.contains(&e.selector))
            .collect();

        let chosen = candidates.choose(&mut rand::thread_rng());
        
        chosen.map(|e| {
            self.explored_selectors.insert(e.selector.clone());
            (*e).clone()
        })
    }

    fn record_interaction(&mut self, _interaction: &InteractionRecord) {
        // 随机策略不需要记录
    }

    fn name(&self) -> &str {
        "RandomStrategy"
    }
}

/// 广度优先策略 - 优先探索不同区域的元素
pub struct BreadthFirstStrategy {
    explored_selectors: HashSet<String>,
    region_count: HashMap<String, usize>,
}

impl BreadthFirstStrategy {
    pub fn new() -> Self {
        Self {
            explored_selectors: HashSet::new(),
            region_count: HashMap::new(),
        }
    }

    /// 确定元素所在区域
    fn get_region(&self, element: &ElementInfo) -> String {
        if let Some(ref bbox) = element.bounding_box {
            if bbox.y < 100.0 {
                "header".to_string()
            } else if bbox.y > 500.0 {
                "footer".to_string()
            } else {
                "main".to_string()
            }
        } else {
            "unknown".to_string()
        }
    }
}

#[async_trait::async_trait]
impl ExplorationStrategy for BreadthFirstStrategy {
    async fn next_element(&mut self, elements: &[ElementInfo]) -> Option<ElementInfo> {
        // 按区域分组
        let mut by_region: HashMap<String, Vec<&ElementInfo>> = HashMap::new();
        
        for element in elements {
            if !self.explored_selectors.contains(&element.selector) {
                let region = self.get_region(element);
                by_region.entry(region).or_default().push(element);
            }
        }

        // 选择探索最少的区域
        let target_region = by_region.keys()
            .min_by_key(|r| self.region_count.get(*r).unwrap_or(&0))?
            .clone();

        by_region.get(&target_region)
            .and_then(|elements| elements.first())
            .map(|e| {
                self.explored_selectors.insert(e.selector.clone());
                *self.region_count.entry(target_region).or_insert(0) += 1;
                (*e).clone()
            })
    }

    fn record_interaction(&mut self, _interaction: &InteractionRecord) {
    }

    fn name(&self) -> &str {
        "BreadthFirstStrategy"
    }
}

/// 智能策略 - 结合多种策略
pub struct SmartStrategy {
    strategies: Vec<Box<dyn ExplorationStrategy>>,
    current_strategy: usize,
    success_rates: HashMap<String, f64>,
}

impl SmartStrategy {
    pub fn new() -> Self {
        let strategies: Vec<Box<dyn ExplorationStrategy>> = vec![
            Box::new(PriorityStrategy::new()),
            Box::new(BreadthFirstStrategy::new()),
        ];

        Self {
            strategies,
            current_strategy: 0,
            success_rates: HashMap::new(),
        }
    }

    /// 切换到最佳策略
    fn switch_to_best_strategy(&mut self) {
        let best = self.success_rates.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(name, _)| name.clone());

        if let Some(best_name) = best {
            for (idx, strategy) in self.strategies.iter().enumerate() {
                if strategy.name() == best_name {
                    self.current_strategy = idx;
                    break;
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl ExplorationStrategy for SmartStrategy {
    async fn next_element(&mut self, elements: &[ElementInfo]) -> Option<ElementInfo> {
        // 定期切换策略
        if rand::random::<f64>() < 0.1 {
            self.current_strategy = (self.current_strategy + 1) % self.strategies.len();
        }

        self.strategies[self.current_strategy]
            .next_element(elements)
            .await
    }

    fn record_interaction(&mut self, interaction: &InteractionRecord) {
        let strategy_name = self.strategies[self.current_strategy].name().to_string();
        
        // 更新成功率
        let entry = self.success_rates.entry(strategy_name).or_insert(1.0);
        let success = if interaction.result.success { 1.0 } else { 0.0 };
        *entry = *entry * 0.9 + success * 0.1;

        // 记录到当前策略
        self.strategies[self.current_strategy].record_interaction(interaction);

        // 偶尔切换到最佳策略
        if rand::random::<f64>() < 0.05 {
            self.switch_to_best_strategy();
        }
    }

    fn name(&self) -> &str {
        "SmartStrategy"
    }
}

/// 表单专用策略
pub struct FormStrategy {
    explored_forms: HashSet<String>,
    current_form: Option<String>,
    form_state: HashMap<String, Vec<FormField>>,
}

#[derive(Debug, Clone)]
struct FormField {
    selector: String,
    field_type: String,
    filled: bool,
}

impl FormStrategy {
    pub fn new() -> Self {
        Self {
            explored_forms: HashSet::new(),
            current_form: None,
            form_state: HashMap::new(),
        }
    }

    /// 查找表单元素
    fn find_form_elements<'a>(&self, elements: &'a [ElementInfo]) -> Vec<&'a ElementInfo> {
        elements.iter()
            .filter(|e| {
                e.tag == "input" || 
                e.tag == "select" || 
                e.tag == "textarea" ||
                e.attributes.get("type") == Some(&"submit".to_string())
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl ExplorationStrategy for FormStrategy {
    async fn next_element(&mut self, elements: &[ElementInfo]) -> Option<ElementInfo> {
        let form_elements = self.find_form_elements(elements);
        
        // 优先找到未填充的输入字段
        for element in &form_elements {
            if self.explored_forms.contains(&element.selector) {
                continue;
            }

            if element.tag == "input" || element.tag == "textarea" {
                self.explored_forms.insert(element.selector.clone());
                return Some((*element).clone());
            }
        }

        // 然后找提交按钮
        for element in &form_elements {
            if element.attributes.get("type") == Some(&"submit".to_string()) {
                return Some((*element).clone());
            }
        }

        None
    }

    fn record_interaction(&mut self, _interaction: &InteractionRecord) {
        // 记录表单填写进度
    }

    fn name(&self) -> &str {
        "FormStrategy"
    }
}
