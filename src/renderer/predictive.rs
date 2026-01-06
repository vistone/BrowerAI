/// Predictive rendering system for optimized page loads
///
/// Uses AI to predict and pre-render content before it's needed
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Priority level for rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RenderPriority {
    Critical = 3,
    High = 2,
    Medium = 1,
    Low = 0,
}

/// Information about a renderable element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderElement {
    pub element_id: String,
    pub element_type: String,
    pub priority: RenderPriority,
    pub viewport_visible: bool,
    pub predicted_visible_soon: bool,
    pub estimated_render_time_ms: f64,
}

impl RenderElement {
    pub fn new(element_id: String, element_type: String) -> Self {
        Self {
            element_id,
            element_type,
            priority: RenderPriority::Medium,
            viewport_visible: false,
            predicted_visible_soon: false,
            estimated_render_time_ms: 0.0,
        }
    }

    pub fn calculate_priority_score(&self) -> i32 {
        let mut score = self.priority as i32 * 100;

        if self.viewport_visible {
            score += 500;
        }
        if self.predicted_visible_soon {
            score += 200;
        }

        // Penalize slow-to-render elements
        score -= (self.estimated_render_time_ms / 10.0) as i32;

        score
    }
}

/// Predictive rendering engine
pub struct PredictiveRenderer {
    /// Queue of elements to render
    render_queue: VecDeque<RenderElement>,
    /// Rendered elements
    rendered: HashMap<String, RenderElement>,
    /// Scroll history for prediction
    scroll_history: VecDeque<f64>,
    /// Max scroll history size
    max_history: usize,
}

impl PredictiveRenderer {
    pub fn new() -> Self {
        Self {
            render_queue: VecDeque::new(),
            rendered: HashMap::new(),
            scroll_history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Add an element to the render queue
    pub fn queue_element(&mut self, mut element: RenderElement) {
        // Predict if element will be visible soon based on scroll patterns
        element.predicted_visible_soon = self.predict_visibility(&element);

        // Insert in priority order
        let score = element.calculate_priority_score();

        let insert_pos = self
            .render_queue
            .iter()
            .position(|existing| score > existing.calculate_priority_score());

        match insert_pos {
            Some(pos) => self.render_queue.insert(pos, element),
            None => self.render_queue.push_back(element),
        }
    }

    /// Process the next element in the queue
    pub fn process_next(&mut self) -> Option<RenderElement> {
        if let Some(element) = self.render_queue.pop_front() {
            let element_id = element.element_id.clone();
            self.rendered.insert(element_id, element.clone());
            Some(element)
        } else {
            None
        }
    }

    /// Process multiple elements up to a time budget
    pub fn process_batch(&mut self, time_budget_ms: f64) -> Vec<RenderElement> {
        let mut processed = Vec::new();
        let mut time_used = 0.0;

        while time_used < time_budget_ms {
            if let Some(element) = self.render_queue.front() {
                if time_used + element.estimated_render_time_ms > time_budget_ms {
                    break;
                }

                if let Some(element) = self.process_next() {
                    time_used += element.estimated_render_time_ms;
                    processed.push(element);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        processed
    }

    /// Update scroll position for prediction
    pub fn update_scroll_position(&mut self, position: f64) {
        self.scroll_history.push_back(position);
        if self.scroll_history.len() > self.max_history {
            self.scroll_history.pop_front();
        }
    }

    /// Predict if an element will be visible soon
    fn predict_visibility(&self, _element: &RenderElement) -> bool {
        if self.scroll_history.len() < 2 {
            return false;
        }

        // Simple prediction: if scrolling down quickly, predict elements below viewport
        let recent: Vec<_> = self.scroll_history.iter().rev().take(5).collect();
        if recent.len() < 2 {
            return false;
        }

        let velocity = recent[0] - recent[recent.len() - 1];

        // If scrolling down with significant velocity, predict visibility
        velocity > 50.0
    }

    /// Get rendering statistics
    pub fn get_stats(&self) -> RenderStats {
        RenderStats {
            queue_size: self.render_queue.len(),
            rendered_count: self.rendered.len(),
            high_priority_queued: self
                .render_queue
                .iter()
                .filter(|e| e.priority >= RenderPriority::High)
                .count(),
        }
    }

    /// Clear the render queue
    pub fn clear_queue(&mut self) {
        self.render_queue.clear();
    }

    /// Check if an element has been rendered
    pub fn is_rendered(&self, element_id: &str) -> bool {
        self.rendered.contains_key(element_id)
    }

    /// Get a rendered element
    pub fn get_rendered(&self, element_id: &str) -> Option<&RenderElement> {
        self.rendered.get(element_id)
    }
}

impl Default for PredictiveRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for predictive rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderStats {
    pub queue_size: usize,
    pub rendered_count: usize,
    pub high_priority_queued: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_element_creation() {
        let element = RenderElement::new("elem1".to_string(), "div".to_string());
        assert_eq!(element.element_id, "elem1");
        assert_eq!(element.priority, RenderPriority::Medium);
    }

    #[test]
    fn test_render_element_priority_score() {
        let mut element = RenderElement::new("elem1".to_string(), "div".to_string());
        let base_score = element.calculate_priority_score();

        element.viewport_visible = true;
        let visible_score = element.calculate_priority_score();
        assert!(visible_score > base_score);
    }

    #[test]
    fn test_predictive_renderer_creation() {
        let renderer = PredictiveRenderer::new();
        let stats = renderer.get_stats();
        assert_eq!(stats.queue_size, 0);
        assert_eq!(stats.rendered_count, 0);
    }

    #[test]
    fn test_queue_element() {
        let mut renderer = PredictiveRenderer::new();
        let element = RenderElement::new("elem1".to_string(), "div".to_string());

        renderer.queue_element(element);
        let stats = renderer.get_stats();
        assert_eq!(stats.queue_size, 1);
    }

    #[test]
    fn test_queue_priority_order() {
        let mut renderer = PredictiveRenderer::new();

        let mut low = RenderElement::new("low".to_string(), "div".to_string());
        low.priority = RenderPriority::Low;

        let mut high = RenderElement::new("high".to_string(), "div".to_string());
        high.priority = RenderPriority::Critical;

        renderer.queue_element(low);
        renderer.queue_element(high);

        let next = renderer.process_next().unwrap();
        assert_eq!(next.element_id, "high");
    }

    #[test]
    fn test_process_next() {
        let mut renderer = PredictiveRenderer::new();
        let element = RenderElement::new("elem1".to_string(), "div".to_string());

        renderer.queue_element(element);
        let processed = renderer.process_next();

        assert!(processed.is_some());
        assert_eq!(processed.unwrap().element_id, "elem1");
        assert!(renderer.is_rendered("elem1"));
    }

    #[test]
    fn test_process_batch() {
        let mut renderer = PredictiveRenderer::new();

        for i in 0..5 {
            let mut element = RenderElement::new(format!("elem{}", i), "div".to_string());
            element.estimated_render_time_ms = 10.0;
            renderer.queue_element(element);
        }

        let processed = renderer.process_batch(30.0);
        assert_eq!(processed.len(), 3); // 3 elements at 10ms each = 30ms
    }

    #[test]
    fn test_update_scroll_position() {
        let mut renderer = PredictiveRenderer::new();

        renderer.update_scroll_position(0.0);
        renderer.update_scroll_position(100.0);

        assert_eq!(renderer.scroll_history.len(), 2);
    }

    #[test]
    fn test_is_rendered() {
        let mut renderer = PredictiveRenderer::new();
        let element = RenderElement::new("elem1".to_string(), "div".to_string());

        assert!(!renderer.is_rendered("elem1"));

        renderer.queue_element(element);
        renderer.process_next();

        assert!(renderer.is_rendered("elem1"));
    }

    #[test]
    fn test_get_rendered() {
        let mut renderer = PredictiveRenderer::new();
        let element = RenderElement::new("elem1".to_string(), "div".to_string());

        renderer.queue_element(element);
        renderer.process_next();

        let rendered = renderer.get_rendered("elem1");
        assert!(rendered.is_some());
        assert_eq!(rendered.unwrap().element_id, "elem1");
    }

    #[test]
    fn test_clear_queue() {
        let mut renderer = PredictiveRenderer::new();
        let element = RenderElement::new("elem1".to_string(), "div".to_string());

        renderer.queue_element(element);
        renderer.clear_queue();

        let stats = renderer.get_stats();
        assert_eq!(stats.queue_size, 0);
    }

    #[test]
    fn test_render_stats() {
        let mut renderer = PredictiveRenderer::new();

        let mut high = RenderElement::new("high".to_string(), "div".to_string());
        high.priority = RenderPriority::High;

        renderer.queue_element(high);

        let stats = renderer.get_stats();
        assert_eq!(stats.high_priority_queued, 1);
    }
}
