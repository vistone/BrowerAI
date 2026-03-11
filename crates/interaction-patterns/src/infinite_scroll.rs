//! 无限滚动模式

use crate::*;
use anyhow::Result;

pub struct InfiniteScrollPattern;

impl InfiniteScrollPattern {
    pub fn new() -> Self {
        Self
    }

    pub fn create_pattern(
        container: &str,
        _item_selector: &str,
        _threshold: u32,
    ) -> InteractionPattern {
        InteractionPattern {
            pattern_type: ComplexPatternType::InfiniteScroll,
            name: "Infinite Scroll".to_string(),
            description: "Load more content as user scrolls".to_string(),
            triggers: vec![
                PatternTrigger {
                    trigger_type: TriggerType::Scroll,
                    selector: container.to_string(),
                    conditions: vec![],
                },
                PatternTrigger {
                    trigger_type: TriggerType::Intersection,
                    selector: ".scroll-sentinel".to_string(),
                    conditions: vec![],
                },
            ],
            behaviors: vec![
                PatternBehavior {
                    behavior_type: BehaviorType::LoadData,
                    target: container.to_string(),
                    animation: None,
                    callback: Some("onLoadMore".to_string()),
                },
                PatternBehavior {
                    behavior_type: BehaviorType::InsertElement,
                    target: container.to_string(),
                    animation: Some(AnimationConfig {
                        duration_ms: 300,
                        easing: "ease-out".to_string(),
                        properties: vec!["opacity".to_string(), "transform".to_string()],
                    }),
                    callback: None,
                },
            ],
            state_machine: PatternStateMachine {
                initial_state: "idle".to_string(),
                states: vec![
                    PatternState {
                        name: "idle".to_string(),
                        description: "Waiting for scroll".to_string(),
                        entry_actions: vec![],
                        exit_actions: vec![],
                    },
                    PatternState {
                        name: "loading".to_string(),
                        description: "Loading more data".to_string(),
                        entry_actions: vec![PatternBehavior {
                            behavior_type: BehaviorType::AddClass,
                            target: ".loading-indicator".to_string(),
                            animation: None,
                            callback: None,
                        }],
                        exit_actions: vec![PatternBehavior {
                            behavior_type: BehaviorType::RemoveClass,
                            target: ".loading-indicator".to_string(),
                            animation: None,
                            callback: None,
                        }],
                    },
                    PatternState {
                        name: "completed".to_string(),
                        description: "No more data to load".to_string(),
                        entry_actions: vec![PatternBehavior {
                            behavior_type: BehaviorType::Show,
                            target: ".end-message".to_string(),
                            animation: None,
                            callback: None,
                        }],
                        exit_actions: vec![],
                    },
                ],
                transitions: vec![
                    StateTransition {
                        from_state: "idle".to_string(),
                        to_state: "loading".to_string(),
                        trigger: "scroll".to_string(),
                        guard: Some("nearBottom".to_string()),
                        actions: vec![],
                    },
                    StateTransition {
                        from_state: "loading".to_string(),
                        to_state: "idle".to_string(),
                        trigger: "dataLoaded".to_string(),
                        guard: Some("hasMoreData".to_string()),
                        actions: vec![PatternBehavior {
                            behavior_type: BehaviorType::InsertElement,
                            target: container.to_string(),
                            animation: Some(AnimationConfig {
                                duration_ms: 300,
                                easing: "ease-out".to_string(),
                                properties: vec!["opacity".to_string()],
                            }),
                            callback: None,
                        }],
                    },
                    StateTransition {
                        from_state: "loading".to_string(),
                        to_state: "completed".to_string(),
                        trigger: "dataLoaded".to_string(),
                        guard: Some("noMoreData".to_string()),
                        actions: vec![],
                    },
                ],
            },
            confidence: 0.85,
        }
    }
}

impl Default for InfiniteScrollPattern {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternImplementation for InfiniteScrollPattern {
    fn pattern_type(&self) -> ComplexPatternType {
        ComplexPatternType::InfiniteScroll
    }

    fn recognize(&self, observations: &[auto_observer::Observation]) -> Option<InteractionPattern> {
        let mut scroll_events = 0;
        let _has_loading_indicator = false;
        let mut has_new_content = false;

        for obs in observations {
            match obs.event_type.as_str() {
                "scroll" => scroll_events += 1,
                "mutation" => {
                    if let Some(details) = obs.details.get("addedNodes") {
                        if details.as_array().map(|a| !a.is_empty()).unwrap_or(false) {
                            has_new_content = true;
                        }
                    }
                }
                _ => {}
            }
        }

        // 如果有多次滚动和新内容添加，可能是无限滚动
        if scroll_events > 5 && has_new_content {
            Some(Self::create_pattern(".infinite-scroll", ".item", 100))
        } else {
            None
        }
    }

    fn generate_code(
        &self,
        _pattern: &InteractionPattern,
        language: CodeLanguage,
    ) -> Result<GeneratedCode> {
        let code = match language {
            CodeLanguage::TypeScript => self.generate_typescript(),
            CodeLanguage::React => self.generate_react(),
            _ => anyhow::bail!("Language not supported"),
        }?;

        Ok(GeneratedCode {
            pattern_type: ComplexPatternType::InfiniteScroll,
            language,
            component_name: "InfiniteScroll".to_string(),
            code,
            css: Some(self.generate_css()),
            tests: None,
            documentation: self.generate_documentation(),
        })
    }

    fn get_template(&self) -> &str {
        ""
    }
}

impl InfiniteScrollPattern {
    fn generate_typescript(&self) -> Result<String> {
        Ok(r#"
interface InfiniteScrollOptions {
    container: HTMLElement;
    loadMore: (page: number) => Promise<any[]>;
    renderItem: (item: any) => HTMLElement;
    threshold?: number;
    pageSize?: number;
}

class InfiniteScroll {
    private options: Required<InfiniteScrollOptions>;
    private page: number = 1;
    private loading: boolean = false;
    private hasMore: boolean = true;
    private sentinel: HTMLElement;
    private observer: IntersectionObserver;

    constructor(options: InfiniteScrollOptions) {
        this.options = {
            threshold: 100,
            pageSize: 20,
            ...options
        };
        this.init();
    }

    private init() {
        // 创建哨兵元素
        this.sentinel = document.createElement('div');
        this.sentinel.className = 'scroll-sentinel';
        this.options.container.appendChild(this.sentinel);

        // 创建 IntersectionObserver
        this.observer = new IntersectionObserver(
            (entries) => this.handleIntersection(entries),
            { rootMargin: `${this.options.threshold}px` }
        );

        this.observer.observe(this.sentinel);

        // 初始加载
        this.loadMore();
    }

    private handleIntersection(entries: IntersectionObserverEntry[]) {
        const entry = entries[0];
        if (entry.isIntersecting && !this.loading && this.hasMore) {
            this.loadMore();
        }
    }

    private async loadMore() {
        if (this.loading || !this.hasMore) return;

        this.loading = true;
        this.showLoading();

        try {
            const items = await this.options.loadMore(this.page);

            if (items.length === 0) {
                this.hasMore = false;
                this.showEndMessage();
            } else {
                items.forEach(item => {
                    const element = this.options.renderItem(item);
                    this.options.container.insertBefore(element, this.sentinel);
                });
                this.page++;
            }
        } catch (error) {
            console.error('Failed to load more:', error);
            this.showError();
        } finally {
            this.loading = false;
            this.hideLoading();
        }
    }

    private showLoading() {
        this.sentinel.classList.add('loading');
    }

    private hideLoading() {
        this.sentinel.classList.remove('loading');
    }

    private showEndMessage() {
        this.sentinel.classList.add('completed');
        this.sentinel.textContent = 'No more items';
    }

    private showError() {
        this.sentinel.classList.add('error');
        this.sentinel.textContent = 'Failed to load. Click to retry.';
        this.sentinel.onclick = () => {
            this.sentinel.classList.remove('error');
            this.loadMore();
        };
    }

    refresh() {
        this.page = 1;
        this.hasMore = true;
        this.options.container.innerHTML = '';
        this.options.container.appendChild(this.sentinel);
        this.loadMore();
    }

    destroy() {
        this.observer.disconnect();
        this.sentinel.remove();
    }
}

export { InfiniteScroll, InfiniteScrollOptions };
"#
        .to_string())
    }

    fn generate_react(&self) -> Result<String> {
        Ok(r#"
import React, { useState, useEffect, useRef, useCallback } from 'react';

interface InfiniteScrollProps<T> {
    items: T[];
    renderItem: (item: T, index: number) => React.ReactNode;
    loadMore: () => Promise<T[]>;
    hasMore: boolean;
    threshold?: number;
    loadingComponent?: React.ReactNode;
    endComponent?: React.ReactNode;
    className?: string;
}

export function InfiniteScroll<T>({
    items,
    renderItem,
    loadMore,
    hasMore,
    threshold = 100,
    loadingComponent,
    endComponent,
    className
}: InfiniteScrollProps<T>) {
    const [loading, setLoading] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const sentinelRef = useRef<HTMLDivElement>(null);

    const handleIntersection = useCallback(async (entries: IntersectionObserverEntry[]) => {
        const entry = entries[0];
        if (entry.isIntersecting && !loading && hasMore) {
            setLoading(true);
            try {
                await loadMore();
            } finally {
                setLoading(false);
            }
        }
    }, [loading, hasMore, loadMore]);

    useEffect(() => {
        const sentinel = sentinelRef.current;
        if (!sentinel) return;

        const observer = new IntersectionObserver(handleIntersection, {
            rootMargin: `${threshold}px`
        });

        observer.observe(sentinel);

        return () => observer.disconnect();
    }, [handleIntersection, threshold]);

    return (
        <div ref={containerRef} className={`infinite-scroll ${className || ''}`}>
            {items.map((item, index) => (
                <div key={index} className="infinite-scroll-item">
                    {renderItem(item, index)}
                </div>
            ))}
            
            <div ref={sentinelRef} className="scroll-sentinel">
                {loading && (loadingComponent || <div className="loading-spinner">Loading...</div>)}
                {!hasMore && !loading && (endComponent || <div className="end-message">No more items</div>)}
            </div>
        </div>
    );
}
"#.to_string())
    }

    fn generate_css(&self) -> String {
        r#"
.infinite-scroll {
    position: relative;
}

.infinite-scroll-item {
    animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.scroll-sentinel {
    padding: 20px;
    text-align: center;
}

.scroll-sentinel.loading::after {
    content: '';
    display: inline-block;
    width: 24px;
    height: 24px;
    border: 2px solid #e0e0e0;
    border-top-color: #007bff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

.scroll-sentinel.completed {
    color: #666;
    font-style: italic;
}

.scroll-sentinel.error {
    color: #dc3545;
    cursor: pointer;
    text-decoration: underline;
}

.loading-spinner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    color: #666;
}

.end-message {
    color: #999;
    font-size: 14px;
}
"#
        .to_string()
    }

    fn generate_documentation(&self) -> String {
        r#"# Infinite Scroll

Load more content as the user scrolls down the page.

## Features

- IntersectionObserver for efficient scroll detection
- Loading states
- Error handling with retry
- End of content detection
- Smooth animations

## Usage

```typescript
const infiniteScroll = new InfiniteScroll({
    container: document.querySelector('.list'),
    loadMore: async (page) => {
        const response = await fetch(`/api/items?page=${page}`);
        return response.json();
    },
    renderItem: (item) => {
        const div = document.createElement('div');
        div.textContent = item.title;
        return div;
    },
    threshold: 100
});
```
"#
        .to_string()
    }
}
