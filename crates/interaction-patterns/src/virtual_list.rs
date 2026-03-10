//! 虚拟列表模式

use crate::*;
use anyhow::Result;

pub struct VirtualListPattern;

impl VirtualListPattern {
    pub fn new() -> Self {
        Self
    }

    pub fn create_pattern(
        container: &str,
        _item_height: u32,
        _total_items: u32,
    ) -> InteractionPattern {
        InteractionPattern {
            pattern_type: ComplexPatternType::VirtualList,
            name: "Virtual List".to_string(),
            description: "Render only visible items for large lists".to_string(),
            triggers: vec![
                PatternTrigger {
                    trigger_type: TriggerType::Scroll,
                    selector: container.to_string(),
                    conditions: vec![],
                },
            ],
            behaviors: vec![
                PatternBehavior {
                    behavior_type: BehaviorType::InsertElement,
                    target: container.to_string(),
                    animation: None,
                    callback: Some("renderVisibleItems".to_string()),
                },
                PatternBehavior {
                    behavior_type: BehaviorType::RemoveElement,
                    target: ".virtual-item".to_string(),
                    animation: None,
                    callback: Some("removeInvisibleItems".to_string()),
                },
            ],
            state_machine: PatternStateMachine {
                initial_state: "idle".to_string(),
                states: vec![
                    PatternState {
                        name: "idle".to_string(),
                        description: "List is stable".to_string(),
                        entry_actions: vec![],
                        exit_actions: vec![],
                    },
                    PatternState {
                        name: "scrolling".to_string(),
                        description: "User is scrolling".to_string(),
                        entry_actions: vec![],
                        exit_actions: vec![
                            PatternBehavior {
                                behavior_type: BehaviorType::SetStyle,
                                target: container.to_string(),
                                animation: None,
                                callback: None,
                            },
                        ],
                    },
                ],
                transitions: vec![
                    StateTransition {
                        from_state: "idle".to_string(),
                        to_state: "scrolling".to_string(),
                        trigger: "scroll".to_string(),
                        guard: None,
                        actions: vec![],
                    },
                    StateTransition {
                        from_state: "scrolling".to_string(),
                        to_state: "idle".to_string(),
                        trigger: "scrollEnd".to_string(),
                        guard: None,
                        actions: vec![
                            PatternBehavior {
                                behavior_type: BehaviorType::InsertElement,
                                target: container.to_string(),
                                animation: None,
                                callback: Some("updateVisibleRange".to_string()),
                            },
                        ],
                    },
                ],
            },
            confidence: 0.8,
        }
    }
}

impl PatternImplementation for VirtualListPattern {
    fn pattern_type(&self) -> ComplexPatternType {
        ComplexPatternType::VirtualList
    }

    fn recognize(&self, observations: &[auto_observer::Observation]) -> Option<InteractionPattern> {
        // 识别虚拟列表的特征：大量滚动但DOM元素数量相对稳定
        let mut scroll_events = 0;
        let mut dom_changes = 0;

        for obs in observations {
            match obs.event_type.as_str() {
                "scroll" => scroll_events += 1,
                "mutation" => dom_changes += 1,
                _ => {}
            }
        }

        // 如果滚动次数远多于DOM变化次数，可能是虚拟列表
        if scroll_events > 10 && dom_changes < scroll_events / 5 {
            Some(Self::create_pattern(".virtual-list", 50, 10000))
        } else {
            None
        }
    }

    fn generate_code(&self, _pattern: &InteractionPattern, language: CodeLanguage) -> Result<GeneratedCode> {
        let code = match language {
            CodeLanguage::TypeScript => self.generate_typescript(),
            CodeLanguage::React => self.generate_react(),
            _ => anyhow::bail!("Language not supported"),
        }?;

        Ok(GeneratedCode {
            pattern_type: ComplexPatternType::VirtualList,
            language,
            component_name: "VirtualList".to_string(),
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

impl VirtualListPattern {
    fn generate_typescript(&self) -> Result<String> {
        Ok(r#"
interface VirtualListOptions {
    container: HTMLElement;
    itemHeight: number;
    totalItems: number;
    renderItem: (index: number) => HTMLElement;
    overscan?: number;
}

interface VisibleRange {
    start: number;
    end: number;
}

class VirtualList {
    private options: Required<VirtualListOptions>;
    private visibleRange: VisibleRange = { start: 0, end: 0 };
    private scrollTop: number = 0;
    private containerHeight: number = 0;
    private itemElements: Map<number, HTMLElement> = new Map();
    private scrollHandler: () => void;
    private resizeHandler: () => void;

    constructor(options: VirtualListOptions) {
        this.options = {
            overscan: 3,
            ...options
        };
        this.init();
    }

    private init() {
        // 设置容器样式
        this.options.container.style.position = 'relative';
        this.options.container.style.overflow = 'auto';

        // 创建内容占位符
        const totalHeight = this.options.totalItems * this.options.itemHeight;
        const spacer = document.createElement('div');
        spacer.className = 'virtual-list-spacer';
        spacer.style.height = `${totalHeight}px`;
        this.options.container.appendChild(spacer);

        // 计算容器高度
        this.containerHeight = this.options.container.clientHeight;

        // 绑定事件
        this.scrollHandler = this.onScroll.bind(this);
        this.resizeHandler = this.onResize.bind(this);

        this.options.container.addEventListener('scroll', this.scrollHandler);
        window.addEventListener('resize', this.resizeHandler);

        // 初始渲染
        this.updateVisibleRange();
    }

    private onScroll() {
        this.scrollTop = this.options.container.scrollTop;
        this.updateVisibleRange();
    }

    private onResize() {
        this.containerHeight = this.options.container.clientHeight;
        this.updateVisibleRange();
    }

    private updateVisibleRange() {
        const { itemHeight, totalItems, overscan } = this.options;

        // 计算可见范围
        const startIndex = Math.floor(this.scrollTop / itemHeight);
        const visibleCount = Math.ceil(this.containerHeight / itemHeight);
        const endIndex = Math.min(startIndex + visibleCount + overscan, totalItems);

        const newRange: VisibleRange = {
            start: Math.max(0, startIndex - overscan),
            end: endIndex
        };

        // 如果范围没有变化，跳过
        if (newRange.start === this.visibleRange.start && 
            newRange.end === this.visibleRange.end) {
            return;
        }

        // 移除不可见的元素
        this.removeInvisibleItems(newRange);

        // 渲染新可见的元素
        this.renderVisibleItems(newRange);

        this.visibleRange = newRange;
    }

    private removeInvisibleItems(newRange: VisibleRange) {
        for (const [index, element] of this.itemElements) {
            if (index < newRange.start || index >= newRange.end) {
                element.remove();
                this.itemElements.delete(index);
            }
        }
    }

    private renderVisibleItems(range: VisibleRange) {
        const spacer = this.options.container.querySelector('.virtual-list-spacer');

        for (let i = range.start; i < range.end; i++) {
            if (this.itemElements.has(i)) continue;

            const element = this.options.renderItem(i);
            element.className = 'virtual-item';
            element.style.position = 'absolute';
            element.style.top = `${i * this.options.itemHeight}px`;
            element.style.left = '0';
            element.style.right = '0';
            element.style.height = `${this.options.itemHeight}px`;
            element.dataset.index = String(i);

            this.options.container.insertBefore(element, spacer);
            this.itemElements.set(i, element);
        }
    }

    scrollToIndex(index: number, behavior: ScrollBehavior = 'smooth') {
        const top = index * this.options.itemHeight;
        this.options.container.scrollTo({ top, behavior });
    }

    getVisibleRange(): VisibleRange {
        return { ...this.visibleRange };
    }

    destroy() {
        this.options.container.removeEventListener('scroll', this.scrollHandler);
        window.removeEventListener('resize', this.resizeHandler);
        
        for (const element of this.itemElements.values()) {
            element.remove();
        }
        this.itemElements.clear();

        const spacer = this.options.container.querySelector('.virtual-list-spacer');
        spacer?.remove();
    }
}

export { VirtualList, VirtualListOptions, VisibleRange };
"#.to_string())
    }

    fn generate_react(&self) -> Result<String> {
        Ok(r#"
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';

interface VirtualListProps<T> {
    items: T[];
    itemHeight: number;
    renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode;
    overscan?: number;
    className?: string;
}

export function VirtualList<T>({
    items,
    itemHeight,
    renderItem,
    overscan = 3,
    className
}: VirtualListProps<T>) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [scrollTop, setScrollTop] = useState(0);
    const [containerHeight, setContainerHeight] = useState(0);

    const { visibleRange, totalHeight, startOffset } = useMemo(() => {
        const startIndex = Math.floor(scrollTop / itemHeight);
        const visibleCount = Math.ceil(containerHeight / itemHeight);
        
        const start = Math.max(0, startIndex - overscan);
        const end = Math.min(items.length, startIndex + visibleCount + overscan);
        
        return {
            visibleRange: { start, end },
            totalHeight: items.length * itemHeight,
            startOffset: start * itemHeight
        };
    }, [scrollTop, containerHeight, itemHeight, items.length, overscan]);

    const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
        setScrollTop(e.currentTarget.scrollTop);
    }, []);

    useEffect(() => {
        if (containerRef.current) {
            setContainerHeight(containerRef.current.clientHeight);
        }
    }, []);

    const visibleItems = useMemo(() => {
        return items.slice(visibleRange.start, visibleRange.end);
    }, [items, visibleRange]);

    return (
        <div
            ref={containerRef}
            className={`virtual-list ${className || ''}`}
            onScroll={handleScroll}
        >
            <div className="virtual-list-spacer" style={{ height: totalHeight }}>
                <div
                    className="virtual-list-content"
                    style={{
                        transform: `translateY(${startOffset}px)`
                    }}
                >
                    {visibleItems.map((item, index) => {
                        const actualIndex = visibleRange.start + index;
                        return (
                            <div
                                key={actualIndex}
                                className="virtual-item"
                                style={{ height: itemHeight }}
                            >
                                {renderItem(item, actualIndex, {})}
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}
"#.to_string())
    }

    fn generate_css(&self) -> String {
        r#"
.virtual-list {
    position: relative;
    overflow: auto;
    height: 100%;
}

.virtual-list-spacer {
    position: relative;
}

.virtual-list-content {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
}

.virtual-item {
    position: absolute;
    left: 0;
    right: 0;
    overflow: hidden;
}
"#.to_string()
    }

    fn generate_documentation(&self) -> String {
        r#"# Virtual List

Efficiently render large lists by only mounting visible items.

## Features

- Only renders visible items
- Smooth scrolling
- Configurable overscan
- Dynamic height support

## Usage

```typescript
const virtualList = new VirtualList({
    container: document.querySelector('.list'),
    itemHeight: 50,
    totalItems: 100000,
    renderItem: (index) => {
        const div = document.createElement('div');
        div.textContent = `Item ${index}`;
        return div;
    },
    overscan: 5
});
```
"#.to_string()
    }
}
