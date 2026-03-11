//! 拖拽模式

use crate::*;
use anyhow::Result;

/// 拖拽模式
pub struct DragDropPattern;

impl DragDropPattern {
    pub fn new() -> Self {
        Self
    }

    pub fn create_pattern(
        drag_handle: &str,
        _drop_zones: Vec<String>,
        _sortable: bool,
    ) -> InteractionPattern {
        InteractionPattern {
            pattern_type: ComplexPatternType::DragAndDrop,
            name: "Drag and Drop".to_string(),
            description: "Drag elements to reorder or move between containers".to_string(),
            triggers: vec![
                PatternTrigger {
                    trigger_type: TriggerType::MouseDown,
                    selector: drag_handle.to_string(),
                    conditions: vec![],
                },
                PatternTrigger {
                    trigger_type: TriggerType::MouseMove,
                    selector: "document".to_string(),
                    conditions: vec![],
                },
                PatternTrigger {
                    trigger_type: TriggerType::MouseUp,
                    selector: "document".to_string(),
                    conditions: vec![],
                },
            ],
            behaviors: vec![
                PatternBehavior {
                    behavior_type: BehaviorType::AddClass,
                    target: drag_handle.to_string(),
                    animation: None,
                    callback: None,
                },
                PatternBehavior {
                    behavior_type: BehaviorType::SetStyle,
                    target: drag_handle.to_string(),
                    animation: None,
                    callback: None,
                },
            ],
            state_machine: PatternStateMachine {
                initial_state: "idle".to_string(),
                states: vec![
                    PatternState {
                        name: "idle".to_string(),
                        description: "Element is not being dragged".to_string(),
                        entry_actions: vec![],
                        exit_actions: vec![],
                    },
                    PatternState {
                        name: "dragging".to_string(),
                        description: "Element is being dragged".to_string(),
                        entry_actions: vec![PatternBehavior {
                            behavior_type: BehaviorType::AddClass,
                            target: drag_handle.to_string(),
                            animation: None,
                            callback: None,
                        }],
                        exit_actions: vec![PatternBehavior {
                            behavior_type: BehaviorType::RemoveClass,
                            target: drag_handle.to_string(),
                            animation: None,
                            callback: None,
                        }],
                    },
                    PatternState {
                        name: "over_dropzone".to_string(),
                        description: "Dragged element is over a drop zone".to_string(),
                        entry_actions: vec![PatternBehavior {
                            behavior_type: BehaviorType::AddClass,
                            target: ".drop-zone".to_string(),
                            animation: None,
                            callback: None,
                        }],
                        exit_actions: vec![PatternBehavior {
                            behavior_type: BehaviorType::RemoveClass,
                            target: ".drop-zone".to_string(),
                            animation: None,
                            callback: None,
                        }],
                    },
                ],
                transitions: vec![
                    StateTransition {
                        from_state: "idle".to_string(),
                        to_state: "dragging".to_string(),
                        trigger: "mousedown".to_string(),
                        guard: None,
                        actions: vec![],
                    },
                    StateTransition {
                        from_state: "dragging".to_string(),
                        to_state: "over_dropzone".to_string(),
                        trigger: "mouseover".to_string(),
                        guard: Some("isOverDropZone".to_string()),
                        actions: vec![],
                    },
                    StateTransition {
                        from_state: "over_dropzone".to_string(),
                        to_state: "dragging".to_string(),
                        trigger: "mouseleave".to_string(),
                        guard: None,
                        actions: vec![],
                    },
                    StateTransition {
                        from_state: "dragging".to_string(),
                        to_state: "idle".to_string(),
                        trigger: "mouseup".to_string(),
                        guard: None,
                        actions: vec![PatternBehavior {
                            behavior_type: BehaviorType::Move,
                            target: drag_handle.to_string(),
                            animation: None,
                            callback: Some("onDrop".to_string()),
                        }],
                    },
                ],
            },
            confidence: 0.9,
        }
    }
}

impl Default for DragDropPattern {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternImplementation for DragDropPattern {
    fn pattern_type(&self) -> ComplexPatternType {
        ComplexPatternType::DragAndDrop
    }

    fn recognize(&self, observations: &[auto_observer::Observation]) -> Option<InteractionPattern> {
        // 识别拖拽模式的特征
        let mut has_drag_start = false;
        let mut has_drag_move = false;
        let mut has_drag_end = false;

        for obs in observations {
            match obs.event_type.as_str() {
                "mousedown" => has_drag_start = true,
                "mousemove" => {
                    if has_drag_start {
                        has_drag_move = true;
                    }
                }
                "mouseup" => {
                    if has_drag_move {
                        has_drag_end = true;
                    }
                }
                _ => {}
            }
        }

        if has_drag_start && has_drag_move && has_drag_end {
            Some(Self::create_pattern(
                ".draggable",
                vec![".drop-zone".to_string()],
                true,
            ))
        } else {
            None
        }
    }

    fn generate_code(
        &self,
        pattern: &InteractionPattern,
        language: CodeLanguage,
    ) -> Result<GeneratedCode> {
        let code = match language {
            CodeLanguage::TypeScript => self.generate_typescript(pattern)?,
            CodeLanguage::JavaScript => self.generate_javascript(pattern)?,
            CodeLanguage::React => self.generate_react(pattern)?,
            _ => anyhow::bail!("Language {:?} not supported for drag-drop", language),
        };

        Ok(GeneratedCode {
            pattern_type: ComplexPatternType::DragAndDrop,
            language,
            component_name: "DragDrop".to_string(),
            code,
            css: Some(self.generate_css()),
            tests: Some(self.generate_tests()),
            documentation: self.generate_documentation(),
        })
    }

    fn get_template(&self) -> &str {
        include_str!("templates/drag_drop.ts")
    }
}

impl DragDropPattern {
    fn generate_typescript(&self, _pattern: &InteractionPattern) -> Result<String> {
        Ok(r#"
interface DragDropOptions {
    dragHandle: string;
    dropZones: string[];
    sortable?: boolean;
    onDragStart?: (element: HTMLElement) => void;
    onDragMove?: (element: HTMLElement, x: number, y: number) => void;
    onDragEnd?: (element: HTMLElement, dropZone: HTMLElement | null) => void;
    onReorder?: (newOrder: HTMLElement[]) => void;
}

class DragDrop {
    private options: DragDropOptions;
    private draggedElement: HTMLElement | null = null;
    private placeholder: HTMLElement | null = null;
    private startX: number = 0;
    private startY: number = 0;
    private initialX: number = 0;
    private initialY: number = 0;
    private currentDropZone: HTMLElement | null = null;

    constructor(options: DragDropOptions) {
        this.options = {
            sortable: true,
            ...options
        };
        this.init();
    }

    private init() {
        const dragHandles = document.querySelectorAll(this.options.dragHandle);
        dragHandles.forEach(handle => {
            handle.addEventListener('mousedown', this.onDragStart.bind(this));
            handle.addEventListener('touchstart', this.onDragStart.bind(this), { passive: false });
        });

        document.addEventListener('mousemove', this.onDragMove.bind(this));
        document.addEventListener('touchmove', this.onDragMove.bind(this), { passive: false });

        document.addEventListener('mouseup', this.onDragEnd.bind(this));
        document.addEventListener('touchend', this.onDragEnd.bind(this));
    }

    private onDragStart(e: MouseEvent | TouchEvent) {
        e.preventDefault();
        
        const target = e.target as HTMLElement;
        this.draggedElement = target.closest('.draggable') as HTMLElement;
        
        if (!this.draggedElement) return;

        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

        this.startX = clientX;
        this.startY = clientY;

        const rect = this.draggedElement.getBoundingClientRect();
        this.initialX = rect.left;
        this.initialY = rect.top;

        // 创建占位符
        this.placeholder = document.createElement('div');
        this.placeholder.className = 'drag-placeholder';
        this.placeholder.style.height = `${rect.height}px`;
        this.draggedElement.parentNode?.insertBefore(this.placeholder, this.draggedElement);

        // 设置拖拽样式
        this.draggedElement.classList.add('dragging');
        this.draggedElement.style.position = 'fixed';
        this.draggedElement.style.left = `${rect.left}px`;
        this.draggedElement.style.top = `${rect.top}px`;
        this.draggedElement.style.width = `${rect.width}px`;
        this.draggedElement.style.zIndex = '1000';

        this.options.onDragStart?.(this.draggedElement);
    }

    private onDragMove(e: MouseEvent | TouchEvent) {
        if (!this.draggedElement) return;

        e.preventDefault();

        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

        const deltaX = clientX - this.startX;
        const deltaY = clientY - this.startY;

        this.draggedElement.style.left = `${this.initialX + deltaX}px`;
        this.draggedElement.style.top = `${this.initialY + deltaY}px`;

        // 检测悬停的放置区域
        this.checkDropZone(clientX, clientY);

        // 更新占位符位置
        this.updatePlaceholder(clientX, clientY);

        this.options.onDragMove?.(this.draggedElement, clientX, clientY);
    }

    private onDragEnd(e: MouseEvent | TouchEvent) {
        if (!this.draggedElement) return;

        // 放置到目标区域
        if (this.currentDropZone && this.placeholder) {
            this.currentDropZone.insertBefore(this.draggedElement, this.placeholder);
        } else if (this.placeholder) {
            this.placeholder.parentNode?.insertBefore(this.draggedElement, this.placeholder);
        }

        // 清理
        this.draggedElement.classList.remove('dragging');
        this.draggedElement.style.position = '';
        this.draggedElement.style.left = '';
        this.draggedElement.style.top = '';
        this.draggedElement.style.width = '';
        this.draggedElement.style.zIndex = '';

        this.placeholder?.remove();
        this.placeholder = null;

        this.options.onDragEnd?.(this.draggedElement, this.currentDropZone);

        // 触发重排序回调
        if (this.options.sortable) {
            const container = this.draggedElement.parentElement;
            if (container) {
                const newOrder = Array.from(container.querySelectorAll('.draggable')) as HTMLElement[];
                this.options.onReorder?.(newOrder);
            }
        }

        this.draggedElement = null;
        this.currentDropZone = null;
    }

    private checkDropZone(x: number, y: number) {
        this.currentDropZone = null;

        for (const selector of this.options.dropZones) {
            const zones = document.querySelectorAll(selector);
            for (const zone of zones) {
                const rect = zone.getBoundingClientRect();
                if (x >= rect.left && x <= rect.right && 
                    y >= rect.top && y <= rect.bottom) {
                    this.currentDropZone = zone as HTMLElement;
                    zone.classList.add('drag-over');
                } else {
                    zone.classList.remove('drag-over');
                }
            }
        }
    }

    private updatePlaceholder(x: number, y: number) {
        if (!this.placeholder || !this.options.sortable) return;

        const container = this.placeholder.parentElement;
        if (!container) return;

        const siblings = Array.from(container.children);
        const placeholderIndex = siblings.indexOf(this.placeholder);

        for (let i = 0; i < siblings.length; i++) {
            if (i === placeholderIndex) continue;

            const sibling = siblings[i] as HTMLElement;
            const rect = sibling.getBoundingClientRect();
            const midY = rect.top + rect.height / 2;

            if (y < midY && i < placeholderIndex) {
                container.insertBefore(this.placeholder, sibling);
                break;
            } else if (y > midY && i > placeholderIndex) {
                container.insertBefore(this.placeholder, sibling.nextSibling);
                break;
            }
        }
    }

    destroy() {
        // 清理事件监听器
    }
}

export { DragDrop, DragDropOptions };
"#.to_string())
    }

    fn generate_javascript(&self, _pattern: &InteractionPattern) -> Result<String> {
        // 简化版的 JavaScript 实现
        Ok(r#"
class DragDrop {
    constructor(options) {
        this.options = { sortable: true, ...options };
        this.draggedElement = null;
        this.placeholder = null;
        this.init();
    }

    init() {
        document.querySelectorAll(this.options.dragHandle).forEach(handle => {
            handle.addEventListener('mousedown', this.onDragStart.bind(this));
        });
        document.addEventListener('mousemove', this.onDragMove.bind(this));
        document.addEventListener('mouseup', this.onDragEnd.bind(this));
    }

    onDragStart(e) {
        this.draggedElement = e.target.closest('.draggable');
        if (!this.draggedElement) return;
        
        this.draggedElement.classList.add('dragging');
        // ... 简化实现
    }

    onDragMove(e) {
        if (!this.draggedElement) return;
        // ... 简化实现
    }

    onDragEnd(e) {
        if (!this.draggedElement) return;
        this.draggedElement.classList.remove('dragging');
        this.options.onDragEnd?.(this.draggedElement);
        this.draggedElement = null;
    }
}

export { DragDrop };
"#
        .to_string())
    }

    fn generate_react(&self, _pattern: &InteractionPattern) -> Result<String> {
        Ok(r#"
import React, { useState, useRef, useCallback } from 'react';

interface DragDropProps {
    items: any[];
    renderItem: (item: any, index: number, dragHandleProps: any) => React.ReactNode;
    onReorder: (items: any[]) => void;
    className?: string;
}

export const DragDrop: React.FC<DragDropProps> = ({ 
    items, 
    renderItem, 
    onReorder,
    className 
}) => {
    const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
    const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
    const dragItem = useRef<number | null>(null);

    const handleDragStart = useCallback((index: number) => {
        dragItem.current = index;
        setDraggingIndex(index);
    }, []);

    const handleDragEnter = useCallback((index: number) => {
        setDragOverIndex(index);
    }, []);

    const handleDragEnd = useCallback(() => {
        if (dragItem.current !== null && dragOverIndex !== null) {
            const newItems = [...items];
            const [removed] = newItems.splice(dragItem.current, 1);
            newItems.splice(dragOverIndex, 0, removed);
            onReorder(newItems);
        }
        setDraggingIndex(null);
        setDragOverIndex(null);
        dragItem.current = null;
    }, [dragOverIndex, items, onReorder]);

    return (
        <div className={`drag-drop-container ${className || ''}`}>
            {items.map((item, index) => (
                <div
                    key={index}
                    className={`drag-item ${
                        draggingIndex === index ? 'dragging' : ''
                    } ${dragOverIndex === index ? 'drag-over' : ''}`}
                    draggable
                    onDragStart={() => handleDragStart(index)}
                    onDragEnter={() => handleDragEnter(index)}
                    onDragEnd={handleDragEnd}
                >
                    {renderItem(item, index, {
                        className: 'drag-handle',
                        'aria-label': 'Drag to reorder'
                    })}
                </div>
            ))}
        </div>
    );
};
"#
        .to_string())
    }

    fn generate_css(&self) -> String {
        r#"
.drag-drop-container {
    position: relative;
}

.drag-item {
    position: relative;
    transition: transform 0.2s ease;
}

.drag-item.dragging {
    opacity: 0.5;
    transform: scale(1.02);
    z-index: 1000;
}

.drag-item.drag-over {
    border-top: 2px solid var(--color-primary, #007bff);
}

.drag-handle {
    cursor: grab;
    user-select: none;
}

.drag-handle:active {
    cursor: grabbing;
}

.drag-placeholder {
    background: rgba(0, 0, 0, 0.05);
    border: 2px dashed rgba(0, 0, 0, 0.1);
    border-radius: 4px;
}

.drop-zone {
    min-height: 50px;
    transition: background-color 0.2s ease;
}

.drop-zone.drag-over {
    background-color: rgba(0, 123, 255, 0.1);
    border: 2px dashed var(--color-primary, #007bff);
}
"#
        .to_string()
    }

    fn generate_tests(&self) -> String {
        r#"
import { DragDrop } from './drag-drop';

describe('DragDrop', () => {
    let container: HTMLElement;
    let dragDrop: DragDrop;

    beforeEach(() => {
        container = document.createElement('div');
        container.innerHTML = `
            <div class="draggable" data-id="1">Item 1</div>
            <div class="draggable" data-id="2">Item 2</div>
            <div class="draggable" data-id="3">Item 3</div>
        `;
        document.body.appendChild(container);
    });

    afterEach(() => {
        dragDrop?.destroy();
        container.remove();
    });

    test('should initialize correctly', () => {
        dragDrop = new DragDrop({
            dragHandle: '.draggable',
            dropZones: ['.drop-zone']
        });
        expect(dragDrop).toBeDefined();
    });

    test('should trigger onDragStart when dragging begins', () => {
        const onDragStart = jest.fn();
        dragDrop = new DragDrop({
            dragHandle: '.draggable',
            dropZones: ['.drop-zone'],
            onDragStart
        });

        const item = container.querySelector('.draggable');
        item?.dispatchEvent(new MouseEvent('mousedown'));

        expect(onDragStart).toHaveBeenCalled();
    });
});
"#
        .to_string()
    }

    fn generate_documentation(&self) -> String {
        r#"# Drag and Drop

A flexible drag and drop implementation supporting sorting and moving between containers.

## Features

- Mouse and touch support
- Visual feedback during drag
- Sortable lists
- Multiple drop zones
- Custom callbacks

## Usage

```typescript
import { DragDrop } from './drag-drop';

const dragDrop = new DragDrop({
    dragHandle: '.draggable',
    dropZones: ['.drop-zone-1', '.drop-zone-2'],
    sortable: true,
    onDragStart: (element) => console.log('Started dragging', element),
    onDragEnd: (element, dropZone) => console.log('Dropped on', dropZone),
    onReorder: (newOrder) => console.log('New order:', newOrder)
});
```

## Options

- `dragHandle`: Selector for draggable elements
- `dropZones`: Array of selectors for drop zones
- `sortable`: Enable sorting within containers
- `onDragStart`: Callback when drag starts
- `onDragMove`: Callback during drag
- `onDragEnd`: Callback when drag ends
- `onReorder`: Callback when order changes
"#
        .to_string()
    }
}
