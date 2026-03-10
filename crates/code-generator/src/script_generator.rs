//! 脚本生成器

use crate::*;
use anyhow::Result;

pub struct ScriptGenerator {
    config: GenerationConfig,
}

impl ScriptGenerator {
    pub fn new(config: &GenerationConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub async fn generate_scripts(&self, behaviors: &[interaction_patterns::InteractionPattern]) -> Result<Vec<GeneratedFile>> {
        let mut scripts = Vec::new();

        // 生成hooks
        let hooks = self.generate_hooks(behaviors).await?;
        scripts.extend(hooks);

        // 生成工具函数
        let utils = self.generate_utils().await?;
        scripts.extend(utils);

        // 生成行为实现
        let behavior_scripts = self.generate_behavior_scripts(behaviors).await?;
        scripts.extend(behavior_scripts);

        Ok(scripts)
    }

    async fn generate_hooks(&self, behaviors: &[interaction_patterns::InteractionPattern]) -> Result<Vec<GeneratedFile>> {
        let mut hooks = Vec::new();

        for behavior in behaviors {
            let hook = self.generate_hook_for_behavior(behavior).await?;
            if let Some(hook) = hook {
                hooks.push(hook);
            }
        }

        // 通用hooks
        hooks.push(self.generate_use_intersection_observer()?);
        hooks.push(self.generate_use_debounce()?);
        hooks.push(self.generate_use_local_storage()?);

        Ok(hooks)
    }

    async fn generate_hook_for_behavior(&self, behavior: &interaction_patterns::InteractionPattern) -> Result<Option<GeneratedFile>> {
        let content = match behavior.pattern_type {
            interaction_patterns::ComplexPatternType::DragAndDrop => Some(self.generate_use_drag_drop()),
            interaction_patterns::ComplexPatternType::InfiniteScroll => Some(self.generate_use_infinite_scroll()),
            interaction_patterns::ComplexPatternType::VirtualList => Some(self.generate_use_virtual_list()),
            _ => None,
        };

        content.map(|c| {
            let hook_name = format!("use_{:?}", behavior.pattern_type).to_lowercase();
            Ok(GeneratedFile {
                path: format!("src/hooks/{}.ts", hook_name),
                content: c,
                file_type: FileType::Script,
            })
        }).transpose()
    }

    fn generate_use_drag_drop(&self) -> String {
        r#"import { useState, useCallback, useRef } from 'react';

interface DragDropState {
  isDragging: boolean;
  draggedItem: any;
  dropTarget: string | null;
}

interface UseDragDropOptions {
  onDragStart?: (item: any) => void;
  onDragEnd?: (item: any, target: string | null) => void;
  onDrop?: (item: any, target: string) => void;
}

export function useDragDrop(options: UseDragDropOptions = {}) {
  const [state, setState] = useState<DragDropState>({
    isDragging: false,
    draggedItem: null,
    dropTarget: null
  });

  const dragRef = useRef<HTMLDivElement>(null);

  const handleDragStart = useCallback((item: any) => {
    setState({
      isDragging: true,
      draggedItem: item,
      dropTarget: null
    });
    options.onDragStart?.(item);
  }, [options]);

  const handleDragOver = useCallback((targetId: string) => {
    setState(prev => ({ ...prev, dropTarget: targetId }));
  }, []);

  const handleDragEnd = useCallback(() => {
    if (state.dropTarget) {
      options.onDrop?.(state.draggedItem, state.dropTarget);
    }
    options.onDragEnd?.(state.draggedItem, state.dropTarget);
    
    setState({
      isDragging: false,
      draggedItem: null,
      dropTarget: null
    });
  }, [state, options]);

  return {
    ...state,
    dragRef,
    handleDragStart,
    handleDragOver,
    handleDragEnd
  };
}
"#.to_string()
    }

    fn generate_use_infinite_scroll(&self) -> String {
        r#"import { useState, useEffect, useRef, useCallback } from 'react';

interface UseInfiniteScrollOptions<T> {
  loadMore: () => Promise<T[]>;
  hasMore: boolean;
  threshold?: number;
}

interface UseInfiniteScrollResult<T> {
  items: T[];
  loading: boolean;
  error: Error | null;
  loadMore: () => void;
}

export function useInfiniteScroll<T>({
  loadMore: loadMoreFn,
  hasMore,
  threshold = 100
}: UseInfiniteScrollOptions<T>): UseInfiniteScrollResult<T> {
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const loadMore = useCallback(async () => {
    if (loading || !hasMore) return;

    setLoading(true);
    setError(null);

    try {
      const newItems = await loadMoreFn();
      setItems(prev => [...prev, ...newItems]);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to load'));
    } finally {
      setLoading(false);
    }
  }, [loadMoreFn, loading, hasMore]);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          loadMore();
        }
      },
      { rootMargin: `${threshold}px` }
    );

    observerRef.current.observe(sentinel);

    return () => observerRef.current?.disconnect();
  }, [loadMore, threshold]);

  return {
    items,
    loading,
    error,
    loadMore
  };
}

export { sentinelRef };
"#.to_string()
    }

    fn generate_use_virtual_list(&self) -> String {
        r#"import { useState, useMemo, useRef, useCallback } from 'react';

interface UseVirtualListOptions {
  itemHeight: number;
  overscan?: number;
}

interface UseVirtualListResult<T> {
  virtualItems: Array<{ item: T; index: number; style: React.CSSProperties }>;
  containerRef: React.RefObject<HTMLDivElement>;
  totalHeight: number;
  scrollToIndex: (index: number) => void;
}

export function useVirtualList<T>(
  items: T[],
  { itemHeight, overscan = 3 }: UseVirtualListOptions
): UseVirtualListResult<T> {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);

  const { virtualItems, totalHeight, startIndex } = useMemo(() => {
    const start = Math.floor(scrollTop / itemHeight);
    const visibleCount = Math.ceil((containerRef.current?.clientHeight || 0) / itemHeight);
    
    const startIndex = Math.max(0, start - overscan);
    const endIndex = Math.min(items.length, start + visibleCount + overscan);
    
    const virtualItems = items.slice(startIndex, endIndex).map((item, idx) => ({
      item,
      index: startIndex + idx,
      style: {
        position: 'absolute',
        top: (startIndex + idx) * itemHeight,
        height: itemHeight,
        left: 0,
        right: 0
      } as React.CSSProperties
    }));

    return {
      virtualItems,
      totalHeight: items.length * itemHeight,
      startIndex
    };
  }, [items, scrollTop, itemHeight, overscan]);

  const handleScroll = useCallback(() => {
    setScrollTop(containerRef.current?.scrollTop || 0);
  }, []);

  const scrollToIndex = useCallback((index: number) => {
    if (containerRef.current) {
      containerRef.current.scrollTop = index * itemHeight;
    }
  }, [itemHeight]);

  return {
    virtualItems,
    containerRef,
    totalHeight,
    scrollToIndex
  };
}
"#.to_string()
    }

    fn generate_use_intersection_observer(&self) -> anyhow::Result<GeneratedFile> {
        Ok(GeneratedFile {
            path: "src/hooks/useIntersectionObserver.ts".to_string(),
            content: r#"import { useEffect, useRef, useState } from 'react';

interface UseIntersectionObserverOptions {
  threshold?: number;
  rootMargin?: string;
  triggerOnce?: boolean;
}

export function useIntersectionObserver<T extends HTMLElement>({
  threshold = 0,
  rootMargin = '0px',
  triggerOnce = false
}: UseIntersectionObserverOptions = {}) {
  const ref = useRef<T>(null);
  const [isIntersecting, setIsIntersecting] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
        
        if (entry.isIntersecting && triggerOnce) {
          observer.unobserve(element);
        }
      },
      { threshold, rootMargin }
    );

    observer.observe(element);

    return () => observer.disconnect();
  }, [threshold, rootMargin, triggerOnce]);

  return { ref, isIntersecting };
}
"#.to_string(),
            file_type: FileType::Script,
        })
    }

    fn generate_use_debounce(&self) -> anyhow::Result<GeneratedFile> {
        Ok(GeneratedFile {
            path: "src/hooks/useDebounce.ts".to_string(),
            content: r#"import { useState, useEffect } from 'react';

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

export function useDebounceCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;

  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => callback(...args), delay);
  };
}
"#.to_string(),
            file_type: FileType::Script,
        })
    }

    fn generate_use_local_storage(&self) -> anyhow::Result<GeneratedFile> {
        Ok(GeneratedFile {
            path: "src/hooks/useLocalStorage.ts".to_string(),
            content: r#"import { useState, useEffect, useCallback } from 'react';

export function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value: T) => {
    try {
      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }, [key]);

  return [storedValue, setValue];
}
"#.to_string(),
            file_type: FileType::Script,
        })
    }

    async fn generate_utils(&self) -> Result<Vec<GeneratedFile>> {
        let mut utils = Vec::new();

        // 事件工具
        utils.push(GeneratedFile {
            path: "src/utils/events.ts".to_string(),
            content: r#"export function throttle<T extends (...args: any[]) => any>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false;
  return function(...args: Parameters<T>) {
    if (!inThrottle) {
      fn.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return function(...args: Parameters<T>) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), delay);
  };
}

export function once<T extends (...args: any[]) => any>(fn: T): (...args: Parameters<T>) => ReturnType<T> {
  let called = false;
  let result: ReturnType<T>;
  return function(...args: Parameters<T>): ReturnType<T> {
    if (!called) {
      called = true;
      result = fn.apply(this, args);
    }
    return result;
  };
}
"#.to_string(),
            file_type: FileType::Script,
        });

        // DOM工具
        utils.push(GeneratedFile {
            path: "src/utils/dom.ts".to_string(),
            content: r#"export function $(selector: string, context: Document | Element = document): Element | null {
  return context.querySelector(selector);
}

export function $$(selector: string, context: Document | Element = document): Element[] {
  return Array.from(context.querySelectorAll(selector));
}

export function addClass(element: Element, ...classes: string[]): void {
  element.classList.add(...classes);
}

export function removeClass(element: Element, ...classes: string[]): void {
  element.classList.remove(...classes);
}

export function toggleClass(element: Element, className: string, force?: boolean): boolean {
  return element.classList.toggle(className, force);
}

export function hasClass(element: Element, className: string): boolean {
  return element.classList.contains(className);
}

export function on<K extends keyof HTMLElementEventMap>(
  element: HTMLElement,
  event: K,
  handler: (event: HTMLElementEventMap[K]) => void,
  options?: boolean | AddEventListenerOptions
): void {
  element.addEventListener(event, handler as EventListener, options);
}

export function off<K extends keyof HTMLElementEventMap>(
  element: HTMLElement,
  event: K,
  handler: (event: HTMLElementEventMap[K]) => void
): void {
  element.removeEventListener(event, handler as EventListener);
}
"#.to_string(),
            file_type: FileType::Script,
        });

        // 类型工具
        utils.push(GeneratedFile {
            path: "src/utils/types.ts".to_string(),
            content: r#"export type Nullable<T> = T | null;
export type Optional<T> = T | undefined;
export type Maybe<T> = T | null | undefined;

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type DeepRequired<T> = {
  [P in keyof T]-?: T[P] extends object ? DeepRequired<T[P]> : T[P];
};

export type EventHandler<E extends Event = Event> = (event: E) => void;

export interface Size {
  width: number;
  height: number;
}

export interface Position {
  x: number;
  y: number;
}

export interface Rect extends Position, Size {}
"#.to_string(),
            file_type: FileType::Script,
        });

        Ok(utils)
    }

    async fn generate_behavior_scripts(&self, behaviors: &[interaction_patterns::InteractionPattern]) -> Result<Vec<GeneratedFile>> {
        let mut scripts = Vec::new();

        for behavior in behaviors {
            let script = self.generate_behavior_script(behavior).await?;
            if let Some(script) = script {
                scripts.push(script);
            }
        }

        Ok(scripts)
    }

    async fn generate_behavior_script(&self, behavior: &interaction_patterns::InteractionPattern) -> Result<Option<GeneratedFile>> {
        // 使用 interaction-patterns 库生成代码
        let generator = interaction_patterns::PatternCodeGenerator::new();
        
        let code = match generator.generate(behavior, interaction_patterns::CodeLanguage::React) {
            Ok(code) => code,
            Err(_) => return Ok(None),
        };

        Some(Ok(GeneratedFile {
            path: format!("src/behaviors/{:?}.tsx", behavior.pattern_type).to_lowercase(),
            content: code.code,
            file_type: FileType::Script,
        })).transpose()
    }
}
