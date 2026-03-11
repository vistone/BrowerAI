//! Resource Management - 资源管理
//!
//! 管理渲染资源，包括：
//! - 图像缓存
//! - 字体管理
//! - 样式表缓存
//! - 内存管理

use browerai_core::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 资源管理器
#[derive(Debug, Clone)]
pub struct ResourceManager {
    /// 配置
    config: ResourceConfig,
    /// 图像缓存
    image_cache: HashMap<String, CachedImage>,
    /// 字体缓存
    font_cache: HashMap<String, CachedFont>,
    /// 样式表缓存
    stylesheet_cache: HashMap<String, CachedStylesheet>,
    /// 缓存统计
    stats: CacheStats,
}

impl ResourceManager {
    /// 创建新的资源管理器
    pub fn new(config: ResourceConfig) -> Self {
        Self {
            config,
            image_cache: HashMap::new(),
            font_cache: HashMap::new(),
            stylesheet_cache: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    /// 加载图像
    pub fn load_image(&mut self, url: &str) -> Result<CachedImage> {
        // 检查缓存
        if let Some(cached) = self.image_cache.get(url) {
            self.stats.hits += 1;
            return Ok(cached.clone());
        }

        self.stats.misses += 1;

        // 简化实现：创建占位图像
        let image = CachedImage {
            url: url.to_string(),
            width: 100,
            height: 100,
            data: Vec::new(),
            loaded_at: Instant::now(),
        };

        // 存入缓存
        if self.image_cache.len() < self.config.max_image_cache_size {
            self.image_cache.insert(url.to_string(), image.clone());
        }

        Ok(image)
    }

    /// 加载字体
    pub fn load_font(&mut self, name: &str) -> Result<CachedFont> {
        if let Some(cached) = self.font_cache.get(name) {
            self.stats.hits += 1;
            return Ok(cached.clone());
        }

        self.stats.misses += 1;

        let font = CachedFont {
            name: name.to_string(),
            family: name.to_string(),
            loaded_at: Instant::now(),
        };

        if self.font_cache.len() < self.config.max_font_cache_size {
            self.font_cache.insert(name.to_string(), font.clone());
        }

        Ok(font)
    }

    /// 加载样式表
    pub fn load_stylesheet(&mut self, url: &str) -> Result<CachedStylesheet> {
        if let Some(cached) = self.stylesheet_cache.get(url) {
            self.stats.hits += 1;
            return Ok(cached.clone());
        }

        self.stats.misses += 1;

        let stylesheet = CachedStylesheet {
            url: url.to_string(),
            content: String::new(),
            loaded_at: Instant::now(),
        };

        if self.stylesheet_cache.len() < self.config.max_stylesheet_cache_size {
            self.stylesheet_cache
                .insert(url.to_string(), stylesheet.clone());
        }

        Ok(stylesheet)
    }

    /// 清除所有缓存
    pub fn clear_cache(&mut self) {
        self.image_cache.clear();
        self.font_cache.clear();
        self.stylesheet_cache.clear();
        self.stats = CacheStats::default();
    }

    /// 清除过期缓存
    pub fn clear_expired(&mut self) {
        let now = Instant::now();
        let max_age = self.config.cache_max_age;

        self.image_cache
            .retain(|_, v| now.duration_since(v.loaded_at) < max_age);
        self.font_cache
            .retain(|_, v| now.duration_since(v.loaded_at) < max_age);
        self.stylesheet_cache
            .retain(|_, v| now.duration_since(v.loaded_at) < max_age);
    }

    /// 获取缓存大小
    pub fn cache_size(&self) -> usize {
        self.image_cache.len() + self.font_cache.len() + self.stylesheet_cache.len()
    }

    /// 获取缓存统计
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// 获取命中率
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.hits + self.stats.misses;
        if total == 0 {
            0.0
        } else {
            self.stats.hits as f64 / total as f64
        }
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new(ResourceConfig::default())
    }
}

/// 缓存的图像
#[derive(Debug, Clone)]
pub struct CachedImage {
    /// URL
    pub url: String,
    /// 宽度
    pub width: u32,
    /// 高度
    pub height: u32,
    /// 图像数据
    pub data: Vec<u8>,
    /// 加载时间
    pub loaded_at: Instant,
}

/// 缓存的字体
#[derive(Debug, Clone)]
pub struct CachedFont {
    /// 名称
    pub name: String,
    /// 字体族
    pub family: String,
    /// 加载时间
    pub loaded_at: Instant,
}

/// 缓存的样式表
#[derive(Debug, Clone)]
pub struct CachedStylesheet {
    /// URL
    pub url: String,
    /// 内容
    pub content: String,
    /// 加载时间
    pub loaded_at: Instant,
}

/// 缓存统计
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    /// 命中次数
    pub hits: u64,
    /// 未命中次数
    pub misses: u64,
}

/// 资源配置
#[derive(Debug, Clone)]
pub struct ResourceConfig {
    /// 最大图像缓存大小
    pub max_image_cache_size: usize,
    /// 最大字体缓存大小
    pub max_font_cache_size: usize,
    /// 最大样式表缓存大小
    pub max_stylesheet_cache_size: usize,
    /// 缓存最大存活时间
    pub cache_max_age: Duration,
    /// 启用缓存
    pub enable_cache: bool,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_image_cache_size: 100,
            max_font_cache_size: 20,
            max_stylesheet_cache_size: 10,
            cache_max_age: Duration::from_secs(3600),
            enable_cache: true,
        }
    }
}

/// 资源类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    /// 图像
    Image,
    /// 字体
    Font,
    /// 样式表
    Stylesheet,
    /// 脚本
    Script,
    /// 其他
    Other,
}

/// 资源加载器
pub trait ResourceLoader: Send + Sync {
    /// 加载资源
    fn load(&self, url: &str, resource_type: ResourceType) -> Result<Vec<u8>>;

    /// 检查是否支持
    fn supports(&self, url: &str) -> bool;

    /// 获取加载器名称
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_manager_creation() {
        let manager = ResourceManager::new(ResourceConfig::default());
        assert_eq!(manager.cache_size(), 0);
    }

    #[test]
    fn test_load_image() {
        let mut manager = ResourceManager::new(ResourceConfig::default());

        let image = manager.load_image("https://example.com/image.png").unwrap();
        assert_eq!(image.url, "https://example.com/image.png");

        // 再次加载应该命中缓存
        let _ = manager.load_image("https://example.com/image.png").unwrap();
        assert_eq!(manager.stats().hits, 1);
        assert_eq!(manager.stats().misses, 1);
    }

    #[test]
    fn test_load_font() {
        let mut manager = ResourceManager::new(ResourceConfig::default());

        let font = manager.load_font("Arial").unwrap();
        assert_eq!(font.name, "Arial");
    }

    #[test]
    fn test_cache_clear() {
        let mut manager = ResourceManager::new(ResourceConfig::default());

        manager.load_image("https://example.com/image.png").unwrap();
        manager.load_font("Arial").unwrap();

        assert_eq!(manager.cache_size(), 2);

        manager.clear_cache();
        assert_eq!(manager.cache_size(), 0);
    }

    #[test]
    fn test_hit_rate() {
        let mut manager = ResourceManager::new(ResourceConfig::default());

        // 2次未命中
        manager.load_image("image1.png").unwrap();
        manager.load_image("image2.png").unwrap();

        // 2次命中
        manager.load_image("image1.png").unwrap();
        manager.load_image("image2.png").unwrap();

        assert_eq!(manager.hit_rate(), 0.5);
    }
}
