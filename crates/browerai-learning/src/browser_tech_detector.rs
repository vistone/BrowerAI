/// 浏览器技术栈检测
///
/// 检测网站使用的现代浏览器技术：
/// - HTML5 特性
/// - CSS 预处理器和框架
/// - WebGL / Canvas
/// - WebAssembly
/// - Service Workers
/// - Web Workers
/// - 高级 API（IndexedDB, File API 等）
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 浏览器技术
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BrowserTechnology {
    // HTML5 特性
    Html5Semantic,
    Html5Canvas,
    Html5Audio,
    Html5Video,
    Html5FormValidation,
    Html5Geolocation,

    // CSS 技术
    CssFlexbox,
    CssGrid,
    CssAnimations,
    CssTransforms,
    CssSass,
    CssLess,
    CssPostCSS,
    CssTailwind,
    CssBootstrap,

    // JavaScript 特性
    EsNext,
    Async,
    Promises,
    Generators,
    Destructuring,

    // 3D 图形
    WebGL,
    WebGL2,
    ThreeJS,
    BabylonJS,

    // 编译技术
    WebAssembly,
    Wasm,

    // 并发机制
    WebWorker,
    SharedArrayBuffer,

    // 离线存储
    ServiceWorker,
    IndexedDB,
    LocalStorage,
    SessionStorage,
    Cache,

    // 网络
    Fetch,
    WebSocket,
    WebRTC,

    // 其他
    ShadowDOM,
    CustomElements,
    MutationObserver,
    IntersectionObserver,
    ResizeObserver,
}

/// 技术检测结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TechnologyDetectionResult {
    /// 发现的所有技术
    pub detected_technologies: HashMap<BrowserTechnology, TechnologyInfo>,

    /// 技术栈复杂度评分（0-100）
    pub complexity_score: f64,

    /// 现代化程度评分（0-100）
    pub modernization_score: f64,

    /// 性能影响评分（0-100）
    pub performance_impact: f64,

    /// 兼容性要求
    pub compatibility_requirements: Vec<String>,

    /// 安全考虑
    pub security_concerns: Vec<String>,

    /// 建议和注意事项
    pub recommendations: Vec<String>,
}

/// 单个技术的信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TechnologyInfo {
    /// 技术名称
    pub name: String,

    /// 在代码中出现的次数
    pub occurrence_count: usize,

    /// 使用的特性
    pub features: Vec<String>,

    /// 浏览器兼容性要求
    pub min_browser_versions: HashMap<String, String>,

    /// 性能影响（-100 到 100，负数表示性能下降）
    pub performance_impact: i32,

    /// 学习难度（1-10）
    pub learning_difficulty: u8,

    /// 是否可以降级处理
    pub degradable: bool,
}

/// 浏览器技术栈检测器
pub struct BrowserTechDetector;

impl BrowserTechDetector {
    /// 检测网站使用的浏览器技术
    pub fn detect_technologies(
        html: &str,
        css: &str,
        js_code: &str,
    ) -> Result<TechnologyDetectionResult> {
        log::info!("🔍 检测浏览器技术栈...");

        let mut detected = HashMap::new();
        let mut complexity_score = 0.0;
        let mut compatibility_reqs = Vec::new();
        let mut security_concerns = Vec::new();
        let mut recommendations = Vec::new();

        // 第1步：检测 HTML5 特性
        Self::detect_html5_features(html, &mut detected, &mut complexity_score)?;

        // 第2步：检测 CSS 技术
        Self::detect_css_technologies(css, &mut detected, &mut complexity_score)?;

        // 第3步：检测 JavaScript 特性
        Self::detect_js_technologies(js_code, &mut detected, &mut complexity_score)?;

        // 第4步：检测 3D 图形
        Self::detect_webgl(js_code, &mut detected)?;

        // 第5步：检测 WebAssembly
        Self::detect_webassembly(js_code, &mut detected, &mut compatibility_reqs)?;

        // 第6步：检测并发机制
        Self::detect_concurrency(js_code, &mut detected)?;

        // 第7步：检测离线存储
        Self::detect_offline_storage(js_code, &mut detected)?;

        // 第8步：检测网络技术
        Self::detect_network_tech(js_code, &mut detected)?;

        // 第9步：检测 DOM 特性
        Self::detect_dom_features(js_code, &mut detected)?;

        // 计算现代化程度
        let modernization_score = Self::calculate_modernization_score(&detected);

        // 识别兼容性问题
        Self::identify_compatibility_issues(&detected, &mut compatibility_reqs);

        // 识别安全问题
        Self::identify_security_concerns(&detected, &mut security_concerns);

        // 生成建议
        Self::generate_recommendations(&detected, &mut recommendations);

        log::info!(
            "  ✓ 发现 {} 个浏览器技术，复杂度 {:.1}/100，现代化 {:.1}/100",
            detected.len(),
            complexity_score,
            modernization_score
        );

        let performance_impact = Self::calculate_performance_impact(&detected);

        Ok(TechnologyDetectionResult {
            detected_technologies: detected,
            complexity_score,
            modernization_score,
            performance_impact,
            compatibility_requirements: compatibility_reqs,
            security_concerns,
            recommendations,
        })
    }

    fn detect_html5_features(
        html: &str,
        detected: &mut HashMap<BrowserTechnology, TechnologyInfo>,
        complexity_score: &mut f64,
    ) -> Result<()> {
        // 检测 semantic HTML
        if html.contains("<article>")
            || html.contains("<section>")
            || html.contains("<nav>")
            || html.contains("<header>")
            || html.contains("<footer>")
        {
            detected.insert(
                BrowserTechnology::Html5Semantic,
                TechnologyInfo {
                    name: "HTML5 Semantic Elements".to_string(),
                    occurrence_count: 1,
                    features: vec!["article".to_string(), "section".to_string()],
                    min_browser_versions: [("IE".to_string(), "9+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: 0,
                    learning_difficulty: 2,
                    degradable: true,
                },
            );
            *complexity_score += 5.0;
        }

        // 检测 Canvas
        if html.contains("<canvas") {
            detected.insert(
                BrowserTechnology::Html5Canvas,
                TechnologyInfo {
                    name: "HTML5 Canvas".to_string(),
                    occurrence_count: html.matches("<canvas").count(),
                    features: vec!["2D drawing".to_string()],
                    min_browser_versions: [("IE".to_string(), "9+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: -20,
                    learning_difficulty: 7,
                    degradable: true,
                },
            );
            *complexity_score += 15.0;
        }

        // 检测 Audio/Video
        if html.contains("<audio>") {
            detected.insert(
                BrowserTechnology::Html5Audio,
                TechnologyInfo {
                    name: "HTML5 Audio".to_string(),
                    occurrence_count: html.matches("<audio>").count(),
                    features: vec!["audio playback".to_string()],
                    min_browser_versions: [("IE".to_string(), "9+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: -10,
                    learning_difficulty: 3,
                    degradable: true,
                },
            );
            *complexity_score += 8.0;
        }

        if html.contains("<video>") {
            detected.insert(
                BrowserTechnology::Html5Video,
                TechnologyInfo {
                    name: "HTML5 Video".to_string(),
                    occurrence_count: html.matches("<video>").count(),
                    features: vec!["video playback".to_string()],
                    min_browser_versions: [("IE".to_string(), "9+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: -15,
                    learning_difficulty: 3,
                    degradable: true,
                },
            );
            *complexity_score += 10.0;
        }

        Ok(())
    }

    fn detect_css_technologies(
        css: &str,
        detected: &mut HashMap<BrowserTechnology, TechnologyInfo>,
        complexity_score: &mut f64,
    ) -> Result<()> {
        // 检测 Flexbox
        if css.contains("flex") || css.contains("display: flex") {
            detected.insert(
                BrowserTechnology::CssFlexbox,
                TechnologyInfo {
                    name: "CSS Flexbox".to_string(),
                    occurrence_count: css.matches("flex").count(),
                    features: vec!["flexible layout".to_string()],
                    min_browser_versions: [("IE".to_string(), "11+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: 0,
                    learning_difficulty: 5,
                    degradable: true,
                },
            );
            *complexity_score += 8.0;
        }

        // 检测 Grid
        if css.contains("grid") || css.contains("display: grid") {
            detected.insert(
                BrowserTechnology::CssGrid,
                TechnologyInfo {
                    name: "CSS Grid".to_string(),
                    occurrence_count: css.matches("grid").count(),
                    features: vec!["grid layout".to_string()],
                    min_browser_versions: [("IE".to_string(), "unsupported".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: 0,
                    learning_difficulty: 6,
                    degradable: true,
                },
            );
            *complexity_score += 10.0;
        }

        // 检测动画
        if css.contains("animation") || css.contains("@keyframes") {
            detected.insert(
                BrowserTechnology::CssAnimations,
                TechnologyInfo {
                    name: "CSS Animations".to_string(),
                    occurrence_count: css.matches("animation").count(),
                    features: vec!["keyframe animations".to_string()],
                    min_browser_versions: [("IE".to_string(), "10+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: -5,
                    learning_difficulty: 4,
                    degradable: true,
                },
            );
            *complexity_score += 6.0;
        }

        Ok(())
    }

    fn detect_js_technologies(
        js_code: &str,
        detected: &mut HashMap<BrowserTechnology, TechnologyInfo>,
        complexity_score: &mut f64,
    ) -> Result<()> {
        // 检测 async/await
        if js_code.contains("async ") && js_code.contains("await ") {
            detected.insert(
                BrowserTechnology::Async,
                TechnologyInfo {
                    name: "Async/Await".to_string(),
                    occurrence_count: js_code.matches("async").count(),
                    features: vec!["async functions".to_string()],
                    min_browser_versions: [("IE".to_string(), "unsupported".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: 0,
                    learning_difficulty: 7,
                    degradable: false,
                },
            );
            *complexity_score += 12.0;
        }

        // 检测 Promises
        if js_code.contains("Promise") {
            detected.insert(
                BrowserTechnology::Promises,
                TechnologyInfo {
                    name: "Promises".to_string(),
                    occurrence_count: js_code.matches("Promise").count(),
                    features: vec!["promise API".to_string()],
                    min_browser_versions: [("IE".to_string(), "unsupported".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: 0,
                    learning_difficulty: 6,
                    degradable: false,
                },
            );
            *complexity_score += 10.0;
        }

        // 检测 Generators
        if js_code.contains("function*") || js_code.contains("yield ") {
            detected.insert(
                BrowserTechnology::Generators,
                TechnologyInfo {
                    name: "Generators".to_string(),
                    occurrence_count: js_code.matches("yield").count(),
                    features: vec!["generator functions".to_string()],
                    min_browser_versions: [("IE".to_string(), "unsupported".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: 0,
                    learning_difficulty: 8,
                    degradable: false,
                },
            );
            *complexity_score += 15.0;
        }

        Ok(())
    }

    fn detect_webgl(
        js_code: &str,
        detected: &mut HashMap<BrowserTechnology, TechnologyInfo>,
    ) -> Result<()> {
        if js_code.contains("WebGLRenderingContext")
            || js_code.contains("getContext('webgl')")
            || js_code.contains("getContext('webgl2')")
        {
            detected.insert(
                BrowserTechnology::WebGL,
                TechnologyInfo {
                    name: "WebGL".to_string(),
                    occurrence_count: 1,
                    features: vec!["3D graphics".to_string()],
                    min_browser_versions: [("IE".to_string(), "11+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: -50,
                    learning_difficulty: 10,
                    degradable: true,
                },
            );
        }

        // 检测 Three.js
        if js_code.contains("THREE.") {
            detected.insert(
                BrowserTechnology::ThreeJS,
                TechnologyInfo {
                    name: "Three.js".to_string(),
                    occurrence_count: js_code.matches("THREE.").count(),
                    features: vec!["3D framework".to_string()],
                    min_browser_versions: Default::default(),
                    performance_impact: -40,
                    learning_difficulty: 8,
                    degradable: true,
                },
            );
        }

        Ok(())
    }

    fn detect_webassembly(
        js_code: &str,
        detected: &mut HashMap<BrowserTechnology, TechnologyInfo>,
        compatibility_reqs: &mut Vec<String>,
    ) -> Result<()> {
        if js_code.contains("WebAssembly") || js_code.contains(".wasm") {
            detected.insert(
                BrowserTechnology::WebAssembly,
                TechnologyInfo {
                    name: "WebAssembly".to_string(),
                    occurrence_count: js_code.matches("WebAssembly").count(),
                    features: vec!["binary code execution".to_string()],
                    min_browser_versions: [("IE".to_string(), "unsupported".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: 30,
                    learning_difficulty: 10,
                    degradable: false,
                },
            );
            compatibility_reqs.push("WASM 支持（现代浏览器默认支持）".to_string());
        }

        Ok(())
    }

    fn detect_concurrency(
        js_code: &str,
        detected: &mut HashMap<BrowserTechnology, TechnologyInfo>,
    ) -> Result<()> {
        if js_code.contains("new Worker") {
            detected.insert(
                BrowserTechnology::WebWorker,
                TechnologyInfo {
                    name: "Web Workers".to_string(),
                    occurrence_count: js_code.matches("new Worker").count(),
                    features: vec!["background processing".to_string()],
                    min_browser_versions: [("IE".to_string(), "10+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: 10,
                    learning_difficulty: 7,
                    degradable: true,
                },
            );
        }

        Ok(())
    }

    fn detect_offline_storage(
        js_code: &str,
        detected: &mut HashMap<BrowserTechnology, TechnologyInfo>,
    ) -> Result<()> {
        if js_code.contains("indexedDB") {
            detected.insert(
                BrowserTechnology::IndexedDB,
                TechnologyInfo {
                    name: "IndexedDB".to_string(),
                    occurrence_count: js_code.matches("indexedDB").count(),
                    features: vec!["offline storage".to_string()],
                    min_browser_versions: [("IE".to_string(), "10+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: -10,
                    learning_difficulty: 8,
                    degradable: true,
                },
            );
        }

        if js_code.contains("localStorage") {
            detected.insert(
                BrowserTechnology::LocalStorage,
                TechnologyInfo {
                    name: "LocalStorage".to_string(),
                    occurrence_count: js_code.matches("localStorage").count(),
                    features: vec!["persistent storage".to_string()],
                    min_browser_versions: [("IE".to_string(), "8+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: 0,
                    learning_difficulty: 2,
                    degradable: true,
                },
            );
        }

        Ok(())
    }

    fn detect_network_tech(
        js_code: &str,
        detected: &mut HashMap<BrowserTechnology, TechnologyInfo>,
    ) -> Result<()> {
        if js_code.contains("WebSocket") {
            detected.insert(
                BrowserTechnology::WebSocket,
                TechnologyInfo {
                    name: "WebSocket".to_string(),
                    occurrence_count: js_code.matches("WebSocket").count(),
                    features: vec!["real-time communication".to_string()],
                    min_browser_versions: [("IE".to_string(), "10+".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: -5,
                    learning_difficulty: 6,
                    degradable: true,
                },
            );
        }

        Ok(())
    }

    fn detect_dom_features(
        js_code: &str,
        detected: &mut HashMap<BrowserTechnology, TechnologyInfo>,
    ) -> Result<()> {
        if js_code.contains("IntersectionObserver") {
            detected.insert(
                BrowserTechnology::IntersectionObserver,
                TechnologyInfo {
                    name: "IntersectionObserver".to_string(),
                    occurrence_count: js_code.matches("IntersectionObserver").count(),
                    features: vec!["element visibility detection".to_string()],
                    min_browser_versions: [("IE".to_string(), "unsupported".to_string())]
                        .iter()
                        .cloned()
                        .collect(),
                    performance_impact: 5,
                    learning_difficulty: 5,
                    degradable: true,
                },
            );
        }

        Ok(())
    }

    fn calculate_modernization_score(detected: &HashMap<BrowserTechnology, TechnologyInfo>) -> f64 {
        let mut score = 0.0;
        let mut count = 0;

        for tech in detected.keys() {
            count += 1;
            score += match tech {
                BrowserTechnology::Html5Semantic
                | BrowserTechnology::CssFlexbox
                | BrowserTechnology::CssGrid
                | BrowserTechnology::Async
                | BrowserTechnology::WebAssembly => 20.0,
                BrowserTechnology::CssAnimations
                | BrowserTechnology::Promises
                | BrowserTechnology::ServiceWorker => 15.0,
                _ => 10.0,
            };
        }

        if count == 0 {
            0.0
        } else {
            (score / (count as f64 * 20.0)).min(100.0)
        }
    }

    fn calculate_performance_impact(detected: &HashMap<BrowserTechnology, TechnologyInfo>) -> f64 {
        let impact: i32 = detected.values().map(|info| info.performance_impact).sum();
        (impact as f64).clamp(-100.0, 100.0)
    }

    fn identify_compatibility_issues(
        detected: &HashMap<BrowserTechnology, TechnologyInfo>,
        requirements: &mut Vec<String>,
    ) {
        for info in detected.values() {
            if info
                .min_browser_versions
                .get("IE")
                .map(|v| v.contains("unsupported"))
                .unwrap_or(false)
            {
                requirements.push(format!("{} 不支持 Internet Explorer", info.name));
            }
        }
    }

    fn identify_security_concerns(
        detected: &HashMap<BrowserTechnology, TechnologyInfo>,
        concerns: &mut Vec<String>,
    ) {
        if detected.contains_key(&BrowserTechnology::WebAssembly) {
            concerns.push("WASM 可以执行任意代码，需要验证源代码".to_string());
        }

        if detected.contains_key(&BrowserTechnology::WebSocket) {
            concerns.push("WebSocket 连接需要 WSS 加密".to_string());
        }
    }

    fn generate_recommendations(
        detected: &HashMap<BrowserTechnology, TechnologyInfo>,
        recommendations: &mut Vec<String>,
    ) {
        if detected.contains_key(&BrowserTechnology::WebGL) {
            recommendations.push("提供 WebGL 降级方案用于不支持的浏览器".to_string());
        }

        if detected.len() > 8 {
            recommendations.push("技术栈复杂，建议提供详细的文档和开发指南".to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_canvas() {
        let html = r#"<canvas id="myCanvas"></canvas>"#;
        let css = "";
        let js = "";
        let result = BrowserTechDetector::detect_technologies(html, css, js).unwrap();
        assert!(result
            .detected_technologies
            .contains_key(&BrowserTechnology::Html5Canvas));
    }

    #[test]
    fn test_modernization_score_calculation() {
        let mut tech_map = HashMap::new();
        tech_map.insert(
            BrowserTechnology::Html5Semantic,
            TechnologyInfo {
                name: "HTML5 Semantic".to_string(),
                occurrence_count: 1,
                features: vec![],
                min_browser_versions: Default::default(),
                performance_impact: 0,
                learning_difficulty: 2,
                degradable: true,
            },
        );

        let score = BrowserTechDetector::calculate_modernization_score(&tech_map);
        assert!(score > 0.0);
    }
}
