use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FrameworkDetection {
    pub react: bool,
    pub vue: bool,
    pub angular: bool,
    pub svelte: bool,
    pub jquery: bool,
    pub next: bool,
    pub nuxt: bool,
    pub remix: bool,
    pub vite: bool,
    pub webpack: bool,
}

impl FrameworkDetection {
    pub fn any(&self) -> bool {
        self.react
            || self.vue
            || self.angular
            || self.svelte
            || self.jquery
            || self.next
            || self.nuxt
            || self.remix
            || self.vite
            || self.webpack
    }
}

fn has_any(text: &str, needles: &[&str]) -> bool {
    let lower = text.to_lowercase();
    needles.iter().any(|n| lower.contains(&n.to_lowercase()))
}

/// 快速框架与构建工具指纹识别（启发式）
pub fn detect(html: &str, scripts: &[String]) -> FrameworkDetection {
    let mut d = FrameworkDetection::default();
    let mut blob = String::new();
    blob.push_str(&html.to_lowercase());
    for s in scripts {
        blob.push('\n');
        blob.push_str(&s.to_lowercase());
    }

    d.react = has_any(
        &blob,
        &[
            "react",
            "reactdom",
            "jsx",
            "__REACT_DEVTOOLS__",
            "useState(",
            "useEffect(",
        ],
    );
    d.vue = has_any(
        &blob,
        &[
            "vue",
            "createapp(",
            "__VUE_DEVTOOLS__",
            "v-if",
            "v-for",
            "v-model",
        ],
    );
    d.angular = has_any(
        &blob,
        &[
            "angular",
            "ngmodule",
            "platformbrowserdynamic",
            "ng-if",
            "ng-for",
        ],
    );
    d.svelte = has_any(&blob, &["svelte", "svelte-hmr", "$$invalidate", "onMount("]);
    d.jquery = has_any(&blob, &["jquery", "$(document)", "$.ajax", "$.fn"]);
    d.next = has_any(
        &blob,
        &["__NEXT_DATA__", "next/router", "next/link", "next/head"],
    );
    d.nuxt = has_any(&blob, &["_nuxt/", "nuxt", "useHead(", "defineNuxtConfig("]);
    d.remix = has_any(&blob, &["@remix-run/", "remix", "useLoaderData("]);
    d.vite = has_any(&blob, &["vite", "import.meta.env", "__vite__"]);
    d.webpack = has_any(
        &blob,
        &[
            "webpack",
            "__webpack_require__",
            "webpackChunk",
            "__webpack_exports__",
        ],
    );

    d
}
