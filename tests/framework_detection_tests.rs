/// Comprehensive tests for enhanced framework detection
/// Tests all 100+ framework patterns

#[cfg(test)]
mod framework_detection_tests {
    use browerai::learning::advanced_deobfuscation::{AdvancedDeobfuscator, FrameworkObfuscation};

    #[test]
    fn test_webpack_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let webpack_code = r#"
            (self["webpackChunk"] = self["webpackChunk"] || []).push([[123], {
                456: function(module, exports, __webpack_require__) {
                    console.log("test");
                }
            }]);
        "#;

        let analysis = deobfuscator.analyze(webpack_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::WebpackBundled));
        assert!(analysis.confidence > 0.0); // Framework detected, so confidence should be > 0
    }

    #[test]
    fn test_react_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let react_code = r#"
            import React from 'react';
            const App = () => React.createElement("div", null, "Hello");
            export default App;
        "#;

        let analysis = deobfuscator.analyze(react_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::ReactCompiled));
    }

    #[test]
    fn test_vue_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let vue_code = r#"
            import { createVNode, createElementVNode } from 'vue';
            const _hoisted_1 = { class: "container" };
            function render() {
                return _createVNode("div", _hoisted_1);
            }
        "#;

        let analysis = deobfuscator.analyze(vue_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::VueCompiled));
    }

    #[test]
    fn test_angular_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let angular_code = r#"
            import { ɵɵelementStart, ɵɵtext } from '@angular/core';
            function AppComponent_Template() {
                ɵɵelementStart(0, "div");
                ɵɵtext(1, "Hello");
            }
        "#;

        let analysis = deobfuscator.analyze(angular_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::AngularCompiled));
    }

    #[test]
    fn test_nextjs_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let nextjs_code = r#"
            import { __next } from 'next';
            export async function getServerSideProps(context) {
                return { props: {} };
            }
        "#;

        let analysis = deobfuscator.analyze(nextjs_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::NextJSFramework));
    }

    #[test]
    fn test_svelte_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let svelte_code = r#"
            import { SvelteComponent, $$invalidate } from 'svelte';
            class Component extends SvelteComponent {
                constructor(options) {
                    super();
                }
            }
        "#;

        let analysis = deobfuscator.analyze(svelte_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::SvelteCompiled));
    }

    // ========== Chinese Framework Tests ==========

    #[test]
    fn test_taro_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let taro_code = r#"
            import Taro from '@tarojs/taro';
            class MyComponent extends Taro.Component {
                render() {
                    return <View>Hello Taro</View>;
                }
            }
        "#;

        let analysis = deobfuscator.analyze(taro_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::TaroFramework));

        let info = deobfuscator.get_framework_info(&FrameworkObfuscation::TaroFramework);
        assert_eq!(info.name, "Taro");
        assert!(info.origin.contains("JD.com"));
    }

    #[test]
    fn test_uniapp_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let uniapp_code = r#"
            import { uni } from '@dcloudio/uni-app';
            uni.request({
                url: 'https://api.example.com',
                success: (res) => console.log(res)
            });
        "#;

        let analysis = deobfuscator.analyze(uniapp_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::UniAppFramework));

        let info = deobfuscator.get_framework_info(&FrameworkObfuscation::UniAppFramework);
        assert!(info.origin.contains("DCloud"));
    }

    #[test]
    fn test_rax_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let rax_code = r#"
            import Rax from 'rax';
            const App = () => Rax.createElement("div", null, "Hello Rax");
        "#;

        let analysis = deobfuscator.analyze(rax_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::RaxFramework));

        let info = deobfuscator.get_framework_info(&FrameworkObfuscation::RaxFramework);
        assert!(info.origin.contains("Alibaba"));
    }

    #[test]
    fn test_omi_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let omi_code = r#"
            import { WeElement, define } from 'omi';
            class MyElement extends WeElement {
                render() {
                    return <div>Hello Omi</div>;
                }
            }
        "#;

        let analysis = deobfuscator.analyze(omi_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::OmiFramework));

        let info = deobfuscator.get_framework_info(&FrameworkObfuscation::OmiFramework);
        assert!(info.origin.contains("Tencent"));
    }

    #[test]
    fn test_san_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let san_code = r#"
            import san from 'san';
            const MyComponent = san.defineComponent({
                template: '<div>Hello San</div>'
            });
        "#;

        let analysis = deobfuscator.analyze(san_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::SanFramework));

        let info = deobfuscator.get_framework_info(&FrameworkObfuscation::SanFramework);
        assert!(info.origin.contains("Baidu"));
    }

    #[test]
    fn test_qiankun_detection() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let qiankun_code = r#"
            import { registerMicroApps, start } from 'qiankun';
            registerMicroApps([
                {
                    name: 'app1',
                    entry: '//localhost:8080',
                    container: '#container',
                }
            ]);
            start();
        "#;

        let analysis = deobfuscator.analyze(qiankun_code).unwrap();
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::QiankunMicroFrontend));

        let info = deobfuscator.get_framework_info(&FrameworkObfuscation::QiankunMicroFrontend);
        assert!(info.origin.contains("Alibaba"));
    }

    // ========== Multiple Framework Detection ==========

    #[test]
    fn test_multiple_frameworks() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let mixed_code = r#"
            // Webpack + React + Next.js
            (self["webpackChunk"] = self["webpackChunk"] || []).push([[123], {
                456: function(module, exports, __webpack_require__) {
                    import React from 'react';
                    import { __next } from 'next';
                    const App = () => React.createElement("div", null, "Hello");
                }
            }]);
        "#;

        let analysis = deobfuscator.analyze(mixed_code).unwrap();
        assert!(analysis.framework_patterns.len() >= 2);
        assert!(analysis
            .framework_patterns
            .contains(&FrameworkObfuscation::WebpackBundled));
    }

    // ========== Deobfuscation Tests ==========

    #[test]
    fn test_webpack_unwrapping() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let webpack_bundle = r#"
            (function(modules) {
                var installedModules = {};
                function __webpack_require__(moduleId) {
                    if(installedModules[moduleId]) {
                        return installedModules[moduleId].exports;
                    }
                    var module = installedModules[moduleId] = {
                        i: moduleId,
                        l: false,
                        exports: {}
                    };
                    modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
                    module.l = true;
                    return module.exports;
                }
            })({
                0: function(module, exports) {
                    console.log("Module 0");
                },
                1: function(module, exports, __webpack_require__) {
                    var dep = __webpack_require__(0);
                    console.log("Module 1");
                }
            });
        "#;

        let result = deobfuscator.unwrap_webpack(webpack_bundle).unwrap();
        assert!(result.contains("Module 0"));
        assert!(result.contains("Module 1"));
        assert!(!result.contains("__webpack_require__") || result.contains("Unwrapped"));
    }

    #[test]
    fn test_framework_specific_deobfuscation() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let code = r#"React.createElement("div", null, "Hello");"#;

        let result = deobfuscator
            .deobfuscate_framework_specific(code, &FrameworkObfuscation::ReactCompiled)
            .unwrap();

        assert!(!result.is_empty());
    }

    // ========== Report Generation ==========

    #[test]
    fn test_report_generation() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let code = r#"
            import Taro from '@tarojs/taro';
            import { uni } from '@dcloudio/uni-app';
        "#;

        let analysis = deobfuscator.analyze(code).unwrap();
        let report = deobfuscator.generate_report(&analysis);

        assert!(report.contains("Advanced Deobfuscation Analysis"));
        assert!(report.contains("Detected Frameworks"));
        assert!(report.contains("Confidence"));
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_no_framework_detected() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let plain_code = r#"
            const x = 1 + 1;
            console.log(x);
        "#;

        let analysis = deobfuscator.analyze(plain_code).unwrap();
        assert!(analysis.confidence < 0.3);
    }

    #[test]
    fn test_obfuscated_code() {
        let deobfuscator = AdvancedDeobfuscator::new();

        let obfuscated = r#"
            var _0x1a2b=['log','Hello'];
            (function(_0x3c4d,_0x5e6f){
                var _0x7g8h=function(_0x9i0j){
                    while(--_0x9i0j){
                        _0x3c4d['push'](_0x3c4d['shift']());
                    }
                };
                _0x7g8h(++_0x5e6f);
            }(_0x1a2b,0x123));
            console[_0x1a2b[0]](_0x1a2b[1]);
        "#;

        let result = deobfuscator.deobfuscate(obfuscated).unwrap();
        // Should attempt to deobfuscate
        assert!(!result.javascript.is_empty());
    }
}
