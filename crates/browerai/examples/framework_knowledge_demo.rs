/// Framework Knowledge Base Demonstration
///
/// This example demonstrates the comprehensive framework knowledge system
/// that can detect and analyze obfuscation patterns from global JavaScript frameworks.

use browerai_learning::{FrameworkKnowledgeBase, FrameworkCategory};

fn main() {
    println!("=== BrowerAI Framework Knowledge Base Demo ===\n");
    
    // Initialize knowledge base
    let kb = FrameworkKnowledgeBase::new();
    
    // Print statistics
    let stats = kb.get_statistics();
    println!("📊 Knowledge Base Statistics:");
    println!("   Total Frameworks: {}", stats.total_frameworks);
    println!("   Total Signatures: {}", stats.total_signatures);
    println!("   Total Patterns: {}", stats.total_patterns);
    println!("   Total Strategies: {}", stats.total_strategies);
    println!();
    
    println!("📁 Frameworks by Category:");
    for (category, count) in &stats.category_counts {
        println!("   {:?}: {}", category, count);
    }
    println!();
    
    // Example 1: Detect React code
    println!("🔍 Example 1: Detecting React Framework");
    let react_code = r#"
        import React from 'react';
        
        function App() {
            return React.createElement("div", { className: "container" }, 
                React.createElement("h1", null, "Hello World"),
                React.createElement("p", null, "This is a React app")
            );
        }
    "#;
    
    let results = kb.analyze_code(react_code).unwrap();
    println!("   Detected {} frameworks:", results.len());
    for result in &results {
        println!("   • {} (confidence: {:.1}%)", result.framework_name, result.confidence * 100.0);
        println!("     Matched signatures: {:?}", result.matched_signatures);
        println!("     Strategies available: {}", result.applicable_strategies.len());
    }
    println!();
    
    // Example 2: Detect Vue 3 code
    println!("🔍 Example 2: Detecting Vue 3 Framework");
    let vue_code = r#"
        import { createApp } from 'vue';
        
        const _hoisted_1 = { class: "container" };
        const _hoisted_2 = { class: "title" };
        
        function render(_ctx) {
            return (_openBlock(), _createElementVNode("div", _hoisted_1, [
                _createElementVNode("h1", _hoisted_2, _toDisplayString(_ctx.title), 1),
                _createTextVNode(" Welcome to Vue 3")
            ]));
        }
    "#;
    
    let results = kb.analyze_code(vue_code).unwrap();
    println!("   Detected {} frameworks:", results.len());
    for result in &results {
        println!("   • {} (confidence: {:.1}%)", result.framework_name, result.confidence * 100.0);
        println!("     Matched signatures: {:?}", result.matched_signatures);
    }
    println!();
    
    // Example 3: Detect Webpack bundle
    println!("🔍 Example 3: Detecting Webpack Bundle");
    let webpack_code = r#"
        (function(modules) {
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
            return __webpack_require__(__webpack_require__.s = 0);
        })([
            function(module, exports) {
                console.log("Hello from Webpack!");
            }
        ]);
    "#;
    
    let results = kb.analyze_code(webpack_code).unwrap();
    println!("   Detected {} frameworks:", results.len());
    for result in &results {
        println!("   • {} (confidence: {:.1}%)", result.framework_name, result.confidence * 100.0);
        if let Some(framework) = kb.get_framework(&result.framework_id) {
            println!("     Origin: {}", framework.origin);
            println!("     Maintainer: {}", framework.maintainer);
        }
    }
    println!();
    
    // Example 4: Detect Chinese frameworks
    println!("🔍 Example 4: Detecting Chinese Frameworks");
    
    // Taro
    let taro_code = r#"
        import Taro from '@tarojs/taro';
        import { View, Text } from '@tarojs/components';
        
        export default function Index() {
            Taro.request({
                url: 'https://api.example.com/data',
                success: (res) => {
                    console.log(res);
                }
            });
            
            return (
                <View className="index">
                    <Text>Taro App</Text>
                </View>
            );
        }
    "#;
    
    let results = kb.analyze_code(taro_code).unwrap();
    println!("   Taro detection:");
    for result in &results {
        if result.framework_id == "taro" || result.framework_id == "react" {
            println!("   • {} (confidence: {:.1}%)", result.framework_name, result.confidence * 100.0);
        }
    }
    
    // Uni-app
    let uniapp_code = r#"
        <template>
            <view class="content">
                <text>{{ title }}</text>
            </view>
        </template>
        
        <script>
        export default {
            data() {
                return {
                    title: 'Uni-app'
                }
            },
            onLoad() {
                uni.request({
                    url: 'https://api.example.com/data',
                    success: (res) => {
                        console.log(res);
                    }
                });
            }
        }
        </script>
    "#;
    
    let results = kb.analyze_code(uniapp_code).unwrap();
    println!("   Uni-app detection:");
    for result in &results {
        if result.framework_id == "uni-app" {
            println!("   • {} (confidence: {:.1}%)", result.framework_name, result.confidence * 100.0);
        }
    }
    println!();
    
    // Example 5: Detect obfuscator patterns
    println!("🔍 Example 5: Detecting Obfuscator Patterns");
    let obfuscated_code = r#"
        var _0xabcd = ['hello', 'world', 'test', 'string1', 'string2', 'string3'];
        (function(_0x1, _0x2) {
            var _0x3 = function(_0x4) {
                while (--_0x4) {
                    _0x1['push'](_0x1['shift']());
                }
            };
            _0x3(++_0x2);
        }(_0xabcd, 0x123));
        
        function test() {
            console.log(_0xabcd[0]);
            if (false) {
                deadCode();
            }
            debugger;
        }
    "#;
    
    let results = kb.analyze_code(obfuscated_code).unwrap();
    println!("   Detected {} obfuscation patterns:", results.len());
    for result in &results {
        println!("   • {} (confidence: {:.1}%)", result.framework_name, result.confidence * 100.0);
        println!("     Available strategies:");
        for strategy in &result.applicable_strategies {
            println!("       - {} (success rate: {:.0}%, priority: {})", 
                strategy.name, strategy.success_rate * 100.0, strategy.priority);
        }
    }
    println!();
    
    // Example 6: Browse frameworks by category
    println!("📚 Example 6: Chinese Mobile Frameworks");
    let chinese_mobile = kb.get_frameworks_by_category(&FrameworkCategory::MobileCrossPlatform);
    println!("   Found {} frameworks:", chinese_mobile.len());
    for framework in chinese_mobile {
        println!("   • {} ({}) - {}", framework.name, framework.origin, framework.maintainer);
        println!("     Signatures: {}, Patterns: {}, Strategies: {}", 
            framework.signatures.len(), 
            framework.obfuscation_patterns.len(),
            framework.strategies.len());
    }
    println!();
    
    println!("✅ Framework Knowledge Base demonstration complete!");
    println!("   This system can detect and analyze {} frameworks worldwide", stats.total_frameworks);
    println!("   with {} deobfuscation strategies available.", stats.total_strategies);
}
