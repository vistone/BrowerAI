//! 框架检测Rust集成测试
//! 
//! 测试Rust HTTP客户端调用Python API的完整流程

use browerai::ai_integration::FrameworkDetectorClient;

#[test]
#[ignore] // 需要Python API服务器运行: python3 training/api_server.py
fn test_rust_python_integration_health_check() {
    let client = FrameworkDetectorClient::default();
    let result = client.health_check();
    assert!(result.is_ok(), "健康检查失败: {:?}", result.err());
    assert!(result.unwrap(), "API服务器不健康");
}

#[test]
#[ignore] // 需要Python API服务器运行
fn test_detect_react_framework() {
    let client = FrameworkDetectorClient::default();
    
    let react_html = r#"
        <!DOCTYPE html>
        <html>
        <head><title>React App</title></head>
        <body>
            <div id="root"></div>
            <script src="/_next/static/chunks/main.js"></script>
            <script>
                const App = () => {
                    const [count, setCount] = React.useState(0);
                    return React.createElement('div', null, count);
                };
                ReactDOM.render(React.createElement(App), document.getElementById('root'));
            </script>
        </body>
        </html>
    "#;

    let result = client.detect(react_html).expect("检测失败");
    
    assert_eq!(result.framework, "React", "框架识别错误");
    assert!(result.confidence > 0.8, "置信度过低: {}", result.confidence);
    assert!(result.success, "检测未成功");
}

#[test]
#[ignore]
fn test_detect_vue_framework() {
    let client = FrameworkDetectorClient::default();
    
    let vue_html = r#"
        <html>
        <head><script src="https://unpkg.com/vue@3"></script></head>
        <body>
            <div id="app" v-if="show">
                <h1>{{ message }}</h1>
                <ul><li v-for="item in items" :key="item.id">{{ item.name }}</li></ul>
            </div>
            <script>
                const { createApp } = Vue;
                createApp({
                    setup() {
                        const message = ref('Hello Vue!');
                        return { message };
                    }
                }).mount('#app');
            </script>
        </body>
        </html>
    "#;

    let result = client.detect(vue_html).expect("检测失败");
    
    assert_eq!(result.framework, "Vue");
    assert!(result.confidence > 0.8);
}

#[test]
#[ignore]
fn test_batch_detection() {
    let client = FrameworkDetectorClient::default();

    let websites = vec![
        (
            "https://react-example.com".to_string(),
            r#"<html><script src="/_next/static/main.js"></script></html>"#.to_string(),
        ),
        (
            "https://vue-example.com".to_string(),
            r#"<html><div v-if="true" v-for="item in items"></div></html>"#.to_string(),
        ),
        (
            "https://angular-example.com".to_string(),
            r#"<html><div ng-app="app" ng-controller="Ctrl"></div></html>"#.to_string(),
        ),
    ];

    let response = client.batch_detect(websites).expect("批量检测失败");
    
    assert_eq!(response.total, 3);
    assert!(response.success);
    
    // 验证所有检测都成功
    for result in &response.results {
        assert!(result.error.is_none(), "检测失败: {:?}", result.error);
        assert!(result.confidence > 0.5, "置信度过低: {}", result.confidence);
    }
    
    // 验证框架类型
    assert_eq!(response.results[0].framework, "React");
    assert_eq!(response.results[1].framework, "Vue");
    assert_eq!(response.results[2].framework, "Angular");
}

#[test]
#[ignore]
fn test_high_accuracy_on_real_websites() {
    let client = FrameworkDetectorClient::default();
    
    // 真实网站的HTML片段
    let test_cases = vec![
        ("React + Next.js", r#"
            <html>
            <head>
                <script src="/_next/static/chunks/webpack.js"></script>
                <script src="/_next/static/chunks/framework.js"></script>
                <script src="/_next/static/chunks/main.js"></script>
            </head>
            <body>
                <div id="__next">
                    <script>
                        const e = React.createElement;
                        ReactDOM.render(e(App), document.getElementById('__next'));
                    </script>
                </div>
            </body>
            </html>
        "#, "React"),
        
        ("Vue 3 Composition API", r#"
            <html>
            <head><script src="https://unpkg.com/vue@3/dist/vue.global.js"></script></head>
            <body>
                <div id="app" v-cloak>
                    <h1 v-if="isVisible">{{ title }}</h1>
                    <button @click="increment">Count: {{ count }}</button>
                    <ul>
                        <li v-for="item in items" :key="item.id">{{ item.name }}</li>
                    </ul>
                </div>
                <script>
                    const { createApp, ref, computed } = Vue;
                    createApp({
                        setup() {
                            const count = ref(0);
                            const increment = () => count.value++;
                            return { count, increment };
                        }
                    }).mount('#app');
                </script>
            </body>
            </html>
        "#, "Vue"),
    ];

    for (name, html, expected_framework) in test_cases {
        let result = client.detect(html).expect(&format!("检测失败: {}", name));
        
        assert_eq!(
            result.framework, expected_framework,
            "测试 '{}' 失败: 期望 {}, 实际 {}",
            name, expected_framework, result.framework
        );
        
        assert!(
            result.confidence > 0.8,
            "测试 '{}' 置信度过低: {:.2}%",
            name, result.confidence * 100.0
        );
        
        println!("✅ {} - 检测成功: {} ({:.1}%)", 
            name, result.framework, result.confidence * 100.0);
    }
}

#[test]
#[ignore]
fn test_error_handling_when_api_unavailable() {
    // 使用错误的端口测试错误处理
    let client = FrameworkDetectorClient::new("http://localhost:9999");
    
    let result = client.health_check();
    assert!(result.is_err(), "应该返回错误");
    
    let html = "<html><body>Test</body></html>";
    let detect_result = client.detect(html);
    assert!(detect_result.is_err(), "检测应该失败");
}

#[test]
#[ignore]
fn test_performance_benchmark() {
    use std::time::Instant;
    
    let client = FrameworkDetectorClient::default();
    
    let test_html = r#"
        <html>
        <head><script src="/_next/static/main.js"></script></head>
        <body><div id="root"></div></body>
        </html>
    "#;
    
    // 预热
    client.detect(test_html).expect("预热失败");
    
    // 基准测试
    let iterations = 100;
    let start = Instant::now();
    
    for _ in 0..iterations {
        client.detect(test_html).expect("检测失败");
    }
    
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
    
    println!("平均检测时间: {:.2}ms ({} 次迭代)", avg_ms, iterations);
    
    // 验证性能在合理范围内 (HTTP调用应该在100ms以内)
    assert!(avg_ms < 100.0, "检测速度过慢: {:.2}ms", avg_ms);
}
