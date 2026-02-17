//! Framework Detection Demo - 演示Rust调用Python API进行框架检测
//!
//! 运行前需要先启动Python API服务器：
//! ```bash
//! cd /home/stone/BrowerAI
//! python training/api_server.py &
//! ```
//!
//! 运行示例：
//! ```bash
//! cargo run --example framework_detection_demo
//! ```

use anyhow::Result;
use browerai::ai_integration::FrameworkDetectorClient;

fn main() -> Result<()> {
    // 初始化日志
    env_logger::init();

    println!("=== Framework Detection Demo ===\n");

    // 创建HTTP客户端
    let client = FrameworkDetectorClient::default();

    // 1. 健康检查
    println!("1. 健康检查...");
    match client.health_check() {
        Ok(healthy) => {
            if healthy {
                println!("✅ API服务正常运行\n");
            } else {
                println!("⚠️  API服务异常\n");
                return Ok(());
            }
        }
        Err(e) => {
            eprintln!("❌ API服务器未运行: {}", e);
            eprintln!("\n请先启动Python API服务器：");
            eprintln!("  cd /home/stone/BrowerAI");
            eprintln!("  python training/api_server.py\n");
            return Ok(());
        }
    }

    // 2. 测试React检测
    println!("2. 测试React框架检测...");
    let react_html = r#"
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>React App</title>
        </head>
        <body>
            <div id="root"></div>
            <script src="/_next/static/chunks/webpack-main.js"></script>
            <script src="/_next/static/chunks/framework.js"></script>
            <script>
                const App = () => {
                    const [count, setCount] = React.useState(0);
                    const [data, setData] = React.useState([]);
                    
                    React.useEffect(() => {
                        fetch('/api/data').then(r => r.json()).then(setData);
                    }, []);
                    
                    return React.createElement('div', null, 
                        React.createElement('h1', null, 'Count: ', count),
                        React.createElement('button', { onClick: () => setCount(count + 1) }, 'Increment')
                    );
                };
                
                ReactDOM.render(React.createElement(App), document.getElementById('root'));
            </script>
        </body>
        </html>
    "#;

    match client.detect(react_html) {
        Ok(result) => {
            println!("✅ 检测成功:");
            println!("   框架: {}", result.framework);
            println!("   置信度: {:.1}%", result.confidence * 100.0);
            println!("   检测方法: {}\n", result.method);
        }
        Err(e) => {
            eprintln!("❌ 检测失败: {}\n", e);
        }
    }

    // 3. 测试Vue检测
    println!("3. 测试Vue框架检测...");
    let vue_html = r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vue App</title>
            <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
        </head>
        <body>
            <div id="app">
                <h1 v-if="showTitle">{{ message }}</h1>
                <ul>
                    <li v-for="item in items" :key="item.id" @click="handleClick(item)">
                        {{ item.name }}
                    </li>
                </ul>
                <button v-on:click="increment">Count: {{ count }}</button>
            </div>
            <script>
                const { createApp, ref, computed } = Vue;
                
                createApp({
                    setup() {
                        const message = ref('Hello Vue!');
                        const count = ref(0);
                        const items = ref([
                            { id: 1, name: 'Item 1' },
                            { id: 2, name: 'Item 2' }
                        ]);
                        
                        const increment = () => count.value++;
                        const handleClick = (item) => console.log(item);
                        
                        return { message, count, items, increment, handleClick };
                    }
                }).mount('#app');
            </script>
        </body>
        </html>
    "#;

    match client.detect(vue_html) {
        Ok(result) => {
            println!("✅ 检测成功:");
            println!("   框架: {}", result.framework);
            println!("   置信度: {:.1}%", result.confidence * 100.0);
            println!("   检测方法: {}\n", result.method);
        }
        Err(e) => {
            eprintln!("❌ 检测失败: {}\n", e);
        }
    }

    // 4. 测试批量检测
    println!("4. 测试批量检测...");
    let websites = vec![
        (
            "https://example1.com".to_string(),
            r#"<html><head><script src="/_next/static/main.js"></script></head></html>"#
                .to_string(),
        ),
        (
            "https://example2.com".to_string(),
            r#"<html><body><div v-if="show" v-for="item in items"></div></body></html>"#
                .to_string(),
        ),
        (
            "https://example3.com".to_string(),
            r#"<html><body><div ng-app="myApp" ng-controller="MainCtrl"></div></body></html>"#
                .to_string(),
        ),
    ];

    match client.batch_detect(websites) {
        Ok(response) => {
            println!("✅ 批量检测成功:");
            println!("   总数: {}", response.total);
            for result in response.results {
                if let Some(err) = result.error {
                    println!("   ❌ {}: {}", result.url, err);
                } else {
                    println!(
                        "   ✓ {}: {} ({:.1}%)",
                        result.url,
                        result.framework,
                        result.confidence * 100.0
                    );
                }
            }
            println!();
        }
        Err(e) => {
            eprintln!("❌ 批量检测失败: {}\n", e);
        }
    }

    // 5. 测试Angular检测
    println!("5. 测试Angular框架检测...");
    let angular_html = r#"
        <!DOCTYPE html>
        <html ng-app="myApp">
        <head>
            <script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.8.2/angular.min.js"></script>
        </head>
        <body ng-controller="MainController">
            <div ng-if="isLoggedIn">
                <h1>{{ title }}</h1>
                <ul>
                    <li ng-repeat="item in items track by $index">
                        {{ item.name }}
                    </li>
                </ul>
                <button ng-click="increment()">Count: {{ count }}</button>
            </div>
            <script>
                angular.module('myApp', [])
                    .controller('MainController', function($scope) {
                        $scope.title = 'Angular App';
                        $scope.count = 0;
                        $scope.isLoggedIn = true;
                        $scope.items = [
                            { name: 'Item 1' },
                            { name: 'Item 2' }
                        ];
                        $scope.increment = function() {
                            $scope.count++;
                        };
                    });
            </script>
        </body>
        </html>
    "#;

    match client.detect(angular_html) {
        Ok(result) => {
            println!("✅ 检测成功:");
            println!("   框架: {}", result.framework);
            println!("   置信度: {:.1}%", result.confidence * 100.0);
            println!("   检测方法: {}\n", result.method);
        }
        Err(e) => {
            eprintln!("❌ 检测失败: {}\n", e);
        }
    }

    println!("=== Demo完成 ===");

    Ok(())
}
