//! 框架检测HTTP客户端集成测试

use browerai_ai_integration::FrameworkDetectorClient;

#[test]
#[ignore] // 需要Python API服务器运行: python3 training/api_server.py
fn test_health_check() {
    let client = FrameworkDetectorClient::default();
    let result = client.health_check();
    assert!(result.is_ok(), "健康检查失败: {:?}", result.err());
    assert!(result.unwrap(), "API服务器不健康");
}

#[test]
#[ignore]
fn test_detect_react() {
    let client = FrameworkDetectorClient::default();

    let react_html = r#"
        <html>
        <head><title>React App</title></head>
        <body>
            <div id="root"></div>
            <script src="/_next/static/chunks/main.js"></script>
            <script>
                const [count, setCount] = React.useState(0);
                ReactDOM.render(React.createElement('div'), document.getElementById('root'));
            </script>
        </body>
        </html>
    "#;

    let result = client.detect(react_html).expect("检测失败");
    assert_eq!(result.framework, "React");
    assert!(result.confidence > 0.8);
}

#[test]
#[ignore]
fn test_batch_detect() {
    let client = FrameworkDetectorClient::default();

    let websites = vec![
        (
            "https://example1.com".to_string(),
            r#"<script src="/_next/static/main.js"></script>"#.to_string(),
        ),
        (
            "https://example2.com".to_string(),
            r#"<div v-if="true"></div>"#.to_string(),
        ),
    ];

    let response = client.batch_detect(websites).expect("批量检测失败");
    assert_eq!(response.total, 2);
    assert!(response.success);
}
