# API 端点文档

完整的BrowerAI API参考指南,包含所有端点、请求格式和响应示例。

## 基础信息

- **主机**: `http://localhost:3000` (开发环境)
- **主机**: `https://api.browerai.com` (生产环境)
- **API版本**: v1
- **认证**: API Key (可选,见[认证](#认证))
- **速率限制**: 1000请求/分钟 (见[速率限制](#速率限制))

---

## API概览

| 端点 | 方法 | 描述 | 版本 |
|------|------|------|------|
| `/api/health` | GET | 健康检查 | 基础 |
| `/api/version` | GET | 版本信息 | 基础 |
| `/api/v1/render` | POST | 渲染HTML/CSS | v1 |
| `/api/v1/parse/html` | POST | 解析HTML | v1 |
| `/api/v1/parse/css` | POST | 解析CSS | v1 |

---

## 端点详解

### 1. 健康检查

**用途**: 检查API服务器是否在运行

**请求**:
```bash
GET /api/health
```

**cURL示例**:
```bash
curl -X GET http://localhost:3000/api/health
```

**Python示例**:
```python
import requests

response = requests.get('http://localhost:3000/api/health')
print(response.status_code)  # 200
print(response.json())
# 输出: {"status": "ok"}
```

**JavaScript示例**:
```javascript
fetch('http://localhost:3000/api/health')
  .then(res => res.json())
  .then(data => console.log(data))
  .catch(err => console.error(err));
```

**响应** (200 OK):
```json
{
  "status": "ok"
}
```

**错误响应** (503 Service Unavailable):
```json
{
  "error": "Service unavailable"
}
```

---

### 2. 版本信息

**用途**: 获取API和应用版本信息

**请求**:
```bash
GET /api/version
```

**cURL示例**:
```bash
curl -X GET http://localhost:3000/api/version
```

**Python示例**:
```python
import requests

response = requests.get('http://localhost:3000/api/version')
data = response.json()
print(f"Version: {data['version']}")
print(f"Build: {data['build_time']}")
```

**响应** (200 OK):
```json
{
  "version": "1.0.0",
  "build_time": "2026-02-17T10:00:00Z",
  "rust_version": "1.70.0",
  "features": ["onnx", "postgres", "redis"]
}
```

---

### 3. 渲染HTML/CSS

**用途**: 渲染HTML文档并获取DOM树和样式信息

**请求**:
```bash
POST /api/v1/render
Content-Type: application/json
```

**请求体**:
```json
{
  "html": "<html><body><h1>Hello World</h1></body></html>",
  "css": "h1 { color: blue; font-size: 24px; }",
  "options": {
    "viewport_width": 1024,
    "viewport_height": 768,
    "enable_ai": true,
    "timeout_ms": 5000
  }
}
```

**cURL示例**:
```bash
curl -X POST http://localhost:3000/api/v1/render \
  -H "Content-Type: application/json" \
  -d '{
    "html": "<html><body><h1>Test</h1></body></html>",
    "css": "h1 { color: red; }",
    "options": {
      "viewport_width": 1024,
      "viewport_height": 768,
      "enable_ai": false
    }
  }'
```

**Python示例**:
```python
import requests
import json

url = 'http://localhost:3000/api/v1/render'
payload = {
    'html': '<html><body><h1>Hello</h1></body></html>',
    'css': 'h1 { color: blue; }',
    'options': {
        'viewport_width': 1024,
        'viewport_height': 768,
        'enable_ai': True
    }
}

response = requests.post(url, json=payload)
result = response.json()
print(f"Status: {response.status_code}")
print(f"DOM nodes: {result.get('node_count', 0)}")
print(f"Render time: {result.get('render_time_ms', 0)}ms")
```

**JavaScript示例**:
```javascript
async function renderPage(html, css) {
  const response = await fetch('http://localhost:3000/api/v1/render', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      html: html,
      css: css,
      options: {
        viewport_width: 1024,
        viewport_height: 768,
        enable_ai: false
      }
    })
  });
  
  const data = await response.json();
  return data;
}

// 使用
renderPage('<h1>Test</h1>', 'h1 { color: blue; }')
  .then(result => console.log(result))
  .catch(err => console.error(err));
```

**响应** (200 OK):
```json
{
  "success": true,
  "node_count": 5,
  "dom_tree": {
    "type": "Document",
    "children": [
      {
        "type": "Element",
        "name": "html",
        "attributes": {},
        "children": [
          {
            "type": "Element",
            "name": "body",
            "attributes": {},
            "children": [
              {
                "type": "Element",
                "name": "h1",
                "attributes": {},
                "text": "Hello World"
              }
            ]
          }
        ]
      }
    ]
  },
  "styles": {
    "h1": {
      "color": "rgb(0, 0, 255)",
      "font-size": "24px"
    }
  },
  "render_time_ms": 45,
  "ai_enhanced": false
}
```

**错误响应** (400 Bad Request):
```json
{
  "error": "Invalid HTML",
  "details": "Unexpected character at position 10"
}
```

**错误响应** (408 Request Timeout):
```json
{
  "error": "Rendering timeout",
  "details": "Rendering exceeded 5000ms timeout"
}
```

---

### 4. 解析HTML

**用途**: 解析HTML并返回DOM树结构

**请求**:
```bash
POST /api/v1/parse/html
Content-Type: application/json
```

**请求体**:
```json
{
  "html": "<html><head><title>Test</title></head><body><div>Content</div></body></html>",
  "options": {
    "strict_mode": false,
    "preserve_whitespace": true,
    "timeout_ms": 3000
  }
}
```

**cURL示例**:
```bash
curl -X POST http://localhost:3000/api/v1/parse/html \
  -H "Content-Type: application/json" \
  -d '{
    "html": "<html><body><p>Hello</p></body></html>",
    "options": {
      "strict_mode": false
    }
  }'
```

**Python示例**:
```python
import requests

url = 'http://localhost:3000/api/v1/parse/html'
payload = {
    'html': '<html><body><h1>Test</h1><p>Content</p></body></html>',
    'options': {
        'strict_mode': False
    }
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    result = response.json()
    print(f"Elements found: {result['element_count']}")
    print(f"Parse time: {result['parse_time_ms']}ms")
```

**响应** (200 OK):
```json
{
  "success": true,
  "element_count": 4,
  "dom_tree": {
    "type": "Document",
    "children": [
      {
        "type": "Element",
        "name": "html",
        "children": [
          {
            "type": "Element",
            "name": "head",
            "children": [
              {
                "type": "Element",
                "name": "title",
                "text": "Test"
              }
            ]
          },
          {
            "type": "Element",
            "name": "body",
            "children": [
              {
                "type": "Element",
                "name": "div",
                "text": "Content"
              }
            ]
          }
        ]
      }
    ]
  },
  "parse_time_ms": 12,
  "warnings": []
}
```

**错误响应** (400 Bad Request):
```json
{
  "error": "Invalid HTML structure",
  "details": "Mismatched closing tag at line 5, column 12"
}
```

---

### 5. 解析CSS

**用途**: 解析CSS并返回规则树结构

**请求**:
```bash
POST /api/v1/parse/css
Content-Type: application/json
```

**请求体**:
```json
{
  "css": "body { margin: 0; } h1 { color: blue; font-size: 24px; }",
  "options": {
    "vendor_prefix": true,
    "minify": false,
    "timeout_ms": 2000
  }
}
```

**cURL示例**:
```bash
curl -X POST http://localhost:3000/api/v1/parse/css \
  -H "Content-Type: application/json" \
  -d '{
    "css": "body { color: black; } h1 { color: red; }"
  }'
```

**Python示例**:
```python
import requests

url = 'http://localhost:3000/api/v1/parse/css'
payload = {
    'css': '''
    body {
      font-family: Arial, sans-serif;
      color: #333;
      margin: 0;
      padding: 0;
    }
    h1 {
      color: #0066cc;
      font-size: 2em;
    }
    '''
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    result = response.json()
    print(f"Rules found: {result['rule_count']}")
    for rule in result['rules'][:3]:  # 显示前3条规则
        print(f"  Selector: {rule['selector']}")
        print(f"  Properties: {rule['properties']}")
```

**JavaScript示例**:
```javascript
async function parseCSS(cssText) {
  const response = await fetch('http://localhost:3000/api/v1/parse/css', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      css: cssText,
      options: {
        vendor_prefix: true
      }
    })
  });
  
  const data = await response.json();
  
  data.rules.forEach(rule => {
    console.log(`Selector: ${rule.selector}`);
    rule.properties.forEach(prop => {
      console.log(`  ${prop.name}: ${prop.value}`);
    });
  });
  
  return data;
}

// 使用
parseCSS('h1 { color: blue; font-size: 2em; }');
```

**响应** (200 OK):
```json
{
  "success": true,
  "rule_count": 2,
  "rules": [
    {
      "selector": "body",
      "specificity": [0, 0, 1],
      "properties": [
        {"name": "margin", "value": "0"},
        {"name": "padding", "value": "0"}
      ]
    },
    {
      "selector": "h1",
      "specificity": [0, 0, 1],
      "properties": [
        {"name": "color", "value": "blue"},
        {"name": "font-size", "value": "24px"}
      ]
    }
  ],
  "parse_time_ms": 8,
  "warnings": []
}
```

**错误响应** (400 Bad Request):
```json
{
  "error": "Invalid CSS syntax",
  "details": "Unexpected character '}' at line 2, column 15",
  "line": 2,
  "column": 15
}
```

---

## 认证

### API Key认证 (可选)

如果启用了API注册表,需要在请求头中提供API Key:

```bash
curl -X GET http://localhost:3000/api/health \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**获取API Key**:
1. 访问: `http://localhost:3000/auth/register`
2. 提交注册表单
3. 使用返回的API Key

### No Auth (默认)

默认配置下,所有端点都无需认证。

---

## 速率限制

### 限制规则

- **基础端点** (`/api/health`, `/api/version`): 无限制
- **v1端点** (`/api/v1/*`): 1000请求/分钟
- **超限响应**: HTTP 429 Too Many Requests

### 响应头

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1708055000
```

### 超限响应

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

---

## 错误处理

### 通用错误格式

所有错误响应都遵循统一格式:

```json
{
  "error": "Error type",
  "details": "Detailed error message",
  "code": "ERROR_CODE",
  "timestamp": "2026-02-17T10:00:00Z"
}
```

### 常见HTTP状态码

| 状态码 | 含义 | 示例 |
|--------|------|------|
| 200 | 成功 | 解析完成 |
| 400 | 请求格式错误 | HTML格式不对 |
| 408 | 请求超时 | 渲染超过5秒 |
| 429 | 速率限制超出 | 请求过于频繁 |
| 500 | 服务器内部错误 | 数据库连接失败 |
| 503 | 服务不可用 | 依赖服务离线 |

---

## 最佳实践

### 1. 错误处理

```python
import requests

try:
    response = requests.post(
        'http://localhost:3000/api/v1/parse/html',
        json={'html': '<html>...</html>'},
        timeout=10
    )
    response.raise_for_status()
    
    result = response.json()
    # 处理成功响应
    
except requests.exceptions.Timeout:
    print("请求超时")
except requests.exceptions.HTTPError as e:
    print(f"HTTP错误: {e.response.status_code}")
    print(f"错误详情: {e.response.json()}")
except Exception as e:
    print(f"通用错误: {e}")
```

### 2. 重试机制

```python
import time
import requests

def api_call_with_retry(url, payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', 60))
                print(f"限制中,等待 {wait_time} 秒...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # 指数退避
            print(f"尝试 {attempt + 1} 失败,{wait_time} 秒后重试...")
            time.sleep(wait_time)
```

### 3. 批量处理

```python
import requests
from concurrent.futures import ThreadPoolExecutor

def batch_parse_html(html_list):
    """并发解析多个HTML文件"""
    url = 'http://localhost:3000/api/v1/parse/html'
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for html in html_list:
            future = executor.submit(
                requests.post,
                url,
                json={'html': html},
                timeout=10
            )
            futures.append(future)
        
        results = []
        for future in futures:
            response = future.result()
            if response.status_code == 200:
                results.append(response.json())
        
        return results
```

---

## 示例项目

### Node.js示例

```javascript
const axios = require('axios');

const api = axios.create({
  baseURL: 'http://localhost:3000',
  timeout: 10000
});

// 健康检查
api.get('/api/health')
  .then(res => console.log('API健康'))
  .catch(err => console.error('API离线'));

// 解析HTML
api.post('/api/v1/parse/html', {
  html: '<html><body><h1>Test</h1></body></html>'
})
  .then(res => {
    console.log(`Found ${res.data.element_count} elements`);
  })
  .catch(err => {
    console.error('解析失败:', err.response.data);
  });
```

### Go示例

```go
package main

import (
  "bytes"
  "encoding/json"
  "fmt"
  "io/ioutil"
  "net/http"
)

func parseHTML(html string) (map[string]interface{}, error) {
  payload := map[string]interface{}{
    "html": html,
  }
  
  data, _ := json.Marshal(payload)
  
  resp, err := http.Post(
    "http://localhost:3000/api/v1/parse/html",
    "application/json",
    bytes.NewBuffer(data),
  )
  if err != nil {
    return nil, err
  }
  defer resp.Body.Close()
  
  body, _ := ioutil.ReadAll(resp.Body)
  
  result := make(map[string]interface{})
  json.Unmarshal(body, &result)
  
  return result, nil
}

func main() {
  result, err := parseHTML("<html><body>Test</body></html>")
  if err != nil {
    fmt.Println("Error:", err)
  } else {
    fmt.Printf("Result: %+v\n", result)
  }
}
```

---

## 监控和日志

### 启用调试日志

```bash
export RUST_LOG=debug
cargo run
```

### 期望的日志输出

```
[2026-02-17T10:00:00Z INFO] 🚀 BrowerAI API Server - Phase 3
[2026-02-17T10:00:00Z INFO] 🌐 Listening on http://0.0.0.0:3000
[2026-02-17T10:00:01Z DEBUG] POST /api/v1/parse/html received
[2026-02-17T10:00:01Z DEBUG] HTML elements: 5
[2026-02-17T10:00:01Z DEBUG] Parse time: 12ms
```

---

## 变更日志

### v1.0.0 (2026-02-17)

- ✅ 生产发布
- ✅ 完整的5个API端点
- ✅ 认证和速率限制支持
- ✅ 完整的错误处理

### v0.2.0 (2026-01-27)

- ✅ Alpha版本发布
- ✅ 基础的REST API
- ✅ HTML/CSS解析

---

## 相关文档

- [快速开始](../guides/QUICK_START_CARD.md)
- [开发指南](../DEVELOPMENT_GUIDE.md)
- [故障排除](../guides/TROUBLESHOOTING.md)
- [部署指南](../guides/DEPLOYMENT_QUICKSTART.md)
