import React, { useState, useEffect } from 'react';
import { apiClient } from './api/client';
import CodeEditor from './components/CodeEditor';
import './App.css';

interface Stats {
  htmlNodes?: number;
  cssRules?: number;
  renderTime?: number;
  error?: string;
}

function App() {
  const [htmlCode, setHtmlCode] = useState(`<!DOCTYPE html>
<html>
<head>
  <title>BrowerAI Demo</title>
</head>
<body>
  <h1>欢迎使用 BrowerAI</h1>
  <p>这是一个AI驱动的浏览器渲染引擎</p>
</body>
</html>`);

  const [cssCode, setCssCode] = useState(`body {
  font-family: Arial, sans-serif;
  margin: 20px;
  background-color: #f5f5f5;
}

h1 {
  color: #333;
  border-bottom: 2px solid #007bff;
  padding-bottom: 10px;
}

p {
  color: #666;
  line-height: 1.6;
}`);

  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<Stats>({});
  const [health, setHealth] = useState<{ status: string; version: string } | null>(null);
  const [activeTab, setActiveTab] = useState<'html' | 'css'>('html');

  // 初始化：检查API健康状态
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const healthData = await apiClient.health();
      setHealth({
        status: healthData.status,
        version: healthData.version || '0.2.0'
      });
    } catch (error) {
      console.error('无法连接到API:', error);
      setHealth({ status: 'error', version: 'unknown' });
    }
  };

  const handleRender = async () => {
    setLoading(true);
    setStats({});
    
    try {
      // 同时解析HTML和CSS
      const [htmlResponse, cssResponse] = await Promise.all([
        apiClient.parseHtml({ html: htmlCode }),
        apiClient.parseCss({ css: cssCode })
      ]);

      // 执行渲染
      const renderResponse = await apiClient.render({
        html: htmlCode,
        css: cssCode,
        use_ai: false
      });

      setStats({
        htmlNodes: htmlResponse.node_count,
        cssRules: cssResponse.rules_count,
        renderTime: renderResponse.duration_ms || 0
      });
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : '未知错误';
      setStats({ error: `渲染失败: ${errorMsg}` });
      console.error('渲染错误:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleParseCss = async () => {
    setLoading(true);
    setStats({});
    
    try {
      const response = await apiClient.parseCss({ css: cssCode });
      setStats({
        cssRules: response.rules_count,
        renderTime: response.duration_ms
      });
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : '未知错误';
      setStats({ error: `CSS解析失败: ${errorMsg}` });
    } finally {
      setLoading(false);
    }
  };

  const handleParseHtml = async () => {
    setLoading(true);
    setStats({});
    
    try {
      const response = await apiClient.parseHtml({ html: htmlCode });
      setStats({
        htmlNodes: response.node_count,
        renderTime: response.duration_ms
      });
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : '未知错误';
      setStats({ error: `HTML解析失败: ${errorMsg}` });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1>🚀 BrowerAI Web 演示</h1>
          <p>真实的AI驱动浏览器渲染引擎</p>
        </div>
        <div className="header-status">
          {health && (
            <>
              <span className={`status-badge ${health.status}`}>
                {health.status === 'ok' ? '✅ 就绪' : '❌ 离线'}
              </span>
              <span className="version">v{health.version}</span>
            </>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="editor-section">
          {/* Tabs */}
          <div className="tabs">
            <button
              className={`tab ${activeTab === 'html' ? 'active' : ''}`}
              onClick={() => setActiveTab('html')}
            >
              HTML 代码
            </button>
            <button
              className={`tab ${activeTab === 'css' ? 'active' : ''}`}
              onClick={() => setActiveTab('css')}
            >
              CSS 样式
            </button>
          </div>

          {/* Editors */}
          {activeTab === 'html' ? (
            <CodeEditor
              title="HTML 输入"
              language="html"
              value={htmlCode}
              onChange={setHtmlCode}
              onSubmit={handleParseHtml}
              loading={loading}
            />
          ) : (
            <CodeEditor
              title="CSS 输入"
              language="css"
              value={cssCode}
              onChange={setCssCode}
              onSubmit={handleParseCss}
              loading={loading}
            />
          )}
        </div>

        <div className="actions-section">
          <button
            className="action-btn primary"
            onClick={handleRender}
            disabled={loading || !htmlCode.trim()}
          >
            {loading ? '⏳ 处理中...' : '🎨 完整渲染'}
          </button>
          <button
            className="action-btn secondary"
            onClick={activeTab === 'html' ? handleParseHtml : handleParseCss}
            disabled={loading}
          >
            {loading ? '⏳ 处理中...' : `📋 仅${activeTab === 'html' ? 'HTML' : 'CSS'}解析`}
          </button>
        </div>

        {/* Results */}
        <div className="results-section">
          <h2>📊 处理结果</h2>
          {stats.error ? (
            <div className="error-box">
              <p>❌ {stats.error}</p>
            </div>
          ) : Object.keys(stats).length > 0 ? (
            <div className="stats-grid">
              {stats.htmlNodes !== undefined && (
                <div className="stat-card">
                  <div className="stat-label">HTML 节点</div>
                  <div className="stat-value">{stats.htmlNodes}</div>
                </div>
              )}
              {stats.cssRules !== undefined && (
                <div className="stat-card">
                  <div className="stat-label">CSS 规则</div>
                  <div className="stat-value">{stats.cssRules}</div>
                </div>
              )}
              {stats.renderTime !== undefined && (
                <div className="stat-card">
                  <div className="stat-label">处理时间</div>
                  <div className="stat-value">{stats.renderTime}ms</div>
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state">
              <p>👇 输入HTML/CSS代码，然后点击"完整渲染"开始处理</p>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>BrowerAI v0.2.0 | 真实的AI浏览器引擎 | 数据已验证</p>
      </footer>
    </div>
  );
}

export default App;
