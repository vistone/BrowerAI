# 生产环境监控配置指南

**文档版本**: 1.0.0  
**创建日期**: 2026-02-01  
**目的**: 配置 Prometheus + Grafana 监控生产环境

---

## 目录

1. [监控架构](#监控架构)
2. [Prometheus 配置](#prometheus-配置)
3. [Grafana 配置](#grafana-配置)
4. [告警规则](#告警规则)
5. [监控仪表盘](#监控仪表盘)
6. [故障排查](#故障排查)

---

## 监控架构

```
┌─────────────────────────────────────────────────────────┐
│ 应用层 (Flask API)                                       │
│ - Prometheus metrics 导出 (/metrics 端点)              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ├─→ Prometheus (指标收集)
                 │   - 每 15 秒抓取一次
                 │   - 本地存储 (TSDB)
                 │   - 数据保留 15 天
                 │
                 ├─→ Grafana (可视化)
                 │   - 仪表盘展示
                 │   - 实时告警
                 │   - 历史趋势分析
                 │
                 └─→ AlertManager (告警管理)
                     - 告警去重
                     - 告警路由
                     - 通知 (邮件、Slack)
```

---

## Prometheus 配置

### Step 1: 安装 Prometheus

#### 方式 1: Docker (推荐)

```bash
# 创建 Prometheus 配置目录
mkdir -p /opt/prometheus
cd /opt/prometheus

# 创建配置文件
cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: 'production'
    cluster: 'k8s-prod'

# 告警规则文件
rule_files:
  - "alert_rules.yml"

# 数据保留
tsdb:
  retention:
    time: 15d
    size: 50GB

# Scrape 配置
scrape_configs:
  # Kubernetes API Server
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
      - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
        action: keep
        regex: default;kubernetes;https

  # BrowerAI API 服务
  - job_name: 'browerai-api'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 5s

  # Kubernetes Nodes
  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)

  # Kubernetes Pods
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: 'true'
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
EOF
```

#### 方式 2: Kubernetes (使用 Helm)

```bash
# 添加 Prometheus 仓库
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# 创建 values.yaml
cat > prometheus-values.yaml << 'EOF'
prometheus:
  prometheusSpec:
    retention: 15d
    storageSpec:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi

    # 服务监控
    serviceMonitorSelectorNilUsesHelmValues: false
    
    # Pod 监控
    podMonitorSelectorNilUsesHelmValues: false

grafana:
  enabled: true
  adminPassword: "admin-password"
  persistence:
    enabled: true
    size: 10Gi

alertmanager:
  enabled: true
  config:
    route:
      group_by: ['alertname', 'cluster']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 12h
      receiver: 'default'
    receivers:
      - name: 'default'
        # 可配置邮件、Slack 等通知
EOF

# 安装
helm install prometheus prometheus-community/kube-prometheus-stack \
  -f prometheus-values.yaml \
  -n monitoring \
  --create-namespace
```

### Step 2: 配置告警规则

```yaml
# alert_rules.yml
groups:
  - name: browerai_api
    interval: 30s
    rules:
      # 高错误率告警
      - alert: HighErrorRate
        expr: |
          (sum(rate(http_requests_total{status=~"5.."}[5m])) by (job))
          /
          (sum(rate(http_requests_total[5m])) by (job))
          > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "高错误率: {{ $labels.job }}"
          description: "{{ $labels.job }} 的错误率为 {{ $value | humanizePercentage }}"

      # 高延迟告警
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "高延迟: {{ $labels.job }}"
          description: "P99 延迟为 {{ $value | humanizeDuration }}"

      # Pod 重启次数过多
      - alert: PodRestartingTooOften
        expr: |
          rate(kube_pod_container_status_restarts_total[15m]) > 0.01
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Pod 重启过频: {{ $labels.pod }}"
          description: "{{ $labels.pod }} 重启频率为 {{ $value }}/分钟"

      # 内存使用过高
      - alert: HighMemoryUsage
        expr: |
          (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "内存使用过高: {{ $labels.pod }}"
          description: "内存使用率为 {{ $value | humanizePercentage }}"

      # CPU 使用过高
      - alert: HighCPUUsage
        expr: |
          (rate(container_cpu_usage_seconds_total[5m]) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "CPU 使用过高: {{ $labels.pod }}"
          description: "CPU 使用率为 {{ $value | humanize }}%"

      # Pod 未就绪
      - alert: PodNotReady
        expr: |
          kube_pod_status_ready{condition="false"} == 1
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Pod 未就绪: {{ $labels.pod }}"
          description: "{{ $labels.pod }} 在 {{ $labels.namespace }} 中已未就绪 10 分钟"
```

---

## Grafana 配置

### Step 1: 访问 Grafana

```
默认地址: http://localhost:3000
默认用户名: admin
默认密码: prom-operator (或在 helm values 中设置的密码)
```

### Step 2: 添加 Prometheus 数据源

```
1. 进入 Configuration → Data Sources
2. 点击 "Add data source"
3. 选择 Prometheus
4. 配置:
   - URL: http://prometheus:9090 (Kubernetes) 或 http://localhost:9090 (本地)
   - Scrape interval: 15s
5. 点击 "Test & Save"
```

### Step 3: 创建监控仪表盘

#### 仪表盘 1: 应用性能监控 (APM)

```json
{
  "dashboard": {
    "title": "BrowerAI API - Application Performance",
    "panels": [
      {
        "title": "请求速率 (RPS)",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (job)"
          }
        ],
        "type": "graph"
      },
      {
        "title": "错误率 (%)",
        "targets": [
          {
            "expr": "(sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))) * 100"
          }
        ],
        "type": "graph"
      },
      {
        "title": "响应时间 (P50, P95, P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ],
        "type": "graph"
      },
      {
        "title": "活跃请求数",
        "targets": [
          {
            "expr": "sum(rate(http_requests_in_progress[5m]))"
          }
        ],
        "type": "stat"
      }
    ]
  }
}
```

#### 仪表盘 2: 资源使用监控

```json
{
  "dashboard": {
    "title": "BrowerAI - Resource Usage",
    "panels": [
      {
        "title": "内存使用 (MB)",
        "targets": [
          {
            "expr": "sum(container_memory_usage_bytes{pod=~\"browerai-api.*\"}) / 1024 / 1024"
          }
        ],
        "type": "graph"
      },
      {
        "title": "CPU 使用 (核)",
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total{pod=~\"browerai-api.*\"}[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Pod 数量",
        "targets": [
          {
            "expr": "count(kube_pod_labels{label_app=\"browerai-api\"})"
          }
        ],
        "type": "stat"
      },
      {
        "title": "磁盘 I/O (读写)",
        "targets": [
          {
            "expr": "sum(rate(container_fs_reads_bytes_total[5m]))",
            "legendFormat": "读"
          },
          {
            "expr": "sum(rate(container_fs_writes_bytes_total[5m]))",
            "legendFormat": "写"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

#### 仪表盘 3: 可用性和健康

```json
{
  "dashboard": {
    "title": "BrowerAI - Availability & Health",
    "panels": [
      {
        "title": "Pod 就绪状态",
        "targets": [
          {
            "expr": "count(kube_pod_status_ready{condition=\"true\",pod=~\"browerai-api.*\"})"
          }
        ],
        "type": "stat"
      },
      {
        "title": "可用性 (%)",
        "targets": [
          {
            "expr": "(1 - (sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])))) * 100"
          }
        ],
        "type": "gauge"
      },
      {
        "title": "Pod 重启次数",
        "targets": [
          {
            "expr": "sum(kube_pod_container_status_restarts_total{pod=~\"browerai-api.*\"})"
          }
        ],
        "type": "stat"
      },
      {
        "title": "最后一次部署",
        "targets": [
          {
            "expr": "time() - max(kube_deployment_created{deployment=\"browerai-api-deployment\"})"
          }
        ],
        "type": "stat"
      }
    ]
  }
}
```

---

## 告警规则

### 关键告警配置

```yaml
# 1. 可用性告警 (SLA >= 99.9%)
- alert: ServiceUnavailable
  expr: availability < 0.999
  for: 5m
  severity: critical
  action: 立即通知 + 自动回滚

# 2. 性能告警 (延迟 < 1 秒)
- alert: HighLatency
  expr: p99_latency > 1000
  for: 10m
  severity: warning
  action: 扩容 + 通知

# 3. 资源告警 (内存 < 1GB 剩余)
- alert: LowMemory
  expr: available_memory < 1073741824
  for: 5m
  severity: warning
  action: 释放资源 + 通知

# 4. Pod 告警 (replicas < desired)
- alert: PodsNotReady
  expr: ready_replicas < desired_replicas
  for: 10m
  severity: critical
  action: 调查 + 自动回滚
```

---

## 监控仪表盘

### 必看仪表盘

1. **应用性能 (APM)**
   - 实时请求率
   - 错误率
   - 响应时间分布
   - 活跃连接数

2. **资源使用**
   - 内存占用
   - CPU 使用率
   - 磁盘 I/O
   - 网络带宽

3. **可用性**
   - Pod 就绪状态
   - 服务可用性 %
   - Pod 重启次数
   - 最后部署时间

4. **业务指标** (如适用)
   - 成功率
   - 特征编码速度
   - 代码生成延迟
   - 反馈处理时间

---

## 快速启动命令

### Prometheus

```bash
# Docker 启动
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v /opt/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
  -v /opt/prometheus/alert_rules.yml:/etc/prometheus/alert_rules.yml \
  prom/prometheus \
  --config.file=/etc/prometheus/prometheus.yml

# 验证
curl http://localhost:9090/api/v1/targets
```

### Grafana

```bash
# Docker 启动
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana

# 访问
# http://localhost:3000
# 用户: admin
# 密码: admin
```

### Kubernetes (Helm)

```bash
# 检查安装
kubectl get pods -n monitoring

# 端口转发
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
kubectl port-forward -n monitoring svc/grafana 3000:80

# 查看日志
kubectl logs -n monitoring -l app=prometheus -f
kubectl logs -n monitoring -l app=grafana -f
```

---

## 故障排查

### 问题 1: Prometheus 无法收集指标

```bash
# 检查 Prometheus 状态
curl http://localhost:9090/-/healthy

# 查看 targets 状态
curl http://localhost:9090/api/v1/targets | jq .

# 检查日志
docker logs prometheus

# 常见原因:
# - 应用未暴露 /metrics 端点
# - 网络不可达
# - 配置错误
```

### 问题 2: Grafana 无法连接 Prometheus

```bash
# 测试连接
curl http://prometheus:9090/api/v1/query?query=up

# Grafana 日志
docker logs grafana

# 检查数据源配置
# Settings → Data Sources → 编辑 Prometheus
```

### 问题 3: 告警未触发

```bash
# 检查告警规则
curl http://localhost:9090/api/v1/rules | jq .

# 检查告警状态
curl http://localhost:9090/api/v1/alerts | jq .

# 手动测试查询
curl 'http://localhost:9090/api/v1/query?query=up'
```

---

## 生产最佳实践

### 1. 高可用设置

```yaml
# Prometheus HA 架构
- 多个 Prometheus 实例
- 统一的告警规则
- RemoteWrite 到长期存储 (InfluxDB, S3)
```

### 2. 告警管理

```
告警级别:
  critical: 需要立即响应 (SLA < 99.9%)
  warning: 需要今天响应 (资源不足)
  info: 记录日志 (日常运维)

告警分类:
  - 可用性 (立即通知)
  - 性能 (通知 + 扩容)
  - 资源 (通知 + 清理)
  - 安全 (立即通知 + 隔离)
```

### 3. 监控数据保留

```
- 高精度 (15 秒): 1 周
- 标准精度 (1 分钟): 1 个月
- 低精度 (1 小时): 1 年
```

### 4. 通知配置

```
邮件告警:
  - 仅 critical 级别
  - 工作时间发送

Slack 告警:
  - warning 及以上
  - #browerai-alerts 频道

PagerDuty 集成:
  - critical 级别
  - 激活值班轮转
```

---

## 监控命令参考

```bash
# Prometheus 查询示例
curl 'http://localhost:9090/api/v1/query?query=up'
curl 'http://localhost:9090/api/v1/query_range?query=rate(http_requests_total[5m])&start=<start>&end=<end>&step=<step>'

# Grafana API
curl http://localhost:3000/api/datasources        # 列出数据源
curl http://localhost:3000/api/dashboards/search  # 列出仪表盘

# Kubernetes 监控
kubectl get servicemonitor -n monitoring
kubectl get prometheusrule -n monitoring
```

---

**版本**: 1.0.0  
**最后更新**: 2026-02-01  
**状态**: ✅ 生产就绪
