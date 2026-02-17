# config/ 目录指南

本目录包含BrowerAI项目的所有配置文件，按用途分类。

## 📋 文件说明

### Docker编排配置

| 文件 | 用途 | 场景 |
|------|------|------|
| `docker-compose.api.yml` | **API服务单独部署** | 微服务部署、仅需API服务 |
| `docker-compose.monitoring.yml` | **监控栈(Prometheus+Grafana)** | 性能监控、告警配置 |

**注**: 主要的开发用docker-compose.yml位于项目根目录

### 监控和告警配置

| 文件 | 用途 |
|------|------|
| `prometheus.yml` | Prometheus数据收集配置 |
| `alertmanager.yml` | 告警规则路由配置 |
| `alert_rules.yml` | 具体告警规则定义 |

---

## 🚀 快速使用

### 开发环境启动(推荐)

```bash
# 从项目根目录执行
cd ..  # 返回项目根目录
docker-compose up -d

# 包括API、Redis、PostgreSQL、监控栈
```

### 仅启动API服务

```bash
docker compose -f config/docker-compose.api.yml up -d
```

### 仅启动监控栈

```bash
docker compose -f config/docker-compose.monitoring.yml up -d
```

### 组合启动(API+监控)

```bash
docker compose \
  -f docker-compose.yml \
  -f config/docker-compose.monitoring.yml \
  up -d
```

---

## 📊 文件归属关系

```
config/
├── Docker编排
│   ├── docker-compose.api.yml         # API单独部署
│   └── docker-compose.monitoring.yml  # 监控栈
│
├── 监控配置
│   ├── prometheus.yml                 # Prometheus
│   ├── alertmanager.yml               # 告警管理
│   └── alert_rules.yml                # 告警规则
│
└── 指南(本文件)
    └── README.md
```

---

## 🔍 配置优先级

1. **docker-compose.yml** (根目录)
   - 优先级: 高
   - 用途: 开发环境完整栈
   - 包含: API + Redis + PostgreSQL + 监控

2. **config/docker-compose.*.yml** (此目录)
   - 优先级: 低(专用/可选)
   - 用途: 精细化部署控制
   - 场景: 微服务拓扑、独立告警监控

---

## ✅ 最佳实践

- ✓ 开发时使用根目录的 `docker-compose.yml`
- ✓ 生产部署时参考 `docker-compose.*.yml` 差异性
- ✓ 根据需要组合使用 `-f` 标志
- ✓ 监控告警配置应与部署配置分离维护

---

**最后更新**: 2026-02-17  
**版本**: 1.0
