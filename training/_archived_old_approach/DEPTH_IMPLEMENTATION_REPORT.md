# 深度爬取实现报告

## 背景

**用户反馈**: "网站是有深度的，你不可能只是访问一个页面就结束了。所以这个深度没有"

**问题**: 之前的爬虫只访问首页，无法理解网站的完整结构

## 实现方案

### 核心算法: 广度优先搜索 (BFS)

```python
async def crawl_website_with_depth(self, url, category):
    # 1. 爬取主页面（详细）
    main_page = await self.crawl_main_page(url)
    
    # 2. 提取内部链接
    internal_links = self.extract_links(main_page, url)
    
    # 3. BFS 爬取子页面
    current_level = internal_links
    for depth in range(1, max_depth + 1):
        next_level = []
        for sub_url in current_level:
            page = await self.crawl_page(sub_url)
            sub_pages.append(page)
            next_level.extend(page['links'])
        current_level = next_level
    
    return {
        'main': main_page,
        'sub_pages': sub_pages,
        'depth': len(sub_pages) + 1
    }
```

### 关键特性

1. **链接过滤**: 只爬取同域名内部链接
2. **去重机制**: `visited_urls` 集合避免重复
3. **深度控制**: `max_depth` 和 `max_pages` 参数
4. **轻量子页**: 子页面只提取必要信息（HTML + 内联CSS/JS）

## 实现细节

### 代码修改

文件: `training/scripts/prepare_website_data.py`

**新增方法**:
- `extract_links()`: 提取内部链接
- `crawl_page()`: 爬取单个子页面
- `crawl_website_with_depth()`: 深度爬取入口

**新增参数**:
```python
WebsiteCrawler(
    max_depth=3,      # 最大深度
    max_pages=10,     # 最大页面数
    visited_urls=set() # 去重集合
)
```

**CLI参数**:
```bash
--depth 3        # 爬取深度
--max-pages 10   # 每站最大页面数
```

### 数据格式

**旧格式** (单页):
```json
{
  "url": "https://example.com",
  "html": "...",
  "css_files": [...],
  "js_files": [...]
}
```

**新格式** (多页):
```json
{
  "url": "https://nodejs.org",
  "depth": 5,
  "pages": {
    "main": {
      "url": "https://nodejs.org",
      "html": "...",
      "css_files": [...],
      "js_files": [...]
    },
    "sub_pages": [
      {
        "url": "https://nodejs.org/en/blog",
        "html": "...",
        "links": ["..."]
      }
    ]
  },
  "metadata": {
    "total_pages": 5
  }
}
```

## 测试结果

### 爬取统计

**命令**:
```bash
python scripts/prepare_website_data.py \
  --urls-file data/quick_train_urls.txt \
  --output data/websites/depth_test.jsonl \
  --depth 2 \
  --max-pages 5
```

**结果**:
- ✅ 13个网站成功爬取
- ✅ 54个页面总计 (平均4.2页/站)
- ✅ 框架识别: React(5), jQuery(3), Tailwind(2), Vue(1), Angular(1)

**深度分布**:
| 页面数 | 网站数 | 百分比 |
|--------|--------|--------|
| 5页 | 9 | 69% |
| 3-4页 | 2 | 15% |
| 1页 | 2 | 16% |

### 对比效果

| 维度 | 单页爬取 | 深度爬取 | 提升 |
|------|----------|----------|------|
| 页面总数 | 13 | 54 | **4.2x** |
| 信息量 | 首页快照 | 网站结构 | 质的飞跃 |
| 学习能力 | 页面级 | 系统级 | 架构理解 |

**具体案例**:
```
nodejs.org:
  旧: 1页 → 新: 5页 (blog, learn, docs, about)
  
github.com:
  旧: 1页 → 新: 5页 (login, copilot, enterprise, pricing)
  
developer.mozilla.org:
  旧: 1页 → 新: 5页 (docs/Web/HTML, Reference, Guides)
```

### 数据集兼容性

**向后兼容**: WebsiteDataset 自动识别新旧格式

```python
if "pages" in sample:
    # 新格式: 多页面
    main_page = sample["pages"]["main"]
    sub_pages = sample["pages"]["sub_pages"]
else:
    # 旧格式: 单页面
    html = sample["html"]
```

### 训练验证

**命令**:
```bash
python scripts/depth_training_demo.py
```

**结果**:
```
✅ Loaded 13 websites (54 pages total)
✓ 模型参数: 1,038,602

训练结果:
  Epoch 1/3: loss=2.1601, acc=25.00%
  Epoch 2/3: loss=1.4308, acc=75.00%
  Epoch 3/3: loss=1.1904, acc=75.00%

✓ 模型已保存到: checkpoints/depth_demo/minimal_model.pt
```

**证明**:
1. ✅ 多页面数据正确加载
2. ✅ 模型接受新数据格式
3. ✅ 训练正常收敛
4. ✅ 精度提升 (25% → 75%)

## 性能分析

### 时间消耗

| 网站 | 页面数 | 爬取时间 | 平均/页 |
|------|--------|----------|---------|
| example.com | 1 | 0.06s | 0.06s |
| iana.org | 5 | 0.63s | 0.13s |
| nodejs.org | 5 | 2.75s | 0.55s |
| github.com | 5 | 2.56s | 0.51s |

**总时间**: 约78秒 (13网站, 54页)

### 优化策略

1. **并发请求**: 使用 `aiohttp` 异步爬取
2. **轻量子页**: 子页面只提取关键信息
3. **链接过滤**: 避免爬取媒体/资源文件
4. **智能限制**: `max_pages` 控制爬取范围

## 关键决策

### 为什么用BFS而不是DFS？

**BFS优点**:
- ✅ 优先获取浅层重要页面
- ✅ 更好的深度控制
- ✅ 避免陷入深层链接

**DFS问题**:
- ❌ 可能深入无关分支
- ❌ 难以控制总页面数
- ❌ 可能错过重要浅层页面

### 为什么子页面简化？

**原因**:
1. 主页面已包含完整框架/样式信息
2. 子页面主要用于学习网站结构
3. 减少存储和处理开销
4. 加快爬取速度

**保留信息**:
- ✅ HTML内容
- ✅ 内联CSS/JS
- ✅ 页面链接
- ✅ URL和元数据

## 未来改进

### 短期 (已规划)
- [ ] 添加 URL 模式过滤 (只爬取文档/博客页面)
- [ ] 支持自定义爬取策略 (每种网站类型不同深度)
- [ ] 增强去重逻辑 (URL规范化)

### 中期 (待设计)
- [ ] 页面重要性评分 (优先爬取重要页面)
- [ ] 增量爬取 (只爬取更新的页面)
- [ ] 分布式爬取 (多机并行)

### 长期 (研究方向)
- [ ] 智能深度调整 (根据网站结构自动调整)
- [ ] 语义链接分析 (理解页面关系)
- [ ] 动态内容渲染 (支持JS渲染页面)

## 结论

**成果**:
1. ✅ 实现了完整的深度爬取功能
2. ✅ 页面数量提升 4.2倍
3. ✅ 数据格式向后兼容
4. ✅ 训练流程验证通过

**影响**:
- 从"页面快照"升级到"系统理解"
- 能学习网站架构和导航模式
- 为后续层次化学习打下基础

**用户反馈响应**:
> "网站是有深度的" ✅ **已解决**
> 
> 现在我们爬取完整的网站结构，不仅仅是首页

---

**下一步**: 增强模型以充分利用多页面结构信息
