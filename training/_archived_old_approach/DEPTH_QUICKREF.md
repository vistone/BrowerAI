# 深度爬取 - 快速参考

## 一键命令

### 基础深度爬取
```bash
cd training
python scripts/prepare_website_data.py \
  --urls-file data/quick_train_urls.txt \
  --output data/websites/my_depth_data.jsonl \
  --depth 2 \
  --max-pages 5
```

### 大规模爬取
```bash
python scripts/prepare_website_data.py \
  --urls-file data/large_urls.txt \
  --output data/websites/large_depth.jsonl \
  --depth 3 \
  --max-pages 10
```

### 检查数据
```bash
# 查看统计
cat data/websites/depth_test.jsonl | jq -r '.url + ": " + (.depth|tostring) + "页"'

# 查看框架分布
cat data/websites/depth_test.jsonl | jq -r '.metadata.framework' | sort | uniq -c

# 查看第一个网站结构
head -1 data/websites/depth_test.jsonl | jq '.'
```

### 使用深度数据训练
```bash
python scripts/depth_training_demo.py
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--depth` | 3 | 爬取深度（层数） |
| `--max-pages` | 10 | 每个网站最大页面数 |
| `--urls-file` | - | URL列表文件 |
| `--output` | - | 输出JSONL文件路径 |

## 数据格式

### 输入 (URLs文件)
```
https://example.com,documentation
https://nodejs.org,documentation
https://github.com,tools
```

### 输出 (JSONL)
```json
{
  "url": "https://nodejs.org",
  "category": "documentation",
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
        "links": [...]
      }
    ]
  },
  "metadata": {
    "framework": "React",
    "total_pages": 5
  }
}
```

## 实际效果

### 对比
- **单页**: 13网站 = 13页
- **深度**: 13网站 = 54页 (**4.2x**)

### 示例网站
```
nodejs.org:     1页 → 5页 (blog, learn, docs, about)
github.com:     1页 → 5页 (login, copilot, enterprise)
vuejs.org:      1页 → 5页 (guide, tutorial, api)
angular.io:     1页 → 5页 (docs, tutorials, resources)
```

## 常见问题

### Q: 如何控制爬取速度？
A: 在 `WebsiteCrawler` 中设置 `rate_limit`

### Q: 如何只爬取特定类型页面？
A: 修改 `extract_links()` 添加URL过滤规则

### Q: 子页面占用太多存储怎么办？
A: 减少 `max_pages` 或只存储子页面元数据

### Q: 如何处理需要登录的页面？
A: 未来可添加cookie/session支持

## 性能提示

✅ **推荐配置**:
- 文档站点: `--depth 3 --max-pages 10`
- 新闻站点: `--depth 2 --max-pages 5`
- 电商站点: `--depth 2 --max-pages 8`

⚠️ **注意事项**:
- 大型网站建议设置较小的 `max_pages`
- 避免爬取媒体/下载链接
- 尊重 robots.txt

## 下一步

查看完整文档: [DEPTH_CRAWLING_SUMMARY.md](DEPTH_CRAWLING_SUMMARY.md)
