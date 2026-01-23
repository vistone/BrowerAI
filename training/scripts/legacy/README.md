# Legacy Scripts Archive

这个目录包含过时或已被新版本取代的训练和爬虫脚本。

## 已弃用的脚本

### 爬虫相关
- **batch_crawl_websites.py** - 已被 `crawlers/scaleable_website_crawler.py` 取代
  - 功能：批量爬取网站数据
  - 弃用原因：可扩展性差，新版本支持分布式爬虫
  - 迁移路径：`training/crawlers/scaleable_website_crawler.py`

### 数据处理
- **extract_website_complete.py** - 遗留的网站提取脚本
  - 功能：完整提取网站内容
  - 弃用原因：功能已合并到 `crawlers/` 中的现代爬虫
  - 替代方案：使用 `crawlers/` 目录下的爬虫

- **prepare_website_data.py** - 数据预处理脚本
  - 功能：准备网站数据以供训练
  - 弃用原因：已迁移到 `training/data_tools/prepare_data.py`
  - 迁移路径：`training/data_tools/prepare_data.py`

### 实验性模型训练
- **train_neuroxide_model.py** - 神经元模型实验
  - 功能：训练基于神经元的网站分析模型
  - 状态：实验性，已被现代的生产训练器取代
  - 推荐替代：`training/trainers/production_trainer.py`

- **train_paired_website_generator.py** - 配对网站生成器实验
  - 功能：训练网站对生成模型
  - 状态：实验性，功能已集成到更新的生成器中
  - 推荐替代：`training/generators/scaleable_data_generator.py`

## 迁移建议

如果需要使用这些脚本中的功能，请：
1. 查看对应的 `training/trainers/`、`training/crawlers/` 或 `training/generators/` 中的现代实现
2. 如果需要特定的历史实现细节，可以从版本控制系统中恢复
3. 建议优先使用 `training/` 中的对应功能模块

## 清理计划

这些脚本可以在以下情况下删除：
- 确认所有功能已在新模块中实现
- 完成旧代码的特征提取和文档化
- 通过生产测试验证新实现的稳定性
