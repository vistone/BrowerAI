# 依赖安全深度清理评估与路线（paste/boa/v8/tokenizers）

## 背景
当前 `cargo audit` 已消除所有漏洞，剩余 1 条“未维护”警告：`paste 1.0.15`。该依赖通过多个链路进入：
- `boa_*` 系列（JS 解析/执行栈）
- `v8`（深度集成栈）
- `tokenizers`（部分 AI 功能链路）

因此，要彻底移除警告需避免上述栈引入 `paste`，涉及较大架构调整。

## 风险评估
- 影响范围：`browerai-js-parser`, `browerai-js-analyzer`, `browerai-dom`, `browerai-html-parser`, `browerai-js-v8`, 以及依赖上述模块的其他 crate。
- 变更风险：高（API 兼容性、行为一致性、性能基准、测试覆盖）。
- 工期预估：3–6 周（取决于替换方案与验证深度）。

## 替换路线选项

### 路线 A：移除 v8 与 boa 执行栈（最低风险版本）
- 解析保留：继续使用 `boa_parser` 或迁移至 `swc`（解析 AST）。
- 执行替换：避免 `boa_engine` 与 `v8`，改为静态分析 + 重写运行时桥接（例如 QuickJS 或 WASM sandbox）。
- 适配影响：
  - Pros：执行层迭代可控，避免 `paste` 来源中的 v8 部分；
  - Cons：`boa_*` 仍可能保留 `paste` 依赖（需验证）。

### 路线 B：全面迁移 JS 栈到 SWC + QuickJS（高风险，高投入）
- 解析：`swc` 全量替代 `boa_*`（已部分在 `browerai-js-analyzer` 使用）。
- 执行：使用 QuickJS/deno_core 或 wasm32 sandbox，自研 API 桥接。
- 适配影响：
  - Pros：有望完全移除 `paste`；
  - Cons：需重构多个 crate，投入大，行为差异需大量回归测试。

### 路线 C：保留栈，审计忽略“未维护”警告（安全合规但非零警告）
- 在 `deny.toml` 中配置允许 `paste` 的 unmaintained 警告（不忽略漏洞）。
- Pros：零功能风险，满足当前安全无漏洞目标；
- Cons：不能实现“零警告”。

## 建议排期（按功能模块分阶段）

- 第 1 周：
  - 详尽依赖映射（已完成 paste 反向树）。
  - 原型验证：移除 `tokenizers` 相关路径（AI-core 已可选，保持禁用）。
- 第 2–3 周：
  - v8 退出方案 POC（替换执行层为 QuickJS/WASM sandbox）。
  - `browerai-js-v8` 标注为实验性特性，默认关闭构建，拆分审计范围。
- 第 4–5 周：
  - 解析栈迁移 POC：`boa_parser` → `swc`（在 `js-analyzer` 先行，扩大覆盖）。
  - API 差异适配层设计与实现。
- 第 6 周：
  - 全量回归测试与性能基准对比。
  - 再次审计，确保零漏洞、尝试降低警告。

## 最小落地变更（建议立即执行）
- 将 `browerai-js-v8` 相关功能标注 feature，可默认关闭；在 CI 审计时仅对核心子集（不含 v8）执行 `cargo audit`。
- AI-Core 保持禁用 `candle`/`tokenizers` 特性（已完成升级与可选化）。

## 验收标准
- 阶段性目标：
  - 阶段 1：仍为零漏洞，减少 `paste` 进入的依赖路径。
  - 阶段 2：可在不包含 v8 的配置下实现零警告（若 `boa_*` 仍引入 `paste`，则进入阶段 3）。
  - 阶段 3：完成解析/执行栈替换并通过全部测试。

## 回滚预案
- 若出现不可接受的兼容性问题，回退到当前稳定版本（零漏洞，1 条警告），并保留审计忽略策略。

## 结论
- 彻底消除 `paste` 警告需对 JS 解析/执行栈进行较大重构。
- 推荐先行按模块分阶段推进，保持“零漏洞”与功能稳定；若对“零警告”有硬性要求，需投入路线 B 并安排 3–6 周改造周期。