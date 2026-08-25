# 训练任务页面重构设计 — 2026-08-25

## 1. 背景与目标

- 现状 `frontend/src/features/training/TrainingPage.tsx:84` 左右 `1fr 1.2fr` 双栏，1 条记录时底部大面积空白，信息稀。
- 用户目标：**信息更密**，核心 4 列一眼可见，左右布局改为更紧凑。
- 约束：内管后台，保持现有风格；`concurrency=1` 串行，需突出进度；不动后端 `training_service / train_worker`。

## 2. 总体布局（B 方案：全宽表格 + 抽屉）

- 容器 `1200px` 居中，页头 `平台状态 / 内部管理台 / 训练任务` 不变。
- 工具栏全宽：左侧 `状态筛选 [全部/运行中/排队中/已完成/失败/已取消]` + `搜索（模型/数据集）`，右侧 `+ 新建`。
- 工具栏下全宽表格，无左右分栏；分页 `20 条/页` 在表底。
- 点行右侧滑出 `480px` 抽屉，不遮工具栏；抽屉内纵向平铺四块（概览/指标/产物/日志），不分页签。

## 3. 表格字段（4 列 + 操作）

| 列 | 内容 | 渲染 |
|---|---|---|
| 任务 | 首行 `parentModelName || id` 12px 600；次行 `datasetName || snapshotId` 9px mono `text-muted` | 2 行压一格 |
| 配置 | `epochs 轮 · YOLOv8 · cpu · imgsz 640` 11px | 单行 |
| 状态 | `badge` 颜色 `running cyan/queued orange/completed green/failed red/cancelled gray`；`running` 行内嵌 `2px 进度条 19/68 27%`；`completed` 显示 `mAP50`；`failed` 红点 hover 浮 `last_error` | 多态 |
| 时间 | 首行 `创建 10:42`；次行 `耗时 8m / 剩余 ~22m`（`now - started_at`） | 2 行 |
| 操作 | `running: 取消` `其余: 删除` 9px，小按钮 hover 强调 | icon |

- 行高 `44px` `padding 8px 12px`，1 屏 12-15 行。
- 排序：默认 `创建时间 desc` 且 `running/queued 置顶`；表头可按 `状态/时间` 排序。
- 空状态：`暂无训练任务 去新建` 居中。

## 4. 抽屉详情

- 头部：`parentModelName · 状态 badge · epochs 轮` + 关闭 `×`。
- 块 1 概览：`状态 badge` `进度条` `当前 Epoch x/y` `尝试 1` `创建/耗时` `数据集/快照`。
- 块 2 指标：`mAP50 / mAP50-95 / Precision / Recall` 4 宫格（`completed` 有值，`running` 显示 —）；`Loss 曲线` 迷你条 `metrics.epochs[].loss` 转宽度，`bestLoss` 标注。
- 块 3 产物：`Checkpoint` 卡片 `epoch / artifactPath / artifactHash / created_at` + `复制路径`；无则 `暂无 Checkpoint`。
- 块 4 日志：`last_error` 红底 `monospace 10px` 可展开，`logPath` 灰字。
- 数据：打开时 `Promise.all(api.getTrainingLogs, getTrainingMetrics, getTrainingCheckpoint)`，骨架屏。
- 关闭：遮罩/×/再点行。

## 5. 交互与边界

- 新建：工具栏 `+ 新建` → `/training/create`（复用现有页，成功回表顶）。
- 取消/删除：`cancel` 乐观置 `cancelled`；`delete` 需 `toast.confirm`，`running` 禁删（后端 400 则 toast）。
- 筛选/搜索：前端过滤 `parentModelName/datasetName`，清空回全部。
- 轮询：`running/queued` 存在时每 `5s` `getTrainingTasks`；抽屉打开时每 `5s` 刷详情。
- 失败：`last_error` 过长截断 `...展开`；加载失败 toast 重试。

## 6. 组件与数据流

- 改动：仅 `frontend/src/features/training/TrainingPage.tsx` 重构；复用 `lib/createApi.ts` `lib/api.ts:TrainingTask/Logs/Metrics/Checkpoint` `lib/toast.tsx` `app/routes.tsx`。
- 组件：`TrainingPage` → `Toolbar` + `TaskTable` + `TaskDrawer`（内含 `Overview/Metrics/Checkpoint/Log`）。
- 数据流：`getTrainingTasks() → tasks → 前端筛选/置顶 → 表`；`selectedId → Promise.all(详情三接口) → 抽屉`。
- 常量：复用 `STATUS_LABEL/STATUS_BADGE`。

## 7. 测试与验收

- 单测：空列表、失败红点、筛选、置顶排序。
- E2E `playwright`：`列表渲染 → 点行抽屉 → 取消/删除 → 新建回显`。
- 验收：`http://localhost:8080/training` 1 条/15 条均无大空白；`running` 进度实时；`failed` hover 可见错误。

## 8. 非目标

- 不改后端 `training_config_json` 字段（`imgsz/batch` 预设后续单独迭代）。
- 不引入新依赖/图表库，Loss 用纯 CSS 条。
