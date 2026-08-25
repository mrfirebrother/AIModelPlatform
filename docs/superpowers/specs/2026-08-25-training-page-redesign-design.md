# 训练任务页面重构设计 — 2026-08-25

## 1. 背景与目标

- 现状 `frontend/src/features/training/TrainingPage.tsx:84` 左右 `1fr 1.2fr` 双栏，1 条记录时底部大面积空白，信息稀。
- 用户目标：**信息更密**，核心 4 列一眼可见，左右布局改为更紧凑。
- 约束：内管后台，保持现有风格；`concurrency=1` 串行，需突出进度；不动后端 `training_service / train_worker`。

## 2. 总体布局（B 方案：全宽表格 + 抽屉）

- 容器 `1200px` 居中，页头 `平台状态 / 内部管理台 / 训练任务` 不变。
- 工具栏全宽：左侧 `状态筛选 [全部/运行中/排队中/已完成/失败/已取消]` + `搜索（模型/数据集）`（`debounce 300ms` 大小写不敏感，清空重置分页到 1），右侧 `+ 新建`。
- 工具栏下全宽表格，无左右分栏；分页 `20 条/页` 在表底，仅 `>20 条` 显示，筛选后 `total` 为过滤后数量，分页与筛选联动重置。
- 点行右侧滑出 `480px` 抽屉：`position:fixed right:0 top:0 bottom:0 z-index 30`，半透明 `overlay z-20` 遮表格不遮工具栏；`ESC` / 点遮罩 / 点 `×` / 再点同行 关闭；焦点锁抽屉内。

## 3. 表格字段（4 列 + 操作）

| 列 | 内容 | 渲染 |
|---|---|---|
| 任务 | 首行 `parentModelName \|\| id.slice(0,8)` 12px 600；次行 `datasetName \|\| snapshotId.slice(0,8)` 9px mono `text-muted` | 2 行压一格 |
| 配置 | `epochs 轮 · modelFamily · trainingConfigJson.device`（如 `1 轮 · YOLOv8 · cpu`）；`imgsz/batch` 暂不显示，后续预设迭代再加，缺省 `epochs 0` 显示 `—` | 单行 11px |
| 状态 | `badge` `running cyan/queued orange/completed green/failed red/cancelled gray`；`running` 行内嵌 `2px 进度条`，进度 `logs.currentEpoch/logs.totalEpochs`（列表不发 N+1，`running` 仅显示 `running` 文字，进度在抽屉；若需列表进度则抽屉已加载后缓存复用）；`completed` 行次行小字 `mAP50 0.24` 取自 `checkpoint.latestCheckpoint.metrics['metrics/mAP50(B)']`，无则不显示；`failed` 不在列表 hover 取 `last_error`（列表仅 `getTrainingTasks`，`last_error` 仅抽屉）| 多态 |
| 时间 | 首行 `创建 MM-DD HH:mm`（跨天带日期）；次行 `running: 耗时 8m`（`now - task.createdAt` 分钟级，`>60m` 显示 `1h 8m`）；`completed: 耗时 33m`（`finished_at - started_at` 取自 `logs.attempts[0]`）；`queued` 仅创建时 | 2 行 10px |
| 操作 | `running: 取消` `其余: 删除` 9px，小按钮 hover 强调；`running` 删除按钮 `disabled`（后端 `DELETE 400 Cannot delete running task` 兜底 toast） | icon |

- 行高 `44px` `padding 8px 12px`，1 屏 12-15 行。
- 排序：默认 `running/queued 置顶` 优先，其次 `createdAt desc`；表头点击 `时间` 切换 `asc/desc`，点击 `状态` 按 `running>queued>failed>completed>cancelled` 循环；排序在前端置顶规则之上二次排序。
- 空状态：`tasks.length===0` → `暂无训练任务 去新建` 居中；`filter 后 0` → `无匹配  清空筛选`。

## 4. 抽屉详情

- 头部：`parentModelName · 状态 badge · epochs 轮` + 关闭 `×`。
- 块 1 概览：`状态 badge` `进度条`（`logs.currentEpoch/logs.totalEpochs` 百分比，`queued/completed` 为 0/100%） `当前 Epoch x/y` `尝试 n` `创建/耗时` `数据集/快照`。
- 块 2 指标：`mAP50 / mAP50-95 / Precision / Recall` 4 宫格取自 `checkpoint.latestCheckpoint.metrics`（`metrics/mAP50(B)` 等），`completed` 有值，`running/queued` 显示 `—`；`Loss 曲线` 迷你条：`metrics.epochs[]` 的 `loss` 按 `min/max 归一化` 转 `width%`（`width = (max - loss)/(max-min)*100`，`max===min` 则 50%），`bestLoss !== null` 在条上方 `最佳 0.12` 标注；`metrics === null` 显示骨架。
- 块 3 产物：`Checkpoint` 卡片 `epoch / artifactPath (9px mono 截断 hover 全路径) / artifactHash (前 8 位) / created_at` + `复制路径`（`navigator.clipboard.writeText` 失败回退 `execCommand` + toast）；无则 `暂无 Checkpoint`。
- 块 4 日志：`last_error` 红底 `monospace 10px` 超 `200 字符` 截断 `…展开`（阈值 200，点击展开/收起），`logPath` 灰字 9px。
- 数据：打开时 `Promise.all(getLogs,getMetrics,getCheckpoint)`，均经 `createApi.ts:79 camelizeKeys` 统一 `snake→camel`，`detail` 接口失败单块显示 `加载失败 重试` 不影响他块，骨架屏 `加载中...`。
- 关闭：遮罩/×/ESC/再点同行。

## 5. 交互与边界

- 新建：工具栏 `+ 新建` → `/training/create`（复用现有页，成功回表顶并 `setTasks` 前插）。
- 取消/删除：`cancel` 乐观置 `cancelled` 失败回滚；`delete` 需 `toast.confirm`，`running` 按钮 `disabled` 且后端 `DELETE 400 Cannot delete running task` 兜底 toast；成功后 `filter` 移除并 `selectedId===id` 则关抽屉。
- 筛选/搜索：前端过滤 `parentModelName/datasetName` 大小写不敏感 `includes`，`debounce 300ms`，清空重置分页到 1。
- 轮询：`tasks.some(s∈{running,queued})` 时每 `5s` `getTrainingTasks`（`setInterval` 清理于 `unmount/空`，失败指数退避 `5s→10s`，`visibility hidden` 暂停）；抽屉打开且 `logs.status∈{running,queued}` 时每 `5s` 刷三接口，竞态用 `abort flag` 丢弃过期响应。
- 失败/加载：`last_error >200` 截断；详情单块失败显示 `加载失败 重试` 按钮；全局 `getTrainingTasks` 失败 toast `加载失败` 不清列表。

## 6. 组件与数据流

- 改动：仅 `frontend/src/features/training/TrainingPage.tsx` 重构，允许同目录拆 `components/TaskDrawer.tsx`（若单文件 >250 行则拆，否则内联），保持 `Toolbar/TaskTable` 内联；复用 `lib/createApi.ts:79 camelizeKeys` `lib/api.ts:76 TrainingTask/105 Logs/120 Metrics/135 Checkpoint` `lib/toast.tsx:confirm/success/error` `app/routes.tsx:/training` `/training/create`。
- 组件：`TrainingPage` → `Toolbar` + `TaskTable` + `TaskDrawer`（内含 `Overview/Metrics/Checkpoint/Log` 四子块，纯函数）。
- 数据流：`getTrainingTasks() → tasks (camelize) → 前端筛选/置顶/排序 → 表`；`selectedId → Promise.all(getLogs/getMetrics/getCheckpoint) → 抽屉`；`metrics/checkpoint` 的 `snake_case` 经 `camelizeKeys` 统一。
- 常量：复用 `STATUS_LABEL/STATUS_BADGE` `taskEpochs/taskDatasetName` 辅助。
- 样式：复用 `card/card-head/badge/page-grid`，不引入图表库，Loss 纯 CSS。

## 7. 测试与验收

- 单测（`vitest` 或手工）：`空列表→暂无`、`筛选 0→无匹配`、`置顶排序`、`running 进度 27%`。
- E2E `playwright`（`VITE_USE_MOCKS=true` + `mockApi`）：`列表渲染 → 筛选无结果 → 点行抽屉 → 轮询刷新 → 复制路径 → 取消/删除 → 新建回显`；覆盖 `筛选零命中`、`详情单块失败重试`。
- 验收：`http://localhost:8080/training`（nginx 代理 `/api→api:8000`，与 `vite --mode mock :3000` 一致）1 条/15 条均无大空白；`running` 进度 5s 刷新；`failed` 详情可见 `last_error`；复制成功 toast。

## 8. 非目标

- 不改后端 `training_config_json` 字段（`imgsz/batch` 预设后续单独迭代）。
- 不引入新依赖/图表库，Loss 用纯 CSS 条。
