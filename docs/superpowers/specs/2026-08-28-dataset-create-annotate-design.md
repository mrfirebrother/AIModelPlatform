# 数据集新建与标注 - 设计文档

## 背景
平台现有数据集仅支持 `zip` 导入，`example2` 的 `crack-seg` 结构曾因校验过严导致统计为 0。用户需要平台内直接新建数据集、上传图片、进行检测/分割标注，并通过类别管理与批量操作快速建集。

## 目标
- 支持空数据集创建与本地上传建集
- 支持 `bbox` + `polygon` 双模式标注
- 提供类别管理与批量操作，提升建集效率
- 复用现有 `BackgroundTasks` 快照链路与 `nginx 600s` 超时

## 非目标
- 集成外部重型标注平台（Label Studio/CVAT）
- 团队协作/权限
- 自动标注（预留）

## 架构
- 前端 `features/datasets/DatasetCreatePage` + `DatasetAnnotatePage`
- 后端 `POST /api/datasets` 新增 `mode: "empty"` 分支：创建空 `Dataset + LabelSchema`
- 上传 `POST /api/datasets/{id}/images` 批量写 `extracted/{id}/images/{split}`
- 标注 `PUT /api/datasets/{id}/annotations/{imageId}` 写 `labels/{split}/{name}.txt`（`class x y w h` 或 `class x1 y1 ...`）
- 类别 `CRUD` 映射 `LabelSchemaClass`，重命名时批量替换标签文件 `class_id`
- 批量 `POST /api/datasets/{id}/batch` 事务处理删图/改类/复制
- 快照 `POST /api/datasets/{id}/snapshot` 复用 `create_dataset_snapshot`，前端轮询 `GET /api/datasets` 直至 `imageCount>0`

## 组件与交互
- **DatasetCreatePage**：`名称 + 描述 + 初始类别列表`，提交后跳标注页
- **AnnotateCanvas**：`bbox` 拖拽、`polygon` 点选闭合、滚轮缩放、`1-9` 切换类别、`Del` 删除
- **ClassPanel**：类别列表（颜色+计数），新增/重命名/删除（同步删除该类所有标签）
- **ImageGrid**：缩略图网格，多选（`Shift` 连选），底部批量条
- **UploadDrop**：拖拽上传，进度条，自动按 `train` 归档

## 数据流与异常
- 上传校验 `2GB`/`zip`，落盘 `extracted`，无 `data.yaml` 时自动生成 `nc/names`
- 标注校验 `class_id < nc`、归一化 `0~1`，失败 `400` + `toast`
- 类别删除：`UPDATE labels` 重映射 + `DELETE LabelSchemaClass`，事务回滚
- 批量删图：同步删 `images` + `labels`，更新计数
- 生成快照：复用现有快照逻辑，前端轮询检测完成

## 原型
- `tmp/dataset_annotate_prototype.html`（三栏：类别/画布/图片网格 + 上传/批量）

## 测试
- 单元：`validate_yolo_dataset` 覆盖多路径 `yaml`、`bbox` 越界
- 集成：空创建 → 上传 → 标注保存 → 快照生成 → `GET` 统计一致
- 前端：`DatasetsPage` 列表与 `AnnotateCanvas` 交互

## 发布与兼容
- 复用 `BackgroundTasks` + `worker_session`，无新增迁移
- 新增路由 `/datasets/create` 与 `/datasets/:id/annotate`
- 空数据集 `imageCount 0` 时列表 `解析中` 样式已兼容
