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
- 前端 `features/datasets/DatasetCreatePage` + `features/datasets/DatasetAnnotatePage` + `AnnotateCanvas.tsx`，路由注册 `app/routes.tsx:pageTitles` 与侧边栏，`X-API-Key` 经 `lib/createApi.ts:camelizeKeys`，新增 `ApiClient` 方法并在 `mockApi.ts` 补 `VITE_USE_MOCKS` 分支
- 后端 `POST /api/datasets` 复用现有 `DatasetCreate`（`sourcePath=null` 时走空数据集分支，跳过 `commit` 前 `BackgroundTasks` 的 `source_path` 检查，`Settings.dataset_dir` 解析为 `/data/datasets/extracted/{id}`，`_locate_yaml`/`_parse_data_yaml` 复用 `dataset_validation.py`，无 `data.yaml` 时在 `snapshot` 前合成 `train/val` 路径）
- 上传 `POST /api/datasets/{id}/images`（`multipart`，`_MAX_DATASET_BYTES 2GB`，`worker_session`）批量写 `extracted/{id}/images/train`，自动生成空 `labels`
- 标注 `PUT /api/datasets/{id}/annotations/{imageId}` 写 `labels/{split}/{name}.txt`（YOLO 归一化 `class x y w h` 或 `class x1 y1 x2 y2 ...`，`len(parts)>=5` 匹配 `dataset_validation.py:231`）
- 类别 `CRUD`：空数据集阶段 `LabelSchema` 未被 `DatasetSnapshot` 引用，允许 `LabelSchemaClass` 增删改；首个 `snapshot` 后 `label_schema.py:128` 密封，后续“重命名”走新建 `LabelSchema` 版本（`parent_schema_id`）而非原地 `UPDATE`
- 批量 `POST /api/datasets/{id}/batch` 事务删图/改类/复制，同步更新 `manifest` 前不生成快照
- 快照 `POST /api/datasets/{id}/snapshot` 复用 `storage/snapshots.py:create_dataset_snapshot(source_dir=extracted/{id}, store_root=..., snapshot_root=...)`，`data.yaml` 在快照前合成 `nc/names + train/val`，`quality_json` 写入 `warnings`

## 组件与交互
- **DatasetCreatePage**：`名称 + 描述 + 初始类别列表`，提交后跳标注页
- **AnnotateCanvas**：`bbox` 拖拽、`polygon` 点选闭合、滚轮缩放、`1-9` 切换类别、`Del` 删除
- **ClassPanel**：类别列表（颜色+计数），新增/重命名/删除（同步删除该类所有标签）
- **ImageGrid**：缩略图网格，多选（`Shift` 连选），底部批量条
- **UploadDrop**：拖拽上传，进度条，自动按 `train` 归档

## 数据流与异常
- 上传：`multipart` 校验 `2GB`，落盘 `extracted/{id}/images/train`，`validate_yolo_dataset` 仅在快照前执行；`nginx 600s` 与 `_MAX_DATASET_BYTES` 复用 `datasets.py:20`
- 标注：前端 `1-9`/`Del` 快捷键需焦点陷阱，校验 `class_id < nc`、`0~1` 归一化，失败 `400` + `toast`，`worker_session` 提交顺序 `commit` 后再 `BackgroundTasks`
- 类别：空阶段可原地改，首快照后走版本化；前端 `isParsing` 需区分 `imageCount 0 && !snapshot` 的空数据集与 `解析中`（新增 `status: empty` 避免 `DatasetsPage.tsx:158` 误判橙标）
- 批量：`Shift` 多选 + 虚拟化（>1k 图），事务删 `images` + `labels`，快照前不更新 `manifest_hash`，`orphan_label` 警告复用 `dataset_validation.py:388`
- 快照：合成 `data.yaml` 后 `create_dataset_snapshot`，`manifest.json` 含 `source_dir` 相对路径，训练/评估经 `yolo_dataset.py` 回退读原图，`store` 不预拷贝以避 `9p` 慢路径

## 原型
- `tmp/dataset_annotate_prototype.html`（三栏：类别/画布/图片网格 + 上传/批量）

## 测试
- 单元：`validate_yolo_dataset` 多路径 `yaml`（`crack-seg.yaml` 缺 `nc` 字典推断）、`bbox` 越界、`polygon` 点数、`class_id` 重映射后不越界、`_locate_yaml` 一层子目录
- 集成：`POST /api/datasets` 空 → `POST images` → `PUT annotations` → `POST snapshot` → `GET` 统计一致（`conftest.py` DB 覆盖，`worker_session` 验证），`python -m pytest -c NUL -p no:cacheprovider` + `verify_bom.py`（`nginx.conf` 去 BOM）
- 前端：`DatasetsPage` 空/`解析中` 区分、`AnnotateCanvas` 框/多边形、`ClassPanel` 重命名同步标签、`UploadDrop` 进度，`mockApi.ts` 补分支以过 `playwright --mode mock`

## 发布与兼容
- 复用 `BackgroundTasks` + `worker_session`，空数据集阶段无 `alembic` 迁移；`zip` 导入链路保持兼容，`operation_log` 新增 `dataset.image.upload` / `annotation.save` / `snapshot.create`
- 路由 `app/routes.tsx` 新增 `/datasets/create` 与 `/datasets/:id/annotate` 及 `pageTitles`，`hotpatch` 走 `docker cp`（`AGENTS.md:86`），回滚仅删空数据集
- 空数据集前端用 `status: empty` 区分 `解析中`，`mockApi` 与 `store` 行为对齐现有 `snapshots.py` 不预拷贝策略
