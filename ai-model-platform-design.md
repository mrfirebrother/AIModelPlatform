# AI 模型服务平台 — 设计文档

## 背景

需要建设独立的 AI 模型服务平台，接收图片或视频，返回识别结果。支持多种异常检测类型（火灾、倒塌、堵车、事故等），结果不满意时可自训练优化，模型有父子谱系和项目迭代发布管理。

## 系统定位

```
调用方（NVR、其他系统）
         │
         │  POST 图片/视频
         ▼
┌─────────────────────────────┐
│     AI 模型服务平台          │     ← 独立项目，独立部署
│     (GPU 服务器)             │
│                             │
│  推理服务 │ 训练服务 │ 模型仓库 │
└─────────────────────────────┘
         │
         │  返回识别结果 (JSON)
         ▼
调用方自行处理（告警、推送、工单等）
```

平台只做两件事：**推理** 和 **训练**。不负责告警推送，不负责业务流程。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     AI 模型服务平台                           │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  推理服务     │  │  训练服务     │  │  模型仓库            │ │
│  │             │  │             │  │                     │ │
│  │ REST API    │  │ 任务队列     │  │ 模型文件存储          │ │
│  │ 模型加载管理  │  │ 训练编排     │  │ 模型谱系与发布管理    │ │
│  │ 并发调度     │  │ 评估报告     │  │ 模型节点与关联发布    │ │
│  │             │  │             │  │ 状态管理              │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┴─────────────────────┘            │
│                          │                                  │
│                   ┌──────┴──────┐                           │
│                   │   数据管理    │                           │
│                   │ 数据集存储    │                           │
│                   │ 标注文件管理  │                           │
│                   │ 训练日志      │                           │
│                   └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## 子系统设计

### 1. 推理服务

对外暴露 REST API，接收图片或视频，返回识别结果。

**API 初稿（暂不定稿）：**

```
POST /api/infer
Content-Type: application/json

{
  "modelBindingId": "mb-fire-001",
  "input": {
    "type": "image",
    "data": "base64...",      // 图片 base64
    "url": "https://..."      // 或图片/视频 URL
  },
  "options": {
    "confidence": 0.5,        // 置信度阈值
    "maxResults": 10          // 最大结果数
  }
}

Response:
{
  "modelBindingId": "mb-fire-001",
  "modelBindingReleaseId": "rel-mb-fire-001-003",
  "modelNodeId": "m-fire-001",
  "runtimeGeneration": 7,
  "inferenceConfigHash": "sha256:...",
  "latencyMs": 120,
  "results": [
    {
      "type": "fire",
      "confidence": 0.92,
      "bbox": [100, 200, 300, 400],
      "label": "火焰",
      "detail": "检测到火焰，置信度92%"
    }
  ]
}
```

```
POST /api/infer/batch
Content-Type: application/json

{
  "modelBindingId": "mb-fire-001",
  "inputs": [
    { "type": "image", "data": "base64..." },
    { "type": "image", "url": "https://..." }
  ]
}

Response:
{
  "modelBindingId": "mb-fire-001",
  "modelBindingReleaseId": "rel-mb-fire-001-003",
  "modelNodeId": "m-fire-001",
  "runtimeGeneration": 7,
  "inferenceConfigHash": "sha256:...",
  "results": [
    { "index": 0, "results": [...] },
    { "index": 1, "results": [...] }
  ]
}
```

**内部机制：**
- 模型是否驻留由管理员指定，平台不自动卸载其他模型
- 请求排队，控制并发数防止 GPU 过载
- 每次请求记录日志：关联 ID、发布记录、模型节点、输入摘要、输出结果和耗时
- 支持 Ultralytics YOLO PT 模型推理

### 2. 训练服务

接收训练任务，执行训练，完成后将模型存入模型仓库。

**API 初稿（暂不定稿）：**

```
POST /api/train
Content-Type: application/json

{
  "datasetSnapshotId": "ds-fire-003",
  "parentModelNodeId": "m-fire-001", // 基于哪个模型节点训练
  "config": {
    "epochs": 100,
    "batchSize": 16,
    "learningRate": 0.001,
    "imageSize": 640
  }
}

Response:
{
  "taskId": "train-20260820-001",
  "status": "queued"
}
```

```
GET /api/train/{taskId}/status

Response:
{
  "taskId": "train-20260820-001",
  "status": "running",        // queued | running | completed | failed
  "progress": {
    "currentEpoch": 45,
    "totalEpochs": 100,
    "loss": 0.032,
    "metrics": {
      "precision": 0.91,
      "recall": 0.88,
      "mAP50": 0.89
    }
  },
  "startedAt": "2026-08-20T10:00:00Z",
  "estimatedRemaining": "01:23:45"
}
```

**训练流程：**
1. 接收训练请求 → 加入任务队列
2. 从数据管理模块加载数据集
3. 基于 parentModelNodeId 加载训练权重
4. 执行训练，定期上报进度
5. 训练完成 → 生成候选 modelNode → 执行自动评估
6. 人工验收通过后创建新的 modelBindingRelease

### 3. 模型仓库

管理所有模型文件、模型节点和父子谱系。

**存储结构：**

```
models/
├── m-root-yolo/
│   ├── model.pt
│   ├── metadata.json
│   └── checksum.sha256
├── m-concrete/
│   ├── model.pt
│   ├── metadata.json
│   └── checksum.sha256
└── m-bridge-a-concrete/
    ├── model.pt
    ├── metadata.json
    └── checksum.sha256
```

**metadata.json 示例：**

```json
{
  "modelNodeId": "m-bridge-a-concrete",
  "parentId": "m-concrete",
  "status": "candidate",
  "createdAt": "2026-08-15T10:00:00Z",
  "labelSchemaId": "schema-concrete-damage",
  "datasetSnapshotId": "ds-bridge-a-003",
  "framework": "pytorch",
  "format": "pt",
  "inputSize": [640, 640],
  "labelOrder": ["crack", "spalling"],
  "metrics": {
    "precision": 0.93,
    "recall": 0.90,
    "mAP50": 0.91
  },
  "trainingConfig": {
    "epochs": 100,
    "batchSize": 16,
    "learningRate": 0.001
  }
}
```

模型目录中的 `metadata.json` 是生成时的不可变文件快照。模型当前状态、运行时加载状态和操作历史以 PostgreSQL 记录为准，不回写模型文件目录中的 metadata。

**模型节点和关联发布状态：**

```
modelNode: candidate → approved / rejected → archived

modelBindingRelease: pending → preparing → active
                              ├→ failed
                              └→ superseded

回滚也创建新的递增发布记录，记录类型为 rollback，避免修改历史发布语义。
```

**API 初稿（暂不定稿）：**

```
模型和关联的具体 API 暂不定稿。核心对象为 modelNode、modelBinding 和
modelBindingRelease，不再使用 model/version 作为主要寻址方式。
```

### 4. 数据管理

管理训练数据集。

**存储结构：**

```
datasets/
├── dataset-fire/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/        # YOLO 格式标注文件
│   │   ├── val/
│   │   └── test/
│   ├── snapshots/
│   │   ├── ds-001/       # 不可变快照
│   │   └── ds-002/
│   └── dataset.json      # 逻辑数据集元信息
```

每次导入或追加数据后生成新的快照。训练任务只引用快照目录和清单，不直接引用可变的逻辑数据集目录。

**API 初稿（暂不定稿）：**

```
POST   /api/datasets                    # 创建数据集（上传）
GET    /api/datasets                    # 列出所有数据集
GET    /api/datasets/{name}             # 获取数据集详情
DELETE /api/datasets/{name}             # 删除数据集
POST   /api/datasets/{name}/images      # 追加图片
POST   /api/datasets/{name}/labels      # 追加标注
POST   /api/datasets/{name}/snapshots   # 校验并生成不可变快照
```

## 模型选型建议

| 检测类型 | 推荐模型 | 训练数据来源 | 难度 |
|---------|---------|------------|------|
| 车辆检测 | YOLOv8 | COCO 预训练直接用 | 低 |
| 火灾/烟雾 | YOLOv8 + 自训练 | 公开火灾数据集 + 自采 | 中 |
| 交通拥堵 | YOLOv8 + 车辆计数逻辑 | 公开交通数据集 | 中 |
| 倒塌检测 | YOLOv8 + 自训练 | 需要大量标注数据 | 高 |
| 事故检测 | YOLOv8 + 自训练 | 需要大量标注数据 | 难度高，数据稀缺 |

**推荐起步路径：**
1. 先用 YOLOv8 预训练模型做车辆检测，跑通端到端
2. 加入火灾/烟雾检测（有公开数据集）
3. 逐步扩展其他类型

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 推理框架 | PyTorch / Ultralytics | 第一阶段生产模型使用 PT，运行在 GPU 服务器 |
| 训练框架 | PyTorch / Ultralytics | 训练和继续微调使用 PT 权重 |
| Web 框架 | FastAPI (Python) | 异步支持好，AI 生态天然契合 |
| 管理后台 | React + TypeScript | 前后端分离，构建为静态文件 |
| 任务队列 | Celery + Redis | 训练任务异步执行 |
| 元数据存储 | PostgreSQL | 保存模型、数据集、任务、评估和发布状态 |
| 缓存与队列状态 | Redis | 任务队列、通知和进度缓存；不作为 GPU 资源权威来源 |
| 模型存储 | 本地文件系统 | 按 modelNodeId 管理不可变模型文件 |
| 数据集存储 | 本地文件系统 | 按目录结构管理 |
| 容器化 | Docker + NVIDIA Container Toolkit | GPU 透传 |

## 项目结构（初始草案，非最终实现目录）

```
ai-model-platform/
├── api/
│   ├── main.py                 # FastAPI 入口
│   ├── routers/
│   │   ├── infer.py            # 推理 API
│   │   ├── train.py            # 训练 API
│   │   ├── models.py           # 模型仓库 API
│   │   └── datasets.py         # 数据管理 API
│   └── schemas/
│       ├── infer.py            # 请求/响应模型
│       ├── train.py
│       ├── model.py
│       └── dataset.py
├── frontend/                     # React + TypeScript 管理后台
├── migrations/                   # PostgreSQL 数据库迁移
├── core/
│   ├── inference/
│   │   ├── engine.py           # Ultralytics YOLO PT 推理引擎
│   │   ├── preprocessor.py     # 图像预处理
│   │   ├── postprocessor.py    # 结果后处理（NMS 等）
│   │   └── model_manager.py    # 模型加载/卸载/缓存
│   ├── training/
│   │   ├── trainer.py          # PyTorch 训练逻辑
│   │   ├── evaluator.py        # 模型评估
│   │   └── validator.py        # 模型加载和推理校验
│   └── repository/
│       ├── model_repo.py       # 模型节点和谱系管理
│       └── dataset_repo.py     # 数据集管理
├── workers/
│   ├── train_worker.py          # Celery 训练任务
│   └── inference_worker.py      # PT 模型加载和推理
├── models/                     # 模型文件存储
├── datasets/                   # 数据集存储
├── config/
│   └── settings.py             # 配置
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

该目录树是早期模块草案，不作为最终实现路径。最终实现统一采用后文“最终技术和部署决策”中的 `backend/`、`frontend/` 和 `infra/` 目录结构。

## 部署方式

```yaml
# docker-compose.yml
version: "3.8"
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: ai_platform
      POSTGRES_USER: ai_platform
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - ./postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ai_platform -d ai_platform"]
      interval: 5s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  migrate:
    build: .
    command: alembic upgrade head
    depends_on:
      postgres:
        condition: service_healthy

  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./datasets:/app/datasets
    depends_on:
      migrate:
        condition: service_completed_successfully
      redis:
        condition: service_healthy

  inference:
    build: .
    command: python -m workers.inference_worker
    volumes:
      - ./models:/app/models
      - ./datasets:/app/datasets
    depends_on:
      migrate:
        condition: service_completed_successfully
      redis:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  worker:
    build: .
    command: celery -A workers.train_worker worker --concurrency=1
    volumes:
      - ./models:/app/models
      - ./datasets:/app/datasets
      - ./logs:/app/logs
    depends_on:
      migrate:
        condition: service_completed_successfully
      redis:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  frontend:
    build: ./frontend
    ports:
      - "8080:80"
    depends_on:
      - api
```

Compose 的 GPU 配置只负责透传设备，不负责训练和推理调度。实际资源准入由 PostgreSQL 事务租约、管理员驻留配置和显存检查共同决定；同一时间最多运行一个训练 attempt。Redis 只承担队列、通知和缓存。

API、推理 Worker 和训练 Worker 都提供应用级健康状态。数据库或 Redis 不可用时 API 进入 `unhealthy`；GPU 或 Worker 不可用时平台进入 `degraded`，管理后台仍可查看故障和资源状态。

## 与 NVR 的对接

以下内容保留为初步对接示意，具体协议暂不定稿。当前确认的模型选择字段是 `modelBindingId`。

NVR 的 AnalysisService（录像分析服务）调用模型平台的方式：

```
AnalysisService                    AI 模型服务平台
     │                                    │
     │  POST /api/infer                   │
      │  { modelBindingId: "mb-fire-001",   │
     │    input: { type: "image",         │
     │             data: "base64..." } }  │
     │  ────────────────────────────────→  │
     │                                    │
     │  { results: [{ type: "fire",       │
     │    confidence: 0.92 }] }           │
     │  ←────────────────────────────────  │
     │                                    │
     │  POST 结果到约定 API 地址           │
     │  写本地日志                         │
```

两个项目完全解耦，通过 HTTP API 通信。

## 开发分期

| 阶段 | 内容 | 周期 |
|------|------|------|
| 第一期 | 单机 GPU 平台基础设施、根模型、图片推理、数据快照、训练、评估、发布和回滚闭环 | 以验收场景为准 |
| 第二期 | 视频推理、标注工具对接、更多模型任务和其他部署格式 | 按需 |

## 参考资料

- 灵境云 2.0 道路智能巡查养护管理解决方案（云-边-端架构、边缘计算主机、模型管理）
- YOLOv8 / Ultralytics（目标检测模型）
- NVIDIA Triton Inference Server（可选的高级推理服务）
- FastAPI（Python Web 框架）

## 当前需求分析结论

本节记录目前头脑风暴阶段已经确认的需求，后续设计以本节为准。对外 HTTP 接口的具体形式暂不在本节展开。

### 平台职责

平台内部闭环包括：

- 模型节点和父子谱系管理
- 数据集导入、校验和快照管理
- 人工发起的训练任务
- GPU 资源调度和训练队列
- 自动指标评估和人工验收
- 模型发布、切换和回滚
- 模型运行、资源监控和操作日志

平台不负责告警推送、工单处理和业务流程编排。

训练和测试阶段只处理已经整理好的图片数据，不处理视频拆帧。视频训练暂不纳入范围。

### 模型节点和谱系

模型不使用 `v1`、`v2` 这类全局版本号作为核心身份。每次导入、训练或生成的模型都是一个不可变的模型节点，由系统生成唯一的 `modelNodeId`。

模型节点之间通过父子关系表达派生关系：

```text
root modelNode
├── child modelNode
└── child modelNode
```

基础模型可以有多个根节点，根节点的 `parentId` 为空。平台支持管理员导入多个 `.pt` 根模型，根模型可以没有项目关联，也可以作为后续训练的父模型。根模型通过文件加载和基础推理校验后直接进入 `approved`，不要求训练数据快照和人工验收。

模型节点至少记录：

- `modelNodeId`
- `parentId`
- 模型文件路径和校验和
- 模型架构、任务类型和框架
- 输入尺寸
- 标签体系，可为空
- 数据集快照，可为空
- 训练配置和评估指标
- 创建时间和状态

第一阶段生产模型格式使用 Ultralytics YOLO 的 `.pt` 文件。`ONNX` 不作为当前必需格式。`.pt` 用于生产推理，也用于后续继续训练；模型文件生成后不可覆盖。

### 标签体系和子模型规则

第一代子模型可以不继承 YOLO 根模型原有的类别体系，直接建立新的标签体系。例如从 YOLO 根模型分别训练火灾模型和混凝土损伤模型。

第一代任务模型建立标签体系后，后续子模型必须保留上一代已有类别，同时允许增加新类别：

```text
父模型：crack、spalling
子模型：crack、spalling、rebar-exposure
```

类别 ID 不能复用给其他含义。类别名称可以调整显示文本，但不能改变类别语义。删除类别或改变类别含义时，应从更早的节点重新创建分支。

标签体系数据库记录是类别定义的唯一权威来源。YOLO `data.yaml` 和模型文件中的类别顺序在导入或训练时转换为 `label_schema_classes`，模型 metadata 中的 `labelOrder` 只作为文件自描述信息，必须与数据库顺序一致。已被模型、数据集快照或评估引用的标签体系和类别记录不可修改，任何类别新增、删除、语义变更或显示名变更都创建新的标签体系。

类别 ID 在标签体系内唯一，不要求跨标签体系全局唯一；数据库约束为 `(label_schema_id, class_id)` 联合唯一。类别语义使用不可变的 `semantic_key` 表示，训练前必须同时校验父模型标签体系、目标标签体系和数据集快照标签体系；第一代任务模型从根模型重新建立标签体系，后续任务模型只能在父体系上追加类别。

训练数据快照的类别覆盖规则取决于标签体系关系：

- 第一代任务模型从根模型建立不兼容的新标签体系时，只要求目标标签体系的类别在 `train` 和 `val` 中有正样本。
- 后续追加类别的模型，`train` 和 `val` 必须包含所有继承类别的正样本。
- 新增类别必须有 `train` 和 `val` 正样本。
- `test` 缺少类别时在导入阶段给出警告；发布评估按评估策略判断是否通过。
- 负样本不计入类别正样本覆盖。

### 模型关联和项目迭代

调用方保存平台生成的稳定 `modelBindingId`，平台内部通过它解析当前使用的模型节点。多个关联可以指向同一个模型节点。

模型关联可以先为空，状态为 `unbound`，待模型准备好后再绑定第一条发布记录。关联 ID 一旦生成不复用。

同一个关联下需要保留项目实际使用模型的迭代记录：

```text
modelBindingId
├── revision 1 → modelNode-A
├── revision 2 → modelNode-B
└── revision 3 → modelNode-C
```

关联发布记录至少保存：

- 迭代序号
- `modelNodeId`
- 推理配置快照
- 创建时间
- 发布原因
- 发布状态

置信度阈值、最大结果数、ROI 和类别过滤等配置属于关联发布记录。即使只修改配置，也创建新的发布记录，保证模型和配置可以一起回滚。

平台不判断模型是“通用”还是“专用”，也不根据该分类执行任何逻辑。模型是否复用、是否分支以及绑定到哪个项目，由管理人员决定。

### 数据集和快照

数据集支持两种来源：

- 导入线下已经整理好的数据集
- 后续通过正样本和负样本通道追加数据

逻辑数据集持续扩展，但每次训练引用不可变的数据集快照：

```text
dataset
├── snapshot-001
├── snapshot-002
└── snapshot-003
```

训练和评估不修改已使用的快照。快照至少记录：

- 图片和标注清单
- 标签体系
- `train`、`val`、`test` 划分
- 正负样本组成
- 数据文件校验信息
- 创建时间和来源

生成快照时，平台将导入目录复制到平台管理的内容寻址存储，生成 manifest 和整体校验和，并将快照文件设为只读。训练和评估只使用快照副本，不再读取可变的导入目录。启动任务前重新校验 manifest；哈希变化时拒绝任务。

同一文件哈希出现在不同数据划分时阻止生成快照。视频帧近似重复和其他疑似泄漏默认给出警告，并在评估策略要求时阻止发布。

YOLO 数据集采用标准目录和标注格式。负样本通过专门通道上传，导出为图片和空的 YOLO 标注文件；普通未标注图片不能自动当作负样本。

数据集导入时，结构性错误阻止生成快照，包括图片损坏、标注格式错误、图片和标注不匹配、类别 ID 越界以及数据配置错误。类别不均衡、重复图片和疑似数据泄漏等质量问题允许生成快照，但必须给出警告。

### 训练任务

训练由人员手动发起，异步进入 GPU 任务队列。训练任务至少记录：

- `parentModelNodeId`，可选
- `datasetSnapshotId`，必填
- 任务类型和模型族
- 完整训练配置
- 资源配置
- 训练日志和指标
- checkpoint
- 输出模型节点
- 失败原因

如果通过 `modelBindingId` 选择父模型，任务创建时立即将当前模型节点写入 `parentModelNodeId`，排队期间不随关联变化。

第一阶段只支持 `YOLO + 目标检测`，但配置结构为后续分类和分割任务预留扩展。配置分为通用参数和任务专属参数，任务创建时保存最终展开后的完整配置。

训练状态包括：

```text
queued → running → completed
                 └→ failed
                 └→ cancelled
```

每个训练任务可以有多个 `trainingAttempt`。失败或取消后，兼容的重试会创建新的 attempt；修改父模型、数据集或关键配置时必须创建新的训练任务。训练过程中按固定间隔保存 checkpoint，保留最近 checkpoint 和指标最好的 checkpoint。

checkpoint 必须记录父模型校验和、数据集快照 ID、训练配置摘要和 epoch。恢复前校验这些信息。每个运行中的 attempt 持有数据库租约、租约过期时间和 fencing token，并通过心跳续租；租约失效后不自动恢复，管理员确认旧进程已停止后才能创建新的 attempt，避免两个 Worker 同时训练。

训练完成只生成候选模型节点，不自动替换当前线上模型。

### GPU 资源策略

推理和训练共用一台 GPU 服务器。训练任务进入低优先级队列，但不做运行中的抢占。第一阶段训练 Worker 的 Celery 并发固定为 1；训练启动前检查已驻留推理模型的显存占用和训练预留显存，只有资源计划允许时才能启动。训练启动后不自动抢占推理，也不自动加载新推理模型；显存不足时训练失败并保留当前推理模型。

模型驻留由管理员手动指定：

- 管理员决定哪些模型加载到 GPU
- 管理员手动加载、卸载和切换模型
- 平台展示 GPU 总显存、已用显存、剩余显存、模型占用和训练占用
- 显存不足时提示异常
- 平台不自动卸载其他模型，也不自动重新分配资源
- 资源不足的训练任务保持排队或提示原因
- 发布操作不自动改变管理员指定的驻留模型集合
- 待发布模型必须先由管理员加载并预热，发布只切换已验证的运行实例
- 当前关联模型未驻留时，推理明确返回模型不可用状态，不自动加载其他模型

`current_release_id` 表示平台期望该关联使用的发布记录，运行时状态单独记录实际已加载的模型节点、加载状态和健康状态。只有发布指针与健康的驻留实例一致时，关联才处于可推理状态。

发布切换采用两阶段协议：

1. 在 binding 行锁内创建新的 `preparing` runtime instance，旧的 `serving` instance 继续处理请求。
2. Worker 加载并预热新实例，使用本次分配的 fencing token 向平台确认 ready。
3. 平台再次锁定 binding，确认 ready 实例和 token 有效后，在一个事务中更新 `current_release_id`、`current_runtime_instance_id` 和 generation；新实例变为 `serving`，旧实例变为 `draining`。
4. Worker 只接受当前 generation 的新请求；正在执行的旧请求使用请求开始时固定的实例快照完成。
5. 旧实例排空后变为 `stopped`。

同一 binding 在切换期间允许一个 `serving` 和一个 `preparing` 实例；同一 binding 只能有一个 `serving` 和一个 `preparing`。过期 generation 的 Worker 写入和新请求全部拒绝。

### 自动评估和人工验收

训练结束后，平台使用带人工标注的测试集执行自动评估。自动评估包括：

- 模型文件加载和基础推理检查
- precision、recall、mAP50、mAP50-95
- 各类别指标
- 误报和漏报统计
- 与父模型或当前模型的回归比较

评估策略保存总体指标、逐类别最低 precision/recall/mAP、每类最少测试图片数、负样本覆盖要求和允许的回归下降幅度。第一代任务模型如果重新建立了与父模型不兼容的标签体系，则不执行父模型逐类别回归比较，只执行自身标签体系的绝对指标检查；后续追加类别的模型才对继承类别执行回归比较。数据集导入时测试集缺少类别可以警告，但发布评估时缺少评估策略要求的测试覆盖必须判定为不通过。

平台展示测试集图片和识别结果，人员可以上传现场图片进行临时验证。临时验证图片不进入正式数据集，也不由平台长期维护。

发布条件为：

```text
自动评估通过 + 人工验收通过 = 允许发布
```

人工确认后，管理员先将新模型加入驻留集合并执行加载、预热和健康检查。验证成功后平台原子切换关联发布记录；切换失败时继续使用旧模型，且不改变驻留集合。

`loading`、`preparing` 或 `draining` 实例超过配置超时时间后标记为 `failed`，不自动接管当前发布。API 重启时保留并核对健康 Worker 心跳；推理 Worker 重启后先将内存实例标记为失联，再按当前发布和管理员驻留集合重新加载并预热，健康后才恢复服务。当前健康发布不受失败候选实例影响。

### 运行管理、回滚和存储

模型切换流程为：

```text
加载新模型 → 预热 → 健康检查 → 原子切换 → 旧请求完成 → 保留旧模型
```

回滚不需要重新训练，也不需要修改调用方保存的 `modelBindingId`。管理员先确认目标历史模型已驻留并通过预热检查，平台再创建新的 rollback 发布记录，将当前指针切换到该记录。

模型文件按 `modelNodeId` 存储，不覆盖已有文件。当前使用、历史发布引用或存在子模型的模型不能直接物理删除。删除采用归档或软删除，物理清理由管理员手动执行。

平台不做自动备份和自动清理。管理员可以手动备份模型文件、数据集快照、训练配置、评估报告以及关联发布历史。

平台启动和管理后台提供文件一致性检查，识别数据库引用的模型、数据集快照、checkpoint 或报告文件是否缺失。缺失文件会使相关对象进入异常状态，但不自动删除数据库记录。

### 日志、监控和验证方式

平台保留以下日志和状态：

- GPU、CPU、内存和磁盘占用
- 当前驻留模型
- 训练任务队列和训练日志
- 模型加载、预热、切换和卸载结果
- 推理耗时和错误摘要
- 模型节点、数据集快照和发布记录的操作记录

普通推理日志不保存原始图片或视频，只保存输入哈希、大小、类型、来源摘要、模型信息、耗时和结果摘要。评估数据与正式数据集分开管理。

平台不做角色管理，验证采用约定的单一平台 Key。Key 只通过请求头传递，不支持 URL 参数，避免 Key 出现在访问日志、浏览器历史和代理记录中。Key 不写入普通日志。

### 当前暂不展开

- 对外推理和训练 API 的具体协议
- 内置标注工具
- 视频训练
- ONNX、TensorRT 等其他模型格式
- 自动训练
- 自动备份和自动清理
- 角色权限体系

### 管理后台

平台提供 Web 管理后台，第一阶段以功能实用为主，不设计复杂的角色体系或业务项目层级。

主要功能区包括：

- 首页总览
- 模型节点
- 模型关联
- 数据集
- 训练任务
- 模型评估
- 发布与回滚
- 系统日志

#### 首页总览

首页展示平台资源和任务状态：

- GPU 总显存、已用显存和剩余显存
- 当前驻留模型及显存占用
- 当前训练任务和等待队列
- 最近完成、失败和取消的训练任务
- 待人工评估的候选模型
- 各关联当前使用的模型
- 最近发布和回滚记录
- 数据集校验失败和质量警告
- 显存不足、磁盘不足、模型切换失败等异常

首页支持进入对应详情页面。第一阶段采用定时刷新，不引入复杂的实时推送机制。

#### 模型节点页面

模型节点以树形谱系展示：

```text
根模型
├── 第一代任务模型
│   ├── 项目迭代模型
│   └── 项目迭代模型
└── 第一代任务模型
```

页面支持多个根模型并列展示，并可按任务类型、标签体系和状态筛选。节点详情展示：

- `modelNodeId`
- 模型名称和描述
- 父节点
- 标签体系
- 训练数据快照
- 评估指标
- 当前状态
- 被哪些 `modelBindingId` 使用
- 模型文件和训练配置

页面不使用“通用模型”或“专用模型”作为树节点分类，只展示实际父子关系。

#### 模型关联页面

模型关联采用扁平列表，不按项目建立平台内部层级。列表展示：

- `modelBindingId`
- 项目备注或外部引用
- 当前模型节点
- 当前发布序号
- 启用状态
- 创建时间

页面支持创建空关联、查看发布历史、手动切换、回滚、停用和归档。关联 ID 生成后不复用。

#### 数据集页面

数据集页面支持从服务器目录导入线下整理好的数据集。平台扫描目录后执行格式校验，并将数据纳入平台管理的存储范围，再生成不可变数据集快照。

页面展示：

- 数据集名称
- 标签体系
- 图片数量
- 正负样本数量
- `train`、`val`、`test` 数量
- 最新快照
- 数据来源
- 校验状态
- 数据质量警告

支持查看快照中的文件清单、类别分布、重复图片和疑似数据泄漏。第一阶段不优先支持压缩包上传，也不提供完整的内置标注工具。

#### 训练任务页面

训练任务创建页面支持选择：

- 父模型节点，或通过 `modelBindingId` 自动解析当前模型
- 数据集快照
- 任务类型和模型族
- 通用训练参数
- YOLO 检测专属参数
- 目标 `modelBindingId`，可选
- 资源需求和 GPU 预估

提交前执行父模型、数据集、标签体系、样本覆盖和 GPU 资源校验。结构性错误阻止提交，数据质量问题显示警告并允许人员确认后继续。

训练任务详情页面实时展示：

- 当前状态
- epoch 进度和预计剩余时间
- loss 和评估指标曲线
- GPU 占用
- 训练日志
- 最近和最佳 checkpoint
- 失败原因
- 输出的 `modelNodeId`

支持取消排队任务、停止运行任务、从兼容 checkpoint 继续训练以及进入模型评估页面。

#### 模型评估页面

评估页面展示候选模型、父模型、数据集快照、自动评估指标和测试集图片结果。测试图片展示预测框、类别、置信度和人工标注框，并支持按类别、结果类型和置信度筛选。

人员可以上传现场图片进行临时验证。临时验证图片不进入正式数据集，也不由平台长期维护。

人工评估结果包括：

- 正确
- 误报
- 漏报
- 错分类
- 需复核

最终由人员提交整批评估结论。自动评估通过且人工验收通过后，模型才允许发布。

#### 发布确认页面

发布前展示：

- 当前线上模型
- 候选模型
- 父子关系
- 数据集快照
- 自动评估指标及差异
- 标签体系变化
- 推理配置变化
- 人工验收结论
- GPU 资源预估

发布前自动检查模型加载、标签和输入结构、GPU 显存和预热推理。人员确认后，管理员先将模型加入驻留集合并完成加载和预热，健康检查通过后平台原子切换关联发布记录。

切换失败时继续使用旧模型，候选模型保留并显示失败原因。发布成功后提供历史发布记录和回滚入口。

### 界面原型策略

正式开发管理后台前，先制作低保真可点击原型，优先验证页面结构、信息层级和操作流程，不先投入复杂视觉设计。

原型必须覆盖以下核心流程：

- 查看多个根模型和父子谱系
- 查看模型节点详情和标签体系
- 创建空的 `modelBindingId`
- 导入线下数据集并查看校验结果
- 创建训练任务并查看任务进度
- 查看测试集图片和自动评估结果
- 上传临时图片进行人工验收
- 加载候选模型、确认发布和回滚
- 查看 GPU 资源和异常状态

原型中的术语、状态和操作名称必须与后端设计保持一致，例如 `modelNodeId`、`modelBindingId`、`datasetSnapshotId`、`candidate`、`approved` 和 `modelBindingRelease`。

第一阶段原型以桌面管理后台为主，不优先设计移动端和复杂视觉效果。原型确认后再输出页面清单、交互说明和前端实现任务。

当前原型文件：`platform-prototype.html`。

### 最终技术和部署决策

后续设计和实现采用以下默认方案，不再为每项基础技术单独增加决策流程：

- 后端和 AI 使用 Python 3.11。
- 管理后台使用 React + TypeScript，与后端前后端分离。
- 后端使用 FastAPI，训练使用 PyTorch 和 Ultralytics YOLO。
- 第一阶段生产模型格式只使用 `.pt`。
- 使用 Celery + Redis 执行训练队列、通知和进度缓存，使用 PostgreSQL 事务租约管理 GPU 资源。
- 使用 PostgreSQL 保存平台元数据和权威业务状态。
- 模型、数据集、checkpoint、日志和评估文件使用本地文件系统。
- 使用 Docker Compose + NVIDIA Container Toolkit 部署单机 GPU 服务。
- 前端由独立静态服务提供，后端 API、推理 Worker 和训练 Worker 分开运行。
- PostgreSQL 和 Redis 先启动，随后执行数据库迁移，再启动 API、推理 Worker、训练 Worker 和前端。
- 存储路径、数据库、Redis、GPU 设备、训练默认值和平台 Key 通过配置文件或环境变量一次性配置。
- 模型驻留、加载、卸载和发布属于运行期运维操作，由管理后台执行并记录到数据库。
- 管理后台不修改基础环境配置，只展示基础配置并管理运行期模型状态。

服务拓扑：

```text
React 静态前端
        │
        ▼
FastAPI 平台服务
   ┌────┼──────┐
   ▼    ▼      ▼
PostgreSQL Redis 推理 Worker
                 │
                 ▼
           训练 Worker
```

推理 Worker 和训练 Worker 共用 GPU，通过 PostgreSQL 资源租约、管理员指定的驻留模型和显存检查共同控制资源；Redis 只承担队列、通知和缓存，不自动卸载模型或重新分配资源。

#### 推理调用链

`/api/infer` 由 FastAPI 接收请求，FastAPI 将请求转发到推理 Worker 的内部 HTTP 接口。推理 Worker 负责解析当前 `modelBindingId` 的 serving runtime instance、执行推理并返回运行实例元数据；FastAPI 原样返回结果并写入摘要日志。

内部调用规则：

- FastAPI 到推理 Worker 使用内部网络，不对外暴露。
- 请求携带关联 ID、请求 ID和输入内容，Worker 不接受调用方直接指定模型节点或发布记录。
- Worker 在开始推理时固定运行实例代际；切换发生在请求执行期间时，当前请求继续使用旧实例并返回旧实例的 binding、release、model、generation 和配置哈希。
- Worker 不存在健康 serving 实例时返回 `model_unavailable`，FastAPI 转换为明确的服务不可用响应。
- 内部调用设置连接超时、推理超时和最大输入大小；超时不改变当前发布指针。
- 训练和评估流程不直接调用对外接口，使用同一套推理引擎执行内部验证。

### 数据库核心表设计

PostgreSQL 是平台元数据的权威来源。模型、数据集、日志和评估文件本体保存在文件系统，数据库只保存路径、校验和、状态、指标和关联关系。

#### `model_nodes`

保存不可变模型节点及其父子谱系。

- `id`：UUID，模型节点唯一标识
- `parent_id`：可为空，自引用父节点
- `label_schema_id`
- `task_type`
- `model_family`
- `artifact_path`
- `artifact_hash`
- `dataset_snapshot_id`
- `training_task_id`
- `status`
- `metadata`
- `created_at`

根模型的 `parent_id` 为空。模型文件和模型节点生成后不覆盖，只允许变更生命周期状态或补充校验结果。

#### `label_schemas` 和 `label_schema_classes`

保存类别体系及其演进关系。

- 标签体系 ID
- 父标签体系 ID
- 类别 ID
- semantic_key
- 类别名称
- 类别描述
- 创建时间

后续模型的标签体系必须保留父模型已有类别，新增类别只能追加，不能复用已有类别 ID。`semantic_key` 在同一标签谱系中不可变；生成子标签体系时复制继承类别并追加新类别。

已被模型节点、数据集快照或评估记录引用的标签体系只能归档，不能修改或物理删除。

#### `datasets` 和 `dataset_snapshots`

`datasets` 表示持续扩展的逻辑数据集，`dataset_snapshots` 表示不可变的训练输入。

快照保存：

- 数据集 ID
- 标签体系 ID
- `train`、`val`、`test` 文件清单
- 正负样本统计
- 清单文件路径
- 数据质量校验结果
- 文件校验信息
- 创建时间

已被训练任务、模型节点或评估记录引用的快照禁止物理删除。

#### `training_tasks`、`training_attempts` 和 `checkpoints`

`training_tasks` 保存一次训练意图，`training_attempts` 保存该任务的具体执行过程，`checkpoints` 保存可恢复的训练检查点。

训练任务保存：

- `parent_model_node_id`
- `dataset_snapshot_id`
- `target_binding_id`，可为空
- 任务类型和模型族
- 完整训练配置
- 任务状态
- 创建人和创建时间

训练尝试保存：

- 尝试序号
- 状态
- GPU 资源
- 当前 epoch
- 日志路径
- 最近 checkpoint
- 最佳 checkpoint
- lease token、fencing token、心跳和租约过期时间
- 失败原因
- 输出模型节点

同一训练任务可以有多个执行尝试，但同一时间只能有一个运行中的尝试。修改父模型、数据集或关键训练配置时，必须创建新的训练任务。

#### `evaluations`

保存模型自动评估和人工验收的结果。

- `model_node_id`
- `dataset_snapshot_id`
- 自动评估状态
- 人工评估状态
- 评估策略和阈值
- evaluation attempt、lease token、fencing token、心跳和租约过期时间
- 指标 JSON
- 报告文件路径
- 人工结论、操作人和备注
- 创建时间

自动评估和人工验收状态分开保存，只有两者都通过时模型节点才能进入 `approved`。

#### `model_bindings` 和 `model_binding_releases`

`model_bindings` 保存稳定的关联 ID，`model_binding_releases` 保存该关联下的项目迭代发布。

关联表保存：

- `id`
- 外部项目备注
- `current_release_id`
- `current_runtime_instance_id`
- 状态
- 创建时间

发布表保存：

- `binding_id`
- `revision_no`
- `model_node_id`
- 推理配置快照
- `release_type`：normal 或 rollback
- `rollback_target_release_id`，rollback 时必填
- `status`
- 发布原因
- 创建时间

同一关联下的 `revision_no` 唯一递增。数据库对同一 `binding_id` 使用行锁，保证发布和回滚串行，并使用联合外键保证 `current_release_id` 必须属于当前 binding。普通发布和回滚都创建新的发布记录，回滚记录通过 `release_type=rollback` 和 `rollback_target_release_id` 指向历史发布，再更新当前指针。

发布流程先创建 `pending` release，再由管理员针对该 release 执行模型驻留、加载和预热。根模型可以使用 `approval_source=root_import` 例外；非根模型必须关联已通过自动评估和人工验收的模型节点。只有 ready runtime instance 经过最终激活事务后，release 才能变为 active。

#### `runtime_instances`

保存实际加载到推理 Worker 的运行实例，解决数据库发布指针和进程内模型状态不一致的问题。

- `id`
- `binding_id`
- `release_id`
- `model_node_id`
- 推理配置哈希
- 代际号
- fencing token
- 状态：`loading`、`preparing`、`ready`、`serving`、`draining`、`stopped`、`failed`
- GPU 设备和显存占用
- Worker 心跳
- 创建时间和失败原因

发布流程先创建并预热 `runtime_instance`，健康检查成功后，在同一数据库事务中更新 `current_release_id`、`current_runtime_instance_id` 和实例代际号。推理 Worker 只服务当前代际的 `serving` 实例，并在结果中返回该实例对应的 binding、release、model 和配置哈希。

#### `gpu_resource_leases`

GPU 资源准入由 PostgreSQL 事务租约控制，不把 Redis 锁作为唯一资源依据。

- GPU 设备 ID
- 资源持有者类型和 ID
- 预留显存
- 实际显存快照
- lease token
- fencing token
- 心跳和过期时间
- 状态

模型驻留和训练启动时在同一事务中锁定 GPU 资源行，检查所有有效预留的总量后再创建租约。训练 Worker 只有拿到有效租约才能启动；租约失效后的写入被拒绝，管理员确认旧进程停止后才释放或重新分配资源。Redis 只用于队列、通知和缓存。

#### `model_residency_plans`

保存管理员希望驻留的模型发布集合，作为 Worker 重启后的恢复依据。

- binding、release 和 model 节点
- GPU 设备
- 预留显存
- 期望状态：`resident` 或 `removed`
- 优先级
- 创建人和更新时间

运行实例是驻留计划的实际执行结果。发布失败不会自动修改驻留计划；管理员需要显式移除或调整计划。

#### 状态和事务约束

- 数据库使用 Alembic 管理迁移。
- 关键外键使用引用限制，不使用级联删除。
- `model_bindings(id, current_release_id)` 对 `model_binding_releases(binding_id, id)` 使用联合外键。
- `model_bindings(id, current_runtime_instance_id)` 对 `runtime_instances(binding_id, id)` 使用联合外键。
- 同一 binding 同时只能存在一个 `serving` 和一个非终态候选实例（`loading`、`preparing` 或 `ready`），发布和回滚使用行锁串行化。
- 有效 GPU 租约使用唯一持有者约束，训练 attempt 的并发数量通过数据库部分唯一索引和 Worker `--concurrency=1` 双重限制。
- 同一训练任务同时只能有一个 `running` 或 `recovering` attempt。
- 发布事务必须引用 `approved` 模型节点和通过的评估记录；根模型的 approved 状态由导入加载校验产生。
- 模型节点、数据集快照和目标标签体系的组合必须在创建模型节点的事务中校验一致。
- 发布、回滚、模型节点状态提升和训练 attempt 状态变更只能通过带事务门禁的服务操作执行，禁止直接修改状态字段绕过校验。
- 训练任务创建时固定父模型和数据集快照。
- Worker 的进度、checkpoint、状态和模型产物写入必须同时校验当前 attempt ID、fencing token 和未过期租约；失效 Worker 的写入全部拒绝。
- 训练和评估启动前重新校验数据集快照 manifest 和文件哈希；校验失败时任务不得启动。
- 模型产物校验通过后才创建模型节点。
- 模型预热和健康检查成功后，才在事务中切换当前发布指针。
- 模型节点、数据集快照和发布记录保留历史，不覆盖旧记录。
- 日志正文、模型文件、数据集清单和评估图片不直接存入数据库。

### 第一阶段范围和验收标准

#### 第一阶段范围

第一阶段实现一套单机 GPU、图片目标检测的完整内部闭环：

- 导入多个 `.pt` 根模型
- 导入服务器目录中的 YOLO 图片数据集
- 校验并生成不可变数据集快照
- 创建和管理模型父子谱系
- 手动创建 YOLO 目标检测训练任务
- 训练排队、GPU 锁、checkpoint 和失败重试
- 生成候选模型并执行自动评估
- 查看测试集图片和识别结果
- 上传临时现场图片进行人工验收
- 发布、切换和回滚模型关联
- 查看模型、数据集、任务、评估和资源状态
- 保存必要的日志和操作记录

#### 第一阶段明确不做

- 视频推理、视频训练和视频数据集处理
- 内置完整标注工具
- 自动触发训练
- ONNX、TensorRT 和其他生产模型格式
- 多 GPU 分布式训练
- 自动卸载模型和自动重新分配 GPU 资源
- 自动备份和自动清理
- 复杂角色权限和项目层级
- 对外 API 的最终协议设计

#### 端到端验收场景

验收使用固定的测试夹具和数据快照，不依赖运行时随机抽样。每个训练任务在创建时保存完整 `evaluationPolicy`，包括总体指标、逐类别阈值、最少测试图片数、负样本覆盖和回归下降幅度；评估过程不得读取未保存的动态默认值。

至少准备以下固定夹具：

- 一个可加载的 YOLO 根模型
- 一个第一代新标签体系数据集
- 一个在父标签体系上追加类别的数据集
- 一个缺少继承类别的非法数据集
- 一个模拟模型加载失败的候选模型
- 一个显存不足的资源配置
- 一个租约过期的训练 attempt
- 一个发布切换期间的并发推理请求
- 两个并发发布或回滚请求
- 一个服务重启后待恢复的 runtime instance

1. 管理员导入多个根模型，平台完成 PT 加载校验并生成根模型节点。
2. 管理员导入线下 YOLO 数据集，平台完成结构校验、质量检查并生成快照。
3. 基于父模型和数据集快照创建训练任务，平台固定父模型和数据集引用。
4. 训练任务进入队列并运行，页面能够查看日志、epoch、指标和 GPU 状态。
5. 训练中断后可以从兼容 checkpoint 创建新的执行尝试并继续训练。
6. 训练成功后生成候选模型节点，不能自动替换当前模型。
7. 平台使用带标注测试集完成自动评估，并展示各类别指标和测试图片。
8. 人员上传临时现场图片完成人工验收，图片不进入正式数据集。
9. 自动评估和人工验收均通过后，管理员先加载并预热新模型，平台再切换关联发布。
10. 新模型加载或预热失败时，旧模型继续提供服务，关联指针不改变。
11. 发布成功后可以查看关联发布历史并回滚到历史模型和配置。
12. GPU 或磁盘资源不足时，平台提示异常，不自动卸载或重新分配模型。
13. 删除操作受到模型、数据集、训练和发布引用约束，历史记录不被级联删除。
14. 普通推理日志不包含原始图片或视频，平台 Key 不写入普通日志。
15. 发布切换期间的请求固定使用请求开始时的 runtime generation，并返回匹配的 binding、release、model 和配置哈希。
16. 并发发布或回滚只能有一个成功分配 revision，另一个必须失败或重新读取最新状态。
17. 旧 Worker、过期 attempt、失联 runtime 和被修改的快照都不能写入新的状态、checkpoint 或模型产物。
18. 服务重启后，旧的 `serving` 状态必须经过 Worker 心跳和实例校验才能恢复；失联实例不能继续接收请求。

#### 验收结果

只有在上述核心场景全部通过后，第一阶段平台才视为具备可用的模型训练和发布闭环。后续新增模型任务、标注工具、视频处理和其他部署格式，应在不破坏当前模型节点、数据集快照和关联发布历史的前提下扩展。
