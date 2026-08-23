# AI Model Platform Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete production hardening, NVR integration, operations management, and deployment readiness for the AI model platform.

**Architecture:** Phase 1 infrastructure and Phase 2 real YOLO training/inference are complete. Phase 3 focuses on external integration, operational tooling, and production deployment.

**Tech Stack:** Python 3.11, FastAPI, PyTorch, Ultralytics YOLO, React, TypeScript, Docker Compose, NVIDIA Container Toolkit.

**Scope:** NVR API integration, operations dashboard, backup/restore, deployment hardening, performance testing. No new model types, no video training, no multi-GPU.

---

## Current State

Phase 1 completed all backend services, database, and admin console.
Phase 2 completed real YOLO training, inference, GPU lifecycle, and evaluation.

```text
Phase 1 (10 tasks): Backend + Frontend + Infrastructure
Phase 2 (10 tasks): Real YOLO Training + Inference + GPU
```

---

## Task 1: External Inference API

**Files:**
- Create: `backend/app/api/routes/inference.py`
- Create: `backend/app/schemas/inference.py`
- Create: `backend/app/services/inference_service.py`
- Create: `backend/tests/unit/test_inference_service.py`
- Create: `backend/tests/integration/test_inference_api.py`

**Purpose:** 提供通用的外部推理接口，支持任意调用方接入。

- [ ] **Step 1: Design inference API contract**
  - POST /api/infer: 接收图片，返回识别结果
  - POST /api/infer/batch: 批量推理
  - GET /api/infer/models: 查询可用模型
  - 请求包含 binding_id 或 model_id + input
  - 响应包含 results、latency、model_version

- [ ] **Step 2: Implement InferenceService**
  - 解析 binding_id 或 model_id
  - 从数据库获取模型和配置
  - 调用推理引擎
  - 格式化响应
  - 记录推理日志

- [ ] **Step 3: Implement API routes**
  - X-API-Key 认证
  - 请求校验（文件大小、类型）
  - 错误处理和超时
  - 结果缓存（可选）

- [ ] **Step 4: Write tests**
  - 单元测试：InferenceService 逻辑
  - 集成测试：完整推理流程
  - 错误场景：模型不存在、GPU 不可用、输入无效

- [ ] **Step 5: Run tests and verify**
- [ ] **Step 6: Commit**

---

## Task 2: Operations Dashboard

**Files:**
- Create: `backend/app/api/routes/operations.py`
- Create: `frontend/src/features/operations/OperationsPage.tsx`
- Create: `backend/tests/unit/test_operations.py`

**Purpose:** 统一运维管理入口，集中查看日志、异常和系统状态。

- [ ] **Step 1: Implement operations API**
  - GET /api/operations/logs: 查询操作日志
  - GET /api/operations/errors: 查询异常记录
  - GET /api/operations/status: 系统健康状态
  - 支持分页、时间范围筛选

- [ ] **Step 2: Implement OperationsPage**
  - 系统健康状态卡片
  - 操作日志时间线
  - 异常记录列表
  - 服务状态指示器

- [ ] **Step 3: Write tests and verify**
- [ ] **Step 4: Commit**

---

## Task 3: Backup and Restore

**Files:**
- Create: `backend/app/services/backup_service.py`
- Create: `backend/app/api/routes/backup.py`
- Create: `backend/tests/unit/test_backup_service.py`
- Create: `backend/tests/integration/test_backup_api.py`

**Purpose:** 手动备份和恢复模型、数据集、配置。

- [ ] **Step 1: Implement BackupService**
  - 备份模型文件和元数据
  - 备份数据集快照
  - 备份数据库状态
  - 恢复指定版本
  - 备份完整性校验

- [ ] **Step 2: Implement API routes**
  - POST /api/backup: 创建备份
  - GET /api/backup/list: 列出备份
  - POST /api/backup/restore: 恢复备份
  - DELETE /api/backup/{id}: 删除备份

- [ ] **Step 3: Write tests and verify**
- [ ] **Step 4: Commit**

---

## Task 4: Performance Testing

**Files:**
- Create: `backend/tests/performance/test_inference_perf.py`
- Create: `backend/tests/performance/test_training_perf.py`
- Create: `docs/performance/benchmark.md`

**Purpose:** 验证推理延迟和训练吞吐量。

- [ ] **Step 1: Write inference performance test**
  - 单张图片推理延迟
  - 批量推理吞吐量
  - 并发请求处理
  - 模型切换开销

- [ ] **Step 2: Write training performance test**
  - 不同数据集大小训练时间
  - 不同模型大小训练时间
  - Checkpoint 恢复时间

- [ ] **Step 3: Run benchmarks and document results**
- [ ] **Step 4: Commit**

---

## Task 5: Deployment Documentation

**Files:**
- Create: `docs/deployment/installation.md`
- Create: `docs/deployment/configuration.md`
- Create: `docs/deployment/troubleshooting.md`
- Create: `docs/deployment/operations.md`

**Purpose:** 完善部署文档，支持独立运维。

- [ ] **Step 1: Installation guide**
  - 环境要求
  - 安装步骤
  - 依赖安装
  - 首次启动

- [ ] **Step 2: Configuration guide**
  - 环境变量说明
  - 配置文件说明
  - GPU 配置
  - 数据库配置

- [ ] **Step 3: Troubleshooting guide**
  - 常见错误
  - 日志查看方法
  - 性能调优

- [ ] **Step 4: Operations runbook**
  - 日常运维
  - 紧急处理
  - 备份恢复
  - 版本升级

- [ ] **Step 5: Commit**

---

## Task 6: Database Backup Integration

**Files:**
- Modify: `backend/app/services/backup_service.py`
- Create: `backend/app/services/db_backup.py`

**Purpose:** 将数据库备份集成到备份流程。

- [ ] **Step 1: Implement DbBackup**
  - PostgreSQL dump
  - PostgreSQL restore
  - 增量备份
  - 备份状态记录

- [ ] **Step 2: Integrate with BackupService**
  - 备份时同时保存数据库
  - 恢复时同时恢复数据库
  - 备份一致性保证

- [ ] **Step 3: Write tests and verify**
- [ ] **Step 4: Commit**

---

## Task 7: Error Alerting

**Files:**
- Create: `backend/app/observability/alerting.py`
- Create: `backend/app/api/routes/alerts.py`
- Create: `backend/tests/unit/test_alerting.py`

**Purpose:** 异常检测和告警通知。

- [ ] **Step 1: Implement AlertingService**
  - GPU 显存不足告警
  - 训练失败告警
  - 推理错误率告警
  - 磁盘空间告警
  - 服务不可用告警

- [ ] **Step 2: Implement alert API**
  - GET /api/alerts: 查询告警
  - POST /api/alerts/{id}/acknowledge: 确认告警
  - GET /api/alerts/stats: 告警统计

- [ ] **Step 3: Write tests and verify**
- [ ] **Step 4: Commit**

---

## Task 8: Data Import Enhancement

**Files:**
- Modify: `backend/app/services/dataset_import.py`
- Create: `backend/app/services/batch_import.py`
- Create: `backend/tests/unit/test_batch_import.py`

**Purpose:** 支持批量数据导入和压缩包导入。

- [ ] **Step 1: Implement BatchImport**
  - 支持压缩包上传（zip/tar）
  - 自动解压和校验
  - 进度跟踪
  - 错误报告

- [ ] **Step 2: Update dataset import API**
  - POST /api/datasets/import: 上传压缩包导入
  - POST /api/datasets/batch: 批量目录导入
  - 任务队列异步处理

- [ ] **Step 3: Write tests and verify**
- [ ] **Step 4: Commit**

---

## Task 9: Model Version Diff

**Files:**
- Create: `backend/app/services/model_diff.py`
- Create: `backend/app/api/routes/model_diff.py`
- Create: `backend/tests/unit/test_model_diff.py`

**Purpose:** 比较不同模型版本的指标和配置差异。

- [ ] **Step 1: Implement ModelDiff**
  - 比较两个模型节点的指标
  - 比较配置差异
  - 比较标签体系变化
  - 生成差异报告

- [ ] **Step 2: Implement API**
  - GET /api/models/{id}/diff?compare={id}: 模型对比
  - GET /api/models/{id}/history: 模型历史

- [ ] **Step 3: Write tests and verify**
- [ ] **Step 4: Commit**

---

## Task 10: Production Readiness

**Files:**
- Create: `docs/production/checklist.md`
- Create: `backend/app/observability/health_probes.py`
- Modify: `infra/docker-compose.yml`

**Purpose:** 生产环境就绪检查。

- [ ] **Step 1: Implement health probes**
  - /health/live: 存活探针
  - /health/ready: 就绪探针
  - /health/startup: 启动探针

- [ ] **Step 2: Update Docker Compose**
  - 添加健康检查配置
  - 配置重启策略
  - 配置资源限制

- [ ] **Step 3: Production checklist**
  - SSL/TLS 配置
  - 访问控制
  - 日志轮转
  - 监控配置
  - 备份策略

- [ ] **Step 4: Run tests and verify**
- [ ] **Step 5: Commit**

---

## Execution Notes

- Tasks 1-2 are core and should be completed first
- Tasks 3-4 depend on Task 1
- Tasks 5-6 are independent
- Tasks 7-8 depend on existing infrastructure
- Tasks 9-10 can be done in parallel
- All files must have UTF-8 BOM for Python
- Frontend files must NOT have BOM
- Run `python -m pytest -c NUL` for test execution
- Task 1 is generic inference API, not NVR-specific
