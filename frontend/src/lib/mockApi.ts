import type {
  ApiClient,
  KpiData,
  ModelNode,
  ModelBinding,
  ModelBindingRelease,
  Dataset,
  TrainingTask,
  TrainingLogs,
  TrainingMetrics,
  TrainingCheckpoint,
  Evaluation,
  ResourceStatus,
} from "./api";

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/* ===== Mock Data ===== */

const modelNodes: ModelNode[] = [
  {
    id: "m-root-yolo",
    name: "YOLO 基础权重",
    parentId: null,
    taskType: "object_detection",
    modelFamily: "YOLOv8",
    labelSchemaId: "schema-yolo-coco",
    labelSchemaName: "COCO 80 类",
    status: "approved",
    createdAt: "2026-07-01T10:00:00Z",
  },
  {
    id: "m-concrete",
    name: "混凝土损伤",
    parentId: "m-root-yolo",
    taskType: "object_detection",
    modelFamily: "YOLOv8",
    labelSchemaId: "schema-concrete-02",
    labelSchemaName: "concrete-02",
    status: "approved",
    metrics: { precision: 0.93, recall: 0.9, mAP50: 0.91 },
    createdAt: "2026-07-15T14:00:00Z",
  },
  {
    id: "m-bridge-a-concrete",
    name: "A 桥 / 混凝土",
    parentId: "m-concrete",
    taskType: "object_detection",
    modelFamily: "YOLOv8",
    labelSchemaId: "schema-concrete-02",
    labelSchemaName: "concrete-02",
    status: "approved",
    metrics: { precision: 0.95, recall: 0.92, mAP50: 0.93 },
    datasetSnapshotId: "ds-bridge-a-004",
    createdAt: "2026-08-10T08:00:00Z",
  },
  {
    id: "m-bridge-b-concrete",
    name: "B 桥 / 混凝土",
    parentId: "m-concrete",
    taskType: "object_detection",
    modelFamily: "YOLOv8",
    labelSchemaId: "schema-concrete-02",
    labelSchemaName: "concrete-02",
    status: "approved",
    metrics: { precision: 0.91, recall: 0.88, mAP50: 0.89 },
    datasetSnapshotId: "ds-bridge-b-002",
    createdAt: "2026-08-05T09:00:00Z",
  },
  {
    id: "m-fire",
    name: "火灾 / 烟雾",
    parentId: "m-root-yolo",
    taskType: "object_detection",
    modelFamily: "YOLOv8",
    labelSchemaId: "schema-fire-01",
    labelSchemaName: "fire-01",
    status: "approved",
    metrics: { precision: 0.89, recall: 0.86, mAP50: 0.88 },
    createdAt: "2026-07-20T11:00:00Z",
  },
  {
    id: "m-fire-yard",
    name: "工业园区 / 火灾",
    parentId: "m-fire",
    taskType: "object_detection",
    modelFamily: "YOLOv8",
    labelSchemaId: "schema-fire-01",
    labelSchemaName: "fire-01",
    status: "candidate",
    metrics: { precision: 0.91, recall: 0.88, mAP50: 0.89 },
    datasetSnapshotId: "ds-fire-yard-02",
    createdAt: "2026-08-18T16:00:00Z",
  },
];

const bindings: ModelBinding[] = [
  {
    id: "mb-bridge-a-concrete-001",
    name: "A 桥 / 混凝土检测",
    description: "桥梁混凝土损伤检测服务",
    currentModelNodeId: "m-bridge-a-concrete",
    currentReleaseId: "rel-003",
    status: "active",
    createdAt: "2026-07-10T10:00:00Z",
  },
  {
    id: "mb-bridge-b-concrete-001",
    name: "B 桥 / 混凝土检测",
    description: "桥梁混凝土损伤检测服务",
    currentModelNodeId: "m-bridge-b-concrete",
    currentReleaseId: "rel-002",
    status: "active",
    createdAt: "2026-07-12T10:00:00Z",
  },
  {
    id: "mb-fire-yard-001",
    name: "工业园区 / 火灾检测",
    description: "工业园区火灾和烟雾检测",
    currentModelNodeId: null,
    currentReleaseId: null,
    status: "unbound",
    createdAt: "2026-08-01T10:00:00Z",
  },
];

const releases: ModelBindingRelease[] = [
  {
    id: "rel-003",
    bindingId: "mb-bridge-a-concrete-001",
    revisionNo: 3,
    modelNodeId: "m-bridge-a-concrete",
    releaseType: "normal",
    status: "active",
    reason: "迭代 03 - 精度提升",
    createdAt: "2026-08-10T09:00:00Z",
  },
  {
    id: "rel-002",
    bindingId: "mb-bridge-a-concrete-001",
    revisionNo: 2,
    modelNodeId: "m-concrete",
    releaseType: "normal",
    status: "superseded",
    reason: "初始发布",
    createdAt: "2026-07-15T15:00:00Z",
  },
  {
    id: "rel-001",
    bindingId: "mb-bridge-a-concrete-001",
    revisionNo: 1,
    modelNodeId: "m-root-yolo",
    releaseType: "normal",
    status: "superseded",
    reason: "根模型导入",
    createdAt: "2026-07-10T11:00:00Z",
  },
  {
    id: "rel-b-002",
    bindingId: "mb-bridge-b-concrete-001",
    revisionNo: 2,
    modelNodeId: "m-bridge-b-concrete",
    releaseType: "normal",
    status: "active",
    reason: "迭代 02 - 回归修复",
    createdAt: "2026-08-05T10:00:00Z",
  },
];

const datasets: Dataset[] = [
  {
    id: "ds-bridge-a-004",
    name: "A 桥混凝土数据集",
    labelSchemaName: "concrete-02",
    imageCount: 1824,
    trainCount: 1280,
    valCount: 360,
    testCount: 184,
    latestSnapshotId: "snap-004",
    source: "服务器目录导入",
    validationStatus: "通过",
    warnings: [],
    createdAt: "2026-08-08T10:00:00Z",
  },
  {
    id: "ds-bridge-a-003",
    name: "A 桥混凝土数据集 (历史)",
    labelSchemaName: "concrete-02",
    imageCount: 1612,
    trainCount: 1128,
    valCount: 320,
    testCount: 164,
    latestSnapshotId: "snap-003",
    source: "服务器目录导入",
    validationStatus: "通过",
    warnings: [],
    createdAt: "2026-07-20T10:00:00Z",
  },
  {
    id: "ds-fire-yard-02",
    name: "工业园区火灾数据集",
    labelSchemaName: "fire-01",
    imageCount: 486,
    trainCount: 340,
    valCount: 98,
    testCount: 48,
    latestSnapshotId: "snap-fire-02",
    source: "服务器目录导入",
    validationStatus: "通过",
    warnings: ["火灾类别正样本偏少"],
    createdAt: "2026-08-15T10:00:00Z",
  },
];

const trainingTasks: TrainingTask[] = [
  {
    id: "train-20260820-001",
    parentModelNodeId: "m-bridge-a-concrete",
    parentModelName: "A 桥 / 混凝土",
    datasetSnapshotId: "ds-bridge-a-004",
    datasetName: "A 桥混凝土数据集",
    targetBindingId: "mb-bridge-a-concrete-001",
    taskType: "object_detection",
    modelFamily: "YOLOv8",
    status: "running",
    epochs: 100,
    currentEpoch: 41,
    loss: 0.032,
    progress: 41,
    createdAt: "2026-08-20T10:00:00Z",
    startedAt: "2026-08-20T10:05:00Z",
  },
  {
    id: "train-20260820-002",
    parentModelNodeId: "m-fire",
    parentModelName: "火灾 / 烟雾",
    datasetSnapshotId: "ds-fire-yard-02",
    datasetName: "工业园区火灾数据集",
    taskType: "object_detection",
    modelFamily: "YOLOv8",
    status: "queued",
    epochs: 80,
    createdAt: "2026-08-20T11:00:00Z",
  },
  {
    id: "train-20260819-003",
    parentModelNodeId: "m-concrete",
    parentModelName: "混凝土损伤",
    datasetSnapshotId: "ds-bridge-a-003",
    datasetName: "A 桥混凝土数据集 (历史)",
    taskType: "object_detection",
    modelFamily: "YOLOv8",
    status: "completed",
    epochs: 100,
    currentEpoch: 100,
    loss: 0.018,
    progress: 100,
    createdAt: "2026-08-19T08:00:00Z",
    startedAt: "2026-08-19T08:05:00Z",
    completedAt: "2026-08-19T14:30:00Z",
  },
  {
    id: "train-20260818-004",
    parentModelNodeId: "m-fire",
    parentModelName: "火灾 / 烟雾",
    datasetSnapshotId: "ds-fire-yard-02",
    datasetName: "工业园区火灾数据集",
    taskType: "object_detection",
    modelFamily: "YOLOv8",
    status: "failed",
    epochs: 80,
    currentEpoch: 12,
    createdAt: "2026-08-18T09:00:00Z",
    startedAt: "2026-08-18T09:05:00Z",
  },
];

const evaluations: Evaluation[] = [
  {
    id: "eval-001",
    modelNodeId: "m-fire-yard",
    modelName: "工业园区 / 火灾",
    datasetSnapshotId: "ds-fire-yard-02",
    datasetName: "工业园区火灾数据集",
    autoStatus: "auto_passed",
    humanStatus: "pending",
    metrics: { precision: 0.91, recall: 0.88, mAP50: 0.89, mAP50_95: 0.72 },
    testImageCount: 48,
    createdAt: "2026-08-19T15:00:00Z",
  },
  {
    id: "eval-002",
    modelNodeId: "m-bridge-a-concrete",
    modelName: "A 桥 / 混凝土 · 迭代 03",
    datasetSnapshotId: "ds-bridge-a-004",
    datasetName: "A 桥混凝土数据集",
    autoStatus: "auto_passed",
    humanStatus: "pending",
    metrics: { precision: 0.95, recall: 0.92, mAP50: 0.93, mAP50_95: 0.78 },
    testImageCount: 184,
    createdAt: "2026-08-20T09:00:00Z",
  },
  {
    id: "eval-003",
    modelNodeId: "m-bridge-b-concrete",
    modelName: "B 桥 / 混凝土 · 迭代 02",
    datasetSnapshotId: "ds-bridge-b-002",
    datasetName: "B 桥混凝土数据集",
    autoStatus: "approved",
    humanStatus: "approved",
    metrics: { precision: 0.91, recall: 0.88, mAP50: 0.89, mAP50_95: 0.74 },
    testImageCount: 164,
    createdAt: "2026-08-05T11:00:00Z",
  },
];

const resourceStatus: ResourceStatus = {
  gpuDevice: "NVIDIA A100 01",
  totalMemory: 40,
  usedMemory: 27.2,
  residentModels: 4,
  inferenceReserved: 18.4,
  trainingReserved: 8.8,
  healthy: true,
};

/* ===== Mock API Client ===== */

let _taskIdCounter = 100;

export const mockApi: ApiClient = {
  async getKpis(): Promise<KpiData> {
    await delay(80);
    return {
      bindingCount: bindings.length,
      bindingDelta: 2,
      residentModels: resourceStatus.residentModels,
      gpuUsage: Math.round((resourceStatus.usedMemory / resourceStatus.totalMemory) * 100),
      gpuUsedMb: resourceStatus.usedMemory * 1024,
      gpuTotalMb: resourceStatus.totalMemory * 1024,
      trainingQueue: trainingTasks.filter((t) => t.status === "queued").length,
      trainingRunning: trainingTasks.filter((t) => t.status === "running").length,
      pendingEval: evaluations.filter((e) => e.humanStatus === "pending").length,
    };
  },

  async getModelNodes(): Promise<ModelNode[]> {
    await delay(60);
    return [...modelNodes];
  },

  async getModelNode(id: string): Promise<ModelNode | null> {
    await delay(40);
    return modelNodes.find((n) => n.id === id) || null;
  },

  async createModelNode(data: Partial<ModelNode>): Promise<ModelNode> {
    await delay(100);
    const node: ModelNode = {
      id: `model-${Date.now()}`,
      name: data.name || "New Model",
      parentId: data.parentId || null,
      taskType: data.taskType || "object_detection",
      modelFamily: data.modelFamily || "yolo",
      labelSchemaId: data.labelSchemaId || "",
      labelSchemaName: data.labelSchemaName || "",
      status: "candidate",
      createdAt: new Date().toISOString(),
    };
    modelNodes.push(node);
    return node;
  },

  async deleteModel(id: string): Promise<boolean> {
    await delay(100);
    const idx = modelNodes.findIndex((n) => n.id === id);
    if (idx >= 0) {
      modelNodes.splice(idx, 1);
      return true;
    }
    return false;
  },

  async getBindings(): Promise<ModelBinding[]> {
    await delay(60);
    return [...bindings];
  },

  async getReleases(bindingId?: string): Promise<ModelBindingRelease[]> {
    await delay(60);
    if (bindingId) return releases.filter((r) => r.bindingId === bindingId);
    return [...releases];
  },

  async getDatasets(): Promise<Dataset[]> {
    await delay(60);
    return [...datasets];
  },

  async createDataset(data: Partial<Dataset>): Promise<Dataset> {
    await delay(100);
    const ds: Dataset = {
      id: `ds-${Date.now()}`,
      name: data.name || "New Dataset",
      labelSchemaName: data.labelSchemaName || "",
      imageCount: 0,
      trainCount: 0,
      valCount: 0,
      testCount: 0,
      latestSnapshotId: "",
      source: data.source || "upload",
      validationStatus: "pending",
      warnings: [],
      createdAt: new Date().toISOString(),
    };
    datasets.push(ds);
    return ds;
  },

  async getTrainingTasks(): Promise<TrainingTask[]> {
    await delay(60);
    return [...trainingTasks];
  },

  async createTrainingTask(data: Partial<TrainingTask>): Promise<TrainingTask> {
    await delay(120);
    _taskIdCounter++;
    const task: TrainingTask = {
      id: `train-20260821-${String(_taskIdCounter).padStart(3, "0")}`,
      parentModelNodeId: data.parentModelNodeId || null,
      parentModelName: data.parentModelName,
      datasetSnapshotId: data.datasetSnapshotId || "",
      datasetName: data.datasetName,
      targetBindingId: data.targetBindingId,
      taskType: data.taskType || "object_detection",
      modelFamily: data.modelFamily || "YOLOv8",
      status: "queued",
      epochs: data.epochs || 100,
      createdAt: new Date().toISOString(),
    };
    trainingTasks.unshift(task);
    return task;
  },

  async getTrainingLogs(taskId: string): Promise<TrainingLogs> {
    await delay(60);
    const task = trainingTasks.find((t) => t.id === taskId);
    if (!task) throw new Error("Task not found");
    return {
      taskId: task.id,
      status: task.status,
      currentEpoch: task.currentEpoch || 0,
      totalEpochs: task.epochs,
      attempts: task.status === "running"
        ? [{ attemptNo: 1, status: "running", currentEpoch: task.currentEpoch || 0, logPath: null, startedAt: task.startedAt || null, finishedAt: null, lastError: null }]
        : [],
    };
  },

  async getTrainingMetrics(taskId: string): Promise<TrainingMetrics> {
    await delay(60);
    const task = trainingTasks.find((t) => t.id === taskId);
    if (!task) throw new Error("Task not found");
    const epochs = [];
    const count = task.currentEpoch || 0;
    for (let e = 1; e <= count; e++) {
      epochs.push({ epoch: e, loss: 1.0 - e * 0.05, lr: 0.001, extra: {} });
    }
    return {
      taskId: task.id,
      epochs,
      bestLoss: epochs.length > 0 ? epochs[epochs.length - 1].loss : null,
    };
  },

  async getTrainingCheckpoint(taskId: string): Promise<TrainingCheckpoint> {
    await delay(60);
    const task = trainingTasks.find((t) => t.id === taskId);
    if (!task) throw new Error("Task not found");
    const epoch = task.currentEpoch || 0;
    return {
      taskId: task.id,
      attemptNo: task.status === "running" ? 1 : null,
      latestCheckpoint: epoch > 0
        ? { epoch, artifactPath: `/artifacts/epoch_${epoch}.pt`, artifactHash: `sha256:ep${epoch}`, metrics: { loss: 1.0 - epoch * 0.05, mAP50: epoch * 0.02 }, createdAt: new Date().toISOString() }
        : null,
    };
  },

  async getEvaluations(): Promise<Evaluation[]> {
    await delay(60);
    return [...evaluations];
  },

  async getResourceStatus(): Promise<ResourceStatus> {
    await delay(40);
    return { ...resourceStatus };
  },

  async loadModel(_modelNodeId: string): Promise<boolean> {
    await delay(500);
    return true;
  },

  async rollbackBinding(_bindingId: string, _targetReleaseId: string): Promise<boolean> {
    await delay(300);
    return true;
  },

  async getGpuStatus() {
    await delay(40);
    return { deviceCount: 1, totalMemoryMb: 8192, usedMemoryMb: 0, freeMemoryMb: 8192 };
  },

  async getGpuModels() {
    await delay(40);
    return [];
  },

  async gpuLoadModel(_modelName: string, _memoryMb: number) {
    await delay(500);
    return { success: true, gpuDevice: "0" };
  },

  async gpuUnloadModel(_modelName: string) {
    await delay(300);
    return { success: true };
  },

  async reviewEvaluation(_id: string, _status: string) {
    await delay(200);
  },

  async getOperationLogs(skip = 0, limit = 20) {
    await delay(60);
    const items = [
      { id: "op-001", operationType: "training.create", actor: "admin", requestId: "r-001", resourceType: "training_task", resourceId: "t-001", status: "success", summaryJson: { detail: "创建训练任务" }, errorSummary: null, createdAt: "2026-08-20T10:00:00Z" },
      { id: "op-002", operationType: "binding.update", actor: "admin", requestId: "r-002", resourceType: "binding", resourceId: "b-001", status: "success", summaryJson: { detail: "更新模型关联" }, errorSummary: null, createdAt: "2026-08-20T11:00:00Z" },
      { id: "op-003", operationType: "gpu.load", actor: "system", requestId: "r-003", resourceType: "gpu", resourceId: null, status: "error", summaryJson: { detail: "加载模型失败" }, errorSummary: "CUDA OOM", createdAt: "2026-08-20T12:00:00Z" },
    ];
    const filtered = items.slice(skip, skip + limit);
    return { items: filtered, total: items.length };
  },

  async getOperationErrors(_skip = 0, _limit = 50) {
    await delay(60);
    return {
      items: [
        { id: "op-003", operationType: "gpu.load", actor: "system", requestId: "r-003", resourceType: "gpu", resourceId: null, status: "error", summaryJson: { detail: "加载模型失败" }, errorSummary: "CUDA OOM", createdAt: "2026-08-20T12:00:00Z" },
      ],
      total: 1,
    };
  },

  async getOperationStatus() {
    await delay(40);
    return { status: "ready", checks: { postgres: true, redis: true, gpu: true, worker: true } };
  },

  async uploadModel(file: File, onProgress?: (pct: number) => void) {
    const totalSteps = 10;
    for (let i = 1; i <= totalSteps; i++) {
      await delay(60);
      if (onProgress) onProgress(Math.round((i / totalSteps) * 100));
    }
    return {
      filename: file.name,
      file_path: `/uploads/models/${file.name}`,
      size: file.size,
      sha256: "sha256:mock_hash",
    };
  },

  async uploadDataset(file: File, onProgress?: (pct: number) => void) {
    const totalSteps = 10;
    for (let i = 1; i <= totalSteps; i++) {
      await delay(80);
      if (onProgress) onProgress(Math.round((i / totalSteps) * 100));
    }
    return {
      filename: file.name,
      file_path: `/uploads/datasets/${file.name}`,
      size: file.size,
      sha256: "sha256:mock_hash",
    };
  },
};
