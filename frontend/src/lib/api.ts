/* ===== Type definitions ===== */

export type ModelNodeStatus = "candidate" | "approved" | "rejected" | "archived";
export type BindingStatus = "active" | "inactive" | "unbound";
export type ReleaseStatus = "pending" | "preparing" | "active" | "superseded" | "failed";
export type TrainingStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type EvalStatus = "pending" | "auto_passed" | "approved" | "rejected";

export interface ModelNode {
  id: string;
  name: string;
  parentId: string | null;
  taskType: string;
  modelFamily: string;
  labelSchemaId: string;
  labelSchemaName: string;
  status: ModelNodeStatus;
  artifactPath?: string;
  metrics?: {
    precision: number;
    recall: number;
    mAP50: number;
  };
  createdAt: string;
  datasetSnapshotId?: string;
}

export interface ModelBinding {
  id: string;
  name: string;
  description: string;
  currentModelNodeId: string | null;
  currentReleaseId: string | null;
  status: BindingStatus;
  createdAt: string;
}

export interface ModelBindingRelease {
  id: string;
  bindingId: string;
  revisionNo: number;
  modelNodeId: string;
  releaseType: "normal" | "rollback";
  status: ReleaseStatus;
  reason: string;
  createdAt: string;
}

export interface Dataset {
  id: string;
  name: string;
  labelSchemaName: string;
  imageCount: number;
  trainCount: number;
  valCount: number;
  testCount: number;
  latestSnapshotId: string;
  source: string;
  sourcePath?: string;
  validationStatus: string;
  warnings: string[];
  createdAt: string;
}

export interface DatasetSnapshot {
  id: string;
  datasetId: string;
  imageCount: number;
  trainCount: number;
  valCount: number;
  testCount: number;
  manifestHash: string;
  createdAt: string;
}

export interface TrainingTask {
  id: string;
  parentModelNodeId: string | null;
  parentModelName?: string;
  datasetSnapshotId: string;
  datasetName?: string;
  targetBindingId?: string;
  taskType: string;
  modelFamily: string;
  status: TrainingStatus;
  epochs: number;
  currentEpoch?: number;
  loss?: number;
  progress?: number;
  trainingConfigJson?: Record<string, unknown>;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
}

export interface TrainingAttemptLog {
  attemptNo: number;
  status: string;
  currentEpoch: number;
  logPath: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  lastError: string | null;
}

export interface TrainingLogs {
  taskId: string;
  status: string;
  currentEpoch: number;
  totalEpochs: number;
  attempts: TrainingAttemptLog[];
}

export interface EpochMetrics {
  epoch: number;
  loss: number;
  lr: number | null;
  extra: Record<string, unknown>;
}

export interface TrainingMetrics {
  taskId: string;
  epochs: EpochMetrics[];
  bestLoss: number | null;
}

export interface CheckpointInfo {
  epoch: number;
  artifactPath: string;
  artifactHash: string;
  metrics: Record<string, unknown>;
  createdAt: string;
}

export interface TrainingCheckpoint {
  taskId: string;
  attemptNo: number | null;
  latestCheckpoint: CheckpointInfo | null;
}

export interface Evaluation {
  id: string;
  modelNodeId: string;
  modelName: string;
  datasetSnapshotId: string;
  datasetName: string;
  autoStatus: EvalStatus;
  humanStatus: EvalStatus;
  metrics?: {
    precision: number;
    recall: number;
    mAP50: number;
    mAP50_95: number;
  };
  testImageCount: number;
  createdAt: string;
}

export interface ResourceStatus {
  gpuDevice: string;
  totalMemory: number;
  usedMemory: number;
  residentModels: number;
  inferenceReserved: number;
  trainingReserved: number;
  healthy: boolean;
}

export interface GpuStatus {
  deviceCount: number;
  totalMemoryMb: number;
  usedMemoryMb: number;
  freeMemoryMb: number;
}

export interface GpuModelEntry {
  modelName: string;
  gpuDevice: string;
  memoryMb: number;
}

export interface GpuLoadResponse {
  success: boolean;
  gpuDevice?: string;
}

export interface GpuUnloadResponse {
  success: boolean;
}

export interface OperationLogEntry {
  id: string;
  operationType: string;
  actor: string | null;
  requestId: string | null;
  resourceType: string | null;
  resourceId: string | null;
  status: string;
  summaryJson: Record<string, unknown>;
  errorSummary: string | null;
  createdAt: string;
}

export interface OperationLogsResponse {
  items: OperationLogEntry[];
  total: number;
}

export interface SystemChecks {
  postgres: boolean;
  redis: boolean;
  gpu: boolean;
  worker: boolean;
}

export interface OperationStatusResponse {
  status: string;
  checks: SystemChecks;
}

export interface KpiData {
  bindingCount: number;
  bindingDelta: number;
  residentModels: number;
  gpuUsage: number;
  gpuUsedMb: number;
  gpuTotalMb: number;
  trainingQueue: number;
  trainingRunning: number;
  pendingEval: number;
}

export interface UploadResponse {
  filename: string;
  file_path: string;
  size: number;
  sha256: string;
}

/* ===== API interface ===== */

export interface ApiClient {
  getKpis(): Promise<KpiData>;
  getModelNodes(): Promise<ModelNode[]>;
  getModelNode(id: string): Promise<ModelNode | null>;
  createModelNode(data: Partial<ModelNode>): Promise<ModelNode>;
  deleteModel(id: string): Promise<boolean>;
  getBindings(): Promise<ModelBinding[]>;
  getReleases(bindingId?: string): Promise<ModelBindingRelease[]>;
  getDatasets(): Promise<Dataset[]>;
  createDataset(data: Partial<Dataset>): Promise<Dataset>;
  deleteDataset(id: string): Promise<void>;
  getTrainingTasks(): Promise<TrainingTask[]>;
  createTrainingTask(data: Partial<TrainingTask>): Promise<TrainingTask>;
  getTrainingLogs(taskId: string): Promise<TrainingLogs>;
  getTrainingMetrics(taskId: string): Promise<TrainingMetrics>;
  getTrainingCheckpoint(taskId: string): Promise<TrainingCheckpoint>;
  getEvaluations(): Promise<Evaluation[]>;
  getResourceStatus(): Promise<ResourceStatus>;
  loadModel(modelNodeId: string): Promise<boolean>;
  rollbackBinding(bindingId: string, targetReleaseId: string): Promise<boolean>;
  getGpuStatus(): Promise<GpuStatus>;
  getGpuModels(): Promise<GpuModelEntry[]>;
  gpuLoadModel(modelName: string, memoryMb: number): Promise<GpuLoadResponse>;
  gpuUnloadModel(modelName: string): Promise<GpuUnloadResponse>;
  reviewEvaluation(id: string, status: string): Promise<void>;
  getOperationLogs(skip?: number, limit?: number): Promise<OperationLogsResponse>;
  getOperationErrors(skip?: number, limit?: number): Promise<OperationLogsResponse>;
  getOperationStatus(): Promise<OperationStatusResponse>;
  uploadModel(file: File, onProgress?: (pct: number) => void): Promise<UploadResponse>;
  uploadDataset(file: File, onProgress?: (pct: number) => void): Promise<UploadResponse>;
}
