import type { ApiClient } from "./api";
import { mockApi } from "./mockApi";

const API_BASE = import.meta.env.VITE_API_URL || "";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": import.meta.env.VITE_API_KEY || "change-me",
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

const realApi: ApiClient = {
  async getKpis() {
    const [bindingsRes, tasksRes, evalsRes, gpuStatus, gpuModels] = await Promise.all([
      fetchJson<any>("/api/bindings").catch(() => ({ bindings: [] })),
      fetchJson<any>("/api/training/tasks").catch(() => ({ tasks: [] })),
      fetchJson<any>("/api/evaluations").catch(() => ({ evaluations: [] })),
      fetchJson<any>("/api/gpu/status").catch(() => ({ used_memory_mb: 0, total_memory_mb: 1 })),
      fetchJson<any>("/api/gpu/models").catch(() => ({ models: [] })),
    ]);
    const bindings = bindingsRes.bindings ?? bindingsRes;
    const tasks = tasksRes.tasks ?? tasksRes;
    const evals = evalsRes.evaluations ?? evalsRes;
    const gpuUsedMb = gpuStatus.used_memory_mb ?? 0;
    const gpuTotalMb = gpuStatus.total_memory_mb ?? 1;
    return {
      bindingCount: Array.isArray(bindings) ? bindings.length : 0,
      bindingDelta: 0,
      residentModels: (gpuModels.models ?? []).length,
      gpuUsage: Math.round((gpuUsedMb / gpuTotalMb) * 100),
      gpuUsedMb,
      gpuTotalMb,
      trainingQueue: Array.isArray(tasks) ? tasks.filter((t: any) => t.status === "queued").length : 0,
      trainingRunning: Array.isArray(tasks) ? tasks.filter((t: any) => t.status === "running").length : 0,
      pendingEval: Array.isArray(evals) ? evals.filter((e: any) => e.auto_status === "pending").length : 0,
    };
  },
  getModelNodes: () => fetchJson<any>("/api/models").then((r) => r.models ?? r),
  getModelNode: (id) => fetchJson<any>(`/api/models/${id}`).catch(() => null),
  createModelNode: (data) => postJson<any>("/api/models", data),
  getBindings: () => fetchJson<any>("/api/bindings").then((r) => r.bindings ?? r),
  getReleases: (bindingId) =>
    fetchJson<any>(bindingId ? `/api/bindings/${bindingId}/releases` : "/api/releases").then((r) => r.releases ?? r),
  getDatasets: () => fetchJson<any>("/api/datasets").then((r) => r.datasets ?? r),
  createDataset: (data) => postJson<any>("/api/datasets", data),
  getTrainingTasks: () => fetchJson<any>("/api/training/tasks").then((r) => r.tasks ?? r),
  createTrainingTask: (data) => postJson("/api/training/tasks", data),
  getTrainingLogs: (taskId: string) => fetchJson<any>(`/api/training/${taskId}/logs`),
  getTrainingMetrics: (taskId: string) => fetchJson<any>(`/api/training/${taskId}/metrics`),
  getTrainingCheckpoint: (taskId: string) => fetchJson<any>(`/api/training/${taskId}/checkpoint`),
  getEvaluations: () => fetchJson<any>("/api/evaluations").then((r) => r.evaluations ?? r),
  getResourceStatus: async () => {
    const gpuStatus = await fetchJson<any>("/api/gpu/status").catch(() => null);
    const gpuModels = await fetchJson<any>("/api/gpu/models").catch(() => ({ models: [] }));
    
    const totalMemoryMb = gpuStatus?.total_memory_mb ?? 8192;
    const usedMemoryMb = gpuStatus?.used_memory_mb ?? 0;
    const models = gpuModels?.models ?? [];
    
    return {
      gpuDevice: `GPU ${gpuStatus?.device_count ?? 0}`,
      totalMemory: Math.round(totalMemoryMb / 1024 * 10) / 10,
      usedMemory: Math.round(usedMemoryMb / 1024 * 10) / 10,
      residentModels: models.length,
      inferenceReserved: 0,
      trainingReserved: 0,
      healthy: (gpuStatus?.device_count ?? 0) > 0,
    };
  },
  loadModel: (modelNodeId) =>
    postJson(`/api/resources/load`, { modelNodeId }).then(() => true),
  rollbackBinding: (bindingId, targetReleaseId) =>
    postJson(`/api/bindings/${bindingId}/rollback`, { targetReleaseId }).then(
      () => true
    ),
  getGpuStatus: () => fetchJson<any>("/api/gpu/status").then((r) => ({
    deviceCount: r.device_count,
    totalMemoryMb: r.total_memory_mb,
    usedMemoryMb: r.used_memory_mb,
    freeMemoryMb: r.free_memory_mb,
  })),
  getGpuModels: () => fetchJson<any>("/api/gpu/models").then((r) =>
    (r.models ?? []).map((m: any) => ({
      modelName: m.model_name,
      gpuDevice: m.gpu_device,
      memoryMb: m.memory_mb,
    }))
  ),
  gpuLoadModel: (modelName, memoryMb) =>
    postJson<any>("/api/gpu/load", { model_name: modelName, memory_mb: memoryMb }).then((r) => ({
      success: r.success,
      gpuDevice: r.gpu_device,
    })),
  gpuUnloadModel: (modelName) =>
    postJson<any>("/api/gpu/unload", { model_name: modelName }).then((r) => ({
      success: r.success,
    })),
  reviewEvaluation: (id, status) =>
    postJson(`/api/evaluations/${id}/review`, { human_status: status }),
  getOperationLogs: (skip = 0, limit = 20) =>
    fetchJson<any>(`/api/operations/logs?skip=${skip}&limit=${limit}`),
  getOperationErrors: (skip = 0, limit = 50) =>
    fetchJson<any>(`/api/operations/errors?skip=${skip}&limit=${limit}`),
  getOperationStatus: () =>
    fetchJson<any>("/api/operations/status"),
};

export function createApi(): ApiClient {
  const useMocks = import.meta.env.VITE_USE_MOCKS === "true";
  if (useMocks) {
    return mockApi;
  }
  return realApi;
}
