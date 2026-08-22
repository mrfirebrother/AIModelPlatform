import { useEffect, useState } from "react";
import { createApi } from "../../lib/createApi";
import type { ResourceStatus, GpuStatus, GpuModelEntry } from "../../lib/api";

export default function ResourcesPage() {
  const api = createApi();
  const [res, setRes] = useState<ResourceStatus | null>(null);
  const [gpuStatus, setGpuStatus] = useState<GpuStatus | null>(null);
  const [gpuModels, setGpuModels] = useState<GpuModelEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadName, setLoadName] = useState("");
  const [loadMem, setLoadMem] = useState(2048);

  const refresh = () => {
    api.getGpuStatus().then(setGpuStatus);
    api.getGpuModels().then(setGpuModels);
  };

  useEffect(() => {
    api.getResourceStatus().then(setRes);
    refresh();
  }, []);

  const handleLoad = async () => {
    if (!loadName.trim()) return;
    setLoading(true);
    try {
      await api.gpuLoadModel(loadName.trim(), loadMem);
      setLoadName("");
      refresh();
    } finally {
      setLoading(false);
    }
  };

  const handleUnload = async (name: string) => {
    setLoading(true);
    try {
      await api.gpuUnloadModel(name);
      refresh();
    } finally {
      setLoading(false);
    }
  };

  if (!res) return <div className="empty-state">加载中...</div>;

  return (
    <>
      <div className="page-grid">
        {/* GPU Card */}
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">GPU 资源总览</div>
              <div className="card-kicker">{res.gpuDevice}</div>
            </div>
            <span className="online">{res.healthy ? "健康" : "异常"}</span>
          </div>
          <div className="card-body">
            <div className="meter-label">
              <span>显存使用</span>
              <b>
                {gpuStatus ? `${(gpuStatus.usedMemoryMb / 1024).toFixed(1)} / ${(gpuStatus.totalMemoryMb / 1024).toFixed(1)} GB` : `${res.usedMemory} / ${res.totalMemory} GB`}
              </b>
            </div>
            <div className="meter">
              <div
                className="meter-fill"
                style={{ width: `${gpuStatus ? (gpuStatus.usedMemoryMb / gpuStatus.totalMemoryMb) * 100 : (res.usedMemory / res.totalMemory) * 100}%` }}
              />
            </div>

            <div style={{ marginTop: 16 }}>
              <div className="meter-label">
                <span>推理预留</span>
                <b>{res.inferenceReserved} GB</b>
              </div>
              <div className="meter">
                <div
                  className="meter-fill cyan"
                  style={{ width: `${(res.inferenceReserved / res.totalMemory) * 100}%` }}
                />
              </div>
            </div>

            <div style={{ marginTop: 16 }}>
              <div className="meter-label">
                <span>训练预留</span>
                <b>{res.trainingReserved} GB</b>
              </div>
              <div className="meter">
                <div
                  className="meter-fill orange"
                  style={{ width: `${(res.trainingReserved / res.totalMemory) * 100}%` }}
                />
              </div>
            </div>

            <hr style={{ border: "none", borderTop: "1px solid #d7e5ed", margin: "16px 0" }} />

            <div className="detail-list">
              <div className="detail-row">
                <span>空闲显存</span>
                <b>{gpuStatus ? `${(gpuStatus.freeMemoryMb / 1024).toFixed(1)} GB` : `${(res.totalMemory - res.usedMemory).toFixed(1)} GB`}</b>
              </div>
              <div className="detail-row">
                <span>驻留模型数</span>
                <b>{gpuModels.length} 个</b>
              </div>
              <div className="detail-row">
                <span>管理方式</span>
                <b>管理员手动</b>
              </div>
            </div>
          </div>
        </div>

        {/* Loaded Models */}
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">已加载模型</div>
              <div className="card-kicker">GPU 驻留模型列表</div>
            </div>
          </div>
          <div className="card-body">
            {gpuModels.length === 0 ? (
              <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "12px 0" }}>
                当前无驻留模型
              </div>
            ) : (
              <div>
                {gpuModels.map((m) => (
                  <div
                    key={m.gpuDevice}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr auto auto",
                      gap: 10,
                      padding: "10px 0",
                      borderBottom: "1px solid #dce8f0",
                      alignItems: "center",
                    }}
                  >
                    <div>
                      <div style={{ color: "var(--text)", fontSize: 12, fontWeight: 600 }}>
                        {m.modelName}
                      </div>
                      <div style={{ color: "var(--text-muted)", fontSize: 9, fontFamily: "Courier New, monospace", marginTop: 2 }}>
                        {m.gpuDevice} / {(m.memoryMb / 1024).toFixed(1)} GB
                      </div>
                    </div>
                    <span className="online" style={{ fontSize: 9 }}>已加载</span>
                    <button
                      className="card-action"
                      onClick={() => handleUnload(m.modelName)}
                      disabled={loading}
                      style={{ fontSize: 10, padding: "4px 8px" }}
                    >
                      卸载
                    </button>
                  </div>
                ))}
              </div>
            )}

            <hr style={{ border: "none", borderTop: "1px solid #d7e5ed", margin: "16px 0" }} />

            <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text)", marginBottom: 8 }}>
              加载模型
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 80px auto", gap: 8, alignItems: "center" }}>
              <input
                type="text"
                placeholder="模型名称"
                value={loadName}
                onChange={(e) => setLoadName(e.target.value)}
                style={{
                  padding: "6px 8px",
                  border: "1px solid #d7e5ed",
                  borderRadius: 4,
                  fontSize: 12,
                  fontFamily: "inherit",
                }}
              />
              <input
                type="number"
                placeholder="MB"
                value={loadMem}
                onChange={(e) => setLoadMem(Number(e.target.value))}
                style={{
                  padding: "6px 8px",
                  border: "1px solid #d7e5ed",
                  borderRadius: 4,
                  fontSize: 12,
                  fontFamily: "Courier New, monospace",
                }}
              />
              <button
                className="card-action"
                onClick={handleLoad}
                disabled={loading || !loadName.trim()}
              >
                加载
              </button>
            </div>
          </div>
        </div>

        {/* Resource Allocation */}
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">资源分配策略</div>
              <div className="card-kicker">GPU 资源准入规则</div>
            </div>
          </div>
          <div className="card-body">
            <div className="detail-list">
              <div className="detail-row">
                <span>推理优先级</span>
                <b>高</b>
              </div>
              <div className="detail-row">
                <span>训练优先级</span>
                <b>低</b>
              </div>
              <div className="detail-row">
                <span>训练并发</span>
                <b>1</b>
              </div>
              <div className="detail-row">
                <span>自动卸载</span>
                <b>关闭</b>
              </div>
              <div className="detail-row">
                <span>自动抢占</span>
                <b>关闭</b>
              </div>
              <div className="detail-row">
                <span>租约管理</span>
                <b>PostgreSQL 事务</b>
              </div>
              <div className="detail-row">
                <span>模型切换</span>
                <b>管理员手动</b>
              </div>
            </div>

            <hr style={{ border: "none", borderTop: "1px solid #d7e5ed", margin: "16px 0" }} />

            <div style={{ color: "var(--text-secondary)", fontSize: 11, lineHeight: 1.6 }}>
              <div style={{ marginBottom: 4, fontWeight: 600, color: "var(--text)" }}>
                说明
              </div>
              推理和训练共用一台 GPU 服务器。训练任务进入低优先级队列，不做运行中的抢占。
              模型驻留由管理员手动指定，平台不自动卸载其他模型。
              显存不足时训练失败并保留当前推理模型。
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
