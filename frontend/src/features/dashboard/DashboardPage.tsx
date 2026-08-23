import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { KpiData, TrainingTask, ResourceStatus, GpuModelEntry } from "../../lib/api";

export default function DashboardPage() {
  const api = useMemo(() => createApi(), []);
  const navigate = useNavigate();
  const [kpis, setKpis] = useState<KpiData | null>(null);
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [resources, setResources] = useState<ResourceStatus | null>(null);
  const [gpuModels, setGpuModels] = useState<GpuModelEntry[]>([]);

  useEffect(() => {
    Promise.all([api.getKpis(), api.getTrainingTasks(), api.getResourceStatus(), api.getGpuModels()])
      .then(([k, t, r, m]) => { setKpis(k); setTasks(t); setResources(r); setGpuModels(m); })
      .catch(() => {});
  }, [api]);

  if (!kpis || !resources) return <div className="empty-state">加载中...</div>;

  const running = tasks.filter((t) => t.status === "running");
  const queued = tasks.filter((t) => t.status === "queued");

  return (
    <>
      <section className="kpis">
        <div className="kpi">
          <div className="kpi-label">当前模型关联</div>
          <strong>
            {String(kpis.bindingCount).padStart(2, "0")}
            <em>本月 +{kpis.bindingDelta}</em>
          </strong>
          <div className="kpi-foot">{kpis.bindingCount} 个服务中</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">驻留模型</div>
          <strong>
            {String(kpis.residentModels).padStart(2, "0")}
            <em>GPU 占用 {kpis.gpuUsage}%</em>
          </strong>
          <div className="kpi-foot">管理员手动维护</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">训练队列</div>
          <strong>
            {String(kpis.trainingQueue + kpis.trainingRunning).padStart(2, "0")}
            <em>{kpis.trainingRunning} 个运行中</em>
          </strong>
          <div className="kpi-foot">下一个资源窗口 00:42:18</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">待人工评估</div>
          <strong>
            {String(kpis.pendingEval).padStart(2, "0")}
            <em>需要处理</em>
          </strong>
          <div className="kpi-foot">候选模型等待验收</div>
        </div>
      </section>

      <section className="page-grid">
        {/* GPU Resources */}
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">GPU 资源</div>
              <div className="card-kicker">管理员手动管理驻留</div>
            </div>
            <span className="online">健康</span>
          </div>
          <div className="card-body">
            <div className="meter-label">
              <span>显存 / {resources.gpuDevice}</span>
              <b>
                {kpis.gpuUsedMb > 0
                  ? `${(kpis.gpuUsedMb / 1024).toFixed(1)} / ${(kpis.gpuTotalMb / 1024).toFixed(1)} GB`
                  : `${resources.usedMemory} / ${resources.totalMemory} GB`}
              </b>
            </div>
            <div className="meter">
              <div
                className="meter-fill"
                style={{ width: `${kpis.gpuUsage}%` }}
              />
            </div>
            <div style={{ marginTop: 12 }}>
              <div className="meter-label">
                <span>推理预留</span>
                <b>{resources.inferenceReserved} GB</b>
              </div>
              <div className="meter">
                <div
                  className="meter-fill cyan"
                  style={{ width: `${(resources.inferenceReserved / resources.totalMemory) * 100}%` }}
                />
              </div>
            </div>
            <div style={{ marginTop: 12 }}>
              <div className="meter-label">
                <span>训练预留</span>
                <b>{resources.trainingReserved} GB</b>
              </div>
              <div className="meter">
                <div
                  className="meter-fill orange"
                  style={{ width: `${(resources.trainingReserved / resources.totalMemory) * 100}%` }}
                />
              </div>
            </div>
            <div className="detail-row" style={{ marginTop: 12 }}>
              <span>驻留集合</span>
              <b>{gpuModels.length} 个模型 / 手动</b>
            </div>
          </div>
        </div>

        {/* Loaded Models */}
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">已加载模型</div>
              <div className="card-kicker">{gpuModels.length} 个驻留</div>
            </div>
            <button className="card-action" onClick={() => navigate("/resources")}>
              管理
            </button>
          </div>
          <div>
            {gpuModels.length === 0 ? (
              <div style={{ padding: "12px 16px", color: "var(--text-muted)", fontSize: 11 }}>
                当前无驻留模型
              </div>
            ) : (
              gpuModels.slice(0, 4).map((m) => (
                <div
                  key={m.gpuDevice}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr auto",
                    gap: 10,
                    padding: "10px 16px",
                    borderBottom: "1px solid #dce8f0",
                  }}
                >
                  <div>
                    <div style={{ color: "var(--text)", fontSize: 12 }}>
                      {m.modelName}
                    </div>
                    <div
                      style={{
                        marginTop: 3,
                        color: "var(--text-muted)",
                        fontSize: 9,
                        fontFamily: "Courier New, monospace",
                      }}
                    >
                      {m.gpuDevice} / {(m.memoryMb / 1024).toFixed(1)} GB
                    </div>
                  </div>
                  <span className="online" style={{ fontSize: 9, alignSelf: "center" }}>
                    已加载
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Training Queue */}
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">训练队列</div>
              <div className="card-kicker">
                {running.length} 个运行中 · {queued.length} 个等待
              </div>
            </div>
            <button className="card-action" onClick={() => navigate("/training")}>
              查看全部
            </button>
          </div>
          <div>
            {tasks.slice(0, 4).map((t) => (
              <div
                key={t.id}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto",
                  gap: 10,
                  padding: "10px 16px",
                  borderBottom: "1px solid #dce8f0",
                }}
              >
                <div>
                  <div style={{ color: "var(--text)", fontSize: 12 }}>
                    {t.parentModelName || t.id}
                  </div>
                  <div
                    style={{
                      marginTop: 3,
                      color: "var(--text-muted)",
                      fontSize: 9,
                      fontFamily: "Courier New, monospace",
                    }}
                  >
                    {t.status === "running"
                      ? `第 ${t.currentEpoch} / ${t.epochs} 轮`
                      : t.status === "queued"
                        ? "等待 GPU"
                        : t.status === "completed"
                          ? "已完成"
                          : "失败"}
                  </div>
                </div>
                <div
                  style={{
                    fontSize: 9,
                    fontFamily: "Courier New, monospace",
                    textTransform: "uppercase",
                    color:
                      t.status === "running"
                        ? "var(--accent-cyan)"
                        : t.status === "queued"
                          ? "var(--accent-orange)"
                          : t.status === "completed"
                            ? "var(--accent-green)"
                            : "#b33",
                    alignSelf: "center",
                  }}
                >
                  {t.status === "running"
                    ? "运行中"
                    : t.status === "queued"
                      ? "排队中"
                      : t.status === "completed"
                        ? "已完成"
                        : "失败"}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
