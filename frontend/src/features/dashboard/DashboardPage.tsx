import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { ModelNode, TrainingTask, Evaluation } from "../../lib/api";

function fmtTime(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export default function DashboardPage() {
  const api = useMemo(() => createApi(), []);
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelNode[]>([]);
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [evals, setEvals] = useState<Evaluation[]>([]);
  const [logs, setLogs] = useState<any[]>([]);

  useEffect(() => {
    Promise.all([
      api.getModelNodes().catch(() => []),
      api.getTrainingTasks().catch(() => []),
      api.getEvaluations().catch(() => []),
      fetch("/api/operations/logs?limit=5", { headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" } }).then((r) => r.json()).catch(() => ({ items: [] })),
    ]).then(([m, t, e, l]) => { setModels(m); setTasks(t); setEvals(e); setLogs(l.items || []); });
  }, [api]);

  const running = tasks.filter((t) => t.status === "running");
  const completed = tasks.filter((t) => t.status === "completed");
  const queued = tasks.filter((t) => t.status === "queued");

  return (
    <>
      <div className="hero-banner">
        <div className="hero-glow" />
        <div className="hero-stats">{models.length} 个模型节点 · {tasks.length} 次训练 · {evals.length} 次评估</div>
        <div className="hero-date">{new Date().toLocaleDateString("zh-CN")}</div>
      </div>

      <section className="kpi-grid">
        {[
          { k: "候选模型", v: models.filter((n) => n.status === "candidate").length, sub: `共 ${models.length} 个`, icon: "◆", bg: "rgba(54,161,189,0.08)", bd: "#36a1bd" },
          { k: "运行中", v: running.length, sub: queued.length ? `${queued.length} 等待` : completed.length ? `${completed.length} 已完成` : "空闲", icon: "▶", bg: running.length ? "rgba(45,131,197,0.08)" : "rgba(0,0,0,0.02)", bd: running.length ? "#2d83c5" : "#d8e5ef" },
          { k: "已完成", v: completed.length, sub: completed.length ? "次训练" : "暂无完成", icon: "✓", bg: completed.length ? "rgba(26,138,117,0.08)" : "rgba(0,0,0,0.02)", bd: completed.length ? "#1a8a75" : "#d8e5ef" },
          { k: "评估", v: evals.length, sub: evals.filter((e: any) => e.autoStatus === "passed").length + " 通过", icon: "◎", bg: evals.length ? "rgba(139,92,246,0.06)" : "rgba(0,0,0,0.02)", bd: evals.length ? "#8b5cf6" : "#d8e5ef" },
        ].map(({ k, v, sub, icon, bg, bd }) => (
          <div key={k} className="grid-card kpi-card">
            <div className="kpi-top">
              <span className="kpi-label">{k}</span>
              <span className="kpi-icon" style={{ background: bd }}>{icon}</span>
            </div>
            <div className="kpi-value">{v}</div>
            <div className="kpi-sub">{sub}</div>
          </div>
        ))}
      </section>

      <section className="dashboard-main">
        <div className="card" style={{ padding: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div className="card-section-header">
            <div className="card-section-title">训练任务</div>
            <div className="card-section-hint">{running.length} 运行中 · {queued.length} 等待 · {completed.length} 完成</div>
          </div>
          <div style={{ flex: 1 }}>
            {tasks.length === 0 ? (
              <div className="empty-placeholder">
                <div className="empty-icon">{"▶"}</div>
                <div className="empty-text">暂无训练，点击新建开始</div>
                <button className="btn primary small" onClick={() => navigate("/training/create")} style={{ marginTop: 10 }}>新建训练</button>
              </div>
            ) : tasks.slice(0, 5).map((t) => {
              const p = t.status === "running" ? 55 : t.status === "completed" ? 100 : 0;
              const pc = t.status === "running" ? "var(--primary)" : t.status === "completed" ? "var(--accent-green)" : t.status === "failed" ? "#b33" : "#d8e5ef";
              return (
                <div key={t.id} className="training-item" onClick={() => navigate("/training")}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                    <span className="training-name">{(t as any).trainingConfigJson?.modelName || (t as any).parentModelName || t.id.slice(0, 8)}</span>
                    <span className={`badge badge-xs ${t.status === "running" ? "badge-cyan" : t.status === "completed" ? "badge-green" : t.status === "failed" ? "badge-red" : "badge-orange"}`} style={{ flexShrink: 0, marginLeft: 8 }}>
                      {t.status === "running" ? "运行中" : t.status === "completed" ? "已完成" : t.status === "failed" ? "失败" : "排队中"}
                    </span>
                  </div>
                  <div className="training-meta">{(t as any).datasetName} · {t.epochs} 轮</div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${p}%`, background: pc }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="dashboard-right">
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div className="card-section-header">
              <div className="card-section-title">最新评估</div>
              <div className="card-section-hint">{evals[0] ? "模型推理质量" : "暂无评估"}</div>
            </div>
            {(() => {
              const latest = evals[0] || null;
              if (!latest) return <div className="empty-msg-sm">暂无评估</div>;
              return (
                <div className="eval-summary">
                  <div className="eval-summary-row">
                    <div className="eval-summary-icon" style={{ background: latest.metrics ? "rgba(26,138,117,0.12)" : "rgba(0,0,0,0.05)" }}>{latest.metrics ? "◆" : "○"}</div>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: 12 }}>{latest.modelName}</div>
                      <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{latest.datasetName}</div>
                    </div>
                    <span className={`badge badge-xs ${latest.metrics ? "badge-green" : (latest as any).autoStatus === "failed" ? "badge-red" : "badge-orange"}`} style={{ marginLeft: "auto" }}>
                      {latest.metrics ? `${(latest.metrics.mAP50 * 100).toFixed(1)}%` : (latest as any).autoStatus}
                    </span>
                  </div>
                  {latest.metrics ? (
                    <div className="metric-mini-grid">
                      {[["mAP50", latest.metrics.mAP50], ["Precision", latest.metrics.precision], ["Recall", latest.metrics.recall]].map(([k, v]) => (
                        <div key={k} className="metric-mini">
                          <div className="metric-mini-label">{k}</div>
                          <div className="metric-mini-value">{((v as number) * 100).toFixed(1)}%</div>
                        </div>
                      ))}
                    </div>
                  ) : <div className="cell-text-muted" style={{ padding: "6px 0" }}>等待评估</div>}
                </div>
              );
            })()}
          </div>

          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div className="card-section-header">
              <div className="card-section-title">最近操作</div>
            </div>
            {logs.length === 0 ? (
              <div className="empty-msg-sm">暂无记录</div>
            ) : logs.slice(0, 4).map((l: any) => (
              <div key={l.id} className="ops-item">
                <div className="ops-bar" style={{ background: l.status === "success" ? "var(--accent-green)" : "#b33" }} />
                <span className="ops-text">{l.operationType}</span>
                <span className="ops-time">{fmtTime(l.createdAt)}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
