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
  const latestEval = evals[0] || null;

  return (
    <>
      {/* Hero banner */}
      <div style={{ background: "linear-gradient(135deg, #0f2a47 0%, #1766ad 45%, #2d83c5 100%)", borderRadius: 8, padding: "10px 16px", display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, color: "#fff", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", right: -20, top: -30, width: 160, height: 160, background: "radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%)" }} />
        <div style={{ fontSize: 11, opacity: 0.85 }}>{models.length} 个模型节点 · {tasks.length} 次训练 · {evals.length} 次评估</div>
        <div style={{ fontSize: 10, opacity: 0.5 }}>{new Date().toLocaleDateString("zh-CN")}</div>
      </div>

      {/* KPI 4 */}
      <section style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 10 }}>
        {[
          { k: "候选模型", v: models.filter((n) => n.status === "candidate").length, sub: `共 ${models.length} 个`, icon: "◆", bg: "rgba(54,161,189,0.08)", bd: "#36a1bd" },
          { k: "运行中", v: running.length, sub: queued.length ? `${queued.length} 等待` : completed.length ? `${completed.length} 已完成` : "空闲", icon: "▶", bg: running.length ? "rgba(45,131,197,0.08)" : "rgba(0,0,0,0.02)", bd: running.length ? "#2d83c5" : "#d8e5ef" },
          { k: "已完成", v: completed.length, sub: completed.length ? "次训练" : "暂无完成", icon: "✓", bg: completed.length ? "rgba(26,138,117,0.08)" : "rgba(0,0,0,0.02)", bd: completed.length ? "#1a8a75" : "#d8e5ef" },
          { k: "评估", v: evals.length, sub: evals.filter((e: any) => e.autoStatus === "passed").length + " 通过", icon: "◎", bg: evals.length ? "rgba(139,92,246,0.06)" : "rgba(0,0,0,0.02)", bd: evals.length ? "#8b5cf6" : "#d8e5ef" },
        ].map(({ k, v, sub, icon, bg, bd }) => (
          <div key={k} className="grid-card" style={{ padding: "8px 12px", borderTop: `2px solid ${bd}`, background: `linear-gradient(180deg, ${bg} 0%, #fff 60%)` }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <span style={{ fontSize: 9, color: "var(--text-muted)", letterSpacing: "0.04em" }}>{k}</span>
              <span style={{ width: 18, height: 18, borderRadius: 4, background: bd, color: "#fff", display: "grid", placeItems: "center", fontSize: 9 }}>{icon}</span>
            </div>
            <div style={{ fontWeight: 700, fontSize: 18, lineHeight: 1 }}>{v}</div>
            <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 1 }}>{sub}</div>
          </div>
        ))}
      </section>

      {/* Main: Training + Right */}
      <section style={{ display: "grid", gridTemplateColumns: "1.25fr 0.85fr", gap: 8 }}>
        {/* Training */}
        <div className="card" style={{ padding: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div style={{ padding: "10px 14px", borderBottom: "1px solid #e6eef6", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ fontWeight: 600, fontSize: 11 }}>训练任务</div>
            <div style={{ fontSize: 9, color: "var(--text-muted)" }}>{running.length} 运行中 · {queued.length} 等待 · {completed.length} 完成</div>
          </div>
          <div style={{ flex: 1 }}>
            {tasks.length === 0 ? (
              <div style={{ padding: 36, textAlign: "center" }}>
                <div style={{ width: 36, height: 36, borderRadius: 8, background: "#f0f7fc", display: "grid", placeItems: "center", margin: "0 auto 8px", fontSize: 16, color: "var(--text-muted)" }}>▶</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)" }}>暂无训练，点击新建开始</div>
                <button className="btn primary small" onClick={() => navigate("/training/create")} style={{ marginTop: 10 }}>新建训练</button>
              </div>
            ) : tasks.slice(0, 5).map((t) => {
              const p = t.status === "running" ? 55 : t.status === "completed" ? 100 : 0;
              const pc = t.status === "running" ? "var(--primary)" : t.status === "completed" ? "var(--accent-green)" : t.status === "failed" ? "#b33" : "#d8e5ef";
              return (
                <div key={t.id} onClick={() => navigate("/training")} style={{ padding: "8px 14px", borderBottom: "1px solid #f0f4f8", cursor: "pointer" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                    <span style={{ fontWeight: 600, fontSize: 12, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", flex: 1 }}>{(t as any).parentModelName || t.id.slice(0, 8)}</span>
                    <span className={`badge ${t.status === "running" ? "badge-cyan" : t.status === "completed" ? "badge-green" : t.status === "failed" ? "badge-red" : "badge-orange"}`} style={{ fontSize: 9, flexShrink: 0, marginLeft: 8 }}>
                      {t.status === "running" ? "运行中" : t.status === "completed" ? "已完成" : t.status === "failed" ? "失败" : "排队中"}
                    </span>
                  </div>
                  <div style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "Courier New, monospace", marginBottom: 5 }}>{(t as any).datasetName} · {t.epochs} 轮</div>
                  <div style={{ height: 3, background: "#e6eef6", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${p}%`, background: pc, borderRadius: 2, transition: "width 0.4s" }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Eval + Ops */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {/* Latest eval */}
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "10px 14px", borderBottom: "1px solid #e6eef6", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontWeight: 600, fontSize: 11 }}>最新评估</div>
              <div style={{ fontSize: 9, color: "var(--text-muted)" }}>{latestEval ? "模型推理质量" : "暂无评估"}</div>
            </div>
            {(() => {
              const latest = evals[0] || null;
              if (!latest) return <div style={{ padding: 28, textAlign: "center", color: "var(--text-muted)", fontSize: 11 }}>暂无评估</div>;
              return (
                <div style={{ padding: 14 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                    <div style={{ width: 32, height: 32, borderRadius: 8, background: latest.metrics ? "rgba(26,138,117,0.12)" : "rgba(0,0,0,0.05)", display: "grid", placeItems: "center", fontSize: 14 }}>{latest.metrics ? "◆" : "○"}</div>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: 12 }}>{latest.modelName}</div>
                      <div style={{ fontSize: 9, color: "var(--text-muted)" }}>{latest.datasetName}</div>
                    </div>
                    <span className={`badge ${latest.metrics ? "badge-green" : (latest as any).autoStatus === "failed" ? "badge-red" : "badge-orange"}`} style={{ marginLeft: "auto", fontSize: 9 }}>
                      {latest.metrics ? `${(latest.metrics.mAP50 * 100).toFixed(1)}%` : (latest as any).autoStatus}
                    </span>
                  </div>
                  {latest.metrics ? (
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
                      <div style={{ background: "#f8fafc", border: "1px solid #e6eef6", borderRadius: 6, padding: "7px 8px" }}><div style={{ fontSize: 9, color: "var(--text-muted)" }}>mAP50</div><div style={{ fontSize: 12, fontWeight: 700 }}>{(latest.metrics.mAP50 * 100).toFixed(1)}%</div></div>
                      <div style={{ background: "#f8fafc", border: "1px solid #e6eef6", borderRadius: 6, padding: "7px 8px" }}><div style={{ fontSize: 9, color: "var(--text-muted)" }}>Precision</div><div style={{ fontSize: 12, fontWeight: 700 }}>{(latest.metrics.precision * 100).toFixed(1)}%</div></div>
                      <div style={{ background: "#f8fafc", border: "1px solid #e6eef6", borderRadius: 6, padding: "7px 8px" }}><div style={{ fontSize: 9, color: "var(--text-muted)" }}>Recall</div><div style={{ fontSize: 12, fontWeight: 700 }}>{(latest.metrics.recall * 100).toFixed(1)}%</div></div>
                    </div>
                  ) : <div style={{ fontSize: 11, color: "var(--text-muted)", padding: "6px 0" }}>等待评估</div>}

                </div>
              );
            })()}
          </div>

          {/* Ops */}
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "10px 14px", borderBottom: "1px solid #e6eef6", fontWeight: 600, fontSize: 11 }}>最近操作</div>
            {logs.length === 0 ? (
              <div style={{ padding: 24, textAlign: "center", color: "var(--text-muted)", fontSize: 11 }}>暂无记录</div>
            ) : logs.slice(0, 4).map((l: any) => (
              <div key={l.id} style={{ padding: "9px 18px", borderBottom: "1px solid #f0f4f8", display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 3, height: 16, borderRadius: 2, background: l.status === "success" ? "var(--accent-green)" : "#b33", flexShrink: 0 }} />
                <span style={{ flex: 1, fontSize: 11, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{l.operationType}</span>
                <span style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "Courier New, monospace" }}>{fmtTime(l.createdAt)}</span>
              </div>
            ))}

          </div>
        </div>
      </section>
    </>
  );
}
