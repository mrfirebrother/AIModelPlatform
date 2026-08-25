import { useEffect, useMemo, useState } from "react";
import { createApi } from "../../lib/createApi";
import type { Evaluation } from "../../lib/api";

const STATUS_LABEL: Record<string, string> = {
  pending: "待处理",
  auto_passed: "自动通过",
  passed: "自动通过",
  approved: "已通过",
  rejected: "已拒绝",
  failed: "已拒绝",
};
const STATUS_BADGE: Record<string, string> = {
  pending: "badge-orange",
  auto_passed: "badge-blue",
  passed: "badge-blue",
  approved: "badge-green",
  rejected: "badge-red",
  failed: "badge-red",
};

function formatTime(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${mm}-${dd} ${hh}:${mi}`;
}

export default function EvaluationsPage() {
  const api = useMemo(() => createApi(), []);
  const [evals, setEvals] = useState<Evaluation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    const t = setTimeout(() => setSearch(searchInput.trim().toLowerCase()), 300);
    return () => clearTimeout(t);
  }, [searchInput]);

  const load = async () => {
    try { setEvals(await api.getEvaluations()); } catch {}
  };
  useEffect(() => { load(); }, [api]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setSelectedId(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const handleApprove = async (id: string) => {
    await api.reviewEvaluation(id, "approved");
    setEvals((prev) => prev.map((e) => e.id === id ? { ...e, humanStatus: "approved" as any } : e));
  };
  const handleReject = async (id: string) => {
    await api.reviewEvaluation(id, "rejected");
    setEvals((prev) => prev.map((e) => e.id === id ? { ...e, humanStatus: "rejected" as any } : e));
  };

  const filtered = useMemo(() => {
    let list = [...evals];
    if (statusFilter !== "all") list = list.filter((e) => e.autoStatus === statusFilter || (e as any).humanStatus === statusFilter);
    if (search) list = list.filter((e) => e.modelName.toLowerCase().includes(search) || e.modelNodeId.toLowerCase().includes(search) || e.datasetName.toLowerCase().includes(search));
    list.sort((a, b) => {
      const ta = new Date(a.createdAt).getTime();
      const tb = new Date(b.createdAt).getTime();
      return sortDir === "asc" ? ta - tb : tb - ta;
    });
    return list;
  }, [evals, statusFilter, search, sortDir]);

  const current = evals.find((e) => e.id === selectedId) || null;
  const pendingCount = evals.filter((e) => (e as any).humanStatus === "pending").length;

  const statusBadge = (s: string) => s === "pending" ? "badge-orange" : s === "auto_passed" || s === "passed" ? "badge-blue" : s === "approved" ? "badge-green" : "badge-red";

  return (
    <>
      <div className="top-actions" style={{ marginBottom: 12, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} style={{ padding: "6px 10px", border: "1px solid #bed2df", borderRadius: 6, fontSize: 12 }}>
          <option value="all">全部状态</option>
          <option value="pending">待处理</option>
          <option value="passed">自动通过</option>
          <option value="approved">已通过</option>
          <option value="rejected">已拒绝</option>
        </select>
        <input value={searchInput} onChange={(e) => setSearchInput(e.target.value)} placeholder="搜索 模型/数据集" style={{ flex: "1 1 200px", maxWidth: 320, padding: "6px 10px", border: "1px solid #dce8f0", borderRadius: 6, fontSize: 12 }} />
        {search && <button className="btn small" onClick={() => { setSearchInput(""); setSearch(""); }}>清空</button>}
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "var(--text-muted)", whiteSpace: "nowrap" }}>{filtered.length} 条 · {pendingCount} 个待人工验收</span>
      </div>

      <div className="card" style={{ overflow: "hidden" }}>
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 100px", gap: 0, padding: "10px 16px", borderBottom: "1px solid #dce8f0", background: "#f8fafc", fontSize: 11, fontWeight: 600, color: "var(--text-muted)" }}>
          <span>模型</span>
          <span>自动评估</span>
          <span>人工验收</span>
          <span>数据集</span>
          <button onClick={() => setSortDir((d) => d === "asc" ? "desc" : "asc")} style={{ border: 0, background: "none", cursor: "pointer", fontWeight: 600, color: "var(--text-muted)", textAlign: "left", fontSize: 11 }}>时间 {sortDir === "asc" ? "↑" : "↓"}</button>
          <span style={{ textAlign: "right" }}>操作</span>
        </div>
        {filtered.map((e) => {
          const sel = selectedId === e.id;
          return (
            <div key={e.id} onClick={() => setSelectedId(sel ? null : e.id)} style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 100px", gap: 0, padding: "10px 16px", borderBottom: "1px solid #eef3f8", background: sel ? "#e7f1fa" : undefined, cursor: "pointer", alignItems: "center" }}>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 12, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{e.modelName}</div>
                <div style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "Courier New, monospace", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{e.modelNodeId.slice(0, 8)}</div>
              </div>
              <div><span className={`badge ${statusBadge(e.autoStatus)}`} style={{ fontSize: 9 }}>{STATUS_LABEL[e.autoStatus] || e.autoStatus}</span></div>
              <div><span className={`badge ${statusBadge((e as any).humanStatus)}`} style={{ fontSize: 9 }}>{STATUS_LABEL[(e as any).humanStatus] || (e as any).humanStatus}</span></div>
              <div style={{ fontSize: 11, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{e.datasetName}</div>
              <div style={{ fontSize: 11 }}>{formatTime(e.createdAt)}</div>
              <div style={{ textAlign: "right" }}><button className="btn small" onClick={(ev) => { ev.stopPropagation(); setSelectedId(e.id); }} style={{ padding: "2px 8px", fontSize: 11 }}>查看</button></div>
            </div>
          );
        })}
        {filtered.length === 0 && <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>{evals.length === 0 ? "暂无评估" : <>无匹配 <button className="btn small" onClick={() => { setStatusFilter("all"); setSearchInput(""); setSearch(""); }} style={{ marginLeft: 8 }}>清空筛选</button></>}</div>}
      </div>

      {current && (
        <>
          <div onClick={() => setSelectedId(null)} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.18)", zIndex: 20 }} />
          <div style={{ position: "fixed", right: 0, top: 0, bottom: 0, width: 480, maxWidth: "92vw", background: "#fff", zIndex: 30, boxShadow: "-8px 0 24px rgba(0,0,0,0.12)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid #dce8f0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <div style={{ fontWeight: 700, fontSize: 13 }}>{current.modelName} · <span className={`badge ${statusBadge(current.autoStatus)}`} style={{ fontSize: 9 }}>{STATUS_LABEL[current.autoStatus]}</span></div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{current.datasetName} · {current.testImageCount} 张</div>
              </div>
              <button className="btn small" onClick={() => setSelectedId(null)}>×</button>
            </div>
            <div style={{ flex: 1, overflowY: "auto", padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ background: "#fff", border: "1px solid #e6eef6", borderRadius: 8, padding: 10 }}>
                <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 6 }}>概览</div>
                <div style={{ fontSize: 11, lineHeight: 1.6 }}>
                  <div>模型：<b style={{ fontFamily: "Courier New, monospace", fontSize: 9 }}>{current.modelNodeId}</b></div>
                  <div>数据集：<b>{current.datasetName}</b> · 测试 {current.testImageCount} 张</div>
                  <div>自动：<span className={`badge ${statusBadge(current.autoStatus)}`}>{STATUS_LABEL[current.autoStatus]}</span> · 人工：<span className={`badge ${statusBadge((current as any).humanStatus)}`}>{STATUS_LABEL[(current as any).humanStatus]}</span></div>
                  <div>创建：{new Date(current.createdAt).toLocaleString("zh-CN")}</div>
                </div>
              </div>
              {current.metrics && (
                <div style={{ background: "#fff", border: "1px solid #e6eef6", borderRadius: 8, padding: 10 }}>
                  <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 6 }}>指标</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                    <div style={{ background: "#f8fafc", border: "1px solid #e6eef6", borderRadius: 6, padding: "8px 10px" }}><div style={{ fontSize: 9, color: "var(--text-muted)" }}>Precision</div><div style={{ fontSize: 12, fontWeight: 600 }}>{(current.metrics.precision * 100).toFixed(1)}%</div></div>
                    <div style={{ background: "#f8fafc", border: "1px solid #e6eef6", borderRadius: 6, padding: "8px 10px" }}><div style={{ fontSize: 9, color: "var(--text-muted)" }}>Recall</div><div style={{ fontSize: 12, fontWeight: 600 }}>{(current.metrics.recall * 100).toFixed(1)}%</div></div>
                    <div style={{ background: "#f8fafc", border: "1px solid #e6eef6", borderRadius: 6, padding: "8px 10px" }}><div style={{ fontSize: 9, color: "var(--text-muted)" }}>mAP50</div><div style={{ fontSize: 12, fontWeight: 600 }}>{(current.metrics.mAP50 * 100).toFixed(1)}%</div></div>
                    <div style={{ background: "#f8fafc", border: "1px solid #e6eef6", borderRadius: 6, padding: "8px 10px" }}><div style={{ fontSize: 9, color: "var(--text-muted)" }}>mAP50-95</div><div style={{ fontSize: 12, fontWeight: 600 }}>{(current.metrics.mAP50_95 * 100).toFixed(1)}%</div></div>
                  </div>
                </div>
              )}
              {(current as any).humanStatus === "pending" && (
                <div style={{ display: "flex", gap: 8 }}>
                  <button className="btn primary" onClick={() => handleApprove(current.id)}>通过验收</button>
                  <button className="btn danger" onClick={() => handleReject(current.id)}>拒绝</button>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </>
  );
}
