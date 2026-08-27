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

const STATUS_COLOR: Record<string, string> = {
  pending: "#d77d59",
  auto_passed: "#36a1bd",
  passed: "#36a1bd",
  approved: "#1a8a75",
  rejected: "#c44",
  failed: "#c44",
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
  const pendingCount = evals.filter((e) => e.autoStatus === "pending").length;

  const statusBadge = (s: string) => s === "pending" ? "badge-orange" : s === "auto_passed" || s === "passed" ? "badge-blue" : s === "approved" ? "badge-green" : "badge-red";

  return (
    <>
      <div className="top-actions" style={{ marginBottom: 12, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} style={{ padding: "6px 10px", border: "1px solid #bed2df", borderRadius: 6, fontSize: 12 }}>
          <option value="all">全部状态</option>
          <option value="pending">待评估</option>
          <option value="passed">自动通过</option>
          <option value="failed">评估失败</option>
        </select>
        <input value={searchInput} onChange={(e) => setSearchInput(e.target.value)} placeholder="搜索 模型/数据集" style={{ flex: "1 1 200px", maxWidth: 320, padding: "6px 10px", border: "1px solid #dce8f0", borderRadius: 6, fontSize: 12 }} />
        {search && <button className="btn small" onClick={() => { setSearchInput(""); setSearch(""); }}>清空</button>}
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "var(--text-muted)", whiteSpace: "nowrap" }}>{filtered.length} 条 · {pendingCount} 个待人工验收</span>
      </div>

      <div className="card" style={{ overflow: "hidden" }}>
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 100px", gap: 0, padding: "10px 16px", borderBottom: "1px solid #dce8f0", background: "#f8fafc", fontSize: 11, fontWeight: 600, color: "var(--text-muted)" }}>
          <span>模型</span>
          <span>状态</span>
          <span>mAP50</span>
          <span>测试集</span>
          <button onClick={() => setSortDir((d) => d === "asc" ? "desc" : "asc")} style={{ border: 0, background: "none", cursor: "pointer", fontWeight: 600, color: "var(--text-muted)", textAlign: "left", fontSize: 11 }}>时间 {sortDir === "asc" ? "↑" : "↓"}</button>
          <span style={{ textAlign: "right" }}>操作</span>
        </div>
        {filtered.map((e) => {
          const sel = selectedId === e.id;
          const mAP = e.metrics?.mAP50 ?? (e as any).auto_metrics_json?.mAP50 ?? null;
          return (
            <div key={e.id} onClick={() => setSelectedId(sel ? null : e.id)} style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 100px", gap: 0, padding: "10px 16px", borderBottom: "1px solid #eef3f8", background: sel ? "#e7f1fa" : undefined, cursor: "pointer", alignItems: "center" }}>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 12, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{e.modelName}</div>
                <div style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "Courier New, monospace", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{e.modelNodeId.slice(0, 8)}</div>
              </div>
              <div><span className={`badge ${statusBadge(e.autoStatus)}`} style={{ fontSize: 9 }}>{STATUS_LABEL[e.autoStatus] || e.autoStatus}</span></div>
              <div style={{ fontSize: 11, fontWeight: mAP !== null ? 600 : 400, color: mAP !== null ? "var(--text)" : "var(--text-muted)" }}>{mAP !== null ? (mAP * 100).toFixed(1) + "%" : "—"}</div>
              <div style={{ fontSize: 11 }}>{e.testImageCount} 张</div>
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

            {/* Header */}
            <div style={{ padding: "16px 20px", borderBottom: "1px solid #dce8f0", background: "#f8fafc" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{current.modelName}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", display: "flex", gap: 8, alignItems: "center" }}>
                    <span>{current.datasetName}</span>
                    <span style={{ width: 1, height: 10, background: "#dce8f0" }} />
                    <span>测试集 {current.testImageCount} 张</span>
                    <span style={{ width: 1, height: 10, background: "#dce8f0" }} />
                    <span>{formatTime(current.createdAt)}</span>
                  </div>
                </div>
                <button onClick={() => setSelectedId(null)} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 18, color: "var(--text-muted)", padding: "0 4px", lineHeight: 1 }}>×</button>
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
                <div style={{ flex: 1, background: "#fff", border: "1px solid #e6eef6", borderRadius: 6, padding: "8px 12px", display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ width: 8, height: 8, borderRadius: 4, background: STATUS_COLOR[current.autoStatus] || "#999", flexShrink: 0 }} />
                  <div>
                    <div style={{ fontSize: 9, color: "var(--text-muted)" }}>评估状态</div>
                    <div style={{ fontSize: 12, fontWeight: 600 }}>{STATUS_LABEL[current.autoStatus]}</div>
                  </div>
                </div>
                <div style={{ flex: 1, background: "#fff", border: "1px solid #e6eef6", borderRadius: 6, padding: "8px 12px", display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ width: 8, height: 8, borderRadius: 4, background: "#999", flexShrink: 0 }} />
                  <div>
                    <div style={{ fontSize: 9, color: "var(--text-muted)" }}>测试集</div>
                    <div style={{ fontSize: 12, fontWeight: 600 }}>{current.testImageCount} 张</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Body */}
            <div style={{ flex: 1, overflowY: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>

              {/* Model Info */}
              <div style={{ background: "#fff", border: "1px solid #e6eef6", borderRadius: 8, padding: "12px 14px" }}>
                <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ width: 3, height: 14, background: "var(--primary)", borderRadius: 2 }} />
                  模型信息
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "80px 1fr", gap: "6px 12px", fontSize: 11 }}>
                  <span style={{ color: "var(--text-muted)" }}>模型 ID</span>
                  <span style={{ fontFamily: "Courier New, monospace", fontSize: 9, color: "var(--text-secondary)" }}>{current.modelNodeId}</span>
                  <span style={{ color: "var(--text-muted)" }}>模型名称</span>
                  <span style={{ fontWeight: 600 }}>{current.modelName}</span>
                  <span style={{ color: "var(--text-muted)" }}>评估策略</span>
                  <span style={{ fontSize: 10, color: "var(--text-secondary)" }}>阈值已配置</span>
                  <span style={{ color: "var(--text-muted)" }}>数据集</span>
                  <span style={{ fontWeight: 600 }}>{current.datasetName}</span>
                  <span style={{ color: "var(--text-muted)" }}>创建时间</span>
                  <span>{new Date(current.createdAt).toLocaleString("zh-CN")}</span>
                </div>
              </div>

              {/* Metrics */}
              {current.metrics ? (
                <div style={{ background: "#fff", border: "1px solid #e6eef6", borderRadius: 8, padding: "12px 14px" }}>
                  <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 3, height: 14, background: "var(--primary)", borderRadius: 2 }} />
                    评估指标
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8 }}>
                    {([
                      ["Precision", (current.metrics.precision * 100), "#1766ad"],
                      ["Recall", (current.metrics.recall * 100), "#1a8a75"],
                      ["mAP50", (current.metrics.mAP50 * 100), "#36a1bd"],
                      ["mAP50-95", (current.metrics.mAP50_95 * 100), "#d77d59"],
                    ] as [string, number, string][]).map(([label, val, color]) => (
                      <div key={label} style={{ background: "#f8fafc", borderRadius: 6, padding: "8px 10px", borderTop: `3px solid ${color}` }}>
                        <div style={{ fontSize: 9, color: "var(--text-muted)", marginBottom: 2 }}>{label}</div>
                        <div style={{ fontSize: 13, fontWeight: 700 }}>{val.toFixed(1)}%</div>
                        <div style={{ marginTop: 4, height: 3, background: "#e6eef6", borderRadius: 2, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${Math.min(val, 100)}%`, background: color, borderRadius: 2 }} />
                        </div>
                      </div>
                    ))}
                  </div>
                  {(current as any).testImageCount > 0 && (
                    <div style={{ marginTop: 8, fontSize: 10, color: "var(--text-muted)", display: "flex", gap: 16 }}>
                      <span>测试集: {(current as any).testImageCount} 张</span>
                      <span>检测框: {(current as any).metrics?.numPredictions ?? (current as any).auto_metrics_json?.num_predictions ?? "—"}</span>
                      <span>标注框: {(current as any).metrics?.numGroundTruths ?? (current as any).auto_metrics_json?.num_ground_truths ?? "—"}</span>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ background: "#fff", border: "1px solid #e6eef6", borderRadius: 8, padding: "12px 14px" }}>
                  <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 3, height: 14, background: "var(--text-muted)", borderRadius: 2 }} />
                    评估指标
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", padding: "8px 0" }}>{(current as any).autoStatus === "failed" ? "评估失败，无指标数据" : "等待自动评估"}</div>
                </div>
              )}

              {/* Error */}
              {current.lastError && (
                <div style={{ background: "#fff5f5", border: "1px solid #fcc", borderRadius: 8, padding: "10px 14px" }}>
                  <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 6, color: "#c44", display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 3, height: 14, background: "#c44", borderRadius: 2 }} />
                    错误
                  </div>
                  <div style={{ fontSize: 11, color: "#833", fontFamily: "Courier New, monospace", wordBreak: "break-all" }}>{current.lastError}</div>
                </div>
              )}

              {/* Actions */}

            </div>
          </div>
        </>
      )}
    </>
  );
}
