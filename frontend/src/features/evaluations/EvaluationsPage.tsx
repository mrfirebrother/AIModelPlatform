import { useEffect, useMemo, useRef, useState } from "react";
import { createApi } from "../../lib/createApi";
import LoadingSpinner from "../../lib/LoadingSpinner";
import type { Evaluation } from "../../lib/api";

const STATUS_LABEL: Record<string, string> = { pending: "待处理", auto_passed: "自动通过", passed: "自动通过", approved: "已通过", rejected: "已拒绝", failed: "已拒绝" };
const STATUS_BADGE: Record<string, string> = { pending: "badge-orange", auto_passed: "badge-blue", passed: "badge-blue", approved: "badge-green", rejected: "badge-red", failed: "badge-red" };
const STATUS_COLOR: Record<string, string> = { pending: "#d77d59", auto_passed: "#36a1bd", passed: "#36a1bd", approved: "#1a8a75", rejected: "#c44", failed: "#c44" };

function formatTime(iso?: string): string { if (!iso) return "—"; const d = new Date(iso); return `${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")} ${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}`; }

export default function EvaluationsPage() {
  const api = useMemo(() => createApi(), []);
  const [evals, setEvals] = useState<Evaluation[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [inferLoading, setInferLoading] = useState(false);
  const [inferResult, setInferResult] = useState<any>(null);
  const [inferError, setInferError] = useState<string | null>(null);
  const [inferImageUrl, setInferImageUrl] = useState<string | null>(null);

  useEffect(() => { const t = setTimeout(() => setSearch(searchInput.trim().toLowerCase()), 300); return () => clearTimeout(t); }, [searchInput]);
  const load = async () => { setLoading(true); try { setEvals(await api.getEvaluations()); } catch {} setLoading(false); };
  useEffect(() => { load(); }, [api]);
  useEffect(() => { const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setSelectedId(null); }; window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey); }, []);

  const filtered = useMemo(() => {
    let list = [...evals];
    if (statusFilter !== "all") list = list.filter((e) => e.autoStatus === statusFilter || (e as any).humanStatus === statusFilter);
    if (search) list = list.filter((e) => e.modelName.toLowerCase().includes(search) || e.modelNodeId.toLowerCase().includes(search) || e.datasetName.toLowerCase().includes(search));
    list.sort((a, b) => { const ta = new Date(a.createdAt).getTime(); const tb = new Date(b.createdAt).getTime(); return sortDir === "asc" ? ta - tb : tb - ta; });
    return list;
  }, [evals, statusFilter, search, sortDir]);

  const handleInfer = async (file: File) => {
    if (!selectedId) return;
    setInferLoading(true); setInferError(null); setInferResult(null);
    const url = URL.createObjectURL(file); setInferImageUrl(url);
    try { setInferResult(await api.inferWithModel(selectedId, file)); } catch (err) { setInferError((err as Error).message); }
    setInferLoading(false);
  };

  const current = evals.find((e) => e.id === selectedId) || null;
  const pendingCount = evals.filter((e) => e.autoStatus === "pending").length;
  const statusBadge = (s: string) => s === "pending" ? "badge-orange" : s === "auto_passed" || s === "passed" ? "badge-blue" : s === "approved" ? "badge-green" : "badge-red";

  function InferResultCanvas({ result, imageUrl }: { result: any; imageUrl: string }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    useEffect(() => {
      const canvas = canvasRef.current; if (!canvas || !imageUrl) return;
      const img = new Image();
      img.onload = () => {
        const maxW = canvas.parentElement?.clientWidth || 448; const maxH = 360;
        let scale = Math.min(1, maxW / img.width, maxH / img.height); scale = Math.max(scale, 0.15);
        canvas.width = Math.round(img.width * scale); canvas.height = Math.round(img.height * scale);
        const ctx = canvas.getContext("2d"); if (!ctx) return;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        const COLORS = ["#36a1bd", "#1a8a75", "#d77d59", "#1766ad", "#c44", "#8b5cf6"];
        for (const det of (result.detections || [])) {
          const [x1, y1, x2, y2] = det.bbox;
          const sx = (x1 / result.image_width) * canvas.width; const sy = (y1 / result.image_height) * canvas.height;
          const sw = ((x2 - x1) / result.image_width) * canvas.width; const sh = ((y2 - y1) / result.image_height) * canvas.height;
          const color = COLORS[det.class_id % COLORS.length];
          ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.strokeRect(sx, sy, sw, sh);
          const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
          ctx.font = "bold 10px sans-serif"; const tw = ctx.measureText(label).width;
          ctx.fillStyle = color; ctx.fillRect(sx, sy - 14, tw + 6, 14);
          ctx.fillStyle = "#fff"; ctx.fillText(label, sx + 3, sy - 3);
        }
      };
      img.src = imageUrl;
    }, [result, imageUrl]);
    return <div className="infer-canvas-wrap"><canvas ref={canvasRef} /></div>;
  }

  return (
    <>
      <div className="toolbar">
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className="input-toolbar">
          <option value="all">全部状态</option>
          <option value="pending">待评估</option>
          <option value="passed">自动通过</option>
          <option value="failed">评估失败</option>
        </select>
        <input value={searchInput} onChange={(e) => setSearchInput(e.target.value)} placeholder="搜索 模型/数据集" className="input-search" />
        {search && <button className="btn small" onClick={() => { setSearchInput(""); setSearch(""); }}>清空</button>}
        <div className="toolbar-spacer" />
        <span className="toolbar-hint">{filtered.length} 条 {"·"} {pendingCount} 个待人工验收</span>
      </div>

      <div className="card" style={{ overflow: "hidden" }}>
        {loading ? (
          <LoadingSpinner text="加载评估列表..." />
        ) : (<>
        <div className="grid-table-header" style={{ gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 100px" }}>
          <span>模型</span><span>状态</span><span>mAP50</span><span>测试集</span>
          <button className="sort-btn" onClick={() => setSortDir((d) => d === "asc" ? "desc" : "asc")}>时间 {sortDir === "asc" ? "↑" : "↓"}</button>
          <span className="cell-actions">操作</span>
        </div>
        {filtered.map((e) => {
          const sel = selectedId === e.id;
          const mAP = e.metrics?.mAP50 ?? (e as any).auto_metrics_json?.mAP50 ?? null;
          return (
            <div key={e.id} onClick={() => setSelectedId(sel ? null : e.id)} className={`grid-table-row ${sel ? "grid-table-row-selected" : ""}`} style={{ gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr 100px" }}>
              <div style={{ minWidth: 0 }}>
                <div className="cell-primary">{e.modelName}</div>
                <div className="cell-mono">{e.modelNodeId.slice(0, 8)}</div>
              </div>
              <div><span className={`badge badge-xs ${statusBadge(e.autoStatus)}`}>{STATUS_LABEL[e.autoStatus] || e.autoStatus}</span></div>
              <div className="cell-text" style={{ fontWeight: mAP !== null ? 600 : 400, color: mAP !== null ? "var(--text)" : "var(--text-muted)", fontFamily: "Courier New, monospace" }}>{mAP !== null ? (mAP * 100).toFixed(1) + "%" : "—"}</div>
              <div className="cell-text">{e.testImageCount} 张</div>
              <div className="cell-text">{formatTime(e.createdAt)}</div>
              <div className="cell-actions"><button className="btn small btn-inline-view" onClick={(ev) => { ev.stopPropagation(); setSelectedId(e.id); }}>查看</button></div>
            </div>
          );
        })}
        {filtered.length === 0 && <div className="empty-msg">{evals.length === 0 ? "暂无评估" : <>{"无匹配"} <button className="btn small" onClick={() => { setStatusFilter("all"); setSearchInput(""); setSearch(""); }} style={{ marginLeft: 8 }}>清空筛选</button></>}</div>}
        </>)}
      </div>

      {current && (
        <>
          <div className="drawer-overlay" onClick={() => setSelectedId(null)} />
          <div className="drawer-panel">
            <div className="eval-drawer-header">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="eval-drawer-title">{current.modelName}</div>
                  <div className="eval-drawer-meta">
                    <span>{current.datasetName}</span>
                    <span className="meta-divider" />
                    <span>{"测试集"} {current.testImageCount} 张</span>
                    <span className="meta-divider" />
                    <span>{formatTime(current.createdAt)}</span>
                  </div>
                </div>
                <button className="close-btn" onClick={() => setSelectedId(null)}>{"×"}</button>
              </div>
              <div className="eval-drawer-stats">
                <div className="eval-stat-card">
                  <div className="eval-stat-dot" style={{ background: STATUS_COLOR[current.autoStatus] || "#999" }} />
                  <div><div className="eval-stat-label">评估状态</div><div className="eval-stat-value">{STATUS_LABEL[current.autoStatus]}</div></div>
                </div>
                <div className="eval-stat-card">
                  <div><div className="eval-stat-label">测试集</div><div className="eval-stat-value">{current.testImageCount} 张</div></div>
                </div>
              </div>
            </div>

            <div className="drawer-body">
              <div className="eval-section">
                <div className="eval-section-title"><div className="accent-bar" />模型信息</div>
                <div className="detail-grid">
                  <span className="label">模型 ID</span><span className="value-mono">{current.modelNodeId}</span>
                  <span className="label">模型名称</span><span style={{ fontWeight: 600 }}>{current.modelName}</span>
                  <span className="label">数据集</span><span style={{ fontWeight: 600 }}>{current.datasetName}</span>
                  <span className="label">创建时间</span><span>{new Date(current.createdAt).toLocaleString("zh-CN")}</span>
                </div>
              </div>

              {current.metrics ? (
                <div className="eval-section">
                  <div className="eval-section-title"><div className="accent-bar" />评估指标</div>
                  <div className="metric-grid-4">
                    {([["Precision", current.metrics.precision * 100, "#1766ad"], ["Recall", current.metrics.recall * 100, "#1a8a75"], ["mAP50", current.metrics.mAP50 * 100, "#36a1bd"], ["mAP50-95", current.metrics.mAP50_95 * 100, "#d77d59"]] as [string, number, string][]).map(([label, val, color]) => (
                      <div key={label} className="metric-card-4" style={{ borderTop: `3px solid ${color}` }}>
                        <div className="mc4-label">{label}</div>
                        <div className="mc4-value">{val.toFixed(1)}%</div>
                        <div className="mc4-bar"><div className="mc4-bar-fill" style={{ width: `${Math.min(val, 100)}%`, background: color }} /></div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {current.metrics?.per_class && Object.keys(current.metrics.per_class).length > 0 && (
                <div className="eval-section">
                  <div className="eval-section-title"><div className="accent-bar" />分类指标</div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 8 }}>{"各类别的 Precision / Recall，识别问题类别"}</div>
                  <div>
                    <div className="class-metrics-header"><span>{"类别"}</span><span>Precision</span><span>Recall</span></div>
                    {Object.entries(current.metrics.per_class as Record<string, { recall: number; precision: number }>).map(([clsId, cls]) => {
                      const p = cls.precision * 100; const r = cls.recall * 100;
                      const pColor = p >= 80 ? "var(--accent-green)" : p >= 50 ? "var(--accent-orange)" : "#c44";
                      const rColor = r >= 80 ? "var(--accent-green)" : r >= 50 ? "var(--accent-orange)" : "#c44";
                      return (
                        <div key={clsId} className="class-metrics-row">
                          <span className="cmr-label">{"类别"} {clsId}</span>
                          <div><div className="cmr-value" style={{ color: pColor }}>{p.toFixed(1)}%</div><div className="cmr-bar"><div className="cmr-bar-fill" style={{ width: `${Math.min(p, 100)}%`, background: pColor }} /></div></div>
                          <div><div className="cmr-value" style={{ color: rColor }}>{r.toFixed(1)}%</div><div className="cmr-bar"><div className="cmr-bar-fill" style={{ width: `${Math.min(r, 100)}%`, background: rColor }} /></div></div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {current.lastError && (
                <div className="drawer-section-error">
                  <div className="error-title"><div style={{ width: 3, height: 14, background: "#c44", borderRadius: 2 }} />错误</div>
                  <div className="error-body">{current.lastError}</div>
                </div>
              )}

              <div className="eval-section">
                <div className="eval-section-title"><div className="section-accent-bar" style={{ background: "var(--accent-cyan)" }} />测试推理</div>
                <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 8 }}>{"上传图片，用该模型做检测"}</div>
                <label className="file-label">
                  <input type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleInfer(f); }} />
                  {"选择图片上传推理"}
                </label>
                {inferLoading && <div style={{ marginTop: 8 }} className="cell-text-muted">{"推理中..."}</div>}
                {inferError && <div style={{ marginTop: 8, fontSize: 11, color: "#c44" }}>{inferError}</div>}
                {inferResult && inferImageUrl && (
                  <div style={{ marginTop: 10 }}>
                    <InferResultCanvas result={inferResult} imageUrl={inferImageUrl} />
                    <div style={{ marginTop: 6, fontSize: 10, color: "var(--text-muted)", display: "flex", gap: 12 }}>
                      <span>{"耗时"} {inferResult.latency_ms}ms</span>
                      <span>{inferResult.detections.length} 个检测框</span>
                    </div>
                    {inferResult.detections.length > 0 && (
                      <div style={{ marginTop: 6, maxHeight: 120, overflowY: "auto" }}>
                        {inferResult.detections.map((d: any, i: number) => (
                          <div key={i} className="detection-item">
                            <span className="det-name">{d.class_name}</span>
                            <span className="det-conf">{(d.confidence * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}
