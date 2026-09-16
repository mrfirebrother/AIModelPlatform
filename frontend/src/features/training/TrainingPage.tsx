import { Fragment, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import { useToast } from "../../lib/toast";
import LoadingSpinner from "../../lib/LoadingSpinner";
import type { TrainingTask, TrainingLogs, TrainingMetrics, TrainingCheckpoint } from "../../lib/api";

const STATUS_LABEL: Record<string, string> = { running: "运行中", queued: "排队中", completed: "已完成", failed: "失败", cancelled: "已取消" };
const STATUS_BADGE: Record<string, string> = { running: "badge-cyan", queued: "badge-orange", completed: "badge-green", failed: "badge-red", cancelled: "badge-gray" };
const STATUS_ORDER: Record<string, number> = { running: 0, queued: 1, failed: 2, completed: 3, cancelled: 4 };

function taskEpochs(task: TrainingTask): number { const v = (task as any).epochs ?? (task as any).trainingConfigJson?.epochs ?? 0; const n = Number(v); return Number.isFinite(n) ? n : 0; }
const RECIPE_LABEL: Record<string, string> = { quick: "快速验证", standard: "标准", highres: "高精度", custom: "自定义" };
const AUG_LABEL: Record<string, string> = { off: "关闭", default: "默认", strong: "强增广" };

/** 本次训练实际生效的参数（创建时写入任务行，行插入后不可变），用于给指标差异归因。 */
function taskParams(task: TrainingTask): [string, string][] {
  const cfg = ((task as any).trainingConfigJson || {}) as Record<string, any>;
  const rows: [string, string][] = [];
  if (cfg.recipe) rows.push(["配方", RECIPE_LABEL[String(cfg.recipe)] ?? String(cfg.recipe)]);
  if (cfg.epochs != null) rows.push(["轮数", String(cfg.epochs)]);
  if (cfg.imgsz != null) rows.push(["分辨率", String(cfg.imgsz)]);
  if (cfg.batch != null) rows.push(["批大小", String(cfg.batch)]);
  if (cfg.patience != null) rows.push(["早停", String(cfg.patience)]);
  if (cfg.augmentation != null) rows.push(["数据增广", AUG_LABEL[String(cfg.augmentation)] ?? String(cfg.augmentation)]);
  if (cfg.cache != null) rows.push(["图像缓存", cfg.cache ? "开" : "关"]);
  if (cfg.device != null) rows.push(["设备", String(cfg.device) === "cpu" ? "CPU" : `GPU ${cfg.device}`]);
  return rows;
}
function taskDatasetName(task: TrainingTask): string { return (task as any).datasetName || (task as any).datasetSnapshotId || "未知数据集"; }
function formatTime(iso?: string): string { if (!iso) return "—"; const d = new Date(iso); return `${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")} ${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}`; }
function durationLabel(start?: string | null, end?: string | null): string { if (!start) return "—"; const s = new Date(start).getTime(); const e = end ? new Date(end).getTime() : Date.now(); const ms = Math.max(0, e - s); if (ms < 60000) return `${Math.floor(ms/1000)}s`; const m = Math.floor(ms/60000); if (m < 60) return `${m}m`; const h = Math.floor(m/60); const rm = m%60; return rm ? `${h}h ${rm}m` : `${h}h`; }

export default function TrainingPage() {
  const api = useMemo(() => createApi(), []);
  const toast = useToast();
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [logs, setLogs] = useState<TrainingLogs | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [checkpoint, setCheckpoint] = useState<TrainingCheckpoint | null>(null);
  const [detailErr, setDetailErr] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [sortKey, setSortKey] = useState<"time" | "status">("time");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(1);
  const [copyOk, setCopyOk] = useState(false);
  const pageSize = 20;

  useEffect(() => { const t = setTimeout(() => setSearch(searchInput.trim().toLowerCase()), 300); return () => clearTimeout(t); }, [searchInput]);

  const fetchTasks = async () => { try { setTasks(await api.getTrainingTasks()); } catch {} setLoading(false); };
  useEffect(() => { fetchTasks(); }, [api]);

  useEffect(() => {
    const hasRunning = tasks.some((t) => t.status === "running" || t.status === "queued");
    if (!hasRunning) return;
    let delay = 5000; let timer: number | undefined;
    const tick = async () => { if (document.visibilityState === "hidden") { timer = window.setTimeout(tick, delay); return; } try { await fetchTasks(); delay = 5000; } catch { delay = Math.min(delay * 2, 10000); } timer = window.setTimeout(tick, delay); };
    timer = window.setTimeout(tick, 5000);
    return () => clearTimeout(timer);
  }, [tasks]);

  useEffect(() => {
    if (!selectedId) { setLogs(null); setMetrics(null); setCheckpoint(null); setDetailErr(null); return; }
    let abort = false; let timer: number | undefined;
    const load = async () => {
      try {
        const [l, m, c] = await Promise.all([api.getTrainingLogs(selectedId).catch(() => null), api.getTrainingMetrics(selectedId).catch(() => null), api.getTrainingCheckpoint(selectedId).catch(() => null)]);
        if (abort) return;
        setLogs(l as any); setMetrics(m as any); setCheckpoint(c as any);
        setDetailErr(!l && !m && !c ? "加载失败" : null);
        const running = (l as any)?.status === "running" || (l as any)?.status === "queued";
        if (running) timer = window.setTimeout(load, 5000);
      } catch (e) { if (!abort) setDetailErr((e as Error).message); }
    };
    load();
    return () => { abort = true; clearTimeout(timer); };
  }, [selectedId, api]);

  useEffect(() => { const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setSelectedId(null); }; window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey); }, []);

  const handleDelete = async (id: string) => { const ok = await toast.confirm("确定删除此训练任务？"); if (!ok) return; try { await api.deleteTrainingTask(id); toast.success("已删除"); setTasks((prev) => prev.filter((t) => t.id !== id)); if (selectedId === id) setSelectedId(null); } catch (e) { toast.error("删除失败: " + (e as Error).message); } };
  const handleCancel = async (id: string) => { const ok = await toast.confirm("确定取消？"); if (!ok) return; const prev = tasks.find((t) => t.id === id)?.status; setTasks((p) => p.map((t) => t.id === id ? { ...t, status: "cancelled" as const } : t)); try { await (api as any).cancelTrainingTask?.(id); toast.success("已取消"); } catch (e) { setTasks((p) => p.map((t) => t.id === id ? { ...t, status: (prev as any) || t.status } : t)); toast.error("取消失败: " + (e as Error).message); } };
  const handleCopy = async (text: string) => { try { await navigator.clipboard.writeText(text); } catch { const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove(); } setCopyOk(true); toast.success("已复制"); setTimeout(() => setCopyOk(false), 1500); };

  const filtered = useMemo(() => {
    let list = [...tasks];
    if (statusFilter !== "all") list = list.filter((t) => t.status === statusFilter);
    if (search) list = list.filter((t) => { const a = (t.parentModelName || t.id).toLowerCase(); const b = taskDatasetName(t).toLowerCase(); return a.includes(search) || b.includes(search); });
    list.sort((a, b) => { const pa = STATUS_ORDER[a.status] ?? 99; const pb = STATUS_ORDER[b.status] ?? 99; if (pa !== pb && sortKey !== "status") return pa - pb; if (sortKey === "status") return sortDir === "asc" ? pa - pb : pb - pa; const ta = new Date((a as any).createdAt || 0).getTime(); const tb = new Date((b as any).createdAt || 0).getTime(); return sortDir === "asc" ? ta - tb : tb - ta; });
    return list;
  }, [tasks, statusFilter, search, sortKey, sortDir]);

  useEffect(() => { setPage(1); }, [statusFilter, search, sortKey, sortDir]);
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const pageItems = filtered.slice((page - 1) * pageSize, page * pageSize);
  const selected = tasks.find((t) => t.id === selectedId) || null;
  const hasRunning = tasks.some((t) => t.status === "running" || t.status === "queued");

  const lossStats = useMemo(() => { const arr = (metrics as any)?.epochs || []; if (!arr.length) return null; const losses = arr.map((e: any) => e.loss).filter((x: any) => typeof x === "number"); return { min: Math.min(...losses), max: Math.max(...losses), arr }; }, [metrics]);

  return (
    <>
      <div className="toolbar">
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className="input-toolbar">
          <option value="all">全部状态</option>
          <option value="running">运行中</option>
          <option value="queued">排队中</option>
          <option value="completed">已完成</option>
          <option value="failed">失败</option>
          <option value="cancelled">已取消</option>
        </select>
        <input value={searchInput} onChange={(e) => setSearchInput(e.target.value)} placeholder="搜索 模型/数据集" className="input-search" />
        {search && <button className="btn small" onClick={() => { setSearchInput(""); setSearch(""); }}>清空</button>}
        <div className="toolbar-spacer" />
        <span className="toolbar-hint">{filtered.length} 条{hasRunning ? " · 自动刷新" : ""}</span>
        <button className="btn primary small" onClick={() => navigate("/training/create")}>+ 新建</button>
      </div>

      <div className="card" style={{ overflow: "hidden" }}>
        {loading ? (
          <LoadingSpinner text="加载训练任务..." />
        ) : (<>
        <div className="grid-table-header" style={{ gridTemplateColumns: "2fr 1fr 1fr 1fr 0.8fr 80px" }}>
          <span>任务</span>
          <span>配置</span>
          <button className="sort-btn" onClick={() => { setSortKey("status"); setSortDir((d) => d === "asc" ? "desc" : "asc"); }}>状态 {sortKey === "status" ? (sortDir === "asc" ? "↑" : "↓") : ""}</button>
          <button className="sort-btn" onClick={() => { setSortKey("time"); setSortDir((d) => d === "asc" ? "desc" : "asc"); }}>时间 {sortKey === "time" ? (sortDir === "asc" ? "↑" : "↓") : ""}</button>
          <span>耗时</span>
          <span className="cell-actions">操作</span>
        </div>
        {pageItems.map((t) => {
          const isSel = selectedId === t.id;
          const dur = t.status === "running" ? durationLabel((t as any).createdAt) : t.status === "completed" || t.status === "failed" || t.status === "cancelled" ? durationLabel((t as any).createdAt, (t as any).updatedAt) : "—";
          return (
            <div key={t.id} onClick={() => setSelectedId(isSel ? null : t.id)} className={`grid-table-row ${isSel ? "grid-table-row-selected" : ""}`} style={{ gridTemplateColumns: "2fr 1fr 1fr 1fr 0.8fr 80px" }}>
              <div style={{ minWidth: 0 }}>
                <div className="cell-primary">{(t as any).trainingConfigJson?.modelName || (t as any).trainingConfigJson?.model_name || (t as any).parentModelName || t.id.slice(0, 8)}</div>
                <div className="cell-mono">{taskDatasetName(t)}</div>
              </div>
              <div className="cell-text" title={`${(t as any).modelFamily || "YOLOv8"} · ${(t as any).trainingConfigJson?.device || "cpu"}`}>{taskEpochs(t)} 轮 · {(t as any).trainingConfigJson?.device || "cpu"}</div>
              <div><span className={`badge badge-xs ${STATUS_BADGE[t.status] || "badge-gray"}`}>{STATUS_LABEL[t.status] || t.status}</span></div>
              <div className="cell-text">{formatTime((t as any).createdAt)}</div>
              <div className="cell-text" style={{ color: t.status === "running" ? "var(--accent-orange)" : "var(--text-muted)" }}>{dur}</div>
              <div className="cell-actions">
                {t.status === "running" ? (
                  <button className="btn small btn-inline-cancel" onClick={(e) => { e.stopPropagation(); handleCancel(t.id); }}>取消</button>
                ) : (
                  <button className="btn small danger btn-inline-danger" onClick={(e) => { e.stopPropagation(); handleDelete(t.id); }}>删除</button>
                )}
              </div>
            </div>
          );
        })}
        {filtered.length === 0 && (
          <div className="empty-msg">
            {tasks.length === 0 ? "暂无训练任务 去新建" : <>{"无匹配"} <button className="btn small" onClick={() => { setStatusFilter("all"); setSearchInput(""); setSearch(""); }} style={{ marginLeft: 8 }}>清空筛选</button></>}
          </div>
        )}
        {filtered.length > pageSize && (
          <div className="pagination">
            <button className="btn small" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>上一页</button>
            <span className="pagination-info">{page} / {totalPages}</span>
            <button className="btn small" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>下一页</button>
          </div>
        )}
        </>)}
      </div>

      {selected && (
        <>
          <div className="drawer-overlay" onClick={() => setSelectedId(null)} />
          <div className="drawer-panel">
            <div className="drawer-header">
              <div>
                <div className="drawer-header-info">{(selected as any).parentModelName || selected.id.slice(0, 8)} · <span className={`badge ${STATUS_BADGE[selected.status]}`}>{STATUS_LABEL[selected.status]}</span></div>
                <div className="drawer-header-meta">{taskDatasetName(selected)} · {taskEpochs(selected)} 轮</div>
              </div>
              <button className="btn small" onClick={() => setSelectedId(null)}>{"×"}</button>
            </div>
            <div className="drawer-body">
              <div className="drawer-section">
                <div className="drawer-section-title">概览</div>
                {logs ? (
                  <div style={{ fontSize: 12, lineHeight: 1.5 }}>
                    <div>{"状态："}<span className={`badge ${STATUS_BADGE[logs.status]}`}>{STATUS_LABEL[logs.status] || logs.status}</span></div>
                    <div>{"进度："}{logs.currentEpoch} / {logs.totalEpochs}</div>
                    <div className="drawer-progress">
                      <div className="drawer-progress-bar"><div className="drawer-progress-fill" style={{ width: `${logs.totalEpochs ? Math.round((logs.currentEpoch / logs.totalEpochs) * 100) : 0}%` }} /></div>
                      <span className="drawer-progress-pct">{logs.totalEpochs ? Math.round((logs.currentEpoch / logs.totalEpochs) * 100) : 0}%</span>
                    </div>
                    <div>{"尝试："}{logs.attempts?.length || 0}</div>
                    <div>{"创建："}{formatTime((selected as any).createdAt)} {"· 耗时"} {selected.status === "running" ? durationLabel((selected as any).createdAt) : logs.attempts?.[0] ? durationLabel(logs.attempts[0].startedAt, logs.attempts[0].finishedAt) : "—"}</div>
                  </div>
                ) : detailErr ? <div style={{ fontSize: 12, color: "#c44" }}>{detailErr} <button className="btn small" onClick={() => setSelectedId(selected.id)}>重试</button></div> : <div className="cell-text-muted">加载中...</div>}
              </div>
              <div className="drawer-section">
                <div className="drawer-section-title">指标</div>
                {checkpoint?.latestCheckpoint ? (
                  <div className="metric-grid-2x2">
                    {[["mAP50", (checkpoint.latestCheckpoint.metrics as any)?.["metrics/mAP50(B)"]], ["mAP50-95", (checkpoint.latestCheckpoint.metrics as any)?.["metrics/mAP50-95(B)"]], ["Precision", (checkpoint.latestCheckpoint.metrics as any)?.["metrics/precision(B)"]], ["Recall", (checkpoint.latestCheckpoint.metrics as any)?.["metrics/recall(B)"]]].map(([k, v]) => (
                      <div key={k} className="metric-tile"><div className="metric-tile-label">{k}</div><div className="metric-tile-value">{typeof v === "number" ? v.toFixed(4) : "—"}</div></div>
                    ))}
                  </div>
                ) : <div className="cell-text-muted" style={{ marginBottom: 8 }}>{selected.status === "running" ? "训练中 —" : "暂无指标"}</div>}
                {metrics && lossStats ? (
                  <div>
                    {metrics.bestLoss !== null && <div style={{ fontSize: 12, marginBottom: 6 }}>{"最佳 Loss: "}{metrics.bestLoss.toFixed(4)}</div>}
                    {lossStats.arr.map((ep: any) => {
                      const w = lossStats.arr.length === 1 ? 100 : lossStats.max === lossStats.min ? 50 : Math.round(((lossStats.max - ep.loss) / (lossStats.max - lossStats.min)) * 100);
                      return (
                        <div key={ep.epoch} className="loss-row">
                          <span className="loss-epoch">E{ep.epoch}</span>
                          <div className="loss-bar"><div className="loss-bar-fill" style={{ width: `${w}%`, background: w > 66 ? "var(--accent-green)" : w > 33 ? "var(--accent-cyan)" : "var(--accent-orange)" }} /></div>
                          <span className="loss-value">{ep.loss.toFixed(4)}</span>
                        </div>
                      );
                    })}
                  </div>
                ) : metrics ? <div className="cell-text-muted">{"暂无 Loss"}</div> : null}
              </div>
              <div className="drawer-section">
                <div className="drawer-section-title">训练参数</div>
                {(() => {
                  const rows = taskParams(selected);
                  if (rows.length === 0) return <div className="cell-text-muted">该任务未记录参数（旧任务）</div>;
                  return (
                    <>
                      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto 1fr", gap: "4px 10px", fontSize: 12 }}>
                        {rows.map(([k, v]) => (
                          <Fragment key={k}>
                            <span style={{ color: "var(--text-muted)" }}>{k}</span>
                            <b>{v}</b>
                          </Fragment>
                        ))}
                      </div>
                      <div className="cell-text-muted" style={{ marginTop: 6, fontSize: 11 }}>本次实际生效的参数，用于对比不同实验</div>
                    </>
                  );
                })()}
              </div>
              <div className="drawer-section">
                <div className="drawer-section-title">产物</div>
                {checkpoint?.latestCheckpoint ? (
                  <div style={{ fontSize: 12, lineHeight: 1.7 }}>
                    <div>Epoch: <b>{checkpoint.latestCheckpoint.epoch}</b></div>
                    <div className="artifact-row">
                      <span className="artifact-path" title={checkpoint.latestCheckpoint.artifactPath}>{checkpoint.latestCheckpoint.artifactPath}</span>
                      <button className="btn small btn-inline-copy" onClick={() => handleCopy(checkpoint.latestCheckpoint!.artifactPath)}>{copyOk ? "已复制" : "复制路径"}</button>
                    </div>
                    <div className="artifact-hash">{checkpoint.latestCheckpoint.artifactHash?.slice(0, 16)} {"·"} {formatTime(checkpoint.latestCheckpoint.createdAt)}</div>
                  </div>
                ) : <div className="cell-text-muted">{"暂无 Checkpoint"}</div>}
              </div>
              <div className="drawer-section">
                <div className="drawer-section-title">日志</div>
                {logs?.attempts?.map((a) => (
                  <div key={a.attemptNo} className={`attempt-card ${a.status === "failed" ? "attempt-card-failed" : "attempt-card-ok"}`}>
                    <div className="attempt-title">{"尝试"} {a.attemptNo} {"·"} <span className={`badge ${STATUS_BADGE[a.status] || "badge-gray"}`}>{a.status}</span></div>
                    <div className="attempt-time">{a.startedAt ? formatTime(a.startedAt) : "—"} {"→"} {a.finishedAt ? formatTime(a.finishedAt) : "进行中"}</div>
                    {a.logPath && <div className="attempt-log-path">{a.logPath}</div>}
                    {a.lastError && <div className="attempt-error">{a.lastError.length > 200 ? <ErrorToggle text={a.lastError} /> : a.lastError}</div>}
                  </div>
                ))}
                {!logs && !detailErr && <div className="cell-text-muted">加载中...</div>}
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}

function ErrorToggle({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (<><span>{open ? text : text.slice(0, 200) + "…"}</span><button className="error-toggle" onClick={() => setOpen((v) => !v)}>{open ? "收起" : "展开"}</button></>);
}
