import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import { useToast } from "../../lib/toast";
import type { TrainingTask, TrainingLogs, TrainingMetrics, TrainingCheckpoint } from "../../lib/api";

const STATUS_LABEL: Record<string, string> = {
  running: "运行中",
  queued: "排队中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};
const STATUS_BADGE: Record<string, string> = {
  running: "badge-cyan",
  queued: "badge-orange",
  completed: "badge-green",
  failed: "badge-red",
  cancelled: "badge-gray",
};
const STATUS_ORDER: Record<string, number> = { running: 0, queued: 1, failed: 2, completed: 3, cancelled: 4 };

function taskEpochs(task: TrainingTask): number {
  const v = (task as any).epochs ?? (task as any).trainingConfigJson?.epochs ?? 0;
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}
function taskDatasetName(task: TrainingTask): string {
  return (task as any).datasetName || (task as any).datasetSnapshotId || "未知数据集";
}
function formatTime(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${mm}-${dd} ${hh}:${mi}`;
}
function durationLabel(start?: string | null, end?: string | null): string {
  if (!start) return "—";
  const s = new Date(start).getTime();
  const e = end ? new Date(end).getTime() : Date.now();
  const ms = Math.max(0, e - s);
  const m = Math.floor(ms / 60000);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `${h}h ${rm}m` : `${h}h`;
}

export default function TrainingPage() {
  const api = useMemo(() => createApi(), []);
  const toast = useToast();
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
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

  // debounce search 300ms
  useEffect(() => {
    const t = setTimeout(() => setSearch(searchInput.trim().toLowerCase()), 300);
    return () => clearTimeout(t);
  }, [searchInput]);

  const fetchTasks = async () => {
    try {
      const list = await api.getTrainingTasks();
      setTasks(list);
    } catch {}
  };
  useEffect(() => { fetchTasks(); }, [api]);

  // polling tasks when running/queued exists
  useEffect(() => {
    const hasRunning = tasks.some((t) => t.status === "running" || t.status === "queued");
    if (!hasRunning) return;
    let delay = 5000;
    let timer: number | undefined;
    const tick = async () => {
      if (document.visibilityState === "hidden") { timer = window.setTimeout(tick, delay); return; }
      try { await fetchTasks(); delay = 5000; } catch { delay = Math.min(delay * 2, 10000); }
      timer = window.setTimeout(tick, delay);
    };
    timer = window.setTimeout(tick, 5000);
    return () => clearTimeout(timer);
  }, [tasks]);

  // detail fetch + polling
  useEffect(() => {
    if (!selectedId) { setLogs(null); setMetrics(null); setCheckpoint(null); setDetailErr(null); return; }
    let abort = false;
    let timer: number | undefined;
    const load = async () => {
      try {
        const [l, m, c] = await Promise.all([
          api.getTrainingLogs(selectedId).catch(() => null),
          api.getTrainingMetrics(selectedId).catch(() => null),
          api.getTrainingCheckpoint(selectedId).catch(() => null),
        ]);
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

  // ESC close drawer
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setSelectedId(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const handleDelete = async (id: string) => {
    const ok = await toast.confirm("确定删除此训练任务？");
    if (!ok) return;
    try {
      await api.deleteTrainingTask(id);
      toast.success("已删除");
      setTasks((prev) => prev.filter((t) => t.id !== id));
      if (selectedId === id) setSelectedId(null);
    } catch (e) { toast.error("删除失败: " + (e as Error).message); }
  };
  const handleCancel = async (id: string) => {
    const ok = await toast.confirm("确定取消？");
    if (!ok) return;
    const prev = tasks.find((t) => t.id === id)?.status;
    setTasks((p) => p.map((t) => t.id === id ? { ...t, status: "cancelled" as const } : t));
    try {
      await (api as any).cancelTrainingTask?.(id);
      toast.success("已取消");
    } catch (e) {
      setTasks((p) => p.map((t) => t.id === id ? { ...t, status: (prev as any) || t.status } : t));
      toast.error("取消失败: " + (e as Error).message);
    }
  };
  const handleCopy = async (text: string) => {
    try { await navigator.clipboard.writeText(text); } catch { const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove(); }
    setCopyOk(true); toast.success("已复制"); setTimeout(() => setCopyOk(false), 1500);
  };

  const filtered = useMemo(() => {
    let list = [...tasks];
    if (statusFilter !== "all") list = list.filter((t) => t.status === statusFilter);
    if (search) {
      list = list.filter((t) => {
        const a = (t.parentModelName || t.id).toLowerCase();
        const b = taskDatasetName(t).toLowerCase();
        return a.includes(search) || b.includes(search);
      });
    }
    list.sort((a, b) => {
      const pa = STATUS_ORDER[a.status] ?? 99;
      const pb = STATUS_ORDER[b.status] ?? 99;
      if (pa !== pb && sortKey !== "status") return pa - pb;
      if (sortKey === "status") return sortDir === "asc" ? pa - pb : pb - pa;
      const ta = new Date((a as any).createdAt || 0).getTime();
      const tb = new Date((b as any).createdAt || 0).getTime();
      return sortDir === "asc" ? ta - tb : tb - ta;
    });
    return list;
  }, [tasks, statusFilter, search, sortKey, sortDir]);

  useEffect(() => { setPage(1); }, [statusFilter, search, sortKey, sortDir]);
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const pageItems = filtered.slice((page - 1) * pageSize, page * pageSize);
  const selected = tasks.find((t) => t.id === selectedId) || null;
  const hasRunning = tasks.some((t) => t.status === "running" || t.status === "queued");

  // metrics helpers
  const lossStats = useMemo(() => {
    const arr = (metrics as any)?.epochs || [];
    if (!arr.length) return null;
    const losses = arr.map((e: any) => e.loss).filter((x: any) => typeof x === "number");
    const min = Math.min(...losses);
    const max = Math.max(...losses);
    return { min, max, arr };
  }, [metrics]);

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto" }}>
      {/* toolbar */}
      <div className="card" style={{ padding: "8px 12px", display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 8, minHeight: 40 }}>
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} style={{ padding: "6px 8px", border: "1px solid #dce8f0", borderRadius: 6, fontSize: 12 }}>
          <option value="all">全部状态</option>
          <option value="running">运行中</option>
          <option value="queued">排队中</option>
          <option value="completed">已完成</option>
          <option value="failed">失败</option>
          <option value="cancelled">已取消</option>
        </select>
        <input value={searchInput} onChange={(e) => setSearchInput(e.target.value)} placeholder="搜索 模型/数据集" style={{ flex: "1 1 200px", maxWidth: 320, padding: "6px 10px", border: "1px solid #dce8f0", borderRadius: 6, fontSize: 12 }} />
        {search && <button className="btn small" onClick={() => { setSearchInput(""); setSearch(""); }}>清空</button>}
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{filtered.length} 条{hasRunning ? " · 自动刷新" : ""}</span>
        <button className="btn primary small" onClick={() => navigate("/training/create")}>+ 新建</button>
      </div>

      {/* table */}
      <div className="card" style={{ overflow: "hidden" }}>
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1.2fr 1.2fr 1fr 80px", gap: 0, padding: "10px 16px", borderBottom: "1px solid #dce8f0", background: "#f8fafc", fontSize: 11, fontWeight: 600, color: "var(--text-muted)" }}>
          <span>任务</span>
          <span>配置</span>
          <button onClick={() => { setSortKey("status"); setSortDir((d) => d === "asc" ? "desc" : "asc"); }} style={{ border: 0, background: "none", cursor: "pointer", fontWeight: 600, color: "var(--text-muted)", textAlign: "left", fontSize: 11 }}>状态 {sortKey === "status" ? (sortDir === "asc" ? "↑" : "↓") : ""}</button>
          <button onClick={() => { setSortKey("time"); setSortDir((d) => d === "asc" ? "desc" : "asc"); }} style={{ border: 0, background: "none", cursor: "pointer", fontWeight: 600, color: "var(--text-muted)", textAlign: "left", fontSize: 11 }}>时间 {sortKey === "time" ? (sortDir === "asc" ? "↑" : "↓") : ""}</button>
          <span style={{ textAlign: "right" }}>操作</span>
        </div>
        {pageItems.map((t) => {
          const isSel = selectedId === t.id;
          return (
            <div key={t.id} onClick={() => setSelectedId(isSel ? null : t.id)} style={{ display: "grid", gridTemplateColumns: "2fr 1.2fr 1.2fr 1fr 80px", gap: 0, padding: "10px 16px", borderBottom: "1px solid #eef3f8", background: isSel ? "#e7f1fa" : undefined, cursor: "pointer", alignItems: "center" }}>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 12, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{(t as any).parentModelName || t.id.slice(0, 8)}</div>
                <div style={{ color: "var(--text-muted)", fontSize: 9, fontFamily: "Courier New, monospace", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{taskDatasetName(t)}</div>
              </div>
              <div style={{ fontSize: 11 }}>{taskEpochs(t)} 轮 · {(t as any).modelFamily || "YOLOv8"} · {(t as any).trainingConfigJson?.device || "cpu"}</div>
              <div>
                <span className={`badge ${STATUS_BADGE[t.status] || "badge-gray"}`} style={{ fontSize: 9 }}>{STATUS_LABEL[t.status] || t.status}</span>
              </div>
              <div style={{ fontSize: 11 }}>
                <div>{formatTime((t as any).createdAt)}</div>
                <div style={{ fontSize: 9, color: "var(--text-muted)" }}>{t.status === "running" ? `耗时 ${durationLabel((t as any).createdAt)}` : t.status === "completed" && logs?.attempts?.[0]?.startedAt ? `耗时 ${durationLabel(logs.attempts[0].startedAt, logs.attempts[0].finishedAt)}` : ""}</div>
              </div>
              <div style={{ textAlign: "right" }}>
                {t.status === "running" ? (
                  <button className="btn small" onClick={(e) => { e.stopPropagation(); handleCancel(t.id); }} style={{ padding: "2px 8px", fontSize: 11, color: "#c47728" }}>取消</button>
                ) : (
                  <button className="btn small danger" onClick={(e) => { e.stopPropagation(); handleDelete(t.id); }} style={{ padding: "2px 8px", fontSize: 11 }}>删除</button>
                )}
              </div>
            </div>
          );
        })}
        {filtered.length === 0 && (
          <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>
            {tasks.length === 0 ? "暂无训练任务 去新建" : <>无匹配 <button className="btn small" onClick={() => { setStatusFilter("all"); setSearchInput(""); setSearch(""); }} style={{ marginLeft: 8 }}>清空筛选</button></>}
          </div>
        )}
        {filtered.length > pageSize && (
          <div style={{ display: "flex", justifyContent: "center", gap: 8, padding: 12, borderTop: "1px solid #dce8f0" }}>
            <button className="btn small" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>上一页</button>
            <span style={{ fontSize: 11, lineHeight: "28px" }}>{page} / {totalPages}</span>
            <button className="btn small" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>下一页</button>
          </div>
        )}
      </div>

      {/* drawer */}
      {selected && (
        <>
          <div onClick={() => setSelectedId(null)} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.18)", zIndex: 20 }} />
          <div style={{ position: "fixed", right: 0, top: 0, bottom: 0, width: 480, maxWidth: "92vw", background: "#fff", zIndex: 30, boxShadow: "-8px 0 24px rgba(0,0,0,0.12)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid #dce8f0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <div style={{ fontWeight: 700, fontSize: 13 }}>{(selected as any).parentModelName || selected.id.slice(0, 8)} · <span className={`badge ${STATUS_BADGE[selected.status]}`}>{STATUS_LABEL[selected.status]}</span></div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{taskDatasetName(selected)} · {taskEpochs(selected)} 轮</div>
              </div>
              <button className="btn small" onClick={() => setSelectedId(null)}>×</button>
            </div>
            <div style={{ flex: 1, overflowY: "auto", padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
              {/* overview */}
              <div style={{ background: "#fff", border: "1px solid #e6eef6", borderRadius: 8, padding: 10, minHeight: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 6 }}>概览</div>
                {logs ? (
                  <div style={{ fontSize: 11, lineHeight: 1.5 }}>
                    <div>状态：<span className={`badge ${STATUS_BADGE[logs.status]}`}>{STATUS_LABEL[logs.status] || logs.status}</span></div>
                    <div>进度：{logs.currentEpoch} / {logs.totalEpochs}</div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                      <div style={{ flex: 1, height: 8, background: "#e6eef6", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${logs.totalEpochs ? Math.round((logs.currentEpoch / logs.totalEpochs) * 100) : 0}%`, background: "var(--primary)", borderRadius: 4, transition: "width 0.3s" }} />
                      </div>
                      <span style={{ fontSize: 10, fontWeight: 600, color: "var(--primary)", minWidth: 32, textAlign: "right" }}>{logs.totalEpochs ? Math.round((logs.currentEpoch / logs.totalEpochs) * 100) : 0}%</span>
                    </div>
                    <div>尝试：{logs.attempts?.length || 0}</div>
                    <div>创建：{formatTime((selected as any).createdAt)} · 耗时 {selected.status === "running" ? durationLabel((selected as any).createdAt) : logs.attempts?.[0] ? durationLabel(logs.attempts[0].startedAt, logs.attempts[0].finishedAt) : "—"}</div>
                  </div>
                ) : detailErr ? <div style={{ fontSize: 11, color: "#c44" }}>{detailErr} <button className="btn small" onClick={() => setSelectedId(selected.id)}>重试</button></div> : <div style={{ fontSize: 11, color: "var(--text-muted)" }}>加载中...</div>}
              </div>
              {/* metrics */}
              <div style={{ background: "#fff", border: "1px solid #e6eef6", borderRadius: 8, padding: 10, minHeight: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 6 }}>指标</div>
                {checkpoint?.latestCheckpoint ? (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 8 }}>
                    {[
                      ["mAP50", (checkpoint.latestCheckpoint.metrics as any)?.["metrics/mAP50(B)"]],
                      ["mAP50-95", (checkpoint.latestCheckpoint.metrics as any)?.["metrics/mAP50-95(B)"]],
                      ["Precision", (checkpoint.latestCheckpoint.metrics as any)?.["metrics/precision(B)"]],
                      ["Recall", (checkpoint.latestCheckpoint.metrics as any)?.["metrics/recall(B)"]],
                    ].map(([k, v]) => (
                      <div key={k} style={{ background: "#f8fafc", border: "1px solid #e6eef6", borderRadius: 6, padding: "8px 10px" }}>
                        <div style={{ fontSize: 9, color: "var(--text-muted)" }}>{k}</div>
                        <div style={{ fontSize: 12, fontWeight: 600 }}>{typeof v === "number" ? v.toFixed(4) : "—"}</div>
                      </div>
                    ))}
                  </div>
                ) : <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>{selected.status === "running" ? "训练中 —" : "暂无指标"}</div>}
                {metrics && lossStats ? (
                  <div>
                    {metrics.bestLoss !== null && <div style={{ fontSize: 10, marginBottom: 6 }}>最佳 Loss: {metrics.bestLoss.toFixed(4)}</div>}
                    {lossStats.arr.map((ep: any) => {
                      const w = lossStats.arr.length === 1 ? 100 : lossStats.max === lossStats.min ? 50 : Math.round(((lossStats.max - ep.loss) / (lossStats.max - lossStats.min)) * 100);
                      return (
                        <div key={ep.epoch} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, fontSize: 10 }}>
                          <span style={{ width: 28, fontFamily: "Courier New, monospace" }}>E{ep.epoch}</span>
                          <div style={{ flex: 1, height: 6, background: "#e6eef6", borderRadius: 3, overflow: "hidden" }}><div style={{ height: "100%", width: `${w}%`, background: w > 66 ? "var(--accent-green)" : w > 33 ? "var(--accent-cyan)" : "var(--accent-orange)", borderRadius: 3 }} /></div>
                          <span style={{ width: 54, fontFamily: "Courier New, monospace" }}>{ep.loss.toFixed(4)}</span>
                        </div>
                      );
                    })}
                  </div>
                ) : metrics ? <div style={{ fontSize: 11, color: "var(--text-muted)" }}>暂无 Loss</div> : null}
              </div>
              {/* checkpoint */}
              <div style={{ background: "#fff", border: "1px solid #e6eef6", borderRadius: 8, padding: 10, minHeight: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 6 }}>产物</div>
                {checkpoint?.latestCheckpoint ? (
                  <div style={{ fontSize: 11, lineHeight: 1.7 }}>
                    <div>Epoch: <b>{checkpoint.latestCheckpoint.epoch}</b></div>
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                      <span title={checkpoint.latestCheckpoint.artifactPath} style={{ flex: 1, fontFamily: "Courier New, monospace", fontSize: 9, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{checkpoint.latestCheckpoint.artifactPath}</span>
                      <button className="btn small" onClick={() => handleCopy(checkpoint.latestCheckpoint!.artifactPath)} style={{ padding: "2px 6px", fontSize: 9 }}>{copyOk ? "已复制" : "复制路径"}</button>
                    </div>
                    <div style={{ fontFamily: "Courier New, monospace", fontSize: 9, color: "var(--text-muted)" }}>{checkpoint.latestCheckpoint.artifactHash?.slice(0, 16)} · {formatTime(checkpoint.latestCheckpoint.createdAt)}</div>
                  </div>
                ) : <div style={{ fontSize: 11, color: "var(--text-muted)" }}>暂无 Checkpoint</div>}
              </div>
              {/* logs */}
              <div style={{ background: "#fff", border: "1px solid #e6eef6", borderRadius: 8, padding: 10, minHeight: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 6 }}>日志</div>
                {logs?.attempts?.map((a) => (
                  <div key={a.attemptNo} style={{ border: "1px solid #e6eef6", borderRadius: 6, padding: 8, marginBottom: 8, background: a.status === "failed" ? "#fff5f5" : "#f8fafc" }}>
                    <div style={{ fontSize: 10, fontWeight: 600 }}>尝试 {a.attemptNo} · <span className={`badge ${STATUS_BADGE[a.status] || "badge-gray"}`}>{a.status}</span></div>
                    <div style={{ fontSize: 9, color: "var(--text-muted)" }}>{a.startedAt ? formatTime(a.startedAt) : "—"} → {a.finishedAt ? formatTime(a.finishedAt) : "进行中"}</div>
                    {a.logPath && <div style={{ fontSize: 9, fontFamily: "Courier New, monospace", color: "var(--text-muted)", marginTop: 4 }}>{a.logPath}</div>}
                    {a.lastError && (
                      <div style={{ marginTop: 6, background: "#fee", border: "1px solid #fcc", borderRadius: 4, padding: 6, fontFamily: "Courier New, monospace", fontSize: 9, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
                        {a.lastError.length > 200 ? <ErrorToggle text={a.lastError} /> : a.lastError}
                      </div>
                    )}
                  </div>
                ))}
                {!logs && !detailErr && <div style={{ fontSize: 11, color: "var(--text-muted)" }}>加载中...</div>}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function ErrorToggle({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <span>{open ? text : text.slice(0, 200) + "…"}</span>
      <button onClick={() => setOpen((v) => !v)} style={{ marginLeft: 6, border: 0, background: "none", color: "var(--primary)", cursor: "pointer", fontSize: 9 }}>{open ? "收起" : "展开"}</button>
    </>
  );
}
