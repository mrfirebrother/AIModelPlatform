import { useEffect, useMemo, useState } from "react";
import { createApi } from "../../lib/createApi";
import LoadingSpinner from "../../lib/LoadingSpinner";

interface OperationLogEntry { id: string; operationType: string; actor: string | null; status: string; summaryJson: Record<string, unknown>; errorSummary: string | null; createdAt: string; }
interface OperationLogsResponse { items: OperationLogEntry[]; total: number; }
interface SystemChecks { postgres: boolean; redis: boolean; gpu: boolean; worker: boolean; }
interface OperationStatusResponse { status: string; checks: SystemChecks; }

function formatTime(iso?: string): string { if (!iso) return "—"; const d = new Date(iso); return `${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")} ${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}`; }

const OP_LABELS: Record<string, string> = { "training.task.create": "创建训练任务", "training.task.delete": "删除训练任务", "training.task.cancel": "取消训练任务", "training.task.dispatch": "调度训练任务", "model.create": "导入模型", "model.update_status": "更新模型状态", "model.delete": "删除模型", "evaluation.create": "创建评估", "evaluation.review": "人工评审", "evaluation.session.create": "创建评估会话" };
const OP_COLORS: Record<string, string> = { "training.task.create": "#1766ad", "training.task.delete": "#c44", "training.task.cancel": "#d77d59", "model.create": "#1a8a75", "model.delete": "#c44", "evaluation.create": "#36a1bd", "evaluation.review": "#8b5cf6" };

export default function OperationsPage() {
  const api = useMemo(() => createApi(), []);
  const [status, setStatus] = useState<OperationStatusResponse | null>(null);
  const [logs, setLogs] = useState<OperationLogEntry[]>([]);
  const [logsTotal, setLogsTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [logPage, setLogPage] = useState(0);
  const pageSize = 10;

  useEffect(() => { fetch("/api/operations/status", { headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" } }).then((r) => r.json()).then((d: OperationStatusResponse) => setStatus(d)).catch(() => {}); }, []);
  useEffect(() => { setLoading(true); fetch(`/api/operations/logs?skip=${logPage * pageSize}&limit=${pageSize}`, { headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" } }).then((r) => r.json()).then((d: OperationLogsResponse) => { setLogs(d.items); setLogsTotal(d.total); }).catch(() => {}).finally(() => setLoading(false)); }, [logPage]);

  const checkItems = [{ key: "postgres", label: "PostgreSQL", icon: "⛁" }, { key: "redis", label: "Redis", icon: "⚡" }, { key: "gpu", label: "GPU", icon: "▣" }, { key: "worker", label: "Worker", icon: "⚙" }];

  return (
    <>
      <section className="ops-layout">
        <div className="card" style={{ padding: "12px 16px" }}>
          <div className="card-section-title" style={{ marginBottom: 8 }}>{"系统健康"}</div>
          {status ? checkItems.map(({ key, label, icon }) => {
            const ok = status.checks[key as keyof SystemChecks];
            return (
              <div key={key} className="health-row">
                <span className="health-label" style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ color: ok ? "var(--accent-green)" : "#b33", fontSize: 12 }}>{icon}</span>{label}</span>
                <span className={`badge badge-xs ${ok ? "badge-green" : "badge-red"}`}>{ok ? "正常" : "异常"}</span>
              </div>
            );
          }) : <div className="cell-text-muted">{"加载中..."}</div>}
        </div>

        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div className="card-section-header">
            <div>
              <div className="card-section-title">{"操作日志"}</div>
              <div className="card-section-hint">{"共"} {logsTotal} {"条 · 第"} {logPage + 1} {"页"}</div>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <button className="btn small" disabled={logPage === 0} onClick={() => setLogPage((p) => Math.max(0, p - 1))} style={{ fontSize: 10 }}>{"上一页"}</button>
              <button className="btn small" disabled={(logPage + 1) * pageSize >= logsTotal} onClick={() => setLogPage((p) => p + 1)} style={{ fontSize: 10 }}>{"下一页"}</button>
            </div>
          </div>
          <div>
            {loading ? (
              <LoadingSpinner text="加载操作日志..." />
            ) : logs.length === 0 ? (
              <div className="empty-msg-sm">{"暂无操作日志"}</div>
            ) : logs.map((log) => (
              <div key={log.id} className="log-entry">
                <div className="log-bar" style={{ background: log.status === "success" ? OP_COLORS[log.operationType] || "var(--accent-green)" : "#b33" }} />
                <div className="log-content">
                  <div className="log-title">
                    <span>{OP_LABELS[log.operationType] || log.operationType}</span>
                    <span className="cell-mono">{log.operationType}</span>
                  </div>
                  <div className="log-meta">{log.actor || "system"} {"·"} {formatTime(log.createdAt)}</div>
                </div>
                <span className={`badge badge-xs ${log.status === "success" ? "badge-green" : "badge-red"}`}>{log.status === "success" ? "成功" : "失败"}</span>
              </div>
            ))}
          </div>
          {logsTotal > pageSize && (
            <div className="pagination">
              <button className="btn small" disabled={logPage === 0} onClick={() => setLogPage((p) => Math.max(0, p - 1))}>{"上一页"}</button>
              <span className="pagination-info">{logPage + 1} / {Math.ceil(logsTotal / pageSize)}</span>
              <button className="btn small" disabled={(logPage + 1) * pageSize >= logsTotal} onClick={() => setLogPage((p) => p + 1)}>{"下一页"}</button>
            </div>
          )}
        </div>
      </section>
    </>
  );
}
