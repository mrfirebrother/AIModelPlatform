import { useEffect, useMemo, useState } from "react";
import { createApi } from "../../lib/createApi";

interface OperationLogEntry {
  id: string;
  operationType: string;
  actor: string | null;
  status: string;
  summaryJson: Record<string, unknown>;
  errorSummary: string | null;
  createdAt: string;
}

interface OperationLogsResponse {
  items: OperationLogEntry[];
  total: number;
}

interface SystemChecks {
  postgres: boolean;
  redis: boolean;
  gpu: boolean;
  worker: boolean;
}

interface OperationStatusResponse {
  status: string;
  checks: SystemChecks;
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

const OP_LABELS: Record<string, string> = {
  "training.task.create": "创建训练任务",
  "training.task.delete": "删除训练任务",
  "training.task.cancel": "取消训练任务",
  "training.task.dispatch": "调度训练任务",
  "model.create": "导入模型",
  "model.update_status": "更新模型状态",
  "model.delete": "删除模型",
  "evaluation.create": "创建评估",
  "evaluation.review": "人工评审",
  "evaluation.session.create": "创建评估会话",
};

const OP_COLORS: Record<string, string> = {
  "training.task.create": "#1766ad",
  "training.task.delete": "#c44",
  "training.task.cancel": "#d77d59",
  "model.create": "#1a8a75",
  "model.delete": "#c44",
  "evaluation.create": "#36a1bd",
  "evaluation.review": "#8b5cf6",
};

export default function OperationsPage() {
  const api = useMemo(() => createApi(), []);
  const [status, setStatus] = useState<OperationStatusResponse | null>(null);
  const [logs, setLogs] = useState<OperationLogEntry[]>([]);
  const [logsTotal, setLogsTotal] = useState(0);
  const [logPage, setLogPage] = useState(0);
  const pageSize = 10;

  useEffect(() => {
    fetch("/api/operations/status", {
      headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
    }).then((r) => r.json()).then((d: OperationStatusResponse) => setStatus(d)).catch(() => {});
  }, []);

  useEffect(() => {
    fetch(`/api/operations/logs?skip=${logPage * pageSize}&limit=${pageSize}`, {
      headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
    }).then((r) => r.json()).then((d: OperationLogsResponse) => { setLogs(d.items); setLogsTotal(d.total); }).catch(() => {});
  }, [logPage]);

  const checkItems = [
    { key: "postgres", label: "PostgreSQL", icon: "⛁" },
    { key: "redis", label: "Redis", icon: "⚡" },
    { key: "gpu", label: "GPU", icon: "▣" },
    { key: "worker", label: "Worker", icon: "⚙" },
  ];

  return (
    <>
      <section style={{ display: "grid", gridTemplateColumns: "minmax(200px, 1fr) 2fr", gap: 12, alignItems: "start" }}>
        {/* System Health */}
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 8 }}>系统健康</div>
          {status ? checkItems.map(({ key, label }) => {
            const ok = status.checks[key as keyof SystemChecks];
            return (
              <div key={key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 0" }}>
                <span style={{ fontSize: 11 }}>{label}</span>
                <span className={`badge ${ok ? "badge-green" : "badge-red"}`} style={{ fontSize: 9 }}>{ok ? "正常" : "异常"}</span>
              </div>
            );
          }) : <div style={{ fontSize: 11, color: "var(--text-muted)" }}>加载中...</div>}
        </div>

        {/* Operation Logs */}
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ padding: "14px 18px", borderBottom: "1px solid #e6eef6", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontWeight: 600, fontSize: 12 }}>操作日志</div>
              <div style={{ fontSize: 10, color: "var(--text-muted)" }}>共 {logsTotal} 条 · 第 {logPage + 1} 页</div>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <button className="btn small" disabled={logPage === 0} onClick={() => setLogPage((p) => Math.max(0, p - 1))} style={{ fontSize: 10 }}>上一页</button>
              <button className="btn small" disabled={(logPage + 1) * pageSize >= logsTotal} onClick={() => setLogPage((p) => p + 1)} style={{ fontSize: 10 }}>下一页</button>
            </div>
          </div>
          <div>
            {logs.length === 0 ? (
              <div style={{ padding: "32px 18px", textAlign: "center", color: "var(--text-muted)", fontSize: 11 }}>暂无操作日志</div>
            ) : logs.map((log) => (
              <div key={log.id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 18px", borderBottom: "1px solid #f0f4f8" }}>
                <div style={{ width: 4, height: 24, borderRadius: 2, background: log.status === "success" ? OP_COLORS[log.operationType] || "var(--accent-green)" : "#b33", flexShrink: 0 }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontWeight: 600, fontSize: 12 }}>{OP_LABELS[log.operationType] || log.operationType}</span>
                    <span style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "Courier New, monospace" }}>{log.operationType}</span>
                  </div>
                  <div style={{ marginTop: 2, fontSize: 10, color: "var(--text-muted)" }}>{log.actor || "system"} · {formatTime(log.createdAt)}</div>
                </div>
                <span className={`badge ${log.status === "success" ? "badge-green" : "badge-red"}`} style={{ fontSize: 9 }}>{log.status === "success" ? "成功" : "失败"}</span>
              </div>
            ))}
          </div>
          {logsTotal > pageSize && (
            <div style={{ padding: "12px 18px", borderTop: "1px solid #e6eef6", display: "flex", justifyContent: "center", gap: 8 }}>
              <button className="btn small" disabled={logPage === 0} onClick={() => setLogPage((p) => Math.max(0, p - 1))}>上一页</button>
              <span style={{ fontSize: 11, lineHeight: "28px" }}>{logPage + 1} / {Math.ceil(logsTotal / pageSize)}</span>
              <button className="btn small" disabled={(logPage + 1) * pageSize >= logsTotal} onClick={() => setLogPage((p) => p + 1)}>下一页</button>
            </div>
          )}
        </div>
      </section>
    </>
  );
}
