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

export default function OperationsPage() {
  const api = useMemo(() => createApi(), []);
  const [status, setStatus] = useState<OperationStatusResponse | null>(null);
  const [logs, setLogs] = useState<OperationLogEntry[]>([]);
  const [logsTotal, setLogsTotal] = useState(0);
  const [errors, setErrors] = useState<OperationLogEntry[]>([]);
  const [errorsTotal, setErrorsTotal] = useState(0);
  const [logPage, setLogPage] = useState(0);
  const pageSize = 20;

  useEffect(() => {
    fetch("/api/operations/status", {
      headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
    })
      .then((r) => r.json())
      .then((data: OperationStatusResponse) => setStatus(data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetch(`/api/operations/logs?skip=${logPage * pageSize}&limit=${pageSize}`, {
      headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
    })
      .then((r) => r.json())
      .then((data: OperationLogsResponse) => {
        setLogs(data.items);
        setLogsTotal(data.total);
      })
      .catch(() => {});
  }, [logPage]);

  useEffect(() => {
    fetch("/api/operations/errors?limit=50", {
      headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
    })
      .then((r) => r.json())
      .then((data: OperationLogsResponse) => {
        setErrors(data.items);
        setErrorsTotal(data.total);
      })
      .catch(() => {});
  }, []);

  const statusLabel: Record<string, string> = {
    ready: "健康",
    degraded: "降级",
    unhealthy: "异常",
  };

  const statusColor: Record<string, string> = {
    ready: "var(--accent-green)",
    degraded: "var(--accent-orange)",
    unhealthy: "#b33",
  };

  const checkLabel: Record<string, string> = {
    postgres: "PostgreSQL",
    redis: "Redis",
    gpu: "GPU",
    worker: "Worker",
  };

  return (
    <>
      <section className="kpis">
        <div className="kpi">
          <div className="kpi-label">系统状态</div>
          <strong>
            {status ? statusLabel[status.status] || status.status : "加载中..."}
            <em
              style={{
                color: status ? statusColor[status.status] : undefined,
              }}
            >
              {status?.status === "ready" ? "正常运行" : ""}
            </em>
          </strong>
          <div className="kpi-foot">服务健康检查</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">操作日志</div>
          <strong>
            {String(logsTotal).padStart(2, "0")}
            <em>最近操作</em>
          </strong>
          <div className="kpi-foot">全部操作记录</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">异常记录</div>
          <strong>
            {String(errorsTotal).padStart(2, "0")}
            <em style={{ color: errorsTotal > 0 ? "#b33" : undefined }}>
              {errorsTotal > 0 ? "需要关注" : "无异常"}
            </em>
          </strong>
          <div className="kpi-foot">错误和异常</div>
        </div>
      </section>

      <section className="page-grid">
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">系统健康</div>
              <div className="card-kicker">服务组件状态</div>
            </div>
          </div>
          <div className="card-body">
            {status
              ? Object.entries(status.checks).map(([key, ok]) => (
                  <div
                    key={key}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr auto",
                      gap: 10,
                      padding: "10px 0",
                      borderBottom: "1px solid #dce8f0",
                    }}
                  >
                    <div style={{ color: "var(--text)", fontSize: 12 }}>
                      {checkLabel[key] || key}
                    </div>
                    <span
                      style={{
                        fontSize: 9,
                        fontFamily: "Courier New, monospace",
                        color: ok ? "var(--accent-green)" : "#b33",
                      }}
                    >
                      {ok ? "正常" : "异常"}
                    </span>
                  </div>
                ))
              : (
                  <div style={{ padding: "12px 0", color: "var(--text-muted)", fontSize: 11 }}>
                    加载中...
                  </div>
                )}
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">操作日志</div>
              <div className="card-kicker">
                共 {logsTotal} 条 · 第 {logPage + 1} 页
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                className="card-action"
                disabled={logPage === 0}
                onClick={() => setLogPage((p) => Math.max(0, p - 1))}
              >
                上一页
              </button>
              <button
                className="card-action"
                disabled={(logPage + 1) * pageSize >= logsTotal}
                onClick={() => setLogPage((p) => p + 1)}
              >
                下一页
              </button>
            </div>
          </div>
          <div>
            {logs.length === 0 ? (
              <div style={{ padding: "12px 16px", color: "var(--text-muted)", fontSize: 11 }}>
                暂无操作日志
              </div>
            ) : (
              logs.map((log) => (
                <div
                  key={log.id}
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
                      {log.operationType}
                    </div>
                    <div
                      style={{
                        marginTop: 3,
                        color: "var(--text-muted)",
                        fontSize: 9,
                        fontFamily: "Courier New, monospace",
                      }}
                    >
                      {log.actor || "system"} · {new Date(log.createdAt).toLocaleString("zh-CN")}
                    </div>
                  </div>
                  <span
                    style={{
                      fontSize: 9,
                      fontFamily: "Courier New, monospace",
                      textTransform: "uppercase",
                      color:
                        log.status === "success"
                          ? "var(--accent-green)"
                          : "#b33",
                      alignSelf: "center",
                    }}
                  >
                    {log.status === "success" ? "成功" : "失败"}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">异常记录</div>
              <div className="card-kicker">共 {errorsTotal} 条</div>
            </div>
          </div>
          <div>
            {errors.length === 0 ? (
              <div style={{ padding: "12px 16px", color: "var(--text-muted)", fontSize: 11 }}>
                暂无异常记录
              </div>
            ) : (
              errors.map((err) => (
                <div
                  key={err.id}
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
                      {err.operationType}
                    </div>
                    <div
                      style={{
                        marginTop: 3,
                        color: "#b33",
                        fontSize: 10,
                      }}
                    >
                      {err.errorSummary || "未知错误"}
                    </div>
                    <div
                      style={{
                        marginTop: 2,
                        color: "var(--text-muted)",
                        fontSize: 9,
                        fontFamily: "Courier New, monospace",
                      }}
                    >
                      {new Date(err.createdAt).toLocaleString("zh-CN")}
                    </div>
                  </div>
                  <span
                    style={{
                      fontSize: 9,
                      fontFamily: "Courier New, monospace",
                      color: "#b33",
                      alignSelf: "center",
                    }}
                  >
                    错误
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </section>
    </>
  );
}
