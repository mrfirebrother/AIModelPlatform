import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type {
  TrainingTask,
  TrainingLogs,
  TrainingMetrics,
  TrainingCheckpoint,
} from "../../lib/api";

export default function TrainingPage() {
  const api = createApi();
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [logs, setLogs] = useState<TrainingLogs | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [checkpoint, setCheckpoint] = useState<TrainingCheckpoint | null>(null);
  const [activeTab, setActiveTab] = useState<"logs" | "metrics" | "checkpoint">("logs");

  useEffect(() => {
    api.getTrainingTasks().then(setTasks);
  }, []);

  const selectTask = useCallback(
    async (id: string) => {
      setSelectedId(id);
      try {
        const [logsData, metricsData, checkpointData] = await Promise.all([
          api.getTrainingLogs(id),
          api.getTrainingMetrics(id),
          api.getTrainingCheckpoint(id),
        ]);
        setLogs(logsData);
        setMetrics(metricsData);
        setCheckpoint(checkpointData);
      } catch {
        setLogs(null);
        setMetrics(null);
        setCheckpoint(null);
      }
    },
    [api]
  );

  const statusLabel = (s: string) => {
    switch (s) {
      case "running": return "运行中";
      case "queued": return "排队中";
      case "completed": return "已完成";
      case "failed": return "失败";
      case "cancelled": return "已取消";
      default: return s;
    }
  };

  const statusBadge = (s: string) => {
    switch (s) {
      case "running": return "badge-blue";
      case "queued": return "badge-orange";
      case "completed": return "badge-green";
      case "failed": return "badge-red";
      case "cancelled": return "badge-gray";
      default: return "badge-gray";
    }
  };

  const renderProgress = (t: TrainingTask) => {
    if (t.status === "running" || t.status === "completed") {
      return (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div className="meter" style={{ width: 80, margin: 0 }}>
            <div
              className="meter-fill"
              style={{ width: `${t.progress || 0}%` }}
            />
          </div>
          <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
            {t.currentEpoch}/{t.epochs}
          </span>
        </div>
      );
    }
    return (
      <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
        {t.epochs} 轮
      </span>
    );
  };

  const renderLogsPanel = () => {
    if (!logs) return <div style={{ padding: 16, color: "var(--text-muted)" }}>加载中...</div>;

    return (
      <div style={{ padding: 16 }}>
        <div style={{ marginBottom: 12, display: "flex", gap: 16, fontSize: 12, color: "var(--text-muted)" }}>
          <span>状态: <span className={`badge ${statusBadge(logs.status)}`}>{statusLabel(logs.status)}</span></span>
          <span>进度: {logs.currentEpoch}/{logs.totalEpochs} epoch</span>
          <span>尝试次数: {logs.attempts.length}</span>
        </div>

        {logs.attempts.length === 0 ? (
          <div style={{ color: "var(--text-muted)", fontSize: 13 }}>暂无尝试记录</div>
        ) : (
          <div style={{ maxHeight: 300, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 6, fontFamily: "monospace", fontSize: 11, lineHeight: 1.6 }}>
            {logs.attempts.map((a) => (
              <div key={a.attemptNo} style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)" }}>
                <div style={{ display: "flex", gap: 12 }}>
                  <span style={{ color: "var(--text-muted)" }}>Attempt #{a.attemptNo}</span>
                  <span className={`badge ${statusBadge(a.status)}`}>{statusLabel(a.status)}</span>
                  <span>Epoch: {a.currentEpoch}</span>
                  {a.logPath && <span style={{ color: "var(--text-muted)" }}>Log: {a.logPath}</span>}
                </div>
                {a.lastError && (
                  <div style={{ color: "var(--danger, #e53e3e)", marginTop: 4 }}>Error: {a.lastError}</div>
                )}
                {a.startedAt && (
                  <div style={{ color: "var(--text-muted)", fontSize: 10 }}>
                    {a.startedAt} {a.finishedAt ? `- ${a.finishedAt}` : "(running)"}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const renderMetricsPanel = () => {
    if (!metrics) return <div style={{ padding: 16, color: "var(--text-muted)" }}>加载中...</div>;

    if (metrics.epochs.length === 0) {
      return <div style={{ padding: 16, color: "var(--text-muted)" }}>暂无指标数据</div>;
    }

    const maxLoss = Math.max(...metrics.epochs.map((e) => e.loss));
    const minLoss = Math.min(...metrics.epochs.map((e) => e.loss));
    const range = maxLoss - minLoss || 1;

    return (
      <div style={{ padding: 16 }}>
        <div style={{ marginBottom: 12, fontSize: 12, color: "var(--text-muted)" }}>
          Best Loss: {metrics.bestLoss?.toFixed(4) ?? "-"} | Epochs: {metrics.epochs.length}
        </div>

        <div style={{ display: "flex", gap: 4, alignItems: "flex-end", height: 120, border: "1px solid var(--border)", borderRadius: 6, padding: "8px 4px", overflowX: "auto" }}>
          {metrics.epochs.map((e) => {
            const height = ((e.loss - minLoss) / range) * 100 + 10;
            return (
              <div
                key={e.epoch}
                title={`Epoch ${e.epoch}: loss=${e.loss.toFixed(4)}, lr=${e.lr ?? "-"}`}
                style={{
                  flex: "1 0 auto",
                  minWidth: 12,
                  maxWidth: 24,
                  height: `${height}%`,
                  background: "var(--primary, #3182ce)",
                  borderRadius: "2px 2px 0 0",
                  transition: "height 0.2s",
                }}
              />
            );
          })}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
          <span>Epoch 1</span>
          <span>Loss: {minLoss.toFixed(4)} ~ {maxLoss.toFixed(4)}</span>
          <span>Epoch {metrics.epochs.length}</span>
        </div>
      </div>
    );
  };

  const renderCheckpointPanel = () => {
    if (!checkpoint) return <div style={{ padding: 16, color: "var(--text-muted)" }}>加载中...</div>;

    return (
      <div style={{ padding: 16 }}>
        <div style={{ marginBottom: 12, fontSize: 12, color: "var(--text-muted)" }}>
          Attempt: #{checkpoint.attemptNo ?? "-"}
        </div>

        {checkpoint.latestCheckpoint ? (
          <div style={{ border: "1px solid var(--border)", borderRadius: 6, padding: 12, fontSize: 12 }}>
            <div style={{ marginBottom: 8, fontWeight: 600 }}>Checkpoint @ Epoch {checkpoint.latestCheckpoint.epoch}</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <div>
                <div style={{ color: "var(--text-muted)", fontSize: 10 }}>Artifact Path</div>
                <div style={{ fontFamily: "monospace", fontSize: 11 }}>{checkpoint.latestCheckpoint.artifactPath}</div>
              </div>
              <div>
                <div style={{ color: "var(--text-muted)", fontSize: 10 }}>Artifact Hash</div>
                <div style={{ fontFamily: "monospace", fontSize: 10, wordBreak: "break-all" }}>{checkpoint.latestCheckpoint.artifactHash}</div>
              </div>
            </div>
            {Object.keys(checkpoint.latestCheckpoint.metrics).length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ color: "var(--text-muted)", fontSize: 10, marginBottom: 4 }}>Metrics</div>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  {Object.entries(checkpoint.latestCheckpoint.metrics).map(([k, v]) => (
                    <div key={k}>
                      <span style={{ color: "var(--text-muted)" }}>{k}: </span>
                      <span style={{ fontFamily: "monospace" }}>{typeof v === "number" ? v.toFixed(4) : String(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div style={{ marginTop: 8, color: "var(--text-muted)", fontSize: 10 }}>
              Created: {checkpoint.latestCheckpoint.createdAt}
            </div>
          </div>
        ) : (
          <div style={{ color: "var(--text-muted)", fontSize: 13 }}>暂无 Checkpoint</div>
        )}
      </div>
    );
  };

  const selectedTask = tasks.find((t) => t.id === selectedId);

  return (
    <>
      <div style={{ marginBottom: 16 }}>
        <button className="btn primary" onClick={() => navigate("/training/new")}>
          + 新建训练
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: selectedId ? "320px 1fr" : "1fr", gap: 16 }}>
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">训练任务队列</div>
              <div className="card-kicker">
                {tasks.filter((t) => t.status === "running").length} 个运行中 ·{" "}
                {tasks.filter((t) => t.status === "queued").length} 个等待
              </div>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>任务 ID</th>
                  <th>父模型</th>
                  <th>数据集</th>
                  <th>进度</th>
                  <th>状态</th>
                  <th>创建时间</th>
                </tr>
              </thead>
              <tbody>
                {tasks.map((t) => (
                  <tr
                    key={t.id}
                    onClick={() => selectTask(t.id)}
                    style={{
                      cursor: "pointer",
                      background: t.id === selectedId ? "var(--bg-hover, rgba(0,0,0,0.04))" : undefined,
                    }}
                  >
                    <td
                      style={{
                        fontFamily: "Courier New, monospace",
                        fontSize: 10,
                      }}
                    >
                      {t.id}
                    </td>
                    <td>{t.parentModelName || t.parentModelNodeId || "-"}</td>
                    <td>{t.datasetName || t.datasetSnapshotId}</td>
                    <td>{renderProgress(t)}</td>
                    <td>
                      <span className={`badge ${statusBadge(t.status)}`}>
                        {statusLabel(t.status)}
                      </span>
                    </td>
                    <td style={{ fontSize: 11 }}>
                      {new Date(t.createdAt).toLocaleString("zh-CN")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {selectedId && selectedTask && (
          <div className="card">
            <div className="card-head">
              <div>
                <div className="card-title">
                  监控 - {selectedTask.modelFamily} ({selectedTask.taskType})
                </div>
                <div className="card-kicker">{selectedTask.id}</div>
              </div>
            </div>
            <div style={{ display: "flex", borderBottom: "1px solid var(--border)" }}>
              {(["logs", "metrics", "checkpoint"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  style={{
                    padding: "8px 16px",
                    border: "none",
                    borderBottom: activeTab === tab ? "2px solid var(--primary, #3182ce)" : "2px solid transparent",
                    background: "none",
                    cursor: "pointer",
                    fontSize: 13,
                    fontWeight: activeTab === tab ? 600 : 400,
                    color: activeTab === tab ? "var(--primary, #3182ce)" : "var(--text-muted)",
                  }}
                >
                  {tab === "logs" && "日志"}
                  {tab === "metrics" && "Loss 曲线"}
                  {tab === "checkpoint" && "Checkpoint"}
                </button>
              ))}
            </div>
            {activeTab === "logs" && renderLogsPanel()}
            {activeTab === "metrics" && renderMetricsPanel()}
            {activeTab === "checkpoint" && renderCheckpointPanel()}
          </div>
        )}
      </div>
    </>
  );
}
