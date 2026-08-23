import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { TrainingTask, TrainingLogs, TrainingMetrics, TrainingCheckpoint } from "../../lib/api";

const STATUS_LABEL: Record<string, string> = {
  running: "\u8fd0\u884c\u4e2d",
  queued: "\u6392\u961f\u4e2d",
  completed: "\u5df2\u5b8c\u6210",
  failed: "\u5931\u8d25",
  cancelled: "\u5df2\u53d6\u6d88",
};
const STATUS_BADGE: Record<string, string> = {
  running: "badge-cyan",
  queued: "badge-orange",
  completed: "badge-green",
  failed: "badge-red",
  cancelled: "badge-gray",
};

export default function TrainingPage() {
  const api = useMemo(() => createApi(), []);
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [logs, setLogs] = useState<TrainingLogs | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [checkpoint, setCheckpoint] = useState<TrainingCheckpoint | null>(null);
  const [tab, setTab] = useState<"logs" | "loss" | "checkpoint">("logs");

  useEffect(() => {
    api.getTrainingTasks().then(setTasks).catch(() => {});
  }, [api]);

  useEffect(() => {
    if (!selectedId) { setLogs(null); setMetrics(null); setCheckpoint(null); return; }
    Promise.all([
      api.getTrainingLogs(selectedId),
      api.getTrainingMetrics(selectedId),
      api.getTrainingCheckpoint(selectedId),
    ]).then(([l, m, c]) => { setLogs(l); setMetrics(m); setCheckpoint(c); }).catch(() => {});
  }, [selectedId, api]);

  const selected = tasks.find((t) => t.id === selectedId);

  return (
    <div className="page-grid" style={{ gridTemplateColumns: "1fr 1.2fr" }}>
      <div className="card">
        <div className="card-head">
          <div>
            <div className="card-title">{"\u8bad\u7ec3\u4efb\u52a1"}</div>
            <div className="card-kicker">{tasks.length} {"\u6761\u8bb0\u5f55"}</div>
          </div>
          <button className="btn primary small" onClick={() => navigate("/training/create")}>{"+ \u65b0\u5efa"}</button>
        </div>
        <div>
          {tasks.map((t) => (
            <div key={t.id} onClick={() => setSelectedId(t.id)} style={{
              padding: "10px 16px", cursor: "pointer", borderBottom: "1px solid #dce8f0",
              background: selectedId === t.id ? "#e7f1fa" : undefined,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontWeight: 600, fontSize: 12 }}>{t.parentModelName || t.id}</div>
                <span className={`badge ${STATUS_BADGE[t.status] || "badge-gray"}`} style={{ fontSize: 9 }}>{STATUS_LABEL[t.status] || t.status}</span>
              </div>
              <div style={{ marginTop: 3, color: "var(--text-muted)", fontSize: 9, fontFamily: "Courier New, monospace" }}>
                {t.datasetName || t.datasetSnapshotId} {"\u00b7"} {t.epochs} {"\u8f6e"}
              </div>
            </div>
          ))}
          {tasks.length === 0 && <div style={{ padding: 20, color: "var(--text-muted)", fontSize: 12, textAlign: "center" }}>{"\u6682\u65e0\u8bad\u7ec3\u4efb\u52a1"}</div>}
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <div>
            <div className="card-title">{selected ? selected.parentModelName || selected.id : "\u8bad\u7ec3\u8be6\u60c5"}</div>
            <div className="card-kicker">{selected ? `${STATUS_LABEL[selected.status] || selected.status} \u00b7 ${selected.epochs} {"\u8f6e"}` : "\u9009\u62e9\u4efb\u52a1\u67e5\u770b"}</div>
          </div>
        </div>
        {selected && (
          <div>
            <div style={{ display: "flex", borderBottom: "1px solid #dce8f0" }}>
              {(["logs", "loss", "checkpoint"] as const).map((t) => (
                <button key={t} onClick={() => setTab(t)} style={{
                  flex: 1, padding: "10px 0", border: 0, background: tab === t ? "#e7f1fa" : undefined,
                  color: tab === t ? "var(--accent)" : "var(--text-muted)", fontSize: 11, fontWeight: tab === t ? 600 : 400, cursor: "pointer",
                }}>
                  {t === "logs" ? "\u65e5\u5fd7" : t === "loss" ? "Loss \u66f2\u7ebf" : "Checkpoint"}
                </button>
              ))}
            </div>
            <div style={{ padding: 16 }}>
              {tab === "logs" && logs && (
                <div style={{ fontFamily: "Courier New, monospace", fontSize: 10, lineHeight: 1.6 }}>
                  <div>{"\u72b6\u6001"}: <span className={`badge ${STATUS_BADGE[logs.status] || "badge-gray"}`}>{STATUS_LABEL[logs.status] || logs.status}</span></div>
                  <div>{"\u5f53\u524d Epoch"}: {logs.currentEpoch} / {logs.totalEpochs}</div>
                  <div>{"\u5c1d\u8bd5\u6b21\u6570"}: {logs.attempts?.length || 0}</div>
                  {logs.attempts?.[0]?.logPath && <div>{"\u65e5\u5fd7\u8def\u5f84"}: {logs.attempts[0].logPath}</div>}
                </div>
              )}
              {tab === "loss" && metrics && (
                <div>
                  {metrics.bestLoss !== null && <div style={{ marginBottom: 8 }}>{"\u6700\u4f73 Loss"}: {metrics.bestLoss.toFixed(4)}</div>}
                  {metrics.epochs?.map((ep) => (
                    <div key={ep.epoch} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, fontSize: 10 }}>
                      <span style={{ width: 24, fontFamily: "Courier New, monospace" }}>E{ep.epoch}</span>
                      <div style={{ flex: 1, height: 4, background: "#dce8f0", borderRadius: 2 }}>
                        <div style={{ height: "100%", width: `${Math.min((1 - ep.loss) * 100, 100)}%`, background: "var(--accent)", borderRadius: 2 }} />
                      </div>
                      <span style={{ width: 50, fontFamily: "Courier New, monospace" }}>{ep.loss.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              )}
              {tab === "checkpoint" && checkpoint && (
                <div>
                  <div style={{ fontSize: 11, marginBottom: 8 }}>{"\u6700\u65b0 Checkpoint"}:</div>
                  {checkpoint.latestCheckpoint ? (
                    <div className="detail-list">
                      <div className="detail-row"><span>{"Epoch"}</span><b>{checkpoint.latestCheckpoint.epoch}</b></div>
                      <div className="detail-row"><span>{"\u6587\u4ef6"}</span><b style={{ fontSize: 9 }}>{checkpoint.latestCheckpoint.artifactPath}</b></div>
                      <div className="detail-row"><span>{"Hash"}</span><b style={{ fontSize: 9 }}>{checkpoint.latestCheckpoint.artifactHash}</b></div>
                    </div>
                  ) : <div style={{ color: "var(--text-muted)", fontSize: 11 }}>{"\u6682\u65e0 Checkpoint"}</div>}
                </div>
              )}
              {!logs && tab === "logs" && <div style={{ color: "var(--text-muted)" }}>{"\u52a0\u8f7d\u4e2d..."}</div>}
              {!metrics && tab === "loss" && <div style={{ color: "var(--text-muted)" }}>{"\u52a0\u8f7d\u4e2d..."}</div>}
              {!checkpoint && tab === "checkpoint" && <div style={{ color: "var(--text-muted)" }}>{"\u52a0\u8f7d\u4e2d..."}</div>}
            </div>
          </div>
        )}
        {!selected && <div className="empty-state">{"\u9009\u62e9\u4efb\u52a1\u67e5\u770b\u8be6\u60c5"}</div>}
      </div>
    </div>
  );
}
