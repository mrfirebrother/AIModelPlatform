import { useEffect, useMemo, useState } from "react";
import { createApi } from "../../lib/createApi";
import type { Evaluation } from "../../lib/api";

const STATUS_LABEL: Record<string, string> = {
  pending: "\u5f85\u5904\u7406",
  auto_passed: "\u81ea\u52a8\u901a\u8fc7",
  approved: "\u5df2\u901a\u8fc7",
  rejected: "\u5df2\u62d2\u7edd",
};

export default function EvaluationsPage() {
  const api = useMemo(() => createApi(), []);
  const [evals, setEvals] = useState<Evaluation[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const current = evals.find((e) => e.id === selected);

  useEffect(() => {
    api.getEvaluations().then(setEvals).catch(() => {});
  }, [api]);

  const statusBadge = (s: string) =>
    s === "pending" ? "badge-orange" : s === "auto_passed" ? "badge-blue" : s === "approved" ? "badge-green" : "badge-red";

  const handleApprove = async (id: string) => {
    await api.reviewEvaluation(id, "approved");
    setEvals((prev) => prev.map((e) => e.id === id ? { ...e, humanStatus: "approved" } : e));
  };

  const handleReject = async (id: string) => {
    await api.reviewEvaluation(id, "rejected");
    setEvals((prev) => prev.map((e) => e.id === id ? { ...e, humanStatus: "rejected" } : e));
  };

  return (
    <div className="page-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
      <div className="card">
        <div className="card-head">
          <div>
            <div className="card-title">{"\u8bc4\u4f30\u6536\u4ef6\u7bb1"}</div>
            <div className="card-kicker">{evals.filter((e) => e.humanStatus === "pending").length} {"\u4e2a\u5f85\u4eba\u5de5\u9a8c\u6536"}</div>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>{"\u6a21\u578b"}</th><th>{"\u81ea\u52a8\u8bc4\u4f30"}</th><th>{"\u4eba\u5de5\u9a8c\u6536"}</th><th>{"\u64cd\u4f5c"}</th></tr>
            </thead>
            <tbody>
              {evals.map((e) => (
                <tr key={e.id} onClick={() => setSelected(e.id)} style={{ cursor: "pointer", background: selected === e.id ? "#e7f1fa" : undefined }}>
                  <td><div style={{ fontWeight: 600 }}>{e.modelName}</div><div style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "Courier New, monospace" }}>{e.modelNodeId}</div></td>
                  <td><span className={`badge ${statusBadge(e.autoStatus)}`}>{STATUS_LABEL[e.autoStatus] || e.autoStatus}</span></td>
                  <td><span className={`badge ${statusBadge(e.humanStatus)}`}>{STATUS_LABEL[e.humanStatus] || e.humanStatus}</span></td>
                  <td><button className="btn small">{"\u67e5\u770b"}</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="card">
        <div className="card-head"><div><div className="card-title">{"\u8bc4\u4f30\u8be6\u60c5"}</div><div className="card-kicker">{current ? current.modelName : "\u9009\u62e9\u8bc4\u4f30\u67e5\u770b\u8be6\u60c5"}</div></div></div>
        <div className="card-body">
          {!current ? <div className="empty-state">{"\u70b9\u51fb\u5de6\u4fa7\u8bc4\u4f30\u67e5\u770b\u8be6\u60c5"}</div> : (
            <div className="detail-list">
              <div className="detail-row"><span>{"\u6a21\u578b\u8282\u70b9"}</span><b>{current.modelNodeId}</b></div>
              <div className="detail-row"><span>{"\u6570\u636e\u96c6\u5feb\u7167"}</span><b>{current.datasetName}</b></div>
              <div className="detail-row"><span>{"\u6d4b\u8bd5\u56fe\u7247\u6570"}</span><b>{current.testImageCount} {"\u5f20"}</b></div>
              <hr style={{ border: "none", borderTop: "1px solid #d7e5ed", margin: "8px 0" }} />
              {current.metrics && (<>
                <div className="detail-row"><span>Precision</span><b>{(current.metrics.precision * 100).toFixed(1)}%</b></div>
                <div className="detail-row"><span>Recall</span><b>{(current.metrics.recall * 100).toFixed(1)}%</b></div>
                <div className="detail-row"><span>mAP50</span><b>{(current.metrics.mAP50 * 100).toFixed(1)}%</b></div>
                <div className="detail-row"><span>mAP50-95</span><b>{(current.metrics.mAP50_95 * 100).toFixed(1)}%</b></div>
              </>)}
              <hr style={{ border: "none", borderTop: "1px solid #d7e5ed", margin: "8px 0" }} />
              <div className="detail-row"><span>{"\u81ea\u52a8\u8bc4\u4f30"}</span><b><span className={`badge ${statusBadge(current.autoStatus)}`}>{STATUS_LABEL[current.autoStatus] || current.autoStatus}</span></b></div>
              <div className="detail-row"><span>{"\u4eba\u5de5\u9a8c\u6536"}</span><b><span className={`badge ${statusBadge(current.humanStatus)}`}>{STATUS_LABEL[current.humanStatus] || current.humanStatus}</span></b></div>
              <div className="detail-row"><span>{"\u521b\u5efa\u65f6\u95f4"}</span><b>{new Date(current.createdAt).toLocaleString("zh-CN")}</b></div>
              {current.humanStatus === "pending" && (
                <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
                  <button className="btn primary" onClick={() => handleApprove(current.id)}>{"\u901a\u8fc7\u9a8c\u6536"}</button>
                  <button className="btn danger" onClick={() => handleReject(current.id)}>{"\u62d2\u7edd"}</button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
