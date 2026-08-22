import { useEffect, useState } from "react";
import { createApi } from "../../lib/createApi";
import type { Evaluation } from "../../lib/api";

export default function EvaluationsPage() {
  const api = createApi();
  const [evals, setEvals] = useState<Evaluation[]>([]);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    api.getEvaluations().then(setEvals);
  }, []);

  const current = evals.find((e) => e.id === selected);

  const statusLabel = (s: string) => {
    switch (s) {
      case "pending": return "待处理";
      case "auto_passed": return "自动通过";
      case "approved": return "已通过";
      case "rejected": return "已拒绝";
      default: return s;
    }
  };

  const statusBadge = (s: string) => {
    switch (s) {
      case "pending": return "badge-orange";
      case "auto_passed": return "badge-blue";
      case "approved": return "badge-green";
      case "rejected": return "badge-red";
      default: return "badge-gray";
    }
  };

  return (
    <div className="page-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
      <div className="card">
        <div className="card-head">
          <div>
            <div className="card-title">评估收件箱</div>
            <div className="card-kicker">
              {evals.filter((e) => e.humanStatus === "pending").length} 个待人工验收
            </div>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>模型</th>
                <th>自动评估</th>
                <th>人工验收</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {evals.map((e) => (
                <tr
                  key={e.id}
                  onClick={() => setSelected(e.id)}
                  style={{
                    cursor: "pointer",
                    background: selected === e.id ? "#e7f1fa" : undefined,
                  }}
                >
                  <td>
                    <div style={{ fontWeight: 600 }}>{e.modelName}</div>
                    <div style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "Courier New, monospace" }}>
                      {e.modelNodeId}
                    </div>
                  </td>
                  <td>
                    <span className={`badge ${statusBadge(e.autoStatus)}`}>
                      {statusLabel(e.autoStatus)}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${statusBadge(e.humanStatus)}`}>
                      {statusLabel(e.humanStatus)}
                    </span>
                  </td>
                  <td>
                    <button className="btn small">查看</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <div>
            <div className="card-title">评估详情</div>
            <div className="card-kicker">{current ? current.modelName : "选择评估查看详情"}</div>
          </div>
        </div>
        <div className="card-body">
          {!current ? (
            <div className="empty-state">点击左侧评估查看详情</div>
          ) : (
            <div className="detail-list">
              <div className="detail-row">
                <span>模型节点</span>
                <b>{current.modelNodeId}</b>
              </div>
              <div className="detail-row">
                <span>数据集快照</span>
                <b>{current.datasetName}</b>
              </div>
              <div className="detail-row">
                <span>测试图片数</span>
                <b>{current.testImageCount} 张</b>
              </div>
              <hr style={{ border: "none", borderTop: "1px solid #d7e5ed", margin: "8px 0" }} />
              {current.metrics && (
                <>
                  <div className="detail-row">
                    <span>Precision</span>
                    <b>{(current.metrics.precision * 100).toFixed(1)}%</b>
                  </div>
                  <div className="detail-row">
                    <span>Recall</span>
                    <b>{(current.metrics.recall * 100).toFixed(1)}%</b>
                  </div>
                  <div className="detail-row">
                    <span>mAP50</span>
                    <b>{(current.metrics.mAP50 * 100).toFixed(1)}%</b>
                  </div>
                  <div className="detail-row">
                    <span>mAP50-95</span>
                    <b>{(current.metrics.mAP50_95 * 100).toFixed(1)}%</b>
                  </div>
                </>
              )}
              <hr style={{ border: "none", borderTop: "1px solid #d7e5ed", margin: "8px 0" }} />
              <div className="detail-row">
                <span>自动评估</span>
                <b>
                  <span className={`badge ${statusBadge(current.autoStatus)}`}>
                    {statusLabel(current.autoStatus)}
                  </span>
                </b>
              </div>
              <div className="detail-row">
                <span>人工验收</span>
                <b>
                  <span className={`badge ${statusBadge(current.humanStatus)}`}>
                    {statusLabel(current.humanStatus)}
                  </span>
                </b>
              </div>
              <div className="detail-row">
                <span>创建时间</span>
                <b>{new Date(current.createdAt).toLocaleString("zh-CN")}</b>
              </div>
              {current.humanStatus === "pending" && (
                <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
                  <button className="btn primary">通过验收</button>
                  <button className="btn danger">拒绝</button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
