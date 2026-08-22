import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { TrainingTask } from "../../lib/api";

export default function TrainingPage() {
  const api = createApi();
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<TrainingTask[]>([]);

  useEffect(() => {
    api.getTrainingTasks().then(setTasks);
  }, []);

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

  return (
    <>
      <div style={{ marginBottom: 16 }}>
        <button className="btn primary" onClick={() => navigate("/training/new")}>
          + 新建训练
        </button>
      </div>

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
                <tr key={t.id}>
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
                  <td>
                    {t.status === "running" || t.status === "completed" ? (
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div className="meter" style={{ width: 80, margin: 0 }}>
                          <div
                            className="meter-fill"
                            style={{
                              width: `${t.progress || 0}%`,
                            }}
                          />
                        </div>
                        <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                          {t.currentEpoch}/{t.epochs}
                        </span>
                      </div>
                    ) : (
                      <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                        {t.epochs} 轮
                      </span>
                    )}
                  </td>
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
    </>
  );
}
