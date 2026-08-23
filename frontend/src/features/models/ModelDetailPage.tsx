import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { ModelNode } from "../../lib/api";

export default function ModelDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const api = useMemo(() => createApi(), []);
  const [node, setNode] = useState<ModelNode | null>(null);

  useEffect(() => {
    if (id) api.getModelNode(id).then(setNode);
  }, [id, api]);

  if (!node) return <div className="empty-state">{"\u52a0\u8f7d\u4e2d..."}</div>;

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title">{"\u6a21\u578b\u8282\u70b9\u8be6\u60c5"}</div>
          <div className="card-kicker">{node.id}</div>
        </div>
        <button className="btn small" onClick={() => navigate("/models")}>{"\u8fd4\u56de\u5217\u8868"}</button>
      </div>
      <div className="card-body">
        <div className="detail-list">
          <div className="detail-row"><span>{"\u6a21\u578b ID"}</span><b>{node.id}</b></div>
          <div className="detail-row"><span>{"\u6a21\u578b\u540d\u79f0"}</span><b>{node.name}</b></div>
          <div className="detail-row"><span>{"\u7236\u8282\u70b9"}</span><b>{node.parentId || "\u65e0\uff08\u6839\u6a21\u578b\uff09"}</b></div>
          <div className="detail-row"><span>{"\u4efb\u52a1\u7c7b\u578b"}</span><b>{node.taskType}</b></div>
          <div className="detail-row"><span>{"\u6a21\u578b\u65cf"}</span><b>{node.modelFamily}</b></div>
          <div className="detail-row"><span>{"\u6807\u7b7e\u4f53\u7cfb"}</span><b>{node.labelSchemaName}</b></div>
          <div className="detail-row">
            <span>{"\u72b6\u6001"}</span>
            <b><span className={`badge ${node.status === "approved" ? "badge-green" : node.status === "candidate" ? "badge-orange" : "badge-gray"}`}>
              {node.status === "approved" ? "\u5df2\u6279\u51c6" : node.status === "candidate" ? "\u5019\u9009" : node.status}
            </span></b>
          </div>
          {node.metrics && (
            <>
              <div className="detail-row"><span>Precision</span><b>{(node.metrics.precision * 100).toFixed(1)}%</b></div>
              <div className="detail-row"><span>Recall</span><b>{(node.metrics.recall * 100).toFixed(1)}%</b></div>
              <div className="detail-row"><span>mAP50</span><b>{(node.metrics.mAP50 * 100).toFixed(1)}%</b></div>
            </>
          )}
          <div className="detail-row"><span>{"\u521b\u5efa\u65f6\u95f4"}</span><b>{new Date(node.createdAt).toLocaleString("zh-CN")}</b></div>
        </div>
      </div>
    </div>
  );
}
