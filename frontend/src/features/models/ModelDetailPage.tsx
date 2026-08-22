import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { ModelNode } from "../../lib/api";

export default function ModelDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const api = createApi();
  const [node, setNode] = useState<ModelNode | null>(null);

  useEffect(() => {
    if (id) api.getModelNode(id).then(setNode);
  }, [id]);

  if (!node) return <div className="empty-state">加载中...</div>;

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title">模型节点详情</div>
          <div className="card-kicker">{node.id}</div>
        </div>
        <button className="btn small" onClick={() => navigate("/models")}>
          返回列表
        </button>
      </div>
      <div className="card-body">
        <div className="detail-list">
          <div className="detail-row">
            <span>模型 ID</span>
            <b>{node.id}</b>
          </div>
          <div className="detail-row">
            <span>模型名称</span>
            <b>{node.name}</b>
          </div>
          <div className="detail-row">
            <span>父节点</span>
            <b>{node.parentId || "无 (根模型)"}</b>
          </div>
          <div className="detail-row">
            <span>任务类型</span>
            <b>{node.taskType}</b>
          </div>
          <div className="detail-row">
            <span>模型族</span>
            <b>{node.modelFamily}</b>
          </div>
          <div className="detail-row">
            <span>标签体系</span>
            <b>{node.labelSchemaName}</b>
          </div>
          <div className="detail-row">
            <span>状态</span>
            <b>
              <span className={`badge ${node.status === "approved" ? "badge-green" : node.status === "candidate" ? "badge-orange" : "badge-gray"}`}>
                {node.status === "approved" ? "已批准" : node.status === "candidate" ? "候选" : node.status}
              </span>
            </b>
          </div>
          {node.metrics && (
            <>
              <div className="detail-row">
                <span>Precision</span>
                <b>{(node.metrics.precision * 100).toFixed(1)}%</b>
              </div>
              <div className="detail-row">
                <span>Recall</span>
                <b>{(node.metrics.recall * 100).toFixed(1)}%</b>
              </div>
              <div className="detail-row">
                <span>mAP50</span>
                <b>{(node.metrics.mAP50 * 100).toFixed(1)}%</b>
              </div>
            </>
          )}
          <div className="detail-row">
            <span>创建时间</span>
            <b>{new Date(node.createdAt).toLocaleString("zh-CN")}</b>
          </div>
        </div>
      </div>
    </div>
  );
}
