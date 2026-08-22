import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { ModelNode } from "../../lib/api";

export default function ModelsPage() {
  const api = createApi();
  const navigate = useNavigate();
  const [nodes, setNodes] = useState<ModelNode[]>([]);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    api.getModelNodes().then(setNodes);
  }, []);

  const roots = nodes.filter((n) => n.parentId === null);
  const byParent = (pid: string) => nodes.filter((n) => n.parentId === pid);

  const filteredRoots =
    filter === "all"
      ? roots
      : roots.filter((r) => r.status === filter);

  return (
    <>
      <div className="top-actions" style={{ marginBottom: 16 }}>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ padding: "6px 10px", border: "1px solid #bed2df", borderRadius: 3, fontSize: 11 }}
        >
          <option value="all">全部状态</option>
          <option value="approved">已批准</option>
          <option value="candidate">候选</option>
          <option value="archived">已归档</option>
        </select>
      </div>

      {filteredRoots.map((root) => (
        <div key={root.id} className="card" style={{ marginBottom: 14 }}>
          <div className="card-head">
            <div>
              <div className="card-title">根模型谱系</div>
              <div className="card-kicker">
                {root.name} · {root.modelFamily} · 不可变节点
              </div>
            </div>
            <span className="badge badge-green">{root.status}</span>
          </div>
          <div className="tree-container">
            <TreeNode
              node={root}
              children={byParent(root.id)}
              allNodes={nodes}
              onNavigate={(id) => navigate(`/models/${id}`)}
            />
          </div>
        </div>
      ))}

      {filteredRoots.length === 0 && (
        <div className="empty-state">暂无模型节点</div>
      )}
    </>
  );
}

function TreeNode({
  node,
  children,
  allNodes,
  onNavigate,
}: {
  node: ModelNode;
  children: ModelNode[];
  allNodes: ModelNode[];
  onNavigate: (id: string) => void;
}) {
  const grandchildren = (pid: string) => allNodes.filter((n) => n.parentId === pid);

  return (
    <div>
      <div
        className={`tree-node ${node.parentId === null ? "root" : ""}`}
        onClick={() => onNavigate(node.id)}
      >
        <div className="node-header">
          <span className="node-kind">
            {node.parentId === null ? "根模型" : "任务模型"}
          </span>
          <span className="node-state">
            {node.status === "approved" ? "已批准" : node.status === "candidate" ? "候选" : node.status}
          </span>
        </div>
        <div className="node-name">{node.name}</div>
        <div className="node-id">
          {node.id} · {node.labelSchemaName}
        </div>
      </div>
      {children.length > 0 && (
        <div className="tree-children">
          {children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              children={grandchildren(child.id)}
              allNodes={allNodes}
              onNavigate={onNavigate}
            />
          ))}
        </div>
      )}
    </div>
  );
}
