import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { ModelNode } from "../../lib/api";

export default function ModelsPage() {
  const api = useMemo(() => createApi(), []);
  const navigate = useNavigate();
  const [nodes, setNodes] = useState<ModelNode[]>([]);
  const [filter, setFilter] = useState<string>("all");
  const [showImport, setShowImport] = useState(false);
  const [importPath, setImportPath] = useState("");
  const [importName, setImportName] = useState("");
  const [importing, setImporting] = useState(false);

  const loadNodes = () => api.getModelNodes().then(setNodes);
  useEffect(() => { loadNodes(); }, [api]);

  const handleImport = async () => {
    if (!importPath) return;
    setImporting(true);
    try {
      await api.createModelNode({
        artifactPath: importPath,
        name: importName || importPath.split("/").pop(),
        taskType: "object_detection",
        modelFamily: "yolo",
      });
      setShowImport(false);
      setImportPath("");
      setImportName("");
      loadNodes();
    } catch (e) {
      alert("导入失败: " + (e as Error).message);
    }
    setImporting(false);
  };

  const roots = nodes.filter((n) => n.parentId === null);
  const byParent = (pid: string) => nodes.filter((n) => n.parentId === pid);

  const filteredRoots =
    filter === "all"
      ? roots
      : roots.filter((r) => r.status === filter);

  return (
    <>
      <div className="top-actions" style={{ marginBottom: 16, display: "flex", gap: 10, alignItems: "center" }}>
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
        <button
          className="btn primary"
          onClick={() => setShowImport(true)}
          style={{ marginLeft: "auto" }}
        >
          + 导入根模型
        </button>
      </div>

      {showImport && (
        <div className="card" style={{ marginBottom: 14, padding: 20 }}>
          <div className="card-title" style={{ marginBottom: 12 }}>导入根模型</div>
          <div style={{ display: "grid", gap: 10, maxWidth: 500 }}>
            <div>
              <label style={{ fontSize: 11, color: "#666" }}>模型文件路径</label>
              <input
                value={importPath}
                onChange={(e) => setImportPath(e.target.value)}
                placeholder="/path/to/yolov8n.pt"
                style={{ width: "100%", padding: "8px 10px", border: "1px solid #bed2df", borderRadius: 3, fontSize: 12, marginTop: 4 }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#666" }}>模型名称（可选）</label>
              <input
                value={importName}
                onChange={(e) => setImportName(e.target.value)}
                placeholder="yolov8n"
                style={{ width: "100%", padding: "8px 10px", border: "1px solid #bed2df", borderRadius: 3, fontSize: 12, marginTop: 4 }}
              />
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn primary" onClick={handleImport} disabled={importing}>
                {importing ? "导入中..." : "确认导入"}
              </button>
              <button className="btn" onClick={() => setShowImport(false)}>取消</button>
            </div>
          </div>
        </div>
      )}

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
