import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { ModelNode } from "../../lib/api";

export default function ModelsPage() {
  const api = useMemo(() => createApi(), []);
  const navigate = useNavigate();
  const [nodes, setNodes] = useState<ModelNode[]>([]);
  const [filter, setFilter] = useState<string>("all");
  const [showImport, setShowImport] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadNodes = () => api.getModelNodes().then(setNodes);
  useEffect(() => { loadNodes(); }, [api]);

  const handleFile = async (file: File) => {
    if (!file.name.endsWith(".pt")) { setUploadError("请选择 .pt 模型文件"); return; }
    if (file.size > 500 * 1024 * 1024) { setUploadError("文件大小不能超过 500MB"); return; }
    setUploadError(null);
    setUploadPct(0);
    setUploading(true);
    try {
      const res = await api.uploadModel(file, (pct) => setUploadPct(pct));
      await api.createModelNode({
        artifactPath: res.file_path,
        name: file.name.replace(".pt", ""),
        taskType: "object_detection",
        modelFamily: "yolo",
      });
      setShowImport(false);
      loadNodes();
    } catch (err) {
      setUploadError("导入失败: " + (err as Error).message);
    }
    setUploading(false);
    setUploadPct(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const roots = nodes.filter((n) => n.parentId === null);
  const byParent = (pid: string) => nodes.filter((n) => n.parentId === pid);
  const filteredRoots = filter === "all" ? roots : roots.filter((r) => r.status === filter);

  return (
    <>
      <div className="top-actions" style={{ marginBottom: 16, display: "flex", gap: 10, alignItems: "center" }}>
        <select value={filter} onChange={(e) => setFilter(e.target.value)} style={{ padding: "6px 10px", border: "1px solid #bed2df", borderRadius: 3, fontSize: 11 }}>
          <option value="all">全部状态</option>
          <option value="approved">已批准</option>
          <option value="candidate">候选</option>
          <option value="archived">已归档</option>
        </select>
        <button className="btn primary" onClick={() => setShowImport(true)} style={{ marginLeft: "auto" }}>+ 导入根模型</button>
      </div>

      {showImport && (
        <div className="card" style={{ marginBottom: 14, padding: 24 }}>
          <div className="card-title" style={{ marginBottom: 16 }}>导入根模型</div>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => !uploading && fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${dragOver ? "#1766ad" : "#bed2df"}`,
              borderRadius: 8,
              padding: "40px 20px",
              textAlign: "center",
              cursor: uploading ? "default" : "pointer",
              background: dragOver ? "#f0f7fc" : "#fafbfc",
              transition: "all 0.2s",
            }}
          >
            <input ref={fileInputRef} type="file" accept=".pt" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
            {uploading ? (
              <>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>{"\u4e0a\u4f20\u4e2d..."}</div>
                <div style={{ width: 200, height: 6, background: "#e0e0e0", borderRadius: 3, margin: "0 auto" }}>
                  <div style={{ width: `${uploadPct ?? 0}%`, height: "100%", background: "#1766ad", borderRadius: 3, transition: "width 0.3s" }} />
                </div>
                <div style={{ marginTop: 6, fontSize: 11, color: "#666" }}>{uploadPct ?? 0}%</div>
              </>
            ) : (
              <>
                <div style={{ fontSize: 40, marginBottom: 8, color: "#ccc" }}>{"\u2b07"}</div>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{"\u62d6\u653e .pt \u6587\u4ef6\u5230\u6b64\u5904"}</div>
                <div style={{ fontSize: 11, color: "#999" }}>{"\u6216\u70b9\u51fb\u9009\u62e9\u6587\u4ef6 \u00b7 \u6700\u5927 500MB"}</div>
              </>
            )}
          </div>
          {uploadError && <div style={{ marginTop: 10, color: "#b33", fontSize: 12 }}>{uploadError}</div>}
          <div style={{ marginTop: 14, display: "flex", gap: 8 }}>
            <button className="btn" onClick={() => { setShowImport(false); setUploadError(null); }}>{"\u53d6\u6d88"}</button>
          </div>
        </div>
      )}

      {filteredRoots.map((root) => (
        <div key={root.id} className="card" style={{ marginBottom: 14 }}>
          <div className="card-head">
            <div>
              <div className="card-title">{"\u6839\u6a21\u578b\u8c31\u7cfb"}</div>
              <div className="card-kicker">{root.name} {"\u00b7"} {root.modelFamily} {"\u00b7"} {"\u4e0d\u53ef\u53d8\u8282\u70b9"}</div>
            </div>
            <span className="badge badge-green">{root.status}</span>
          </div>
          <div className="tree-container">
            <TreeNode node={root} children={byParent(root.id)} allNodes={nodes} onNavigate={(id) => navigate(`/models/${id}`)} />
          </div>
        </div>
      ))}

      {filteredRoots.length === 0 && <div className="empty-state">{"\u6682\u65e0\u6a21\u578b\u8282\u70b9"}</div>}
    </>
  );
}

function TreeNode({ node, children, allNodes, onNavigate }: { node: ModelNode; children: ModelNode[]; allNodes: ModelNode[]; onNavigate: (id: string) => void }) {
  const grandchildren = (pid: string) => allNodes.filter((n) => n.parentId === pid);
  return (
    <div>
      <div className={`tree-node ${node.parentId === null ? "root" : ""}`} onClick={() => onNavigate(node.id)}>
        <div className="node-header">
          <span className="node-kind">{node.parentId === null ? "\u6839\u6a21\u578b" : "\u4efb\u52a1\u6a21\u578b"}</span>
          <span className="node-state">{node.status === "approved" ? "\u5df2\u6279\u51c6" : node.status === "candidate" ? "\u5019\u9009" : node.status}</span>
        </div>
        <div className="node-name">{node.name}</div>
        <div className="node-id">{node.id} {"\u00b7"} {node.labelSchemaName}</div>
      </div>
      {children.length > 0 && (
        <div className="tree-children">
          {children.map((child) => (
            <TreeNode key={child.id} node={child} children={grandchildren(child.id)} allNodes={allNodes} onNavigate={onNavigate} />
          ))}
        </div>
      )}
    </div>
  );
}
