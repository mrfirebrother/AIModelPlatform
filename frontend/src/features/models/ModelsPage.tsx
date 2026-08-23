import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import { useToast } from "../../lib/toast";
import type { ModelNode } from "../../lib/api";

export default function ModelsPage() {
  const api = useMemo(() => createApi(), []);
  const toast = useToast();
  const navigate = useNavigate();
  const [nodes, setNodes] = useState<ModelNode[]>([]);
  const [filter, setFilter] = useState<string>("all");
  const [showImport, setShowImport] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
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
      toast.success("\u6a21\u578b\u5df2\u5bfc\u5165");
      setShowImport(false);
      loadNodes();
    } catch (err) {
      toast.error("\u5bfc\u5165\u5931\u8d25: " + (err as Error).message);
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

  const handleDelete = async (modelId: string) => {
    const confirmed = await toast.confirm("确定要删除此模型吗？");
    if (!confirmed) return;
    setDeleting(modelId);
    try {
      await api.deleteModel(modelId);
      toast.success("模型已删除");
      loadNodes();
    } catch (err) {
      toast.error("删除失败: " + (err as Error).message);
    }
    setDeleting(null);
  };

  const roots = nodes.filter((n) => n.parentId === null);
  const filteredRoots = filter === "all" ? roots : roots.filter((r) => r.status === filter);
  const hasRoot = roots.length > 0;

  return (
    <>
      <div className="top-actions" style={{ marginBottom: 16, display: "flex", gap: 10, alignItems: "center" }}>
        <select value={filter} onChange={(e) => setFilter(e.target.value)} style={{ padding: "6px 10px", border: "1px solid #bed2df", borderRadius: 3, fontSize: 11 }}>
          <option value="all">全部状态</option>
          <option value="approved">已批准</option>
          <option value="candidate">候选</option>
          <option value="archived">已归档</option>
        </select>
        <button
          className="btn primary"
          disabled={hasRoot}
          onClick={() => setShowImport(true)}
          style={{ marginLeft: "auto", opacity: hasRoot ? 0.5 : 1, cursor: hasRoot ? "not-allowed" : "pointer" }}
          title={hasRoot ? "已存在根模型，请先删除后再导入" : ""}
        >
          + 导入根模型
        </button>
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
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>上传中...</div>
                <div style={{ width: 200, height: 6, background: "#e0e0e0", borderRadius: 3, margin: "0 auto" }}>
                  <div style={{ width: `${uploadPct ?? 0}%`, height: "100%", background: "#1766ad", borderRadius: 3, transition: "width 0.3s" }} />
                </div>
                <div style={{ marginTop: 6, fontSize: 11, color: "#666" }}>{uploadPct ?? 0}%</div>
              </>
            ) : (
              <>
                <div style={{ fontSize: 40, marginBottom: 8, color: "#ccc" }}>⬆</div>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>拖放 .pt 文件到此处</div>
                <div style={{ fontSize: 11, color: "#999" }}>或点击选择文件 · 最大 500MB</div>
              </>
            )}
          </div>
          {uploadError && <div style={{ marginTop: 10, color: "#b33", fontSize: 12 }}>{uploadError}</div>}
          <div style={{ marginTop: 14, display: "flex", gap: 8 }}>
            <button className="btn" onClick={() => { setShowImport(false); setUploadError(null); }}>取消</button>
          </div>
        </div>
      )}

      {filteredRoots.map((root) => (
        <ModelTree
          key={root.id}
          root={root}
          allNodes={nodes}
          onNavigate={(id) => navigate(`/models/${id}`)}
          onDelete={handleDelete}
          deleting={deleting}
        />
      ))}

      {filteredRoots.length === 0 && <div className="empty-state">暂无模型节点</div>}
    </>
  );
}

function ModelTree({ root, allNodes, onNavigate, onDelete, deleting }: {
  root: ModelNode;
  allNodes: ModelNode[];
  onNavigate: (id: string) => void;
  onDelete: (id: string) => void;
  deleting: string | null;
}) {
  const byParent = (pid: string) => allNodes.filter((n) => n.parentId === pid);
  const children = byParent(root.id);
  const hasChildren = children.length > 0;

  return (
    <div className="model-tree-card">
      <div className="tree-card-header">
        <div className="tree-card-info">
          <div className="tree-card-title">模型谱系</div>
          <div className="tree-card-subtitle">{root.name || root.modelFamily} · {root.modelFamily}</div>
        </div>
        <div className="tree-card-actions">
          <span className={`badge badge-${root.status === "approved" ? "green" : root.status === "candidate" ? "orange" : "gray"}`}>
            {root.status === "approved" ? "已批准" : root.status === "candidate" ? "候选" : root.status}
          </span>
          {!hasChildren && (
            <button
              className="btn small danger"
              onClick={() => onDelete(root.id)}
              disabled={deleting === root.id}
              style={{ marginLeft: 8 }}
            >
              {deleting === root.id ? "删除中..." : "删除"}
            </button>
          )}
        </div>
      </div>

      <div className="tree-visual">
        <TreeNodeCard
          node={root}
          isRoot={true}
          onNavigate={onNavigate}
        />

        {children.length > 0 && (
          <div className="tree-subtree">
            <div className="tree-connector-vertical" />
            <div className="tree-children-row">
              {children.map((child, idx) => (
                <TreeNodeBranch
                  key={child.id}
                  node={child}
                  allNodes={allNodes}
                  onNavigate={onNavigate}
                  isLast={idx === children.length - 1}
                  isFirst={idx === 0}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function TreeNodeCard({ node, isRoot, onNavigate }: {
  node: ModelNode;
  isRoot: boolean;
  onNavigate: (id: string) => void;
}) {
  return (
    <div
      className={`tree-node-card ${isRoot ? "root" : "child"}`}
      onClick={() => onNavigate(node.id)}
    >
      <div className="node-type">{isRoot ? "根模型" : "子模型"}</div>
      <div className="node-name">{node.name || node.modelFamily || "未命名"}</div>
      <div className="node-meta">{node.modelFamily} · {node.id.slice(0, 8)}...</div>
    </div>
  );
}

function TreeNodeBranch({ node, allNodes, onNavigate, isLast, isFirst }: {
  node: ModelNode;
  allNodes: ModelNode[];
  onNavigate: (id: string) => void;
  isLast: boolean;
  isFirst: boolean;
}) {
  const byParent = (pid: string) => allNodes.filter((n) => n.parentId === pid);
  const children = byParent(node.id);
  const hasChildren = children.length > 0;

  return (
    <div className="tree-branch">
      <div className="tree-branch-connector">
        <svg width="100%" height="30" viewBox="0 0 100 30" preserveAspectRatio="none">
          <path d="M 50 0 L 50 30" stroke="#2d83c5" strokeWidth="2" fill="none" />
        </svg>
      </div>

      <TreeNodeCard node={node} isRoot={false} onNavigate={onNavigate} />

      {hasChildren && (
        <div className="tree-subtree">
          <div className="tree-connector-vertical" />
          <div className="tree-children-row">
            {children.map((child, idx) => (
              <TreeNodeBranch
                key={child.id}
                node={child}
                allNodes={allNodes}
                onNavigate={onNavigate}
                isLast={idx === children.length - 1}
                isFirst={idx === 0}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
