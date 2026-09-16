import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import { useToast } from "../../lib/toast";
import LoadingSpinner from "../../lib/LoadingSpinner";
import type { ModelNode } from "../../lib/api";
import dagre from "dagre";

const NODE_W = 160;
const NODE_H = 80;
const ROOT_W = 200;
const ROOT_H = 90;

function buildLayout(nodes: ModelNode[], rootId: string) {
  const g = new dagre.graphlib.Graph({ directed: true });
  g.setGraph({ rankdir: "TB", ranksep: 60, nodesep: 30, marginx: 20, marginy: 20 });
  g.setDefaultEdgeLabel(() => ({}));
  const map = new Map(nodes.map((n) => [n.id, n]));
  for (const n of nodes) {
    const isRoot = n.parentId === null;
    g.setNode(n.id, { width: isRoot ? ROOT_W : NODE_W, height: isRoot ? ROOT_H : NODE_H, label: n.id });
  }
  for (const n of nodes) {
    if (n.parentId && map.has(n.parentId)) g.setEdge(n.parentId, n.id);
  }
  dagre.layout(g);
  const positions = new Map<string, { x: number; y: number }>();
  for (const n of nodes) {
    const d = g.node(n.id);
    if (d) positions.set(n.id, { x: d.x - (n.parentId === null ? ROOT_W : NODE_W) / 2, y: d.y - (n.parentId === null ? ROOT_H : NODE_H) / 2 });
  }
  const edges: { from: string; to: string }[] = [];
  g.edges().forEach((e: any) => { edges.push({ from: e.v, to: e.w }); });
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of positions.values()) {
    minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x + ROOT_W); maxY = Math.max(maxY, p.y + ROOT_H);
  }
  return { positions, edges, width: maxX - minX + 40, height: maxY - minY + 40 };
}

export default function ModelsPage() {
  const api = useMemo(() => createApi(), []);
  const toast = useToast();
  const navigate = useNavigate();
  const [nodes, setNodes] = useState<ModelNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [showImport, setShowImport] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadNodes = () => { setLoading(true); api.getModelNodes().then(setNodes).finally(() => setLoading(false)); };
  useEffect(() => { loadNodes(); }, [api]);

  const handleFile = async (file: File) => {
    if (!file.name.endsWith(".pt")) { setUploadError("请选择 .pt 模型文件"); return; }
    if (file.size > 500 * 1024 * 1024) { setUploadError("文件大小不能超过 500MB"); return; }
    setUploadError(null); setUploadPct(0); setUploading(true);
    try {
      const res = await api.uploadModel(file, (pct) => setUploadPct(pct));
      await api.createModelNode({ artifactPath: res.file_path, name: file.name.replace(".pt", ""), taskType: "object_detection", modelFamily: "yolo" });
      toast.success("模型已导入"); setShowImport(false); loadNodes();
    } catch (err) { toast.error("导入失败: " + (err as Error).message); }
    setUploading(false); setUploadPct(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDrop = (e: React.DragEvent) => { e.preventDefault(); setDragOver(false); const file = e.dataTransfer.files[0]; if (file) handleFile(file); };

  const handleDelete = async (modelId: string) => {
    const confirmed = await toast.confirm("确定要删除此模型吗？");
    if (!confirmed) return;
    setDeleting(modelId);
    try { await api.deleteModel(modelId); toast.success("模型已删除"); loadNodes(); } catch (err) { toast.error("删除失败: " + (err as Error).message); }
    setDeleting(null);
  };

  const roots = nodes.filter((n) => n.parentId === null);
  const filteredRoots = roots;
  const hasRoot = roots.length > 0;

  return (
    <>
      <div className="toolbar">
        <button className="btn primary" disabled={hasRoot} onClick={() => setShowImport(true)} style={{ marginLeft: "auto", opacity: hasRoot ? 0.5 : 1, cursor: hasRoot ? "not-allowed" : "pointer" }} title={hasRoot ? "已存在根模型，请先删除后再导入" : ""}>+ 导入根模型</button>
      </div>

      {showImport && (
        <div className="card" style={{ marginBottom: 14, padding: 24 }}>
          <div className="card-title" style={{ marginBottom: 16 }}>导入根模型</div>
          <div onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={handleDrop} onClick={() => !uploading && fileInputRef.current?.click()} className={`dropzone ${dragOver ? "dropzone-active" : "dropzone-idle"}`}>
            <input ref={fileInputRef} type="file" accept=".pt" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
            {uploading ? (
              <>
                <div className="upload-label">上传中...</div>
                <div className="upload-progress"><div className="upload-progress-fill" style={{ width: `${uploadPct ?? 0}%` }} /></div>
                <div className="upload-hint">{uploadPct ?? 0}%</div>
              </>
            ) : (
              <>
                <div className="dropzone-icon">{"⬆"}</div>
                <div className="dropzone-title">拖放 .pt 文件到此处</div>
                <div className="dropzone-hint">或点击选择文件 · 最大 500MB</div>
              </>
            )}
          </div>
          {uploadError && <div style={{ marginTop: 10, color: "#b33", fontSize: 12 }}>{uploadError}</div>}
          <div style={{ marginTop: 14, display: "flex", gap: 8 }}><button className="btn" onClick={() => { setShowImport(false); setUploadError(null); }}>取消</button></div>
        </div>
      )}

      {filteredRoots.map((root) => (
        <ModelLineage key={root.id} root={root} allNodes={nodes} onNavigate={(id) => navigate(`/models/${id}`)} onDelete={handleDelete} deleting={deleting} />
      ))}
      {loading ? (
        <LoadingSpinner text="加载模型列表..." />
      ) : filteredRoots.length === 0 ? (
        <div className="empty-state">暂无模型节点</div>
      ) : null}
    </>
  );
}

function ModelLineage({ root, allNodes, onNavigate, onDelete, deleting }: { root: ModelNode; allNodes: ModelNode[]; onNavigate: (id: string) => void; onDelete: (id: string) => void; deleting: string | null; }) {
  const byParent = (pid: string) => allNodes.filter((n) => n.parentId === pid);
  const children = byParent(root.id);
  const { positions, edges, width, height } = useMemo(() => buildLayout(allNodes, root.id), [allNodes, root.id]);
  const nodeById = useMemo(() => new Map(allNodes.map((n) => [n.id, n])), [allNodes]);
  const nodeW = (id: string) => (nodeById.get(id)?.parentId === null ? ROOT_W : NODE_W);
  const nodeH = (id: string) => (nodeById.get(id)?.parentId === null ? ROOT_H : NODE_H);

  return (
    <div className="model-tree-card">
      <div className="tree-card-header">
        <div className="tree-card-info">
          <div className="tree-card-title">模型谱系</div>
          <div className="tree-card-subtitle">{root.name || root.modelFamily} · {root.modelFamily}</div>
        </div>
        <div className="tree-card-actions">
          {children.length === 0 && <button className="btn small danger" onClick={() => onDelete(root.id)} disabled={deleting === root.id}>{deleting === root.id ? "删除中..." : "删除"}</button>}
        </div>
      </div>
      <div className="tree-canvas">
        <div className="tree-canvas-inner" style={{ width, height }}>
          <svg className="tree-svg" style={{ width, height }}>
            {edges.map(({ from, to }) => {
              const fp = positions.get(from); const tp = positions.get(to);
              if (!fp || !tp) return null;
              const fx = fp.x + nodeW(from) / 2; const fy = fp.y + nodeH(from);
              const tx = tp.x + nodeW(to) / 2; const ty = tp.y; const my = (fy + ty) / 2;
              return <path key={`${from}-${to}`} d={`M ${fx} ${fy} L ${fx} ${my} L ${tx} ${my} L ${tx} ${ty}`} stroke="#2d83c5" strokeWidth="2" fill="none" />;
            })}
          </svg>
          {allNodes.map((n) => {
            const pos = positions.get(n.id);
            if (!pos) return null;
            const isRoot = n.parentId === null;
            return (
              <div key={n.id} onClick={() => onNavigate(n.id)} className={`tree-node-card ${isRoot ? "root" : "child"}`} style={{ position: "absolute", left: pos.x, top: pos.y, width: isRoot ? ROOT_W : NODE_W, zIndex: 1 }}>
                {!isRoot && byParent(n.id).length === 0 && (
                  <button className="btn small danger" onClick={(e) => { e.stopPropagation(); onDelete(n.id); }} disabled={deleting === n.id} title="删除" style={{ position: "absolute", top: 4, right: 4, padding: "0 4px", fontSize: 12, lineHeight: "16px", minWidth: 0 }}>{"×"}</button>
                )}
                <div className="node-type">{isRoot ? "根模型" : "子模型"}</div>
                <div className="node-name">{n.name || n.modelFamily || "未命名"}</div>
                <div className="node-meta">{n.code || n.id.slice(0, 8)} · {n.modelFamily}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
