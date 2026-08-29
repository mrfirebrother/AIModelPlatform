import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import { useToast } from "../../lib/toast";
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
    if (n.parentId && map.has(n.parentId)) {
      g.setEdge(n.parentId, n.id);
    }
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
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x + ROOT_W);
    maxY = Math.max(maxY, p.y + ROOT_H);
  }

  return { positions, edges, width: maxX - minX + 40, height: maxY - minY + 40 };
}

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
      toast.success("模型已导入");
      setShowImport(false);
      loadNodes();
    } catch (err) {
      toast.error("导入失败: " + (err as Error).message);
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
        <ModelLineage
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

function ModelLineage({ root, allNodes, onNavigate, onDelete, deleting }: {
  root: ModelNode;
  allNodes: ModelNode[];
  onNavigate: (id: string) => void;
  onDelete: (id: string) => void;
  deleting: string | null;
}) {
  const byParent = (pid: string) => allNodes.filter((n) => n.parentId === pid);
  const children = byParent(root.id);

  const { positions, edges, width, height } = useMemo(
    () => buildLayout(allNodes, root.id),
    [allNodes, root.id],
  );

  const nodeById = useMemo(() => new Map(allNodes.map((n) => [n.id, n])), [allNodes]);
  const nodeW = (id: string) => (nodeById.get(id)?.parentId === null ? ROOT_W : NODE_W);
  const nodeH = (id: string) => (nodeById.get(id)?.parentId === null ? ROOT_H : NODE_H);

  const statusBadge = (s: string) => s === "approved" ? "badge-green" : s === "candidate" ? "badge-orange" : "badge-gray";
  const statusLabel = (s: string) => s === "approved" ? "已批准" : s === "candidate" ? "候选" : s;

  return (
    <div className="model-tree-card">
      <div className="tree-card-header">
        <div className="tree-card-info">
          <div className="tree-card-title">模型谱系</div>
          <div className="tree-card-subtitle">{root.name || root.modelFamily} · {root.modelFamily}</div>
        </div>
        <div className="tree-card-actions">
          <span className={`badge ${statusBadge(root.status)}`}>{statusLabel(root.status)}</span>
          {children.length === 0 && (
            <button className="btn small danger" onClick={() => onDelete(root.id)} disabled={deleting === root.id} style={{ marginLeft: 8 }}>
              {deleting === root.id ? "删除中..." : "删除"}
            </button>
          )}
        </div>
      </div>

      <div style={{ overflowX: "auto", padding: "24px 20px" }}>
        <div style={{ position: "relative", width, height, margin: "0 auto" }}>
          {/* Edges */}
          <svg style={{ position: "absolute", top: 0, left: 0, width, height, pointerEvents: "none" }}>
            {edges.map(({ from, to }) => {
              const fp = positions.get(from);
              const tp = positions.get(to);
              if (!fp || !tp) return null;
              const fx = fp.x + nodeW(from) / 2;
              const fy = fp.y + nodeH(from);
              const tx = tp.x + nodeW(to) / 2;
              const ty = tp.y;
              const my = (fy + ty) / 2;
              return (
                <path
                  key={`${from}-${to}`}
                  d={`M ${fx} ${fy} L ${fx} ${my} L ${tx} ${my} L ${tx} ${ty}`}
                  stroke="#2d83c5"
                  strokeWidth="2"
                  fill="none"
                />
              );
            })}
          </svg>

          {/* Nodes */}
          {allNodes.map((n) => {
            const pos = positions.get(n.id);
            if (!pos) return null;
            const isRoot = n.parentId === null;
            return (
              <div
                key={n.id}
                onClick={() => onNavigate(n.id)}
                style={{
                  position: "absolute",
                  left: pos.x,
                  top: pos.y,
                  width: isRoot ? ROOT_W : NODE_W,
                  padding: isRoot ? "12px 16px" : "10px 14px",
                  borderRadius: 8,
                  cursor: "pointer",
                  transition: "all 0.2s",
                  textAlign: "center",
                  background: isRoot ? "linear-gradient(135deg, #1766ad 0%, #0f4f8b 100%)" : "white",
                  color: isRoot ? "white" : undefined,
                  border: isRoot ? "2px solid #0d4a7a" : "2px solid #d3e2ec",
                  boxShadow: isRoot ? "0 4px 12px rgba(23,102,173,0.3)" : "0 2px 8px rgba(0,0,0,0.06)",
                  zIndex: 1,
                }}
              >
                {!isRoot && byParent(n.id).length === 0 && (
                  <button
                    className="btn small danger"
                    onClick={(e) => { e.stopPropagation(); onDelete(n.id); }}
                    disabled={deleting === n.id}
                    title="删除"
                    style={{ position: "absolute", top: 4, right: 4, padding: "0 4px", fontSize: 11, lineHeight: "16px", minWidth: 0 }}
                  >
                    ×
                  </button>
                )}
                <div style={{ fontSize: 8, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 3, color: isRoot ? "rgba(255,255,255,0.7)" : "var(--primary)" }}>{isRoot ? "根模型" : "子模型"}</div>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>{n.name || n.modelFamily || "未命名"}</div>
                <div style={{ fontSize: 8, color: isRoot ? "rgba(255,255,255,0.7)" : "var(--text-muted)" }}>{n.code || n.id.slice(0, 8)} · {n.modelFamily}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
