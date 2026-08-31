import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import { useToast } from "../../lib/toast";
import type { ModelNode, Dataset } from "../../lib/api";

export default function TrainingCreatePage() {
  const api = useMemo(() => createApi(), []);
  const toast = useToast();
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelNode[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [parentId, setParentId] = useState("");
  const [selectedDs, setSelectedDs] = useState<Set<string>>(new Set());
  const [modelName, setModelName] = useState("");
  const [epochs, setEpochs] = useState(100);

  useEffect(() => {
    Promise.all([api.getModelNodes(), api.getDatasets()]).then(([m, d]) => {
      setModels(m);
      setDatasets(d);
      if (m.length > 0) setParentId(m[0].id);
    }).catch(() => {});
  }, [api]);

  const toggleDs = (id: string) => {
    setSelectedDs((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const selectedList = datasets.filter((d) => selectedDs.has(d.id));
  const totalImages = selectedList.reduce((s, d) => s + (d.imageCount ?? 0), 0);
  const allWithSnapshot = selectedList.every((d) => !!d.latestSnapshotId);

  const [submitting, setSubmitting] = useState(false);
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    if (selectedDs.size === 0) { toast.error("请选择至少一个数据集"); return; }
    if (!allWithSnapshot) { toast.error("所选数据集中有未生成快照的，请先生成快照"); return; }
    const parent = models.find((m) => m.id === parentId);
    const dsNames = selectedList.map((d) => d.name).join(", ");
    const ok = await toast.confirm(`确定加入训练？\n模型：${modelName.trim() || `${parent?.name || "yolo"}-finetune`} \u00b7 ${epochs} 轮 \u00b7 ${selectedDs.size} 个数据集 (${totalImages} 张)`);
    if (!ok) return;
    setSubmitting(true);
    try {
      const name = modelName.trim() || `${parent?.name || "yolo"}-finetune`;
      const snapshotIds = selectedList.map((d) => d.latestSnapshotId!);
      await api.createTrainingTask({
        datasetSnapshotId: snapshotIds[0],
        parentModelNodeId: parentId || undefined,
        taskType: parent?.taskType || "object_detection",
        modelFamily: parent?.modelFamily || "YOLOv8",
        trainingConfigJson: { epochs, model_name: name, dataset_snapshot_ids: snapshotIds },
      });
      toast.success("训练任务已创建");
      navigate("/training");
    } catch (err) {
      toast.error("创建失败: " + (err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const parentName = models.find((m) => m.id === parentId)?.name || "yolo";

  return (
    <div className="card form-card">
      <div className="card-head">
        <div>
          <div className="card-title">新建训练任务</div>
          <div className="card-kicker">选择父模型、数据集和训练参数</div>
        </div>
        <button className="btn small" onClick={() => navigate("/training")}>返回</button>
      </div>
      <form onSubmit={handleSubmit}>
        <div className="card-body">
          <div className="form-group">
            <label>父模型节点</label>
            <select value={parentId} onChange={(e) => setParentId(e.target.value)}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>{m.name} ({m.code || m.id.slice(0, 8)})</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>数据集（可多选）</label>
            <div style={{ border: "1px solid #bed2df", borderRadius: 3, maxHeight: 180, overflowY: "auto" }}>
              {datasets.map((d) => (
                <label key={d.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", borderBottom: "1px solid #f0f4f8", cursor: "pointer", fontSize: 12, background: selectedDs.has(d.id) ? "#e7f1fa" : undefined }}>
                  <input type="checkbox" checked={selectedDs.has(d.id)} onChange={() => toggleDs(d.id)} />
                  <span style={{ flex: 1 }}>{d.name}</span>
                  <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{(d.imageCount ?? 0)} 张</span>
                  {!d.latestSnapshotId && <span className="badge badge-orange badge-xs">无快照</span>}
                </label>
              ))}
            </div>
            {selectedDs.size > 0 && (
              <div className="form-hint" style={{ color: "var(--primary)" }}>已选 {selectedDs.size} 个，共 {totalImages} 张图片</div>
            )}
          </div>
          <div className="form-group">
            <label>模型名称</label>
            <input type="text" value={modelName} onChange={(e) => setModelName(e.target.value)} placeholder={`${parentName}-finetune`} maxLength={40} />
            <div className="form-hint">为空则自动命名</div>
          </div>
          <div className="form-group">
            <label>训练轮数</label>
            <input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} min={1} max={1000} />
          </div>
        </div>
        <div className="modal-foot">
          <button type="button" className="btn" onClick={() => navigate("/training")}>取消</button>
          <button type="submit" className="btn primary" disabled={submitting}>{submitting ? "提交中..." : "加入训练队列"}</button>
        </div>
      </form>
    </div>
  );
}
