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
  const [datasetId, setDatasetId] = useState("");
  const [modelName, setModelName] = useState("");
  const [epochs, setEpochs] = useState(100);

  useEffect(() => {
    Promise.all([api.getModelNodes(), api.getDatasets()]).then(([m, d]) => {
      setModels(m);
      setDatasets(d);
      if (m.length > 0) setParentId(m[0].id);
      if (d.length > 0) setDatasetId(d[0].id);
    }).catch(() => {});
  }, [api]);

  const [submitting, setSubmitting] = useState(false);
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    const parent = models.find((m) => m.id === parentId);
    const dataset = datasets.find((d) => d.id === datasetId);
    const snapshotId = dataset?.latestSnapshotId;
    if (!snapshotId) {
      toast.error("数据集没有快照，请先导入并验证数据集");
      return;
    }
    const ok = await toast.confirm(`确定加入训练？\n模型：${modelName.trim() || `${parent?.name || "yolo"}-finetune`} · ${epochs} 轮 · ${dataset?.name || ""}`, );
    if (!ok) return;
    setSubmitting(true);
    try {
      const name = modelName.trim() || `${parent?.name || "yolo"}-finetune`;
      await api.createTrainingTask({
        datasetSnapshotId: snapshotId,
        parentModelNodeId: parentId || undefined,
        taskType: parent?.taskType || "object_detection",
        modelFamily: parent?.modelFamily || "YOLOv8",
        trainingConfigJson: { epochs, model_name: name },
      });
      toast.success("训练任务已创建");
      navigate("/training");
    } catch (err) {
      toast.error("创建失败: " + (err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="card" style={{ maxWidth: 560 }}>
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
                <option key={m.id} value={m.id}>
                  {m.name} ({m.code || m.id.slice(0, 8)})
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>数据集快照</label>
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({(d.imageCount ?? 0)} 张图片)
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>模型名称</label>
            <input type="text" value={modelName} onChange={(e) => setModelName(e.target.value)} placeholder={`${models.find((m) => m.id===parentId)?.name || "yolo"}-finetune`} maxLength={40} />
            <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>为空则自动命名，产出模型显示此名称而非 YOLOv8</div>
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
