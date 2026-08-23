import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { ModelNode, Dataset } from "../../lib/api";

export default function TrainingCreatePage() {
  const api = useMemo(() => createApi(), []);
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelNode[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [parentId, setParentId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [epochs, setEpochs] = useState(100);
  const [bindingId, setBindingId] = useState("");

  useEffect(() => {
    Promise.all([api.getModelNodes(), api.getDatasets()]).then(([m, d]) => {
      setModels(m.filter((n) => n.status === "approved"));
      setDatasets(d);
      if (m.length > 0) setParentId(m[0].id);
      if (d.length > 0) setDatasetId(d[0].id);
    }).catch(() => {});
  }, [api]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const parent = models.find((m) => m.id === parentId);
    const dataset = datasets.find((d) => d.id === datasetId);
    await api.createTrainingTask({
      parentModelNodeId: parentId,
      parentModelName: parent?.name,
      datasetSnapshotId: datasetId,
      datasetName: dataset?.name,
      targetBindingId: bindingId || undefined,
      epochs,
      taskType: "object_detection",
      modelFamily: "YOLOv8",
    });
    navigate("/training");
  };

  return (
    <div className="card" style={{ maxWidth: 560 }}>
      <div className="card-head">
        <div>
          <div className="card-title">新建训练任务</div>
          <div className="card-kicker">选择父模型、数据集和训练参数</div>
        </div>
        <button className="btn small" onClick={() => navigate("/training")}>
          返回
        </button>
      </div>
      <form onSubmit={handleSubmit}>
        <div className="card-body">
          <div className="form-group">
            <label>父模型节点</label>
            <select value={parentId} onChange={(e) => setParentId(e.target.value)}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id} | {m.name} | {m.labelSchemaName}
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>数据集快照</label>
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.id} | {d.name} | {d.imageCount} 张图片
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>目标关联 (可选)</label>
            <input
              value={bindingId}
              onChange={(e) => setBindingId(e.target.value)}
              placeholder="留空则生成独立模型"
            />
          </div>
          <div className="form-group">
            <label>训练轮数</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              min={1}
              max={1000}
            />
          </div>
        </div>
        <div className="modal-foot">
          <button type="button" className="btn" onClick={() => navigate("/training")}>
            取消
          </button>
          <button type="submit" className="btn primary">
            加入训练队列
          </button>
        </div>
      </form>
    </div>
  );
}
