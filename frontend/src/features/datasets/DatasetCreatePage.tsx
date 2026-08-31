import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import { useToast } from "../../lib/toast";

export default function DatasetCreatePage() {
  const api = createApi();
  const toast = useToast();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [classes, setClasses] = useState("crack");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) { toast.error("名称必填"); return; }
    setSubmitting(true);
    try {
      const classList = classes.split(",").map((c) => c.trim()).filter(Boolean);
      const ds = await api.createDataset({ name: name.trim(), description: description.trim() || undefined, classes: classList } as any);
      const id = (ds as any).id || (ds as any).dataset_id;
      toast.success("数据集已创建");
      navigate(`/datasets/${id}/annotate`);
    } catch (err) { toast.error("创建失败: " + (err as Error).message); }
    setSubmitting(false);
  };

  return (
    <div className="card form-card">
      <div className="card-head">
        <div>
          <div className="card-title">新建数据集</div>
          <div className="card-kicker">填写基本信息后进入标注</div>
        </div>
        <button className="btn small" onClick={() => navigate("/datasets")}>返回</button>
      </div>
      <form onSubmit={handleSubmit}>
        <div className="card-body">
          <div className="form-group">
            <label>名称</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="例如 crack-seg" maxLength={80} />
          </div>
          <div className="form-group">
            <label>描述</label>
            <input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="可选" maxLength={200} />
          </div>
          <div className="form-group">
            <label>{"初始类别（逗号分隔）"}</label>
            <input value={classes} onChange={(e) => setClasses(e.target.value)} placeholder="crack, background" />
            <div className="form-hint">{"用于生成 LabelSchema"}</div>
          </div>
        </div>
        <div className="modal-foot">
          <button type="button" className="btn" onClick={() => navigate("/datasets")}>取消</button>
          <button type="submit" className="btn primary" disabled={submitting}>{submitting ? "创建中..." : "创建并标注"}</button>
        </div>
      </form>
    </div>
  );
}
