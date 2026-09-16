import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import { useToast } from "../../lib/toast";
import type { ModelNode, Dataset } from "../../lib/api";

// 吞吐实测：GTX 745 (4 GB, sm_50)、imgsz 416、workers 0、cache 关，
// example1（1083 张训练图）逐轮实测 151~219 秒，稳态约 165 秒/轮 → 2.75 分钟/轮。
// 单轮耗时 ∝ 每张图的像素数（batch 不改变总计算量），所以按 (imgsz/416)² 换算；
// 再按训练集张数线性缩放。换卡或数据规模差异较大时需重新标定。
const MINUTES_PER_EPOCH_AT_416 = 2.75;
const MEASURED_TRAIN_IMAGES = 1083;

// 显存估算与后端 training_params.estimate_train_vram_mb 保持一致：
// 实测 imgsz 416 + batch 8 → 1280 MB（ultralytics 自报 GPU_mem），随「像素数 × batch」线性放大。
// 4096 MB 的卡要留给桌面和 CUDA 上下文约 800 MB，所以可用上限取 3300 MB。
const VRAM_LIMIT_MB = 3300;
const IMGSZ_OPTIONS = [320, 416, 512, 640, 768];
const BATCH_OPTIONS = [2, 4, 8, 16];

// 三档配方：每组都是一套完整且已验证的参数组合，选它＝选一整套，不必逐个理解参数。
const RECIPES = [
  {
    key: "quick",
    label: "快速验证",
    epochs: 30,
    imgsz: 416,
    batch: 8,
    augmentation: "default",
    patience: 30,
    title: "先探路：看指标趋势和参数是否合理，1~2 小时出结果",
  },
  {
    key: "standard",
    label: "标准",
    epochs: 100,
    imgsz: 416,
    batch: 8,
    augmentation: "default",
    patience: 30,
    title: "已实测标定（约 2.75 分钟/轮），推荐起点",
  },
  {
    key: "highres",
    label: "高精度",
    epochs: 100,
    imgsz: 640,
    batch: 4,
    augmentation: "default",
    patience: 30,
    title: "提高分辨率，细长/小缺陷更容易检出；耗时约为标准的 2.4 倍",
  },
] as const;

const AUGMENTATION_OPTIONS = [
  { key: "off", label: "关闭增广", hint: "数据量足够、或想验证标注质量时用" },
  { key: "default", label: "默认", hint: "库的基线，一般不用改" },
  { key: "strong", label: "强增广", hint: "数据少、训练指标高但验证指标低（过拟合）时用" },
];

function estimateVramMb(imgsz: number, batch: number) {
  return Math.round(160 * Math.pow(imgsz / 416, 2) * batch);
}

export default function TrainingCreatePage() {
  const api = useMemo(() => createApi(), []);
  const toast = useToast();
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelNode[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [parentId, setParentId] = useState("");
  const [selectedDs, setSelectedDs] = useState<Set<string>>(new Set());
  const [modelName, setModelName] = useState("");
  const [recipe, setRecipe] = useState<string>("standard");
  const [epochs, setEpochs] = useState(100);
  const [imgsz, setImgsz] = useState(416);
  const [batch, setBatch] = useState(8);
  const [patience, setPatience] = useState(30);
  const [augmentation, setAugmentation] = useState("default");
  const [showAdvanced, setShowAdvanced] = useState(false);

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

  // 手改任一参数就退出配方（快照里记为 custom），避免"选了标准却跑了别的参数"这种无从归因的情况。
  const applyRecipe = (r: typeof RECIPES[number]) => {
    setRecipe(r.key);
    setEpochs(r.epochs);
    setImgsz(r.imgsz);
    setBatch(r.batch);
    setAugmentation(r.augmentation);
    setPatience(r.patience);
  };
  const markCustom = () => setRecipe("custom");

  const selectedList = datasets.filter((d) => selectedDs.has(d.id));
  const totalImages = selectedList.reduce((s, d) => s + (d.imageCount ?? 0), 0);
  const allWithSnapshot = selectedList.every((d) => !!d.latestSnapshotId);

  const totalTrainImages = selectedList.reduce((s, d) => s + (d.trainCount ?? d.imageCount ?? 0), 0);
  const minutesPerEpoch =
    MINUTES_PER_EPOCH_AT_416 *
    Math.pow(imgsz / 416, 2) *
    (totalTrainImages > 0 ? totalTrainImages / MEASURED_TRAIN_IMAGES : 1);
  const estimatedMinutes = Math.round(epochs * minutesPerEpoch);
  const estimateText = estimatedMinutes >= 60 ? `${(estimatedMinutes / 60).toFixed(1)} 小时` : `${estimatedMinutes} 分钟`;
  const vramMb = estimateVramMb(imgsz, batch);
  const vramTooHigh = vramMb > VRAM_LIMIT_MB;
  const vramTight = !vramTooHigh && vramMb > VRAM_LIMIT_MB * 0.75;

  const [submitting, setSubmitting] = useState(false);
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    if (selectedDs.size === 0) { toast.error("请选择至少一个数据集"); return; }
    if (!allWithSnapshot) { toast.error("所选数据集中有未生成快照的，请先生成快照"); return; }
    if (vramTooHigh) { toast.error(`当前组合预计占用 ${vramMb} MB 显存，超过本机可用的 ${VRAM_LIMIT_MB} MB，请调小分辨率或批大小`); return; }
    const parent = models.find((m) => m.id === parentId);
    const recipeLabel = RECIPES.find((r) => r.key === recipe)?.label ?? "自定义";
    const ok = await toast.confirm(
      `确定加入训练？\n模型：${modelName.trim() || `${parent?.name || "yolo"}-finetune`}\n` +
      `参数：${recipeLabel} · ${epochs} 轮 · ${imgsz} 分辨率 · batch ${batch} · 增广${AUGMENTATION_OPTIONS.find((a) => a.key === augmentation)?.label}\n` +
      `数据：${selectedDs.size} 个数据集（${totalImages} 张）· 预计约 ${estimateText}`
    );
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
        trainingConfigJson: {
          epochs, model_name: name, dataset_snapshot_ids: snapshotIds,
          // 参数快照：配方名 + 每个生效参数都记在任务上，否则两次实验的指标差异无从归因
          recipe, imgsz, batch, patience, augmentation,
        },
        // 不再下发评测门槛：mAP50 是个连续指标，用阈值切一刀得到的"通过/未通过"没有决策价值，
        // 看指标本身即可（后端仍会按 TRAINING_DEFAULT_MIN_MAP50 计算 auto_status，未改动）
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
          <div className="card-kicker">选择父模型、数据集和训练配方</div>
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
            <div className="form-hint">从该模型继续训练（微调）；换更大的模型请先在「模型谱系」页上传 .pt</div>
          </div>
          <div className="form-group">
            <label>数据集（可多选）</label>
            <div style={{ border: "1px solid var(--surface-border)", borderRadius: 3, maxHeight: 180, overflowY: "auto" }}>
              {datasets.map((d) => (
                <label key={d.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", borderBottom: "1px solid var(--surface-border)", cursor: "pointer", fontSize: 12, background: selectedDs.has(d.id) ? "#e6f4ff" : undefined }}>
                  <input type="checkbox" checked={selectedDs.has(d.id)} onChange={() => toggleDs(d.id)} />
                  <span style={{ flex: 1 }}>{d.name}</span>
                  <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{(d.imageCount ?? 0)} 张</span>
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
            <label>训练配方</label>
            <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
              {RECIPES.map((r) => (
                <button
                  key={r.key}
                  type="button"
                  className={`btn small${recipe === r.key ? " primary" : ""}`}
                  title={r.title}
                  onClick={() => applyRecipe(r)}
                >
                  {r.label}
                </button>
              ))}
              {recipe === "custom" && <span className="badge badge-gray badge-xs">自定义</span>}
            </div>
            <div className="form-hint">
              {RECIPES.find((r) => r.key === recipe)?.title
                ?? "已手动调整参数；下面「高级参数」里可以看到具体值"}
              {" 预计 "}
              <b>{estimateText}</b>
              {totalTrainImages > 0
                ? `（${epochs} 轮 × 约 ${minutesPerEpoch.toFixed(1)} 分钟/轮，按 ${totalTrainImages} 张训练图估算）`
                : "（请先选择数据集）"}
              ，验证指标连续 {patience} 轮不提升会自动早停，实际通常更短。
            </div>
            <div className="form-hint" style={{ color: vramTooHigh ? "#c44" : vramTight ? "var(--accent-orange)" : undefined }}>
              预计显存占用约 {vramMb} MB（可用上限 {VRAM_LIMIT_MB} MB）
              {vramTooHigh ? " — 超出上限，请调小分辨率或批大小" : vramTight ? " — 已接近上限" : ""}
            </div>
          </div>

          <div className="form-group">
            <button type="button" className="btn small" onClick={() => setShowAdvanced((v) => !v)}>
              {showAdvanced ? "收起高级参数" : "展开高级参数"}
            </button>
            <div className="form-hint">一般不需要动；改这里适合做"一次只改一个变量"的对比实验</div>
          </div>

          {showAdvanced && (
            <div style={{ border: "1px solid var(--surface-border)", borderRadius: 3, padding: "10px 12px" }}>
              <div className="form-group">
                <label>训练轮数</label>
                <input type="number" value={epochs} onChange={(e) => { setEpochs(Number(e.target.value)); markCustom(); }} min={1} max={1000} />
              </div>
              <div className="form-group">
                <label>输入分辨率（imgsz）</label>
                <select value={imgsz} onChange={(e) => { setImgsz(Number(e.target.value)); markCustom(); }}>
                  {IMGSZ_OPTIONS.map((v) => <option key={v} value={v}>{v}{v === 416 ? "（实测基线）" : v === 640 ? "（细长/小缺陷更友好，耗时约 2.4 倍）" : ""}</option>)}
                </select>
                <div className="form-hint">影响最大的一项：调大后小目标更容易检出，代价是显存和时间按平方增长</div>
              </div>
              <div className="form-group">
                <label>批大小（batch）</label>
                <select value={batch} onChange={(e) => { setBatch(Number(e.target.value)); markCustom(); }}>
                  {BATCH_OPTIONS.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
                <div className="form-hint">只影响显存占用和收敛稳定性，不改变每轮耗时（总计算量不变）</div>
              </div>
              <div className="form-group">
                <label>早停耐心值（patience）</label>
                <input type="number" value={patience} onChange={(e) => { setPatience(Number(e.target.value)); markCustom(); }} min={0} max={1000} />
                <div className="form-hint">连续多少轮验证指标不提升就停止；设为 0 表示不早停（跑满配置轮数）</div>
              </div>
              <div className="form-group">
                <label>数据增广</label>
                <select value={augmentation} onChange={(e) => { setAugmentation(e.target.value); markCustom(); }}>
                  {AUGMENTATION_OPTIONS.map((a) => <option key={a.key} value={a.key}>{a.label}</option>)}
                </select>
                <div className="form-hint">{AUGMENTATION_OPTIONS.find((a) => a.key === augmentation)?.hint}</div>
              </div>
            </div>
          )}

        </div>
        <div className="modal-foot">
          <button type="button" className="btn" onClick={() => navigate("/training")}>取消</button>
          <button type="submit" className="btn primary" disabled={submitting || vramTooHigh}>{submitting ? "提交中..." : "加入训练队列"}</button>
        </div>
      </form>
    </div>
  );
}
