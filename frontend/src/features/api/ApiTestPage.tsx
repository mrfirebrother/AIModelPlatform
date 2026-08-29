import { useEffect, useMemo, useRef, useState } from "react";
import { createApi } from "../../lib/createApi";
import type { ModelNode } from "../../lib/api";

interface Detection {
  class_name: string;
  confidence: number;
  bbox: number[];
  class_id: number;
}

interface InferResult {
  model_code: string;
  model_name: string;
  detections: Detection[];
  latency_ms: number;
  image_width: number;
  image_height: number;
  overlay_image?: string | null;
}

export default function ApiTestPage() {
  const api = useMemo(() => createApi(), []);
  const [models, setModels] = useState<ModelNode[]>([]);
  const [selectedCode, setSelectedCode] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getModelNodes().then((m) => {
      setModels(m);
      if (m.length > 0) setSelectedCode(m[0].code || "");
    }).catch(() => {});
  }, [api]);

  const selectedModel = models.find((m) => m.code === selectedCode);

  const handleInfer = async () => {
    if (!selectedCode || !file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`/api/models/${selectedCode}/infer`, {
        method: "POST",
        headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
        body: fd,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `${res.status} ${res.statusText}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError((err as Error).message);
    }
    setLoading(false);
  };

  return (
    <>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 10, color: "var(--primary)", letterSpacing: "0.15em", textTransform: "uppercase", fontWeight: 600 }}>API 测试</div>
        <div style={{ fontSize: 20, fontWeight: 600, marginTop: 4 }}>模型推理测试</div>
      </div>

      <section style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1.6fr)", gap: 16, alignItems: "start" }}>
        {/* Left: Parameter Documentation */}
        <div className="card" style={{ padding: 0, overflow: "hidden", minWidth: 0 }}>
          <div style={{ padding: "12px 16px", borderBottom: "1px solid #e6eef6", fontWeight: 600, fontSize: 11 }}>接口说明</div>
          <div style={{ padding: "14px 16px" }}>
            <div style={{ fontFamily: "Courier New, monospace", fontSize: 10, color: "var(--text-secondary)", background: "#f8fafc", padding: "10px 12px", borderRadius: 6, border: "1px solid #e6eef6" }}>
              <div style={{ marginBottom: 6 }}><span style={{ color: "var(--primary)" }}>POST</span> /api/models/<span style={{ color: "var(--accent-orange)" }}>{'{code}'}</span>/infer</div>
              <div style={{ marginBottom: 6 }}>Content-Type: multipart/form-data</div>
              <div style={{ marginBottom: 10 }}>参数: <span style={{ color: "var(--accent-green)" }}>file</span> (图片文件)</div>
              <div style={{ color: "var(--text-muted)", marginBottom: 4 }}>响应字段:</div>
              <div style={{ fontSize: 9, lineHeight: 1.7 }}>
                <div><span style={{ color: "var(--primary)" }}>model_code</span> — 模型编号</div>
                <div><span style={{ color: "var(--primary)" }}>model_name</span> — 模型名称</div>
                <div><span style={{ color: "var(--primary)" }}>detections</span> — 检测结果数组</div>
                <div style={{ paddingLeft: 12 }}><span style={{ color: "var(--accent-green)" }}>class_name</span> — 类别名称</div>
                <div style={{ paddingLeft: 12 }}><span style={{ color: "var(--accent-green)" }}>confidence</span> — 置信度</div>
                <div style={{ paddingLeft: 12 }}><span style={{ color: "var(--accent-green)" }}>bbox</span> — 检测框 [x1,y1,x2,y2]</div>
                <div style={{ paddingLeft: 12 }}><span style={{ color: "var(--accent-green)" }}>class_id</span> — 类别 ID</div>
                <div><span style={{ color: "var(--primary)" }}>latency_ms</span> — 推理耗时(ms)</div>
                <div><span style={{ color: "var(--primary)" }}>image_width/height</span> — 图片尺寸</div>
                <div><span style={{ color: "var(--primary)" }}>overlay_image</span> — 检测结果图(base64)</div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Inference Test */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0, overflow: "hidden" }}>
          {/* Model Selector + Upload */}
          <div className="card" style={{ padding: 0, overflow: "hidden", minHeight: 100 }}>
            <div style={{ padding: "8px 12px", borderBottom: "1px solid #e6eef6", fontWeight: 600, fontSize: 11 }}>推理测试</div>
            <div style={{ padding: "10px 12px" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 10, alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: 9, color: "var(--text-muted)", marginBottom: 4 }}>选择模型</div>
                  <select value={selectedCode} onChange={(e) => { setSelectedCode(e.target.value); setResult(null); setError(null); }} style={{ width: "100%", padding: "7px 10px", border: "1px solid #bed2df", borderRadius: 3, fontSize: 11, background: "#f6fafc" }}>
                    {models.map((m) => <option key={m.id} value={m.code || ""}>{m.code} — {m.name}</option>)}
                  </select>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: "var(--text-muted)", marginBottom: 4 }}>上传图片</div>
                  <label style={{ display: "flex", alignItems: "center", gap: 6, padding: "7px 10px", border: "1px dashed #bed2df", borderRadius: 6, cursor: "pointer", fontSize: 11, color: "var(--text-muted)", background: "#fafbfc" }}>
                    <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) { setFile(f); setResult(null); setError(null); } }} />
                    {file ? file.name : "选择图片"}
                  </label>
                </div>
                <div style={{ paddingTop: 14 }}>
                  <button className="btn primary" onClick={handleInfer} disabled={!file || loading} style={{ fontSize: 11, minWidth: 80 }}>
                    {loading ? "推理中..." : "开始推理"}
                  </button>
                </div>
              </div>
              {error && <div style={{ fontSize: 11, color: "#c44", marginTop: 8 }}>{error}</div>}
            </div>
          </div>

          {/* Result: Image + Detections */}
          {result && (
            <div className="card" style={{ padding: 0, overflow: "hidden" }}>
              <div style={{ padding: "8px 12px", borderBottom: "1px solid #e6eef6", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontWeight: 600, fontSize: 11 }}>检测结果</div>
                <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{result.model_code} · 耗时 {result.latency_ms}ms · {result.image_width}×{result.image_height} · {result.detections.length} 个检测框</div>
              </div>
              <div style={{ padding: "10px 12px", display: "grid", gridTemplateColumns: result.overlay_image ? "minmax(0,1fr) 180px" : "1fr", gap: 10, alignItems: "start" }}>
                {result.overlay_image ? (
                  <div style={{ overflow: "hidden", borderRadius: 6, border: "1px solid #e6eef6", width: "100%", height: 240, display: "flex", alignItems: "center", justifyContent: "center", background: "#fafbfc", minWidth: 0 }}>
                    <img src={`data:image/jpeg;base64,${result.overlay_image}`} style={{ display: "block", maxWidth: "100%", maxHeight: 240, width: "100%", height: 240, objectFit: "contain" }} />
                  </div>
                ) : (
                  <div style={{ padding: 20, textAlign: "center", color: "var(--text-muted)", fontSize: 11, border: "1px solid #e6eef6", borderRadius: 6 }}>无检测结果图片</div>
                )}
                <div>
                  <div style={{ fontSize: 10, fontWeight: 600, marginBottom: 6, color: "var(--text-secondary)" }}>检测列表</div>
                  {result.detections.length === 0 ? (
                    <div style={{ fontSize: 10, color: "var(--text-muted)" }}>未检测到目标</div>
                  ) : result.detections.map((d, i) => (
                    <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #f0f4f8", fontSize: 10 }}>
                      <span style={{ fontWeight: 600 }}>{d.class_name}</span>
                      <span style={{ color: "var(--text-muted)" }}>{(d.confidence * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Result JSON */}
          {result && (
            <div className="card" style={{ padding: 0, overflow: "hidden" }}>
              <div style={{ padding: "8px 12px", borderBottom: "1px solid #e6eef6", fontWeight: 600, fontSize: 11 }}>返回参数</div>
              <div style={{ padding: "10px 12px", fontFamily: "Courier New, monospace", fontSize: 10, background: "#fafbfc", wordBreak: "break-all", whiteSpace: "pre-wrap" }}>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{JSON.stringify(result, null, 2)}</pre>
              </div>
            </div>
          )}

          {!result && !loading && (
            <div style={{ padding: 20, textAlign: "center", color: "var(--text-muted)", fontSize: 11, border: "1px dashed #bed2df", borderRadius: 6 }}>
              选择模型 → 上传图片 → 点击"开始推理"查看结果
            </div>
          )}
        </div>
      </section>
    </>
  );
}
