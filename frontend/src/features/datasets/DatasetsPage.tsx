import { useEffect, useMemo, useRef, useState } from "react";
import { createApi } from "../../lib/createApi";
import type { Dataset } from "../../lib/api";

export default function DatasetsPage() {
  const api = useMemo(() => createApi(), []);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [showImport, setShowImport] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDatasets = () => api.getDatasets().then(setDatasets);
  useEffect(() => { loadDatasets(); }, [api]);

  const handleFile = async (file: File) => {
    const lower = file.name.toLowerCase();
    if (!lower.endsWith(".zip") && !lower.endsWith(".tar.gz") && !lower.endsWith(".tgz")) {
      setUploadError("仅支持 .zip 或 .tar.gz 文件");
      return;
    }
    if (file.size > 2 * 1024 * 1024 * 1024) {
      setUploadError("文件大小不能超过 2GB");
      return;
    }
    setUploadError(null);
    setUploadPct(0);
    setUploading(true);
    try {
      const res = await api.uploadDataset(file, (pct) => setUploadPct(pct));
      await api.createDataset({
        name: file.name.replace(/\.(zip|tar\.gz|tgz)$/i, ""),
        sourcePath: res.file_path,
      });
      setShowImport(false);
      loadDatasets();
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

  return (
    <>
      <div className="top-actions" style={{ marginBottom: 16, display: "flex", gap: 10, alignItems: "center" }}>
        <button className="btn primary" onClick={() => setShowImport(true)}>+ 导入数据集</button>
      </div>

      {showImport && (
        <div className="card" style={{ marginBottom: 14, padding: 24 }}>
          <div className="card-title" style={{ marginBottom: 16 }}>导入 YOLO 数据集</div>
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
            <input ref={fileInputRef} type="file" accept=".zip,.tar.gz,.tgz" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
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
                <div style={{ fontSize: 40, marginBottom: 8, color: "#ccc" }}>{"\u2b07"}</div>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>拖放 .zip / .tar.gz 文件到此处</div>
                <div style={{ fontSize: 11, color: "#999" }}>或点击选择文件 {"\u00b7"} 最大 2GB</div>
              </>
            )}
          </div>
          {uploadError && <div style={{ marginTop: 10, color: "#b33", fontSize: 12 }}>{uploadError}</div>}
          <div style={{ marginTop: 14 }}><button className="btn" onClick={() => { setShowImport(false); setUploadError(null); }}>取消</button></div>
        </div>
      )}

      <div className="card">
        <div className="card-head">
          <div>
            <div className="card-title">数据集管理</div>
            <div className="card-kicker">{datasets.length} 个数据集</div>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>数据集名称</th>
                <th>标签体系</th>
                <th>图片总数</th>
                <th>训练集</th>
                <th>验证集</th>
                <th>测试集</th>
                <th>来源</th>
                <th>校验</th>
                <th>快照</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.id}>
                  <td>
                    <div style={{ fontWeight: 600 }}>{d.name}</div>
                    <div style={{ color: "var(--text-muted)", fontSize: 9, fontFamily: "Courier New, monospace" }}>{d.id}</div>
                  </td>
                  <td>{d.labelSchemaName ?? "-"}</td>
                  <td style={{ fontWeight: 600 }}>{(d.imageCount ?? 0).toLocaleString()}</td>
                  <td>{(d.trainCount ?? 0).toLocaleString()}</td>
                  <td>{(d.valCount ?? 0).toLocaleString()}</td>
                  <td>{(d.testCount ?? 0).toLocaleString()}</td>
                  <td>{d.source ?? "-"}</td>
                  <td><span className={`badge ${(d.validationStatus ?? "") === "通过" ? "badge-green" : "badge-red"}`}>{d.validationStatus ?? "-"}</span></td>
                  <td style={{ fontFamily: "Courier New, monospace", fontSize: 10 }}>{d.latestSnapshotId ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
