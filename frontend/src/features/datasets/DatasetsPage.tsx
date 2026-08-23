import { useEffect, useMemo, useState, useRef } from "react";
import { createApi } from "../../lib/createApi";
import type { Dataset } from "../../lib/api";

export default function DatasetsPage() {
  const api = useMemo(() => createApi(), []);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [showImport, setShowImport] = useState(false);
  const [importPath, setImportPath] = useState("");
  const [importName, setImportName] = useState("");
  const [importing, setImporting] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDatasets = () => api.getDatasets().then(setDatasets);
  useEffect(() => { loadDatasets(); }, [api]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

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
    setImportName(file.name.replace(/\.(zip|tar\.gz|tgz)$/i, ""));

    try {
      const res = await api.uploadDataset(file, (pct) => setUploadPct(pct));
      setImportPath(res.file_path);
      setUploadPct(null);
    } catch (err) {
      setUploadError("上传失败: " + (err as Error).message);
      setUploadPct(null);
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleImport = async () => {
    if (!importPath) return;
    setImporting(true);
    try {
      await api.createDataset({
        name: importName || importPath.split("/").pop(),
        sourcePath: importPath,
      });
      setShowImport(false);
      setImportPath("");
      setImportName("");
      loadDatasets();
    } catch (e) {
      alert("导入失败: " + (e as Error).message);
    }
    setImporting(false);
  };

  return (
    <>
      <div className="top-actions" style={{ marginBottom: 16, display: "flex", gap: 10, alignItems: "center" }}>
        <button className="btn primary" onClick={() => setShowImport(true)}>
          + 导入数据集
        </button>
      </div>

      {showImport && (
        <div className="card" style={{ marginBottom: 14, padding: 20 }}>
          <div className="card-title" style={{ marginBottom: 12 }}>导入 YOLO 数据集</div>
          <div style={{ display: "grid", gap: 10, maxWidth: 500 }}>
            <div>
              <label style={{ fontSize: 11, color: "#666" }}>选择压缩包文件</label>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 4 }}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".zip,.tar.gz,.tgz"
                  onChange={handleFileSelect}
                  style={{ fontSize: 11, flex: 1 }}
                />
              </div>
              {uploadPct !== null && (
                <div style={{ marginTop: 6 }}>
                  <div style={{ background: "#e0e8f0", borderRadius: 3, height: 6, overflow: "hidden" }}>
                    <div
                      style={{
                        background: "#2563eb",
                        height: "100%",
                        width: `${uploadPct}%`,
                        transition: "width 0.2s",
                      }}
                    />
                  </div>
                  <div style={{ fontSize: 10, color: "#666", marginTop: 2 }}>{uploadPct}%</div>
                </div>
              )}
              {uploadError && (
                <div style={{ fontSize: 11, color: "#dc2626", marginTop: 4 }}>{uploadError}</div>
              )}
            </div>
            {importPath && (
              <div style={{ fontSize: 11, color: "#16a34a", background: "#f0fdf4", padding: "6px 10px", borderRadius: 3 }}>
                已上传: {importPath}
              </div>
            )}
            <div>
              <label style={{ fontSize: 11, color: "#666" }}>数据集目录路径（也可手动输入）</label>
              <input
                value={importPath}
                onChange={(e) => setImportPath(e.target.value)}
                placeholder="/path/to/yolo-dataset"
                style={{ width: "100%", padding: "8px 10px", border: "1px solid #bed2df", borderRadius: 3, fontSize: 12, marginTop: 4 }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#666" }}>数据集名称（可选）</label>
              <input
                value={importName}
                onChange={(e) => setImportName(e.target.value)}
                placeholder="fire-dataset"
                style={{ width: "100%", padding: "8px 10px", border: "1px solid #bed2df", borderRadius: 3, fontSize: 12, marginTop: 4 }}
              />
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn primary" onClick={handleImport} disabled={importing || !importPath || uploadPct !== null}>
                {importing ? "导入中..." : "确认导入"}
              </button>
              <button className="btn" onClick={() => { setShowImport(false); setImportPath(""); setImportName(""); setUploadPct(null); setUploadError(null); }}>取消</button>
            </div>
          </div>
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
                  <div
                    style={{
                      color: "var(--text-muted)",
                      fontSize: 9,
                      fontFamily: "Courier New, monospace",
                    }}
                  >
                    {d.id}
                  </div>
                </td>
                <td>{d.labelSchemaName}</td>
                <td style={{ fontWeight: 600 }}>{d.imageCount.toLocaleString()}</td>
                <td>{d.trainCount.toLocaleString()}</td>
                <td>{d.valCount.toLocaleString()}</td>
                <td>{d.testCount.toLocaleString()}</td>
                <td>{d.source}</td>
                <td>
                  <span
                    className={`badge ${d.validationStatus === "通过" ? "badge-green" : "badge-red"}`}
                  >
                    {d.validationStatus}
                  </span>
                </td>
                <td
                  style={{
                    fontFamily: "Courier New, monospace",
                    fontSize: 10,
                  }}
                >
                  {d.latestSnapshotId}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {datasets.some((d) => d.warnings.length > 0) && (
        <div
          className="card-body"
          style={{ borderTop: "1px solid #dce8f0" }}
        >
          <div style={{ color: "var(--accent-orange)", fontSize: 11, fontWeight: 600, marginBottom: 6 }}>
            数据质量警告
          </div>
          {datasets
            .filter((d) => d.warnings.length > 0)
            .map((d) =>
              d.warnings.map((w, i) => (
                <div
                  key={`${d.id}-${i}`}
                  style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 2 }}
                >
                  {d.name}: {w}
                </div>
              )),
            )}
        </div>
      )}
      </div>
    </>
  );
}
