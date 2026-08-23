import { useEffect, useMemo, useState } from "react";
import { createApi } from "../../lib/createApi";
import type { Dataset } from "../../lib/api";

export default function DatasetsPage() {
  const api = useMemo(() => createApi(), []);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [showImport, setShowImport] = useState(false);
  const [importPath, setImportPath] = useState("");
  const [importName, setImportName] = useState("");
  const [importing, setImporting] = useState(false);

  const loadDatasets = () => api.getDatasets().then(setDatasets);
  useEffect(() => { loadDatasets(); }, [api]);

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
              <label style={{ fontSize: 11, color: "#666" }}>数据集目录路径</label>
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
              <button className="btn primary" onClick={handleImport} disabled={importing}>
                {importing ? "导入中..." : "确认导入"}
              </button>
              <button className="btn" onClick={() => setShowImport(false)}>取消</button>
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
