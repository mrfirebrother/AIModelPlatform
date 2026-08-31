import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import { useToast } from "../../lib/toast";
import type { Dataset } from "../../lib/api";

export default function DatasetsPage() {
  const api = useMemo(() => createApi(), []);
  const toast = useToast();
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [showImport, setShowImport] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDatasets = () => api.getDatasets().then(setDatasets);
  useEffect(() => { loadDatasets(); }, [api]);

  const handleDelete = async (id: string) => {
    const confirmed = await toast.confirm("确定删除此数据集？");
    if (!confirmed) return;
    try {
      await api.deleteDataset(id);
      toast.success("数据集已删除");
      loadDatasets();
    } catch (err) {
      toast.error("删除失败: " + (err as Error).message);
    }
  };

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
      setUploadPct(100);
      await api.createDataset({
        name: file.name.replace(/\.(zip|tar\.gz|tgz)$/i, ""),
        sourcePath: res.file_path,
      });
      toast.success("已提交，后台解析中");
      setShowImport(false);
      loadDatasets();
      const targetName = file.name.replace(/\.(zip|tar\.gz|tgz)$/i, "");
      let polls = 0;
      const timer = setInterval(async () => {
        polls += 1;
        if (polls > 60) { clearInterval(timer); return; }
        try {
          const list = await api.getDatasets();
          setDatasets(list);
          const d = list.find((x) => x.name === targetName || x.name.startsWith(targetName));
          if (d && d.validationStatus !== "-" && (d.imageCount ?? 0) > 0) {
            clearInterval(timer);
            toast.success(`数据集 ${d.name} 解析完成：${d.imageCount} 张`);
          }
        } catch { /* ignore */ }
      }, 3000);
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

  return (
    <>
      <div className="toolbar">
        <button className="btn primary" onClick={() => setShowImport(true)}>+ 导入数据集</button>
        <button className="btn" onClick={() => { window.location.hash = "#/datasets/create"; window.location.href = "/#/datasets/create"; window.location.pathname = "/datasets/create"; }}>+ 新建数据集</button>
      </div>

      {showImport && (
        <div className="card" style={{ marginBottom: 14, padding: 24 }}>
          <div className="card-title" style={{ marginBottom: 16 }}>导入 YOLO 数据集</div>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => !uploading && fileInputRef.current?.click()}
            className={`dropzone ${dragOver ? "dropzone-active" : "dropzone-idle"}`}
          >
            <input ref={fileInputRef} type="file" accept=".zip,.tar.gz,.tgz" style={{ display: "none" }} onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
            {uploading ? (
              <>
                <div className="upload-label">{(uploadPct ?? 0) >= 100 ? "后台解析中..." : "上传中..."}</div>
                <div className="upload-progress">
                  <div className="upload-progress-fill" style={{ width: `${uploadPct ?? 0}%` }} />
                </div>
                <div className="upload-hint">{(uploadPct ?? 0) >= 100 ? "已上传，等待解析" : `${uploadPct ?? 0}%`}</div>
              </>
            ) : (
              <>
                <div className="dropzone-icon">{"⬇"}</div>
                <div className="dropzone-title">拖放 .zip / .tar.gz 文件到此处</div>
                <div className="dropzone-hint">或点击选择文件 {"·"} 最大 2GB</div>
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
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => {
                const isParsing = (d.imageCount ?? 0) === 0 && (d.validationStatus ?? "-") === "-" && !d.latestSnapshotId;
                return (
                <tr key={d.id} className={isParsing ? "row-parsing" : undefined}>
                  <td>
                    <div style={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 6 }}>{d.name}{isParsing && <span className="badge badge-orange badge-xs">解析中</span>}</div>
                    <div className="cell-mono">{d.id}</div>
                  </td>
                  <td>{isParsing ? <span className="parsing-text">{"—"}</span> : (d.labelSchemaName ?? "-")}</td>
                  <td style={{ fontWeight: 600 }}>{isParsing ? <span className="parsing-text">解析中</span> : (d.imageCount ?? 0).toLocaleString()}</td>
                  <td>{isParsing ? "—" : (d.trainCount ?? 0).toLocaleString()}</td>
                  <td>{isParsing ? "—" : (d.valCount ?? 0).toLocaleString()}</td>
                  <td>{isParsing ? "—" : (d.testCount ?? 0).toLocaleString()}</td>
                  <td>{d.source ?? "-"}</td>
                  <td>{isParsing ? <span className="badge badge-orange">解析中</span> : <span className={`badge ${(d.validationStatus ?? "") === "通过" ? "badge-green" : "badge-red"}`}>{d.validationStatus ?? "-"}</span>}</td>
                  <td className="cell-mono">{isParsing ? <span className="parsing-text">生成中</span> : (d.latestSnapshotId ?? "-")}</td>
                  <td>
                    <div style={{ display: "flex", gap: 6 }}>
                      {!d.source && <button className="btn small" onClick={() => navigate(`/datasets/${d.id}/annotate`)}>标注</button>}
                      <button className="btn small danger" onClick={() => handleDelete(d.id)}>删除</button>
                    </div>
                  </td>
                </tr>
              );})}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
