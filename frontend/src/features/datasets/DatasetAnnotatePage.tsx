import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import { useToast } from "../../lib/toast";
import AnnotateCanvas from "./AnnotateCanvas";

export default function DatasetAnnotatePage() {
  const { id } = useParams();
  const api = useMemo(() => createApi(), []);
  const toast = useToast();
  const navigate = useNavigate();
  const [images, setImages] = useState<string[]>([]);
  const [current, setCurrent] = useState(0);
  const [mode, setMode] = useState<"bbox" | "polygon">("bbox");
  const [classId, setClassId] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0);
  const [classNames, setClassNames] = useState<string[]>([]);

  const imageUrl = id ? `/api/datasets/${id}/images/${images[current]}` : "";

  // Load image list on mount
  useEffect(() => {
    if (!id) return;
    Promise.all([
      fetch(`/api/datasets/${id}/image_list`, { headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" } }).then((r) => r.json()).catch(() => ({ images: [] })),
      fetch(`/api/datasets/${id}`, { headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" } }).then((r) => r.json()).catch(() => ({})),
    ]).then(([imgData, dsData]) => {
      if (imgData.images && imgData.images.length > 0) {
        setImages(imgData.images);
        setCurrent(0);
      }
      // Load class names from label schema (always try, not only when snapshot exists)
      fetch(`/api/datasets/${id}/label_classes`, { headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" } })
        .then((r) => r.json())
        .then((data) => {
          if (data.classes && data.classes.length > 0) {
            setClassNames(data.classes);
          } else {
            // Fallback: show single default class
            setClassNames(["object"]);
          }
        })
        .catch(() => { setClassNames(["object"]); });
    });
  }, [id]);

  const handleUpload = async (files: FileList | null) => {
    if (!files || !id) return;
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    try {
      const res = await fetch(`/api/datasets/${id}/images`, {
        method: "POST",
        headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());
      toast.success("上传成功");
      const newNames = Array.from(files).map((f) => f.name.replace(/\.[^/.]+$/, ""));
      // New images go to front of list, select the first new one
      setImages((prev) => [...newNames, ...prev]);
      setCurrent(0);
    } catch (err) {
      toast.error("上传失败: " + (err as Error).message);
    }
  };

  const handleSave = async (ann: any) => {
    try {
      await fetch(`/api/datasets/${id}/annotations/${images[current]}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json", "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
        body: JSON.stringify({ annotations: [ann] }),
      });
      toast.success("已保存");
    } catch (err) {
      toast.error("保存失败");
    }
  };

  const handleSnapshot = async () => {
    try {
      await fetch(`/api/datasets/${id}/snapshot`, {
        method: "POST",
        headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
      });
      toast.success("快照生成中，后台处理");
      setTimeout(() => navigate("/datasets"), 1000);
    } catch (err) {
      toast.error("失败: " + (err as Error).message);
    }
  };

  const handleDeleteAnnotation = async () => {
    if (!id || !images[current]) return;
    try {
      await fetch(`/api/datasets/${id}/annotations/${images[current]}`, {
        method: "DELETE",
        headers: { "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" },
      });
      toast.success("已清空");
      setRefreshKey((k) => k + 1);
    } catch (err) {
      toast.error("失败");
    }
  };

  return (
    <>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <button className="btn" onClick={() => navigate("/datasets")}>返回列表</button>
        <button className="btn primary" onClick={handleSnapshot}>生成快照</button>
        <button className="btn" onClick={handleDeleteAnnotation}>清空标注</button>
        <label className="btn" style={{ cursor: "pointer" }}>
          上传图片
          <input type="file" multiple accept="image/*" style={{ display: "none" }} onChange={(e) => handleUpload(e.target.files)} />
        </label>
        <select value={mode} onChange={(e) => setMode(e.target.value as any)} style={{ padding: "6px 8px", border: "1px solid #bed2df", borderRadius: 3 }}>
          <option value="bbox">框选</option>
          <option value="polygon">多边形</option>
        </select>
        <select value={classId} onChange={(e) => setClassId(Number(e.target.value))} style={{ padding: "6px 8px", border: "1px solid #bed2df", borderRadius: 3 }}>
          {classNames.map((c, i) => <option key={i} value={i}>{c}</option>)}
        </select>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 200px", gap: 12 }}>
          <div className="card" style={{ padding: 12, display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ width: "100%", maxWidth: 600, display: "flex", justifyContent: "center" }}>
            <AnnotateCanvas key={`${images[current]}-${refreshKey}`} imageUrl={imageUrl} mode={mode} classId={classId} onSave={handleSave} datasetId={id} imageId={images[current]} />
          </div>
          <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
            <button className="btn small" onClick={() => setCurrent((c) => Math.max(0, c - 1))}>上一张</button>
            <button className="btn small" onClick={() => setCurrent((c) => Math.min(images.length - 1, c + 1))}>下一张</button>
            <span style={{ fontSize: 11, color: "var(--text-muted)", alignSelf: "center" }}>{current + 1} / {images.length}</span>
          </div>
        </div>
        <div className="card" style={{ padding: 8 }}>
          <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 8 }}>图片列表 (Shift 多选)</div>
          {images.map((img, idx) => (
            <div key={img} onClick={() => setCurrent(idx)} style={{ padding: "6px 8px", borderBottom: "1px solid #f0f4f8", cursor: "pointer", background: idx === current ? "#e7f1fa" : undefined, fontSize: 11 }}>{img}</div>
          ))}
          <div style={{ marginTop: 8, display: "flex", gap: 6 }}>
            <button className="btn small" onClick={async () => {
              await fetch(`/api/datasets/${id}/batch`, { method: "POST", headers: { "Content-Type": "application/json", "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" }, body: JSON.stringify({ operation: "delete", image_ids: [images[current]] }) });
              toast.success("已删除");
            }}>批量删图</button>
          </div>
        </div>
      </div>
    </>
  );
}
