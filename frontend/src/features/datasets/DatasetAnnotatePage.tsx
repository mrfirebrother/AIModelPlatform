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
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const imageUrl = id && images.length > 0 ? `/api/datasets/${id}/images/${images[current]}` : "";

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
            setClassNames([]);
          }
        })
        .catch(() => { setClassNames([]); });
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
        {classNames.length > 0 && (
          <select value={classId} onChange={(e) => setClassId(Number(e.target.value))} style={{ padding: "6px 8px", border: "1px solid #bed2df", borderRadius: 3 }}>
            {classNames.map((c, i) => <option key={i} value={i}>{c}</option>)}
          </select>
        )}
        {classNames.length === 0 && (
          <span style={{ fontSize: 11, color: "var(--text-muted)", padding: "6px 0" }}>无标注类别（负例模式）</span>
        )}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 200px", gap: 12 }}>
          <div className="card" style={{ padding: 12, display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ width: "100%", maxWidth: 600, display: "flex", justifyContent: "center" }}>
            {images.length > 0 ? (
              <AnnotateCanvas key={`${images[current]}-${refreshKey}`} imageUrl={imageUrl} mode={mode} classId={classId} onSave={handleSave} datasetId={id} imageId={images[current]} />
            ) : (
              <div style={{ padding: 60, textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>暂无图片，请先上传</div>
            )}
          </div>
          <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
            <button className="btn small" onClick={() => setCurrent((c) => Math.max(0, c - 1))}>上一张</button>
            <button className="btn small" onClick={() => setCurrent((c) => Math.min(images.length - 1, c + 1))}>下一张</button>
            <span style={{ fontSize: 11, color: "var(--text-muted)", alignSelf: "center" }}>{current + 1} / {images.length}</span>
          </div>
        </div>
        <div className="card" style={{ padding: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <label style={{ fontSize: 11, fontWeight: 600, display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
              <input type="checkbox" checked={selected.size === images.length && images.length > 0} onChange={(e) => { if (e.target.checked) setSelected(new Set(images)); else setSelected(new Set()); }} />
              全选 ({selected.size}/{images.length})
            </label>
            {selected.size > 0 && (
              <button className="btn small danger" onClick={async () => {
                const ok = await toast.confirm(`确定删除选中的 ${selected.size} 张图片？`);
                if (!ok) return;
                await fetch(`/api/datasets/${id}/batch`, { method: "POST", headers: { "Content-Type": "application/json", "X-API-Key": import.meta.env.VITE_API_KEY || "change-me" }, body: JSON.stringify({ operation: "delete", image_ids: Array.from(selected) }) });
                toast.success(`已删除 ${selected.size} 张`);
                const remaining = images.filter((img) => !selected.has(img));
                setImages(remaining);
                setSelected(new Set());
                if (current >= remaining.length) setCurrent(Math.max(0, remaining.length - 1));
              }}>删除选中 ({selected.size})</button>
            )}
          </div>
          <div style={{ maxHeight: 400, overflowY: "auto" }}>
            {images.map((img, idx) => (
              <div key={img} style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 6px", borderBottom: "1px solid #f0f4f8", background: idx === current ? "#e7f1fa" : undefined }}>
                <input type="checkbox" checked={selected.has(img)} onChange={(e) => {
                  setSelected((prev) => { const next = new Set(prev); if (e.target.checked) next.add(img); else next.delete(img); return next; });
                }} onClick={(e) => e.stopPropagation()} />
                <span onClick={() => setCurrent(idx)} style={{ flex: 1, cursor: "pointer", fontSize: 11, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{img}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
