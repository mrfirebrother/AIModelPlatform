import { useCallback, useEffect, useRef, useState } from "react";

type BBox = { classId: number; bbox: [number, number, number, number] };
type Polygon = { classId: number; polygon: [number, number][] };

export default function AnnotateCanvas({ imageUrl, mode, onSave, classId, datasetId, imageId }: { imageUrl: string; mode: "bbox" | "polygon"; onSave: (ann: BBox | Polygon) => void; classId: number; datasetId?: string; imageId?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const lastBoxRef = useRef<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const polyRef = useRef<[number, number][]>([]);
  const [poly, setPoly] = useState<[number, number][]>([]);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!imageUrl) { setBlobUrl(null); imgRef.current = null; polyRef.current = []; setPoly([]); return; }
    const apiKey = (import.meta as any).env?.VITE_API_KEY || "change-me";
    fetch(imageUrl, { headers: { "X-API-Key": apiKey } })
      .then((r) => r.blob())
      .then((b) => {
        const url = URL.createObjectURL(b);
        setBlobUrl(url);
        const img = new Image();
        img.onload = () => {
          imgRef.current = img;
          lastBoxRef.current = null;
          polyRef.current = [];
          setPoly([]);
          const c = canvasRef.current;
          if (!c) return;
          const ctx = c.getContext("2d");
          if (!ctx) return;
          c.width = img.width;
          c.height = img.height;
          ctx.drawImage(img, 0, 0);
          // Load saved annotations
          if (datasetId && imageId) {
            fetch(`/api/datasets/${datasetId}/annotations/${imageId}`, {
              headers: { "X-API-Key": apiKey },
            }).then((r) => r.json()).then((data) => {
              if (data.annotations && data.annotations.length > 0 && c && imgRef.current) {
                const ctx2 = c.getContext("2d");
                if (!ctx2) return;
                data.annotations.forEach((ann: any) => {
                  if (ann.bbox) {
                    const [x, y, w, h] = ann.bbox;
                    ctx2.strokeStyle = "#2d83c5";
                    ctx2.lineWidth = 2;
                    ctx2.strokeRect(x * img.width, y * img.height, w * img.width, h * img.height);
                  } else if (ann.polygon) {
                    const pts = ann.polygon;
                    if (pts.length >= 2) {
                      ctx2.strokeStyle = "#8b5cf6";
                      ctx2.lineWidth = 2;
                      ctx2.beginPath();
                      ctx2.moveTo(pts[0][0] * img.width, pts[0][1] * img.height);
                      pts.slice(1).forEach(([px, py]: [number, number]) => ctx2.lineTo(px * img.width, py * img.height));
                      ctx2.closePath();
                      ctx2.stroke();
                      ctx2.fillStyle = "rgba(139,92,246,0.15)";
                      ctx2.fill();
                      pts.forEach(([px, py]: [number, number]) => {
                        ctx2.beginPath();
                        ctx2.arc(px * img.width, py * img.height, 3, 0, Math.PI * 2);
                        ctx2.fillStyle = "#8b5cf6";
                        ctx2.fill();
                      });
                    }
                  }
                });
              }
            }).catch(() => {});
          }
        };
        img.src = url;
      })
      .catch(() => { setBlobUrl(null); imgRef.current = null; });
  }, [imageUrl, datasetId, imageId]);

  const toNorm = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current!;
    const rect = c.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    return [x, y] as [number, number];
  };

  const redrawImage = useCallback(() => {
    const c = canvasRef.current;
    const img = imgRef.current;
    if (!c || !img) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    c.width = img.width;
    c.height = img.height;
    ctx.drawImage(img, 0, 0);
    if (lastBoxRef.current) {
      const { x1, y1, x2, y2 } = lastBoxRef.current;
      ctx.strokeStyle = "#2d83c5";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
  }, []);

  const drawPolygon = (pts: [number, number][], closed = false) => {
    const c = canvasRef.current;
    const img = imgRef.current;
    if (!c || !img) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    c.width = img.width;
    c.height = img.height;
    ctx.drawImage(img, 0, 0);
    if (pts.length >= 2) {
      ctx.strokeStyle = "#8b5cf6";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pts[0][0] * img.width, pts[0][1] * img.height);
      pts.slice(1).forEach(([px, py]) => ctx.lineTo(px * img.width, py * img.height));
      if (closed) ctx.closePath();
      ctx.stroke();
      if (closed) {
        ctx.fillStyle = "rgba(139,92,246,0.15)";
        ctx.fill();
      }
    }
    pts.forEach(([px, py]) => {
      ctx.beginPath();
      ctx.arc(px * img.width, py * img.height, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#8b5cf6";
      ctx.fill();
    });
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 0 && mode === "bbox") {
      e.preventDefault();
      const [x, y] = toNorm(e);
      dragRef.current = { x, y };
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragRef.current || mode !== "bbox") return;
    const [x2, y2] = toNorm(e);
    const { x: x1, y: y1 } = dragRef.current;
    const c = canvasRef.current!;
    const ctx = c.getContext("2d")!;
    const img = imgRef.current!;
    c.width = img.width;
    c.height = img.height;
    ctx.drawImage(img, 0, 0);
    const lx = Math.min(x1, x2) * img.width;
    const ly = Math.min(y1, y2) * img.height;
    const lw = Math.abs(x2 - x1) * img.width;
    const lh = Math.abs(y2 - y1) * img.height;
    ctx.strokeStyle = "#2d83c5";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(lx, ly, lw, lh);
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(45,131,197,0.12)";
    ctx.fillRect(lx, ly, lw, lh);
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragRef.current || mode !== "bbox") return;
    const [x2, y2] = toNorm(e);
    const { x: x1, y: y1 } = dragRef.current;
    dragRef.current = null;
    const x = Math.min(x1, x2);
    const y = Math.min(y1, y2);
    const w = Math.abs(x2 - x1);
    const h = Math.abs(y2 - y1);
    if (w > 0.02 && h > 0.02) {
      const img = imgRef.current!;
      lastBoxRef.current = {
        x1: x * img.width,
        y1: y * img.height,
        x2: (x + w) * img.width,
        y2: (y + h) * img.height,
      };
      onSave({ classId, bbox: [x, y, w, h] });
    }
    redrawImage();
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode !== "polygon") return;
    const [x, y] = toNorm(e);
    polyRef.current = [...polyRef.current, [x, y]];
    setPoly([...polyRef.current]);
    drawPolygon(polyRef.current);
  };

  const handleContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (mode === "polygon" && polyRef.current.length >= 3) {
      lastBoxRef.current = null;
      onSave({ classId, polygon: polyRef.current });
      drawPolygon(polyRef.current, true);
      polyRef.current = [];
      setPoly([]);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
      style={{ width: 600, height: 600, objectFit: "contain", border: "1px solid #d3e2ec", borderRadius: 6, cursor: "crosshair", background: "#fafbfc" }}
    />
  );
}
