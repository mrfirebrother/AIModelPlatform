"use strict";

const API_KEY = "change-me";
function uiPassword() { return sessionStorage.getItem("ui_password") || ""; }
function apiHeaders(extra) {
  return Object.assign({
    "X-API-Key": API_KEY,
    "X-UI-Password": uiPassword(),
  }, extra || {});
}
async function apiFetch(path, options) {
  const opts = options || {};
  const res = await apiFetch(path, Object.assign({}, opts, { headers: apiHeaders(opts.headers || {}) }));
  if (res.status === 401) { window.location.href = "/"; throw new Error("unauthorized"); }
  return res;
}

const ROW = 34;
const folderInput = document.querySelector("#folder");
const statsEl = document.querySelector("#stats");
const listEl = document.querySelector("#list");
const canvas = document.querySelector("#view");
const emptyEl = document.querySelector("#empty");
const metaEl = document.querySelector("#meta");
const statusEl = document.querySelector("#status");
const boxListEl = document.querySelector("#boxes");
const ctx = canvas.getContext("2d");

const state = {
  images: [],
  filter: "all",
  query: "",
  index: -1,
  boxes: [],
  selected: -1,
  seen: false,
  img: null,
  view: { scale: 1, ox: 0, oy: 0 },
  fitted: 1,
  userView: false,
  undo: [],
  tool: null,
  drag: null,
  ready: false,
};

const spacer = document.createElement("div");
const windowEl = document.createElement("div");
spacer.style.position = "relative";
windowEl.style.position = "absolute";
windowEl.style.left = "0";
windowEl.style.right = "0";
windowEl.style.top = "0";
spacer.appendChild(windowEl);
listEl.appendChild(spacer);

function setStatus(text, bad = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("bad", bad);
}

function current() {
  return state.index >= 0 ? state.images[state.index] : null;
}

function filteredIndexes() {
  const query = state.query.trim().toLowerCase();
  const indexes = [];
  state.images.forEach((image, index) => {
    if (query && !image.rel.toLowerCase().includes(query) && !image.name.toLowerCase().includes(query)) return;
    if (state.filter === "todo" && image.count !== null) return;
    if (state.filter === "done" && image.count === null) return;
    indexes.push(index);
  });
  return indexes;
}

function renderStats() {
  const total = state.images.length;
  const seen = state.images.filter((image) => image.count !== null).length;
  const here = state.index >= 0 ? state.index + 1 : 0;
  statsEl.textContent = total ? `${String(here).padStart(3, "0")} / ${total}    已看 ${seen}` : "未打开";
}

function renderList() {
  const indexes = filteredIndexes();
  const viewHeight = listEl.clientHeight || 1;
  const start = Math.max(0, Math.floor(listEl.scrollTop / ROW) - 6);
  const end = Math.min(indexes.length, Math.ceil((listEl.scrollTop + viewHeight) / ROW) + 6);
  spacer.style.height = `${indexes.length * ROW}px`;
  windowEl.style.transform = `translateY(${start * ROW}px)`;
  windowEl.innerHTML = "";
  indexes.slice(start, end).forEach((imageIndex) => {
    const image = state.images[imageIndex];
    const button = document.createElement("button");
    button.type = "button";
    button.className = `row${imageIndex === state.index ? " on" : ""}`;
    button.dataset.i = String(imageIndex);
    const dot = document.createElement("i");
    dot.className = "dot" + (image.count === null ? "" : image.count > 0 ? " seen" : " empty");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = image.rel;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = image.count === null ? "" : String(image.count);
    button.append(dot, name, count);
    windowEl.appendChild(button);
  });
}

function scrollCurrentIntoView() {
  const indexes = filteredIndexes();
  const pos = indexes.indexOf(state.index);
  if (pos < 0) return;
  const top = pos * ROW;
  if (top < listEl.scrollTop || top > listEl.scrollTop + listEl.clientHeight - ROW) {
    listEl.scrollTop = Math.max(0, top - listEl.clientHeight / 2);
  }
}

function renderBoxes() {
  boxListEl.innerHTML = "";
  state.boxes.forEach((box, index) => {
    const item = document.createElement("li");
    if (index === state.selected) item.className = "on";
    const label = document.createElement("button");
    label.type = "button";
    label.textContent = `轮 ${index + 1}`;
    label.addEventListener("click", () => {
      state.selected = index;
      draw();
      renderBoxes();
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "删除";
    remove.addEventListener("click", () => {
      pushUndo();
      state.boxes.splice(index, 1);
      state.selected = -1;
      commit();
    });
    item.append(label, remove);
    boxListEl.appendChild(item);
  });
  if (!state.boxes.length) {
    const item = document.createElement("li");
    item.textContent = "还没有框";
    boxListEl.appendChild(item);
  }
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  if (state.img && !state.userView) fit();
  draw();
}

function fit() {
  if (!state.img) return;
  const margin = 28 * (window.devicePixelRatio || 1);
  const scale = Math.min(
    (canvas.width - margin * 2) / state.img.naturalWidth,
    (canvas.height - margin * 2) / state.img.naturalHeight,
  );
  state.fitted = scale;
  state.view.scale = scale;
  state.view.ox = (canvas.width - state.img.naturalWidth * scale) / 2;
  state.view.oy = (canvas.height - state.img.naturalHeight * scale) / 2;
}

function screenToImage(x, y) {
  return [(x - state.view.ox) / state.view.scale, (y - state.view.oy) / state.view.scale];
}

function eventToCanvas(event) {
  const rect = canvas.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) / rect.width) * canvas.width,
    ((event.clientY - rect.top) / rect.height) * canvas.height,
  ];
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#e7f0f8";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!state.img) return;
  const width = state.img.naturalWidth * state.view.scale;
  const height = state.img.naturalHeight * state.view.scale;
  ctx.drawImage(state.img, state.view.ox, state.view.oy, width, height);
  state.boxes.forEach((box, index) => {
    const x = state.view.ox + box.x * state.view.scale;
    const y = state.view.oy + box.y * state.view.scale;
    const w = box.w * state.view.scale;
    const h = box.h * state.view.scale;
    const selected = index === state.selected;
    ctx.strokeStyle = selected ? "#0b4f93" : "#1d6fbf";
    ctx.lineWidth = (selected ? 2.4 : 1.6) * (window.devicePixelRatio || 1);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = selected ? "#0b4f93" : "#1d6fbf";
    ctx.font = `${12 * (window.devicePixelRatio || 1)}px Consolas, monospace`;
    ctx.fillText(String(index + 1), x + 4, y + 14 * (window.devicePixelRatio || 1));
    if (!selected) return;
    const handle = 7 * (window.devicePixelRatio || 1);
    [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].forEach(([hx, hy]) => {
      ctx.fillRect(hx - handle / 2, hy - handle / 2, handle, handle);
    });
  });
  if (state.tool === "draw" && state.drag) {
    const box = normalizedBox(state.drag.x1, state.drag.y1, state.drag.x2, state.drag.y2);
    ctx.strokeStyle = "#0b4f93";
    ctx.lineWidth = 1.5 * (window.devicePixelRatio || 1);
    ctx.strokeRect(
      state.view.ox + box.x * state.view.scale,
      state.view.oy + box.y * state.view.scale,
      box.w * state.view.scale,
      box.h * state.view.scale,
    );
  }
}

function normalizedBox(x1, y1, x2, y2) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  return { x: left, y: top, w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) };
}

function clampBox(box) {
  if (!state.img) return box;
  const width = state.img.naturalWidth;
  const height = state.img.naturalHeight;
  let x = Math.min(Math.max(box.x, 0), width);
  let y = Math.min(Math.max(box.y, 0), height);
  let right = Math.min(Math.max(box.x + box.w, 0), width);
  let bottom = Math.min(Math.max(box.y + box.h, 0), height);
  return { x, y, w: Math.max(0, right - x), h: Math.max(0, bottom - y) };
}

function hitHandle(ix, iy, box) {
  const reach = 8 / state.view.scale;
  const points = {
    nw: [box.x, box.y],
    ne: [box.x + box.w, box.y],
    sw: [box.x, box.y + box.h],
    se: [box.x + box.w, box.y + box.h],
  };
  for (const [name, point] of Object.entries(points)) {
    if (Math.abs(ix - point[0]) <= reach && Math.abs(iy - point[1]) <= reach) return name;
  }
  return null;
}

function hitBox(ix, iy) {
  for (let index = state.boxes.length - 1; index >= 0; index -= 1) {
    const box = state.boxes[index];
    if (ix >= box.x && iy >= box.y && ix <= box.x + box.w && iy <= box.y + box.h) return index;
  }
  return -1;
}

function pushUndo() {
  state.undo.push(state.boxes.map((box) => ({ ...box })));
  if (state.undo.length > 40) state.undo.shift();
}

function undo() {
  const previous = state.undo.pop();
  if (!previous) return;
  state.boxes = previous;
  state.selected = -1;
  commit();
}

let saveQueue = Promise.resolve();

function commit() {
  const image = current();
  if (!image) return;
  image.count = state.boxes.length;
  state.seen = true;
  renderStats();
  renderList();
  renderBoxes();
  draw();
  const rel = image.rel;
  const boxes = state.boxes.map((box) => ({ ...box }));
  saveQueue = saveQueue.then(async () => {
    const response = await apiFetch(`/api/annotate/labels?rel=${encodeURIComponent(rel)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setStatus(payload.error || "保存失败", true);
      return;
    }
    if (current() && current().rel === rel) setStatus(`已保存 ${payload.count} 个轮子框`);
  }).catch(() => setStatus("保存失败", true));
}

async function goto(index) {
  if (index < 0 || index >= state.images.length) return;
  await saveQueue;
  state.index = index;
  state.selected = -1;
  state.undo = [];
  state.userView = false;
  state.ready = false;
  state.boxes = [];
  const image = current();
  metaEl.textContent = image.rel;
  emptyEl.classList.add("hide");
  renderStats();
  renderList();
  renderBoxes();
  draw();
  scrollCurrentIntoView();
  setStatus("正在打开…");
  const img = new Image();
  const token = index;
  img.onload = async () => {
    if (state.index !== token) return;
    state.img = img;
    fit();
    const response = await apiFetch(`/api/annotate/labels?rel=${encodeURIComponent(image.rel)}`);
    const payload = await response.json();
    if (state.index !== token) return;
    state.boxes = payload.boxes || [];
    state.seen = payload.seen;
    state.ready = true;
    draw();
    renderBoxes();
    setStatus(state.boxes.length ? `${state.boxes.length} 个框，可直接改` : "没有框。按 G 提议，或直接画。");
  };
  img.onerror = () => setStatus("图片读取失败", true);
  const imgRes = await apiFetch(`/api/annotate/image?rel=${encodeURIComponent(image.rel)}&t=${Date.now()}`);
  if (imgRes.ok) {
    img.src = URL.createObjectURL(await imgRes.blob());
  } else {
    setStatus("图片读取失败", true);
  }
}

function step(delta) {
  const indexes = filteredIndexes();
  if (!indexes.length) return;
  const pos = indexes.indexOf(state.index);
  const next = indexes[Math.min(indexes.length - 1, Math.max(0, pos + delta))];
  if (pos < 0) goto(indexes[0]);
  else if (next !== state.index) goto(next);
}

async function openFolder(path) {
  setStatus("正在读取图片…");
  const response = await apiFetch("/api/annotate/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  const payload = await response.json();
  if (!response.ok) {
    setStatus(payload.error || "打不开文件夹", true);
    return;
  }
  folderInput.textContent = payload.root;
  localStorage.setItem("axle-folder", payload.root);
  state.images = payload.images;
  state.index = -1;
  state.img = null;
  state.boxes = [];
  emptyEl.classList.toggle("hide", state.images.length > 0);
  renderStats();
  renderList();
  if (!state.images.length) {
    setStatus("这个文件夹里没有图片", true);
    return;
  }
  setStatus(`共 ${state.images.length} 张。先改未看的图。`);
  const firstTodo = state.images.findIndex((image) => image.count === null);
  goto(firstTodo >= 0 ? firstTodo : 0);
}

async function propose() {
  const image = current();
  if (!image) return;
  setStatus("正在找轮子…");
  const response = await apiFetch(`/api/annotate/propose?rel=${encodeURIComponent(image.rel)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      param2: Number(document.querySelector("#loose").value),
      lowerOnly: document.querySelector("#lower-only").checked,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    setStatus(payload.error || "提议失败", true);
    return;
  }
  pushUndo();
  state.boxes = payload.boxes || [];
  state.selected = -1;
  if (!state.boxes.length) {
    setStatus("没有找到够圆的轮子。把滑块拨向宽松，或直接画。", true);
    draw();
    renderBoxes();
    return;
  }
  commit();
  setStatus(`提议了 ${state.boxes.length} 个框。错的删掉，漏的补上。`);
}

let browserPath = "";
function renderBrowser() {
  return apiFetch("/api/annotate/browse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: browserPath }),
  }).then((r) => r.json()).then((payload) => {
    if (!payload || payload.error) { setStatus(payload.error || "浏览失败", true); return; }
    browserPath = payload.path || "";
    document.querySelector("#browser-path").textContent =
      payload.at_root ? "驱动器" : (payload.path || "驱动器");
    const list = document.querySelector("#browser-list");
    list.innerHTML = "";
    if (payload.parent) {
      const item = document.createElement("div");
      item.className = "browser-item";
      item.textContent = ".. 上一级";
      item.addEventListener("click", () => { browserPath = payload.parent; renderBrowser(); });
      list.appendChild(item);
    }
    (payload.entries || []).forEach((entry) => {
      const item = document.createElement("div");
      item.className = "browser-item";
      item.textContent = entry.name + "\\";
      item.addEventListener("click", () => { browserPath = entry.path; renderBrowser(); });
      list.appendChild(item);
    });
  }).catch(() => setStatus("浏览失败", true));
}
document.querySelector("#pick").addEventListener("click", async () => {
  document.querySelector("#browser").classList.remove("hide");
  browserPath = folderInput.textContent.trim();
  renderBrowser();
});
document.querySelector("#browser-close").addEventListener("click", () =>
  document.querySelector("#browser").classList.add("hide"));
document.querySelector("#browser-choose").addEventListener("click", () => {
  if (!browserPath) {
    setStatus("请先进入一个目录，再选这个文件夹", true);
    return;
  }
  document.querySelector("#browser").classList.add("hide");
  folderInput.textContent = browserPath;
  openFolder(browserPath);
});

// 回到驱动器：直接回到盘符列表（目录列表里的 ".. 上一级" 负责逐级回退）
document.querySelector("#browser-parent").addEventListener("click", () => {
  browserPath = "";
  renderBrowser();
});

document.querySelector("#filter").addEventListener("click", (event) => {
  const button = event.target.closest("[data-filter]");
  if (!button) return;
  state.filter = button.dataset.filter;
  document.querySelectorAll("#filter button").forEach((item) => item.classList.toggle("on", item === button));
  renderList();
});

document.querySelector("#query").addEventListener("input", (event) => {
  state.query = event.target.value;
  renderList();
});

listEl.addEventListener("scroll", renderList);
listEl.addEventListener("click", (event) => {
  const button = event.target.closest("[data-i]");
  if (!button) return;
  goto(Number(button.dataset.i));
});

document.querySelector("#propose").addEventListener("click", propose);
document.querySelector("#prev").addEventListener("click", () => step(-1));
document.querySelector("#next").addEventListener("click", () => step(1));
document.querySelector("#clear").addEventListener("click", () => {
  if (!current()) return;
  pushUndo();
  state.boxes = [];
  state.selected = -1;
  commit();
});
document.querySelector("#forget").addEventListener("click", async () => {
  const image = current();
  if (!image) return;
  await apiFetch(`/api/annotate/labels?rel=${encodeURIComponent(image.rel)}`, { method: "DELETE" });
  image.count = null;
  state.boxes = [];
  state.undo = [];
  renderStats();
  renderList();
  renderBoxes();
  draw();
  setStatus("已标为未看");
});
document.querySelector("#export").addEventListener("click", async () => {
  setStatus("正在导出…");
  const response = await apiFetch("/api/annotate/export", { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    setStatus(payload.error || "导出失败", true);
    return;
  }
  setStatus(`已导出 ${payload.images} 张到 ${payload.dir}`);
});

canvas.addEventListener("mousedown", (event) => {
  if (!state.img || !state.ready) return;
  const [cx, cy] = eventToCanvas(event);
  const [ix, iy] = screenToImage(cx, cy);
  if (event.button === 1 || event.altKey) {
    state.tool = "pan";
    state.drag = { cx, cy, ox: state.view.ox, oy: state.view.oy };
    event.preventDefault();
    return;
  }
  if (event.button !== 0) return;
  if (state.selected >= 0) {
    const handle = hitHandle(ix, iy, state.boxes[state.selected]);
    if (handle) {
      pushUndo();
      state.tool = "resize";
      state.drag = { handle, box: { ...state.boxes[state.selected] } };
      return;
    }
  }
  const hit = hitBox(ix, iy);
  if (hit >= 0) {
    pushUndo();
    state.selected = hit;
    state.tool = "move";
    const box = state.boxes[hit];
    state.drag = { dx: ix - box.x, dy: iy - box.y };
    draw();
    renderBoxes();
    return;
  }
  state.selected = -1;
  state.tool = "draw";
  state.drag = { x1: ix, y1: iy, x2: ix, y2: iy };
  renderBoxes();
});

window.addEventListener("mousemove", (event) => {
  if (!state.tool || !state.drag || !state.img) return;
  const [cx, cy] = eventToCanvas(event);
  const [ix, iy] = screenToImage(cx, cy);
  if (state.tool === "pan") {
    state.view.ox = state.drag.ox + (cx - state.drag.cx);
    state.view.oy = state.drag.oy + (cy - state.drag.cy);
    state.userView = true;
    draw();
    return;
  }
  if (state.tool === "draw") {
    state.drag.x2 = ix;
    state.drag.y2 = iy;
    draw();
    return;
  }
  if (state.tool === "move") {
    const box = state.boxes[state.selected];
    box.x = ix - state.drag.dx;
    box.y = iy - state.drag.dy;
    Object.assign(box, clampBox(box));
    draw();
    return;
  }
  if (state.tool === "resize") {
    const box = { ...state.drag.box };
    if (state.drag.handle.includes("n")) {
      box.h = box.y + box.h - iy;
      box.y = iy;
    }
    if (state.drag.handle.includes("s")) box.h = iy - box.y;
    if (state.drag.handle.includes("w")) {
      box.w = box.x + box.w - ix;
      box.x = ix;
    }
    if (state.drag.handle.includes("e")) box.w = ix - box.x;
    state.boxes[state.selected] = clampBox(normalizedBox(box.x, box.y, box.x + box.w, box.y + box.h));
    draw();
  }
});

window.addEventListener("mouseup", () => {
  if (!state.tool) return;
  const tool = state.tool;
  const drag = state.drag;
  state.tool = null;
  state.drag = null;
  if (tool === "draw" && drag) {
    const box = clampBox(normalizedBox(drag.x1, drag.y1, drag.x2, drag.y2));
    if (box.w >= 4 && box.h >= 4) {
      pushUndo();
      state.boxes.push(box);
      state.selected = state.boxes.length - 1;
      commit();
      return;
    }
  }
  if (tool === "move" || tool === "resize") commit();
  else draw();
});

canvas.addEventListener("wheel", (event) => {
  if (!state.img) return;
  event.preventDefault();
  const [cx, cy] = eventToCanvas(event);
  const [ix, iy] = screenToImage(cx, cy);
  const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
  const scale = Math.min(Math.max(state.view.scale * factor, state.fitted * 0.25), state.fitted * 14);
  state.view.ox = cx - ix * scale;
  state.view.oy = cy - iy * scale;
  state.view.scale = scale;
  state.userView = true;
  draw();
}, { passive: false });

window.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA"].includes(event.target.tagName);
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undo();
    return;
  }
  if (typing) return;
  if (event.key === "Delete" || event.key === "Backspace") {
    if (state.selected < 0) return;
    event.preventDefault();
    pushUndo();
    state.boxes.splice(state.selected, 1);
    state.selected = -1;
    commit();
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    step(1);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    step(-1);
  } else if (event.key.toLowerCase() === "g") {
    propose();
  } else if (event.key === "Escape") {
    state.selected = -1;
    draw();
    renderBoxes();
  }
});

window.addEventListener("resize", resizeCanvas);
new ResizeObserver(resizeCanvas).observe(document.querySelector("#stage"));

async function boot() {
  const response = await apiFetch("/api/annotate/state");
  const payload = await response.json();
  if (payload.root && payload.images) {
    folderInput.textContent = payload.root;
    state.images = payload.images;
    renderStats();
    renderList();
    if (state.images.length) {
      const firstTodo = state.images.findIndex((image) => image.count === null);
      goto(firstTodo >= 0 ? firstTodo : 0);
      return;
    }
  }
  const remembered = localStorage.getItem("axle-folder");
  if (remembered) folderInput.textContent = remembered;
  resizeCanvas();
}

boot();
