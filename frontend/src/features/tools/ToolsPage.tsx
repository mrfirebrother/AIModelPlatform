/** 标注工具入口：宫格展示，每个工具一张卡片。 */
const TOOLS = [
  {
    key: "wheel-labeler",
    name: "车轮标注工具",
    desc: "打开图片文件夹，自动按圆提议车轮框，人工修正后导出 YOLO 数据集",
    icon: "◎",
    href: "/wheel-labeler/index.html",
    enabled: true,
  },
];

export default function ToolsPage() {
  return (
    <div className="tool-grid">
      {TOOLS.map((tool) => (
        <div
          key={tool.key}
          className="grid-card tool-card"
          style={{ cursor: tool.enabled ? "pointer" : "not-allowed", opacity: tool.enabled ? 1 : 0.55 }}
          onClick={() => { if (tool.enabled) window.open(tool.href, "_blank"); }}
        >
          <div className="tool-card-icon" style={{ background: "rgba(54,161,189,0.1)", color: "#36a1bd" }}>
            {tool.icon}
          </div>
          <div className="tool-card-name">{tool.name}</div>
          <div className="tool-card-desc">{tool.desc}</div>
          <div className="tool-card-action">{tool.enabled ? "进入工具 →" : "即将上线"}</div>
        </div>
      ))}
      <div
        className="grid-card tool-card"
        style={{ cursor: "not-allowed", opacity: 0.55 }}
      >
        <div className="tool-card-icon" style={{ background: "rgba(139,92,246,0.08)", color: "#8b5cf6" }}>
          {"＋"}
        </div>
        <div className="tool-card-name">更多工具</div>
        <div className="tool-card-desc">后续标注工具将在这里以宫格形式加入</div>
        <div className="tool-card-action">敬请期待</div>
      </div>
    </div>
  );
}
