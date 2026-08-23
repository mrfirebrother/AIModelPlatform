import { Routes, Route, NavLink, Navigate, useLocation } from "react-router-dom";
import DashboardPage from "../features/dashboard/DashboardPage";
import ModelsPage from "../features/models/ModelsPage";
import ModelDetailPage from "../features/models/ModelDetailPage";
import BindingsPage from "../features/bindings/BindingsPage";
import DatasetsPage from "../features/datasets/DatasetsPage";
import TrainingPage from "../features/training/TrainingPage";
import TrainingCreatePage from "../features/training/TrainingCreatePage";
import EvaluationsPage from "../features/evaluations/EvaluationsPage";
import ReleasesPage from "../features/releases/ReleasesPage";
import ResourcesPage from "../features/resources/ResourcesPage";

const navItems = [
  { to: "/dashboard", label: "总览", icon: "▦" },
  { to: "/models", label: "模型谱系", icon: "◈" },
  { to: "/bindings", label: "模型关联", icon: "◇" },
  { to: "/datasets", label: "数据集", icon: "◉" },
  { to: "/training", label: "训练任务", icon: "▶" },
  { to: "/evaluations", label: "模型评估", icon: "◆" },
];

const navItems2 = [
  { to: "/releases", label: "发布管理", icon: "↗" },
  { to: "/resources", label: "资源控制", icon: "≡" },
];

const pageTitles: Record<string, string> = {
  "/dashboard": "模型运营总览",
  "/models": "模型谱系",
  "/bindings": "模型关联",
  "/datasets": "数据集",
  "/training": "训练任务",
  "/evaluations": "模型评估",
  "/releases": "发布管理",
  "/resources": "资源控制",
};

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <i /> AI Model Platform
        </div>
        <h1>AI 模型<br />服务平台</h1>
        <p>模型运维台</p>
      </div>

      <div className="nav-label">工作台</div>
      <nav className="nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <span>{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="nav-label" style={{ marginTop: 20 }}>控制中心</div>
      <nav className="nav">
        {navItems2.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <span>{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-foot">
        <div className="server-state">
          <span>GPU 服务器 / 01</span>
          <span className="online">在线</span>
        </div>
        <div className="server-id">nvidia-a100 · cuda 12.4 · 2026.08.20</div>
      </div>
    </aside>
  );
}

export function AppRoutes() {
  const location = useLocation();
  const title = pageTitles[location.pathname] || "管理后台";

  return (
    <>
      <Sidebar />
      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">平台状态 / 内部管理台</div>
            <h2>{title}</h2>
          </div>
        </header>

        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/models/:id" element={<ModelDetailPage />} />
          <Route path="/bindings" element={<BindingsPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/training/new" element={<TrainingCreatePage />} />
          <Route path="/evaluations" element={<EvaluationsPage />} />
          <Route path="/releases" element={<ReleasesPage />} />
          <Route path="/resources" element={<ResourcesPage />} />
        </Routes>
      </main>
    </>
  );
}
