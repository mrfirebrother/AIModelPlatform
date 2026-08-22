import { BrowserRouter } from "react-router-dom";
import { AppRoutes } from "./routes";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <AppRoutes />
      </div>
    </BrowserRouter>
  );
}
