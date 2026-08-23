import { BrowserRouter } from "react-router-dom";
import { AppRoutes } from "./routes";
import { ToastProvider } from "../lib/toast";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <div className="app-shell">
          <AppRoutes />
        </div>
      </ToastProvider>
    </BrowserRouter>
  );
}
