import { BrowserRouter } from "react-router-dom";
import { AppRoutes } from "./routes";
import { ToastProvider } from "../lib/toast";
import LoginGate from "./LoginGate";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <LoginGate>
          <div className="app-shell">
            <AppRoutes />
          </div>
        </LoginGate>
      </ToastProvider>
    </BrowserRouter>
  );
}
