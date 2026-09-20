import { useEffect, useState } from "react";
import { createApi } from "../lib/createApi";

/** 单一密码登录门：密码存 sessionStorage，通过后每次请求带 X-UI-Password 头。 */
export default function LoginGate({ children }: { children: React.ReactNode }) {
  const [authed, setAuthed] = useState(() => !!sessionStorage.getItem("ui_password"));
  const [pw, setPw] = useState("");
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    const onRequired = () => setAuthed(false);
    window.addEventListener("ui-auth-required", onRequired);
    return () => window.removeEventListener("ui-auth-required", onRequired);
  }, []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (checking) return;
    setChecking(true);
    setError("");
    sessionStorage.setItem("ui_password", pw);
    try {
      await createApi().getModelNodes();
      setAuthed(true);
    } catch {
      sessionStorage.removeItem("ui_password");
      setError("密码错误");
    } finally {
      setChecking(false);
    }
  };

  if (!authed) {
    return (
      <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#f0f5f9" }}>
        <form onSubmit={submit} className="card" style={{ width: 340, padding: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <div className="logo-mark">AI</div>
            <div className="card-title">AI 模型平台</div>
          </div>
          <div className="card-kicker" style={{ marginBottom: 16 }}>请输入访问密码</div>
          <input
            type="password"
            value={pw}
            onChange={(e) => setPw(e.target.value)}
            placeholder="访问密码"
            autoFocus
            style={{ width: "100%", padding: "9px 12px", fontSize: 14, borderRadius: 4, border: "1px solid #c8d8e4", marginBottom: 12, boxSizing: "border-box" }}
          />
          {error && <div style={{ color: "#b33", fontSize: 12, marginBottom: 8 }}>{error}</div>}
          <button type="submit" className="btn primary" disabled={checking || !pw} style={{ width: "100%" }}>
            {checking ? "验证中..." : "进入平台"}
          </button>
        </form>
      </div>
    );
  }
  return <>{children}</>;
}
