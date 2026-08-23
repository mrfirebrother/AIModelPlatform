import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createApi } from "../../lib/createApi";
import type { ModelBinding, ModelBindingRelease } from "../../lib/api";

export default function BindingsPage() {
  const api = useMemo(() => createApi(), []);
  const navigate = useNavigate();
  const [bindings, setBindings] = useState<ModelBinding[]>([]);
  const [releases, setReleases] = useState<ModelBindingRelease[]>([]);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.getBindings(), api.getReleases()]).then(([b, r]) => {
      setBindings(b);
      setReleases(r);
    }).catch(() => {});
  }, [api]);

  const bindingReleases = selected ? releases.filter((r) => r.bindingId === selected) : [];

  return (
    <>
      <div className="page-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">{"\u6a21\u578b\u5173\u8054\u5217\u8868"}</div>
              <div className="card-kicker">{bindings.length} {"\u4e2a\u5173\u8054"}</div>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{"\u5173\u8054 ID"}</th>
                  <th>{"\u540d\u79f0"}</th>
                  <th>{"\u72b6\u6001"}</th>
                  <th>{"\u64cd\u4f5c"}</th>
                </tr>
              </thead>
              <tbody>
                {bindings.map((b) => (
                  <tr key={b.id} onClick={() => setSelected(b.id)} style={{ cursor: "pointer", background: selected === b.id ? "#e7f1fa" : undefined }}>
                    <td style={{ fontFamily: "Courier New, monospace", fontSize: 10 }}>{b.id}</td>
                    <td>{b.name}</td>
                    <td>
                      <span className={`badge ${b.status === "active" ? "badge-green" : b.status === "unbound" ? "badge-gray" : "badge-orange"}`}>
                        {b.status === "active" ? "\u6d3b\u8dc3" : b.status === "unbound" ? "\u672a\u7ed1\u5b9a" : b.status}
                      </span>
                    </td>
                    <td><button className="btn small" onClick={(e) => { e.stopPropagation(); navigate("/releases"); }}>{"\u53d1\u5e03\u5386\u53f2"}</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">{"\u53d1\u5e03\u8bb0\u5f55"}</div>
              <div className="card-kicker">{selected ? `${bindingReleases.length} {"\u6761\u8bb0\u5f55"}` : "\u9009\u62e9\u5173\u8054\u67e5\u770b\u8be6\u60c5"}</div>
            </div>
          </div>
          <div className="card-body">
            {!selected ? (
              <div className="empty-state">{"\u70b9\u51fb\u5de6\u4fa7\u5173\u8054\u67e5\u770b\u53d1\u5e03\u5386\u53f2"}</div>
            ) : bindingReleases.length === 0 ? (
              <div className="empty-state">{"\u6682\u65e0\u53d1\u5e03\u8bb0\u5f55"}</div>
            ) : (
              <div className="detail-list">
                {bindingReleases.map((r) => (
                  <div key={r.id} style={{ padding: "10px 12px", border: "1px solid #d3e2ec", borderRadius: 3, marginBottom: 8 }}>
                    <div className="detail-row"><span>{"\u4fee\u8ba2"}</span><b>{`{"\u7b2c"}${r.revisionNo}{"\u6b21"}`}</b></div>
                    <div className="detail-row"><span>{"\u6a21\u578b\u8282\u70b9"}</span><b>{r.modelNodeId}</b></div>
                    <div className="detail-row"><span>{"\u7c7b\u578b"}</span><b><span className={`badge ${r.releaseType === "rollback" ? "badge-red" : "badge-blue"}`}>{r.releaseType === "rollback" ? "\u56de\u6eda" : "\u6b63\u5e38"}</span></b></div>
                    <div className="detail-row"><span>{"\u72b6\u6001"}</span><b><span className={`badge ${r.status === "active" ? "badge-green" : "badge-gray"}`}>{r.status === "active" ? "\u6d3b\u8dc3" : r.status === "superseded" ? "\u5df2\u66ff\u6362" : r.status}</span></b></div>
                    <div className="detail-row"><span>{"\u539f\u56e0"}</span><b>{r.reason}</b></div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
