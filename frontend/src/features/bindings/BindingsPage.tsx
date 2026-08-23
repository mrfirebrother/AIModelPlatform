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
              <div className="card-title">模型关联列表</div>
              <div className="card-kicker">{bindings.length} 个关联</div>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>关联 ID</th>
                  <th>名称</th>
                  <th>状态</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {bindings.map((b) => (
                  <tr key={b.id} onClick={() => setSelected(b.id)} style={{ cursor: "pointer", background: selected === b.id ? "#e7f1fa" : undefined }}>
                    <td style={{ fontFamily: "Courier New, monospace", fontSize: 10 }}>{b.id}</td>
                    <td>{b.name}</td>
                    <td>
                      <span className={`badge ${b.status === "active" ? "badge-green" : b.status === "unbound" ? "badge-gray" : "badge-orange"}`}>
                        {b.status === "active" ? "活跃" : b.status === "unbound" ? "未绑定" : b.status}
                      </span>
                    </td>
                    <td>
                      <button className="btn small" onClick={(e) => { e.stopPropagation(); navigate("/releases"); }}>
                        发布历史
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">发布记录</div>
              <div className="card-kicker">
                {selected ? `${bindingReleases.length} 条记录` : "选择关联查看详情"}
              </div>
            </div>
          </div>
          <div className="card-body">
            {!selected ? (
              <div className="empty-state">点击左侧关联查看发布历史</div>
            ) : bindingReleases.length === 0 ? (
              <div className="empty-state">暂无发布记录</div>
            ) : (
              <div className="detail-list">
                {bindingReleases.map((r) => (
                  <div key={r.id} style={{ padding: "10px 12px", border: "1px solid #d3e2ec", borderRadius: 3, marginBottom: 8 }}>
                    <div className="detail-row"><span>修订</span><b>第 {r.revisionNo} 次</b></div>
                    <div className="detail-row"><span>模型节点</span><b>{r.modelNodeId}</b></div>
                    <div className="detail-row"><span>类型</span><b><span className={`badge ${r.releaseType === "rollback" ? "badge-red" : "badge-blue"}`}>{r.releaseType === "rollback" ? "回滚" : "正常"}</span></b></div>
                    <div className="detail-row"><span>状态</span><b><span className={`badge ${r.status === "active" ? "badge-green" : "badge-gray"}`}>{r.status === "active" ? "活跃" : r.status === "superseded" ? "已替换" : r.status}</span></b></div>
                    <div className="detail-row"><span>原因</span><b>{r.reason}</b></div>
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
