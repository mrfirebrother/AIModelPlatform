import { useEffect, useMemo, useState } from "react";
import { createApi } from "../../lib/createApi";
import type { ModelBindingRelease } from "../../lib/api";

export default function ReleasesPage() {
  const api = useMemo(() => createApi(), []);
  const [releases, setReleases] = useState<ModelBindingRelease[]>([]);

  useEffect(() => {
    api.getReleases().then(setReleases);
  }, [api]);

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title">发布与回滚</div>
          <div className="card-kicker">{releases.length} 条发布记录</div>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>发布 ID</th>
              <th>关联 ID</th>
              <th>修订</th>
              <th>模型节点</th>
              <th>类型</th>
              <th>状态</th>
              <th>原因</th>
              <th>创建时间</th>
            </tr>
          </thead>
          <tbody>
            {releases.map((r) => (
              <tr key={r.id}>
                <td style={{ fontFamily: "Courier New, monospace", fontSize: 12 }}>{r.id}</td>
                <td style={{ fontFamily: "Courier New, monospace", fontSize: 12 }}>{r.bindingId}</td>
                <td>第 {r.revisionNo} 次</td>
                <td>{r.modelNodeId}</td>
                <td>
                  <span className={`badge ${r.releaseType === "rollback" ? "badge-red" : "badge-blue"}`}>
                    {r.releaseType === "rollback" ? "回滚" : "正常"}
                  </span>
                </td>
                <td>
                  <span
                    className={`badge ${
                      r.status === "active"
                        ? "badge-green"
                        : r.status === "superseded"
                          ? "badge-gray"
                          : "badge-orange"
                    }`}
                  >
                    {r.status === "active" ? "活跃" : r.status === "superseded" ? "已替换" : r.status}
                  </span>
                </td>
                <td>{r.reason}</td>
                <td style={{ fontSize: 12 }}>{new Date(r.createdAt).toLocaleString("zh-CN")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
