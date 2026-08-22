import { useEffect, useState } from "react";
import { createApi } from "../../lib/createApi";
import type { Dataset } from "../../lib/api";

export default function DatasetsPage() {
  const api = createApi();
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  useEffect(() => {
    api.getDatasets().then(setDatasets);
  }, []);

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title">数据集管理</div>
          <div className="card-kicker">{datasets.length} 个数据集</div>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>数据集名称</th>
              <th>标签体系</th>
              <th>图片总数</th>
              <th>训练集</th>
              <th>验证集</th>
              <th>测试集</th>
              <th>来源</th>
              <th>校验</th>
              <th>快照</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.id}>
                <td>
                  <div style={{ fontWeight: 600 }}>{d.name}</div>
                  <div
                    style={{
                      color: "var(--text-muted)",
                      fontSize: 9,
                      fontFamily: "Courier New, monospace",
                    }}
                  >
                    {d.id}
                  </div>
                </td>
                <td>{d.labelSchemaName}</td>
                <td style={{ fontWeight: 600 }}>{d.imageCount.toLocaleString()}</td>
                <td>{d.trainCount.toLocaleString()}</td>
                <td>{d.valCount.toLocaleString()}</td>
                <td>{d.testCount.toLocaleString()}</td>
                <td>{d.source}</td>
                <td>
                  <span
                    className={`badge ${d.validationStatus === "通过" ? "badge-green" : "badge-red"}`}
                  >
                    {d.validationStatus}
                  </span>
                </td>
                <td
                  style={{
                    fontFamily: "Courier New, monospace",
                    fontSize: 10,
                  }}
                >
                  {d.latestSnapshotId}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {datasets.some((d) => d.warnings.length > 0) && (
        <div
          className="card-body"
          style={{ borderTop: "1px solid #dce8f0" }}
        >
          <div style={{ color: "var(--accent-orange)", fontSize: 11, fontWeight: 600, marginBottom: 6 }}>
            数据质量警告
          </div>
          {datasets
            .filter((d) => d.warnings.length > 0)
            .map((d) =>
              d.warnings.map((w, i) => (
                <div
                  key={`${d.id}-${i}`}
                  style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 2 }}
                >
                  {d.name}: {w}
                </div>
              )),
            )}
        </div>
      )}
    </div>
  );
}
