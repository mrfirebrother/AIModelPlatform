import { useEffect, useMemo, useState } from "react";
import { createApi } from "./createApi";

/** 轮询 logs 接口取运行中任务的实时轮数（后端从 results.csv 行数推算），15 秒一次。 */
export function useLiveEpoch(taskId: string): number {
  const api = useMemo(() => createApi(), []);
  const [cur, setCur] = useState(0);
  useEffect(() => {
    let alive = true;
    const tick = () => {
      api.getTrainingLogs(taskId).then((l) => { if (alive) setCur(l.currentEpoch ?? 0); }).catch(() => {});
    };
    tick();
    const timer = setInterval(tick, 15000);
    return () => { alive = false; clearInterval(timer); };
  }, [api, taskId]);
  return cur;
}

/** 运行中任务轮数文案：有实时数据时显示「第 X/N 轮」，否则显示「N 轮」。 */
export function LiveEpoch({ taskId, totalEpochs }: { taskId: string; totalEpochs: number }) {
  const cur = useLiveEpoch(taskId);
  if (cur > 0) return <span>第 {cur}/{totalEpochs} 轮</span>;
  return <span>{totalEpochs} 轮</span>;
}
