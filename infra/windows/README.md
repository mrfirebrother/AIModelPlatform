# Windows 原生部署（一键）

平台不依赖 Docker：PostgreSQL 与 Redis 装成 Windows 服务，四个应用进程由**计划任务**托管。
本目录的脚本把整套东西装出来，并且是**幂等**的——重复执行只会覆盖配置，不会重复安装。

```powershell
# 一键安装（需要管理员权限；会自动重启平台以验证）
powershell -ExecutionPolicy Bypass -File infra\windows\deploy.ps1 install

# 日常
powershell -ExecutionPolicy Bypass -File infra\windows\deploy.ps1 status
powershell -ExecutionPolicy Bypass -File infra\windows\deploy.ps1 restart
powershell -ExecutionPolicy Bypass -File infra\windows\deploy.ps1 logs
powershell -ExecutionPolicy Bypass -File infra\windows\deploy.ps1 uninstall   # 保留数据与日志
```

## 它装了什么

| 组件 | 托管方式 | 说明 |
|---|---|---|
| PostgreSQL / Redis | Windows 服务（原有） | 脚本额外配置 **失败自动重启**：`restart/5s, restart/10s, restart/30s`（一天归零） |
| AIP-API | 计划任务 `AtStartup` + SYSTEM + 最高权限 | `uvicorn backend.app.main:app`，延迟 45s 起（等数据库/Redis） |
| AIP-Worker | 同上 | `celery worker -Q training,evaluation,default --pool=solo` |
| AIP-Beat | 同上 | 定时清扫：评测、租约回收、排队任务 |
| AIP-Frontend | 同上 | `npm run dev`（Vite :3000） |
| AIP-Watchdog | 计划任务，**每 1 分钟** | 巡检进程/端口/`/health/ready`，连续失败达阈值就重启对应任务 |

产物位置：

```
<DataRoot>\deploy.json              安装配置（status/uninstall 读它）
<DataRoot>\run-{api,worker,beat,frontend}.cmd   启动器（可手工执行）
<DataRoot>\run-hidden.vbs           备用隐藏启动器（-LaunchMode HiddenVbs 时用）
<DataRoot>\logs\{api,worker,beat,frontend,watchdog}.log
<DataRoot>\watchdog.state.json      连续失败计数（防抖动）
```

## 为什么 worker 不交给 IIS 应用池

IIS 已经装在这台机器上，但它适合承担的是**前端静态站点 + 反向代理**，不是长驻后台进程：

* 应用池默认**空闲 20 分钟回收**，会直接杀掉正在跑的训练；要当守护用就得关回收、关空闲超时、关定期重启——那等于手搓一个更差的 Windows 服务。
* Celery 不是 HTTP 应用，塞进 `HttpPlatformHandler` 属于滥用。
* 本平台是**单 GPU + Celery solo + GPU 租约**，天生单实例，不存在负载均衡需求。

如果要把前端从 Vite dev server 换成 IIS 静态站点：`cd frontend && npm run build`，IIS 建站点指向 `frontend/dist`，
再装 **URL Rewrite + ARR** 把 `/api` 反代到 `127.0.0.1:8000`（注意 `maxAllowedContentLength` 要放到 2GB 以匹配数据集上传，
ARR `proxyTimeout` 对齐现有 nginx 的 600s）。这一步没做，前端仍由 Vite 托管。

## 自愈实测（2026-09-16，本机）

杀掉 API 与 Beat 进程后观察恢复：

```
+  0s  杀掉 uvicorn 与 celery beat
+ 60s  看门狗第 1 次巡检：进程不存在（连续 1 次）→ 只记日志，不动手
+120s  看门狗第 2 次巡检：连续 2 次 → 触发重启 AIP-API / AIP-Beat
+235s  API 与 Beat 均已恢复，/health = ready
```

**重要结论：任务计划程序的「失败重启」（RestartCount/RestartInterval）在本场景实测没有生效**——
Direct 模式下 `cmd.exe /c` 是任务动作，子进程被杀后任务理应算失败，但实际没有自动重启。
因此**恢复完全依赖看门狗**，这也是把巡检间隔设为 1 分钟的原因（恢复窗口约 2 分钟）。
`RestartCount` 仍保留在任务设置里，作为额外一层（不指望它）。

## 已知取舍与注意事项

* **`-LaunchMode Direct`（默认）**：任务是 `cmd.exe /c <cmd>`，任务计划程序持有整棵进程树，
  「结束任务」会连带杀掉 python/node；进程在会话 0 运行，**没有窗口可被误关**。
  `-LaunchMode HiddenVbs` 是旧方案（wscript + 隐藏窗口），wscript 立刻退出、进程成为孤儿，只在 Direct 出问题时回退用。
* **重启 worker = 中断正在跑的训练**。Celery 配了 `task_acks_late` + `task_reject_on_worker_lost`，
  任务会被重新投递并**从头再跑**。所以看门狗只在确认进程真的不存在时才重启 worker，并写明显日志。
* **别让机器睡眠**：睡眠会把训练停住（`powercfg /change standby-timeout-ac 0`，已设置）。
  Windows Update 的自动重启同理，建议配成不自动重启。
* **端口**：`ApiPort=8000`、`FrontendPort=3000` 可在安装时改；`ApiHost` 默认 `127.0.0.1`（只本机可访问）。
  要让别的机器访问，把 `-ApiHost 0.0.0.0` 并自行加防火墙规则——注意平台只有 `X-API-Key` 一层鉴权。
* **前端首次请求会慢**：Vite 刚启动时第一笔 `/api` 代理请求可能要几秒预热（实测之后为 30~50ms）。
* **`.ps1` 必须带 UTF-8 BOM**：计划任务用 `powershell.exe`(5.1) 调用，不带 BOM 时中文会被按 ANSI 解码而报语法错误（踩过）。
* **看门狗日志**超过 2MB 自动只保留最后 400 行；进程日志（api/worker/beat）目前**不轮转**，长期运行需人工关注大小。
* 旧的 `%USERPROFILE%\aip-data\start-platform.ps1` / `stop-platform.ps1` 已被本目录脚本取代，
  不要再用它们（它们按登录触发，且不配置自愈）。
