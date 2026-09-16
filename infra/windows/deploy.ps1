<#
.SYNOPSIS
    AI 模型平台 —— Windows 原生部署（一键安装 / 启停 / 状态 / 卸载）

.DESCRIPTION
    把 API、Celery Worker、Celery Beat、前端 四个进程注册为「开机自启的计划任务」，
    以 SYSTEM 身份在会话 0 运行（无窗口、可注销、早于登录），并额外配置：

      * 失败自动重启（RestartCount / RestartInterval，仅在 Direct 模式下由任务计划程序真正生效）
      * PostgreSQL / Redis 服务恢复策略（崩溃自动拉起）
      * 每 3 分钟一次的看门狗（AIP-Watchdog），巡检进程与 /health/ready 并自愈

    为什么 worker 不交给 IIS 应用池：应用池默认空闲 20 分钟回收，会杀掉正在跑的训练；
    Celery 是长驻后台进程而不是 HTTP 应用。IIS 的合理职责是「前端静态站点 + 反向代理」，
    见 README.md 的说明。

.PARAMETER Action
    install   生成启动器、注册计划任务、配置服务恢复与看门狗（默认）
    uninstall 停止并删除计划任务与看门狗（保留数据、日志与启动器）
    start     启动全部任务
    stop      停止全部任务及其子进程
    restart   先停后启
    status    查看任务状态、进程、端口、/health、日志大小
    logs      打印各日志文件尾部
    watchdog  立刻执行一次巡检

.EXAMPLE
    # 一键安装（仓库根目录已 clone、.venv 已建好）
    powershell -ExecutionPolicy Bypass -File infra\windows\deploy.ps1 install

.EXAMPLE
    # 换机器：显式指定仓库、数据目录与解释器
    .\deploy.ps1 install -RepoRoot D:\Work\AIModelPlatform -DataRoot C:\aip-data `
        -PythonExe D:\Work\AIModelPlatform\.venv\Scripts\python.exe -ApiPort 8000
#>
[CmdletBinding()]
param(
    [ValidateSet('install', 'uninstall', 'start', 'stop', 'restart', 'status', 'logs', 'watchdog')]
    [string]$Action = 'install',

    [string]$RepoRoot,
    [string]$DataRoot,
    [string]$PythonExe,

    [string]$ApiHost = '127.0.0.1',
    [int]$ApiPort = 8000,
    [int]$FrontendPort = 3000,

    # Direct   : 任务直接执行 .cmd，任务计划程序持有进程树 —— 崩溃重启与「结束任务」都可靠
    # HiddenVbs: 旧的 wscript+隐藏窗口方式（进程成为孤儿，重启只能靠看门狗）
    [ValidateSet('Direct', 'HiddenVbs')]
    [string]$LaunchMode = 'Direct',

    [int]$StartupDelaySeconds = 45,
    [string[]]$DependencyServices = @('postgresql-x64-16', 'Redis'),

    [switch]$SkipServiceRecovery,
    [switch]$SkipWatchdog,

    # 安装完默认会重启平台以验证新任务定义；加 -NoRestart 可跳过（例如你正想保住现有进程）
    [switch]$NoRestart
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --------------------------------------------------------------------------- 输出
function Write-Head([string]$text) { Write-Host ''; Write-Host "== $text" -ForegroundColor Cyan }
function Write-Ok([string]$text) { Write-Host "   [ok]   $text" -ForegroundColor Green }
function Write-Info([string]$text) { Write-Host "   [--]   $text" -ForegroundColor Gray }
function Write-Warn2([string]$text) { Write-Host "   [警告] $text" -ForegroundColor Yellow }
function Write-Err([string]$text) { Write-Host "   [错误] $text" -ForegroundColor Red }

# --------------------------------------------------------------------------- 默认值
function Resolve-Defaults {
    if (-not $RepoRoot) {
        # infra\windows\deploy.ps1 -> 仓库根
        $script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    } else {
        $script:RepoRoot = (Resolve-Path $RepoRoot).Path
    }
    if (-not $DataRoot) { $script:DataRoot = Join-Path $env:USERPROFILE 'aip-data' }
    if (-not $PythonExe) { $script:PythonExe = Join-Path $script:RepoRoot '.venv\Scripts\python.exe' }

    $script:LogDir = Join-Path $script:DataRoot 'logs'
    $script:SettingsFile = Join-Path $script:DataRoot 'deploy.json'
    $script:WatchdogScript = Join-Path $PSScriptRoot 'watchdog.ps1'
    $script:Tasks = @('AIP-API', 'AIP-Worker', 'AIP-Beat', 'AIP-Frontend')
}

function Get-PlatformSettings {
    Resolve-Defaults
    return [ordered]@{
        repoRoot       = $script:RepoRoot
        dataRoot       = $script:DataRoot
        pythonExe      = $script:PythonExe
        apiHost        = $ApiHost
        apiPort        = $ApiPort
        frontendPort   = $FrontendPort
        launchMode     = $LaunchMode
        startupDelay   = $StartupDelaySeconds
        dependencySvcs = $DependencyServices
        installedAt    = (Get-Date).ToString('s')
    }
}

function Save-Settings($settings) {
    $settings | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $script:SettingsFile -Encoding UTF8
}

function Load-Settings {
    Resolve-Defaults
    if (Test-Path -LiteralPath $script:SettingsFile) {
        $raw = Get-Content -LiteralPath $script:SettingsFile -Raw | ConvertFrom-Json
        $script:RepoRoot = $raw.repoRoot
        $script:DataRoot = $raw.dataRoot
        $script:PythonExe = $raw.pythonExe
        $script:LogDir = Join-Path $script:DataRoot 'logs'
        $script:ApiPort = [int]$raw.apiPort
        $script:ApiHost = [string]$raw.apiHost
        $script:FrontendPort = [int]$raw.frontendPort
        $script:LaunchMode = [string]$raw.launchMode
        return $raw
    }
    # 还没安装过：用参数/默认值兜底，这样 status/logs 之类的只读动作也能跑
    return Get-PlatformSettings
}

# --------------------------------------------------------------------------- 预检
function Test-Preflight($settings) {
    Write-Head '环境检查'
    $ok = $true

    if (Test-Path -LiteralPath (Join-Path $settings.repoRoot 'backend\app\main.py')) {
        Write-Ok "仓库根: $($settings.repoRoot)"
    } else {
        Write-Err "仓库根下找不到 backend\app\main.py: $($settings.repoRoot)"; $ok = $false
    }

    if (Test-Path -LiteralPath $settings.pythonExe) {
        Write-Ok "Python: $($settings.pythonExe)"
    } else {
        Write-Err "解释器不存在（先建 .venv 并装依赖）: $($settings.pythonExe)"; $ok = $false
    }

    if (Test-Path -LiteralPath (Join-Path $settings.repoRoot '.env')) {
        Write-Ok '.env 存在于仓库根（Settings 按工作目录读取它）'
    } else {
        Write-Warn2 '.env 不存在 —— 平台会退回默认配置（数据库/API Key/数据目录都会不对）'
    }

    if (Get-Command node -ErrorAction SilentlyContinue) { Write-Ok "node: $((Get-Command node).Source)" }
    else { Write-Warn2 'node 不在 PATH —— 前端任务会启动失败（只用 API 可忽略）' }

    foreach ($svc in $settings.dependencySvcs) {
        $s = Get-Service -Name $svc -ErrorAction SilentlyContinue
        if ($s) {
            if ($s.Status -eq 'Running') { Write-Ok "依赖服务 $svc 正在运行" }
            else { Write-Warn2 "依赖服务 $svc 状态为 $($s.Status)（不是 Running）" }
        } else {
            Write-Warn2 "找不到依赖服务 $svc（若数据库/Redis 是别的服务名，用 -DependencyServices 指定）"
        }
    }

    foreach ($p in @($settings.apiPort, $settings.frontendPort)) {
        $busy = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue
        if ($busy) { Write-Info "端口 $p 已被 PID $($busy[0].OwningProcess) 占用（若那是本平台的旧进程，install 后会接管）" }
    }

    if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Err '需要管理员权限（注册 SYSTEM 计划任务与设置服务恢复）'; $ok = $false
    } else {
        Write-Ok '管理员权限'
    }
    return $ok
}

# --------------------------------------------------------------------------- 启动器
function New-Launchers($settings) {
    Write-Head '生成启动器'
    New-Item -ItemType Directory -Force -Path $settings.dataRoot, $script:LogDir | Out-Null

    $vbs = Join-Path $settings.dataRoot 'run-hidden.vbs'
    @'
' Launch a command line with a hidden window, so there is no console for anyone to close.
' Usage: wscript.exe run-hidden.vbs <command-line>
Set shell = CreateObject("WScript.Shell")
If WScript.Arguments.Count = 0 Then WScript.Quit 1
shell.Run WScript.Arguments(0), 0, False
'@ | Set-Content -LiteralPath $vbs -Encoding ASCII

    $py = $settings.pythonExe
    $logs = $script:LogDir
    $repo = $settings.repoRoot

    $cmds = [ordered]@{
        'run-api.cmd' = @"
@echo off
cd /d $repo
set PYTHONUNBUFFERED=1
"$py" -m uvicorn backend.app.main:app --host $($settings.apiHost) --port $($settings.apiPort) >> "$logs\api.log" 2>&1
"@
        'run-worker.cmd' = @"
@echo off
cd /d $repo
set PYTHONUNBUFFERED=1
"$py" -m celery -A backend.app.workers.celery_app worker -Q training,evaluation,default --concurrency=1 --pool=solo --loglevel=info >> "$logs\worker.log" 2>&1
"@
        'run-beat.cmd' = @"
@echo off
cd /d $repo
set PYTHONUNBUFFERED=1
"$py" -m celery -A backend.app.workers.celery_app beat --loglevel=info --schedule "$($settings.dataRoot)\celerybeat-schedule" >> "$logs\beat.log" 2>&1
"@
        'run-frontend.cmd' = @"
@echo off
cd /d $repo\frontend
call npm run dev >> "$logs\frontend.log" 2>&1
"@
    }

    foreach ($name in $cmds.Keys) {
        $path = Join-Path $settings.dataRoot $name
        # cmd 需要 CRLF，否则某些行会被吞掉
        ($cmds[$name] -replace "`r?`n", "`r`n") | Set-Content -LiteralPath $path -Encoding ASCII -NoNewline
        Write-Ok "$name"
    }
    Write-Ok 'run-hidden.vbs'
}

# --------------------------------------------------------------------------- 计划任务
function New-TaskActionFor([string]$taskName, $settings) {
    $cmd = switch ($taskName) {
        'AIP-API' { 'run-api.cmd' }
        'AIP-Worker' { 'run-worker.cmd' }
        'AIP-Beat' { 'run-beat.cmd' }
        'AIP-Frontend' { 'run-frontend.cmd' }
    }
    $cmdPath = Join-Path $settings.dataRoot $cmd
    if ($settings.launchMode -eq 'HiddenVbs') {
        $vbs = Join-Path $settings.dataRoot 'run-hidden.vbs'
        return New-ScheduledTaskAction -Execute 'wscript.exe' -Argument "`"$vbs`" `"$cmdPath`""
    }
    return New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$cmdPath`""
}

function Register-PlatformTasks($settings) {
    Write-Head "注册计划任务（开机自启 + SYSTEM + 启动延迟 $($settings.startupDelay)s + 失败重启）"

    $settingsObj = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -MultipleInstances IgnoreNew

    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

    foreach ($name in $script:Tasks) {
        $trigger = New-ScheduledTaskTrigger -AtStartup
        $trigger.Delay = "PT$($settings.startupDelay)S"
        $action = New-TaskActionFor $name $settings

        Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
            -Settings $settingsObj -Principal $principal -Force | Out-Null
        Write-Ok "$name  （$($action.Execute) $($action.Arguments)）"
    }

    Write-Info '注意：Direct 模式下任务计划程序持有进程树，「结束任务」会连带杀掉子进程；'
    Write-Info '      HiddenVbs 模式下 wscript 会立刻退出，崩溃重启只能靠看门狗。'
}

function Register-WatchdogTask($settings) {
    if ($SkipWatchdog) { Write-Warn2 '按参数要求跳过看门狗注册'; return }
    Write-Head '注册看门狗（每 3 分钟巡检一次，自愈）'

    $psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $arg = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$($script:WatchdogScript)`" -DataRoot `"$($settings.dataRoot)`""
    $action = New-ScheduledTaskAction -Execute $psExe -Argument $arg

    # 只给 -RepetitionInterval、不给 -RepetitionDuration：任务计划程序即视为「无限期重复」。
    # 传 [TimeSpan]::MaxValue 会被序列化成 P99999999DT23H59M59S，注册时报 out of range（踩过）。
    # 间隔取 1 分钟：实测「任务计划程序的失败重启」在本场景并不生效（见 README 的自愈实测），
    # 恢复完全靠看门狗，所以跑密一点，把恢复窗口从 ~4 分钟压到 ~2 分钟。
    $t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 1)
    $t2 = New-ScheduledTaskTrigger -AtStartup
    $t2.Delay = 'PT2M'

    $settingsObj = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

    Register-ScheduledTask -TaskName 'AIP-Watchdog' -Action $action -Trigger @($t1, $t2) `
        -Settings $settingsObj -Principal $principal -Force | Out-Null
    Write-Ok 'AIP-Watchdog（每 1 分钟巡检；开机 2 分钟后开始）'
}

# --------------------------------------------------------------------------- 服务恢复
function Set-DependencyRecovery($settings) {
    if ($SkipServiceRecovery) { Write-Warn2 '按参数要求跳过服务恢复配置'; return }
    Write-Head '配置依赖服务的崩溃自动重启'
    foreach ($svc in $settings.dependencySvcs) {
        if (-not (Get-Service -Name $svc -ErrorAction SilentlyContinue)) {
            Write-Warn2 "$svc 不存在，跳过"
            continue
        }
        # 第一次失败 5s 后重启，第二次 10s，其后 30s；计数器一天归零
        & sc.exe failure $svc reset= 86400 actions= restart/5000/restart/10000/restart/30000 | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Ok "$svc 已配置 restart/5s,10s,30s" }
        else { Write-Warn2 "$svc 配置失败（exit $LASTEXITCODE）" }
    }
}

# --------------------------------------------------------------------------- 启停
function Get-PlatformProcesses($settings) {
    $patterns = 'uvicorn|celery|vite'
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='node.exe' OR Name='cmd.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match $patterns) -and ($_.CommandLine -notmatch 'deploy\.ps1|watchdog\.ps1') }
}

function Stop-Platform($settings) {
    Write-Head '停止平台'
    foreach ($name in $script:Tasks) {
        $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($t -and $t.State -eq 'Running') {
            Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
            Write-Ok "$name 已停止"
        }
    }
    Start-Sleep -Seconds 2
    $strays = Get-PlatformProcesses $settings
    foreach ($p in $strays) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    if ($strays.Count -gt 0) { Write-Ok "清理残留进程 $($strays.Count) 个" } else { Write-Info '无残留进程' }
}

function Start-Platform($settings) {
    Write-Head '启动平台'
    foreach ($name in @('AIP-API', 'AIP-Worker', 'AIP-Beat', 'AIP-Frontend')) {
        Start-ScheduledTask -TaskName $name
        Write-Ok "$name 已触发"
    }
    Write-Info '进程以 SYSTEM 身份在会话 0 运行，无窗口；等待健康检查…'
    $deadline = (Get-Date).AddSeconds(90)
    do {
        Start-Sleep -Seconds 3
        $ready = $null
        try { $ready = Invoke-RestMethod "http://$($settings.apiHost):$($settings.apiPort)/health" -TimeoutSec 5 } catch { }
    } while (-not $ready -and (Get-Date) -lt $deadline)
    if ($ready) { Write-Ok "/health = $($ready.status)  $($ready.checks | ConvertTo-Json -Compress)" }
    else { Write-Warn2 '90 秒内 /health 未响应，用 status 看看日志' }
}

function Show-Status($settings) {
    Write-Head '计划任务'
    foreach ($name in ($script:Tasks + 'AIP-Watchdog')) {
        $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if (-not $t) { Write-Warn2 "$name 未注册"; continue }
        $info = Get-ScheduledTaskInfo -TaskName $name -ErrorAction SilentlyContinue
        $trig = ($t.Triggers | ForEach-Object { $_.CimClass.CimClassName -replace 'MSFT_Task', '' }) -join '+'
        Write-Host ("   {0,-14} {1,-8} {2,-14} 上次结果={3}" -f $name, $t.State, $trig, $(if ($info) { $info.LastTaskResult } else { 'n/a' }))
    }

    Write-Head '进程'
    $procs = Get-PlatformProcesses $settings
    if ($procs) {
        foreach ($p in $procs) {
            $desc = switch -Regex ($p.CommandLine) {
                # 注意：beat 的命令行里也含 "backend.app.workers"，粗匹配 'celery.*worker' 会把它算成 Worker
                'uvicorn' { 'API'; break }
                'celery_app\s+worker' { 'Worker'; break }
                'celery_app\s+beat' { 'Beat'; break }
                'vite' { 'Frontend'; break }
                default { 'Other' }
            }
            Write-Host ("   {0,-9} pid={1,-7} 启动于 {2}" -f $desc, $p.ProcessId, $p.CreationDate)
        }
    } else { Write-Warn2 '没有任何平台进程在跑' }

    Write-Head '端口与健康'
    foreach ($p in @($settings.apiPort, $settings.frontendPort)) {
        $c = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue
        Write-Host ("   {0,-6} {1}" -f $p, $(if ($c) { "监听中 (pid $($c[0].OwningProcess))" } else { '未监听' }))
    }
    try {
        $h = Invoke-RestMethod "http://$($settings.apiHost):$($settings.apiPort)/health" -TimeoutSec 5
        Write-Host "   /health  $($h.status)  $($h.checks | ConvertTo-Json -Compress)"
    } catch { Write-Warn2 "/health 无响应: $($_.Exception.Message)" }

    Write-Head '依赖服务'
    foreach ($svc in $settings.dependencySvcs) {
        $s = Get-Service -Name $svc -ErrorAction SilentlyContinue
        if ($s) { Write-Host ("   {0,-22} {1}" -f $svc, $s.Status) }
    }

    Write-Head '日志大小'
    if (Test-Path $script:LogDir) {
        Get-ChildItem $script:LogDir -Filter *.log | Sort-Object Length -Descending |
            ForEach-Object { Write-Host ("   {0,-16} {1,8:N1} MB   最后写入 {2}" -f $_.Name, ($_.Length / 1MB), $_.LastWriteTime) }
    }
}

function Show-Logs($settings) {
    foreach ($name in @('api', 'worker', 'beat', 'frontend', 'watchdog')) {
        $path = Join-Path $script:LogDir "$name.log"
        Write-Head "$name.log"
        if (Test-Path $path) { Get-Content $path -Tail 15 } else { Write-Info '暂无' }
    }
}

function Uninstall-Platform($settings) {
    Stop-Platform $settings
    Write-Head '删除计划任务'
    foreach ($name in ($script:Tasks + 'AIP-Watchdog')) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Ok "$name 已删除"
        }
    }
    if (-not $SkipServiceRecovery) {
        Write-Head '还原依赖服务的恢复策略'
        foreach ($svc in $settings.dependencySvcs) {
            if (Get-Service -Name $svc -ErrorAction SilentlyContinue) {
                & sc.exe failure $svc reset= 0 actions= "" | Out-Null
                Write-Ok "$svc 已还原为不自动重启"
            }
        }
    }
    Write-Info "数据、日志与启动器保留在 $($settings.dataRoot)（要清干净就手工删这个目录）"
}

# --------------------------------------------------------------------------- 主流程
Resolve-Defaults

switch ($Action) {
    'install' {
        $settings = Get-PlatformSettings
        if (-not (Test-Preflight $settings)) { Write-Err '预检未通过，已中止'; exit 1 }
        New-Launchers $settings
        Register-PlatformTasks $settings
        Set-DependencyRecovery $settings
        Register-WatchdogTask $settings
        Save-Settings $settings
        Write-Head '安装完成'
        Write-Info "配置已写入 $($script:SettingsFile)（status/uninstall 会读它）"
        if (-not $NoRestart) {
            Write-Info '正在重启平台以验证新任务定义…'
            Stop-Platform $settings
            Start-Platform $settings
        }
        Show-Status $settings
    }
    'uninstall' { Uninstall-Platform (Load-Settings) }
    'start' { Start-Platform (Load-Settings) }
    'stop' { Stop-Platform (Load-Settings) }
    'restart' {
        $s = Load-Settings
        Stop-Platform $s
        Start-Platform $s
    }
    'status' { Show-Status (Load-Settings) }
    'logs' { Show-Logs (Load-Settings) }
    'watchdog' {
        & $script:WatchdogScript -DataRoot $DataRoot
    }
}
