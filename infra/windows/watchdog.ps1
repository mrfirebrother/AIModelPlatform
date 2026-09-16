<#
.SYNOPSIS
    AI 模型平台看门狗 —— 巡检 API / Worker / Beat / 前端，必要时重启对应计划任务

.DESCRIPTION
    由计划任务 AIP-Watchdog 每 3 分钟调用一次（也可手动跑）。
    判定依据是「进程是否还在」与「/health/ready 是否正常」，并带连续失败计数以避免抖动：
    单次巡检失败只记日志，连续失败达到阈值才动手重启。

    为什么需要它：Direct 模式下任务计划程序能对「进程退出」做失败重启，
    但覆盖不了「进程活着但卡住」（例如数据库不可达时 API 请求挂住）以及被外部杀掉的场景。

.NOTES
    Worker 被重启会中断正在跑的训练（Celery task_acks_late + reject_on_worker_lost
    会把任务重新投递，从头再跑），所以只在确认进程真的不存在时才重启它，并写明显日志。
#>
[CmdletBinding()]
param(
    [string]$DataRoot = (Join-Path $env:USERPROFILE 'aip-data'),
    [int]$ApiUnhealthyThreshold = 3,
    [int]$ProcessMissingThreshold = 2,
    [switch]$DryRun
)

Set-StrictMode -Off
$ErrorActionPreference = 'Continue'

# --------------------------------------------------------------------------- 配置
$settingsPath = Join-Path $DataRoot 'deploy.json'
$settings = $null
if (Test-Path -LiteralPath $settingsPath) {
    try { $settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json } catch { }
}
$apiHost = if ($settings -and $settings.apiHost) { [string]$settings.apiHost } else { '127.0.0.1' }
$apiPort = if ($settings -and $settings.apiPort) { [int]$settings.apiPort } else { 8000 }
$frontPort = if ($settings -and $settings.frontendPort) { [int]$settings.frontendPort } else { 3000 }

$logDir = Join-Path $DataRoot 'logs'
$logFile = Join-Path $logDir 'watchdog.log'
$stateFile = Join-Path $DataRoot 'watchdog.state.json'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# --------------------------------------------------------------------------- 工具
function Write-Wd([string]$level, [string]$message) {
    $line = "[{0}] {1,-5} {2}" -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'), $level, $message
    Add-Content -LiteralPath $logFile -Value $line -Encoding UTF8
    Write-Host $line
    # 日志超过 2MB 就只保留最后 400 行，避免无限增长
    if ((Get-Item -LiteralPath $logFile).Length -gt 2MB) {
        $tail = Get-Content -LiteralPath $logFile -Tail 400
        Set-Content -LiteralPath $logFile -Value $tail -Encoding UTF8
    }
}

function Get-State {
    if (Test-Path -LiteralPath $stateFile) {
        try { return Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json } catch { }
    }
    return [pscustomobject]@{ apiUnhealthy = 0; apiMissing = 0; workerMissing = 0; beatMissing = 0; frontMissing = 0 }
}

function Save-State($state) {
    $state | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding UTF8
}

function Get-MatchingProcess([string]$pattern) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='node.exe' OR Name='cmd.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $pattern -and $_.CommandLine -notmatch 'watchdog\.ps1' }
}

function Get-Health {
    try {
        return Invoke-RestMethod "http://${apiHost}:${apiPort}/health/ready" -TimeoutSec 8
    } catch {
        return $null
    }
}

function Restart-Task([string]$name, [string]$why) {
    if ($DryRun) { Write-Wd 'DRY' "会重启 $name（$why）"; return }
    try {
        Start-ScheduledTask -TaskName $name
        Write-Wd 'FIX' "已触发重启 $name（$why）"
    } catch {
        Write-Wd 'ERR' "重启 $name 失败：$($_.Exception.Message)"
    }
}

# --------------------------------------------------------------------------- 巡检
$state = Get-State
$problems = 0

# --- API：进程在不在 + 健康检查 -------------------------------------------------
$apiProc = Get-MatchingProcess 'uvicorn'
if (-not $apiProc) {
    $state.apiMissing++
    $problems++
    Write-Wd 'WARN' "API 进程不存在（连续 $($state.apiMissing) 次）"
    if ($state.apiMissing -ge $ProcessMissingThreshold) {
        Restart-Task 'AIP-API' '进程不存在'
        $state.apiMissing = 0
    }
} else {
    $state.apiMissing = 0
    $health = Get-Health
    if ($null -eq $health) {
        $state.apiUnhealthy++
        $problems++
        Write-Wd 'WARN' "API 进程在（pid $($apiProc[0].ProcessId)）但 /health/ready 无响应（连续 $($state.apiUnhealthy) 次）"
        if ($state.apiUnhealthy -ge $ApiUnhealthyThreshold) {
            # 卡住的 API（例如数据库不可达导致请求挂起）只有重启才能恢复
            Restart-Task 'AIP-API' '健康检查连续失败'
            $state.apiUnhealthy = 0
        }
    } else {
        $state.apiUnhealthy = 0
        if ($health.status -ne 'ok') {
            # 依赖服务（Postgres/Redis）不可用：重启 API 没用，只告警
            Write-Wd 'WARN' "/health/ready = $($health.status) $($health.checks | ConvertTo-Json -Compress)（依赖服务问题，不重启 API）"
            $problems++
        }
    }
}

# --- Worker：进程在不在（重启会中断训练，阈值更保守） ---------------------------
# 必须匹配子命令 "celery_app worker"：beat 的命令行里也含 backend.app.workers，
# 粗匹配 'celery.*worker' 会在 worker 已死、只剩 beat 时误判为存活。
$workerProc = Get-MatchingProcess 'celery_app\s+worker'
if (-not $workerProc) {
    $state.workerMissing++
    $problems++
    Write-Wd 'WARN' "Worker 进程不存在（连续 $($state.workerMissing) 次）"
    if ($state.workerMissing -ge $ProcessMissingThreshold) {
        Restart-Task 'AIP-Worker' '进程不存在'
        Write-Wd 'WARN' 'Worker 重启会中断正在跑的训练；该训练会被重新投递并从头再跑'
        $state.workerMissing = 0
    }
} else {
    $state.workerMissing = 0
}

# --- Beat：进程在不在 -----------------------------------------------------------
$beatProc = Get-MatchingProcess 'celery_app\s+beat'
if (-not $beatProc) {
    $state.beatMissing++
    $problems++
    Write-Wd 'WARN' "Beat 进程不存在（连续 $($state.beatMissing) 次）—— 定时清扫（评测/租约/排队任务）会停摆"
    if ($state.beatMissing -ge $ProcessMissingThreshold) {
        Restart-Task 'AIP-Beat' '进程不存在'
        $state.beatMissing = 0
    }
} else {
    $state.beatMissing = 0
}

# --- 前端：端口是否在听 ---------------------------------------------------------
$front = Get-NetTCPConnection -State Listen -LocalPort $frontPort -ErrorAction SilentlyContinue
if (-not $front) {
    $state.frontMissing++
    $problems++
    Write-Wd 'WARN' "前端端口 $frontPort 未监听（连续 $($state.frontMissing) 次）"
    if ($state.frontMissing -ge $ProcessMissingThreshold) {
        Restart-Task 'AIP-Frontend' '端口未监听'
        $state.frontMissing = 0
    }
} else {
    $state.frontMissing = 0
}

Save-State $state
if ($problems -eq 0) { Write-Wd 'OK' '四项巡检正常' }
exit 0
