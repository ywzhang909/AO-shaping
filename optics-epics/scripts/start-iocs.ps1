# =============================================================================
# start-iocs.ps1 - 在 Windows Host 上以离线模式启动全部 5 个 IOC
#
# 用法:
#   .\scripts\start-iocs.ps1              # 启动全部
#   .\scripts\start-iocs.ps1 -Ioc slm    # 仅启动 slm
#   .\scripts\stop-iocs.ps1               # 停止全部
#
# 说明:
#   - 每个 IOC 独立进程 + 独立 EPICS_CA_PORT(config/ioc.yaml 的 ca_port)
#   - 无硬件时 IOC 自动降级离线模式(仅注册 PV)
#   - 日志输出到 $env:TEMP\opencode\ioc-<name>.log
# =============================================================================
param(
    [string[]]$Ioc = @("slm", "dm", "dhcam", "mii", "wfs")
)

$root = Split-Path -Parent $PSScriptRoot        # optics-epics
$repo = Split-Path -Parent $root                # AO-shaping
$py = Join-Path $repo ".venv\Scripts\python.exe"
$logDir = Join-Path $env:TEMP "opencode"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$ports = @{ slm = 5065; dm = 5066; dhcam = 5067; mii = 5068; wfs = 5069 }
$classes = @{
    slm   = "src.slm_ioc.SlmIoc"
    dm    = "src.dm_ioc.DmIoc"
    dhcam = "src.dhcam_ioc.DhcamIoc"
    mii   = "src.miicam_ioc.MiiCamIoc"
    wfs   = "src.wfs_ioc.WfsIoc"
}

foreach ($name in $Ioc) {
    $iocDir = Join-Path $root "iocs\ioc-$name"
    if (-not (Test-Path (Join-Path $iocDir "config\ioc.yaml"))) {
        Write-Host "SKIP $name (ioc-$name 不存在)" -ForegroundColor Yellow
        continue
    }
    $env:PYTHONPATH = "$root\iocs\common;$iocDir\src;$repo\src;$repo\libs"
    $env:EPICS_CA_ADDR_LIST = "127.0.0.1"
    $env:EPICS_CA_SERVER_PORT = "$($ports[$name])"
    $log = Join-Path $logDir "ioc-$name.log"
    $args = @("-m", "ao_epics_common.serve", "config/ioc.yaml", $classes[$name])
    Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $iocDir `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden
    Write-Host "START $name (port $($ports[$name])) log=$log"
}

Write-Host "提示:等待 3 秒后可用 scripts\check-iocs.ps1 或 caget 验证"
