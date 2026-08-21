# =============================================================================
# stop-iocs.ps1 - 停止全部(或指定)IOC 进程
# =============================================================================
param(
    [string[]]$Ioc = @("slm", "dm", "dhcam", "mii", "wfs")
)

foreach ($name in $Ioc) {
    $log = Join-Path $env:TEMP "opencode\ioc-$name.log"
    if (Test-Path $log) {
        # 通过日志文件关联进程(由 start-iocs.ps1 启动的 python 进程)
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -like "*ioc-$name*" -and $_.CommandLine -like "*ao_epics_common.serve*" } |
            ForEach-Object {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
                Write-Host "STOP $name (pid $($_.ProcessId))"
            }
    }
}
