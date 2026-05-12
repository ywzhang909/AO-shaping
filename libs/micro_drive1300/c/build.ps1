# Build Script for DM Control DLL
# Run: .\build.ps1

param(
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DM Control DLL Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$ProjectDir = "D:\Projects\TIFO\ao\SDKs\微驱动器\软件\c"
$VSInstallDir = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$VCVarsPath = "$VSInstallDir\VC\Auxiliary\Build\vcvars64.bat"

# Check if VS exists
if (-not (Test-Path $VCVarsPath)) {
    Write-Error "Visual Studio 2022 not found at $VSInstallDir"
    exit 1
}

Write-Host "Setting up Visual Studio environment..." -ForegroundColor Yellow

# Call vcvars via cmd
$env:Path = "C:\Windows\System32;C:\Windows;$env:Path"  # Reset PATH
$cmdScript = @"
cd /d "$ProjectDir"
call "$VCVarsPath"
if exist dm_control.dll del /f dm_control.dll
if exist dm_control.lib del /f dm_control.lib
if exist dm_control.exp del /f dm_control.exp
cl /LD /I. /Fe:dm_control.dll dm_control.c /Ws2_32.lib /DM_CONTROL_EXPORTS /EHsc /MD $(if($Debug){'/Zi /Od'}else{''})
echo.
if exist dm_control.dll (
    echo DLL built successfully!
) else (
    echo DLL build FAILED!
    exit /b 1
)
if exist main.obj del /f main.obj
cl /Fe:dm_test.exe main.c /I. /EHsc /MD /W4 dm_control.lib /Ws2_32.lib $(if($Debug){'/Zi /Od'}else{''})
if exist dm_test.exe (
    echo EXE built successfully!
) else (
    echo EXE build FAILED!
)
"@

# Run via cmd
cmd /c $cmdScript

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# List output files
Get-ChildItem -Path $ProjectDir -Filter "dm_control.*" -File | ForEach-Object {
    Write-Host "  $($_.Name) ($([math]::Round($_.Length/1KB, 1)) KB)" -ForegroundColor White
}
Get-ChildItem -Path $ProjectDir -Filter "dm_test.exe" -File | ForEach-Object {
    Write-Host "  $($_.Name) ($([math]::Round($_.Length/1KB, 1)) KB)" -ForegroundColor White
}

Write-Host ""
Write-Host "To run tests:" -ForegroundColor Yellow
Write-Host "  .\dm_test.exe" -ForegroundColor White
Write-Host ""
Write-Host "For interactive mode:" -ForegroundColor Yellow
Write-Host "  .\dm_test.exe -i" -ForegroundColor White