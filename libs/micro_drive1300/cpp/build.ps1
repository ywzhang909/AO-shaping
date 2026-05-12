# DM Control C++ DLL Build Script (PowerShell)
# Usage: .\build.ps1

param(
    [switch]$Clean,
    [switch]$Test
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Building DM Control C++ DLL" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Find Visual Studio
$vsPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
)

$vsPath = $null
foreach ($path in $vsPaths) {
    if (Test-Path "$path\VC\Auxiliary\Build\vcvars64.bat") {
        $vsPath = $path
        break
    }
}

if (-not $vsPath) {
    Write-Host "[ERROR] Visual Studio not found" -ForegroundColor Red
    Write-Host "Please install Visual Studio with C++ support" -ForegroundColor Yellow
    exit 1
}

# Initialize Visual Studio
Write-Host "[INFO] Initializing Visual Studio..." -ForegroundColor Gray
$env:VSINSTALLDIR = $vsPath
$vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"

# Run vcvars and capture environment
$vcvarsOutput = & cmd /c "`"$vcvars`" && set" 2>&1 | Out-String
$vcvarsOutput -split "`n" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $name = $matches[1]
        $value = $matches[2]
        Set-Item -Path "env:$name" -Value $value
    }
}

# Clean if requested
if ($Clean) {
    Write-Host "[INFO] Cleaning..." -ForegroundColor Gray
    Remove-Item -Path "*.dll", "*.lib", "*.exp", "*.obj", "*.exe" -Force -ErrorAction SilentlyContinue
}

# Build DLL
Write-Host "[INFO] Building DLL..." -ForegroundColor Gray
$clArgs = @(
    "/LD",
    "/Iinclude",
    "src\dm_control.cpp",
    "src\controller.cpp",
    "src\thread_pool.cpp",
    "ws2_32.lib",
    "/DM_CONTROL_EXPORTS",
    "/EHsc",
    "/MD",
    "/W4",
    "/Fe:dm_control.dll"
)

$clResult = & cl $clArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] DLL build failed" -ForegroundColor Red
    Write-Host $clResult -ForegroundColor Red
    exit 1
}

Write-Host "[OK] DLL built successfully" -ForegroundColor Green

# Build test if requested
if ($Test) {
    Write-Host "[INFO] Building test..." -ForegroundColor Gray
    
    $testArgs = @(
        "/Fe:dm_test.exe",
        "src\test_main.cpp",
        "dm_control.lib",
        "ws2_32.lib",
        "/EHsc",
        "/MD",
        "/W4"
    )
    
    $testResult = & cl $testArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Test build failed" -ForegroundColor Yellow
    } else {
        Write-Host "[OK] Test built successfully" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Build Complete" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Output files:" -ForegroundColor Gray
Get-ChildItem -Path "." -Filter "*.dll", "*.lib" | ForEach-Object {
    Write-Host "  $($_.Name)" -ForegroundColor White
}