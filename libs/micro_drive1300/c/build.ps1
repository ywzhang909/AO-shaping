# Build Script for DM Control DLL
# Supports: clang-cl (default) or MSVC cl
# Run: .\build.ps1 [-UseMSVC] [-Debug]

param(
    [switch]$UseMSVC,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DM ControlDLL Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

if ($UseMSVC) {
    Write-Host "Using MSVC (cl.exe)" -ForegroundColor Yellow
    $VSInstallDir = "C:\Program Files\Microsoft Visual Studio\2022\Community"
    $VCVarsPath = "$VSInstallDir\VC\Auxiliary\Build\vcvars64.bat"

    if (-not (Test-Path $VCVarsPath)) {
        Write-Error "Visual Studio 2022 not found at $VSInstallDir"
        exit 1
    }

    Write-Host "Setting up Visual Studio environment..." -ForegroundColor Yellow
    $env:Path = "C:\Windows\System32;C:\Windows;$env:Path"
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
    cmd /c $cmdScript
} else {
    Write-Host "Using Clang (clang-cl.exe)" -ForegroundColor Yellow

    $LlvmDir = "$env:USERPROFILE\scoop\apps\llvm\current\bin"
    $ClangCl = "$LlvmDir\clang-cl.exe"

    if (-not (Test-Path $ClangCl)) {
        Write-Error "clang-cl not found at $ClangCl"
        exit 1
    }

    $env:Path = "$LlvmDir;$env:Path"

    Write-Host ""
    & clang-cl --version | Select-Object -First 2
    Write-Host ""

    # Clean old artifacts
    @("dm_control.dll", "dm_control.lib", "dm_control.exp", "dm_control.obj", "dm_test.exe", "main.obj") | ForEach-Object {
        if (Test-Path $_) { Remove-Item $_ -Force }
    }

    Write-Host "Building dm_control.dll with clang-cl..." -ForegroundColor Yellow
    $extraFlags = if ($Debug) { "/Zi /Od" } else { "" }
    & clang-cl /LD /I. /Fe:dm_control.dll dm_control.c ws2_32.lib /DM_CONTROL_EXPORTS /EHsc /MD /W4 $extraFlags.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries) /link /DEF:dm_control.def
    if ($LASTEXITCODE -ne 0) {
        Write-Error "DLL build FAILED!"
        exit 1
    }
    Write-Host "DLL built successfully!" -ForegroundColor Green

    Write-Host ""
    Write-Host "Building dm_test.exe with clang-cl..." -ForegroundColor Yellow
    & clang-cl /Fe:dm_test.exe main.c /I. dm_control.lib ws2_32.lib /EHsc /MD /W4 $extraFlags.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "EXE build FAILED!"
        exit 1
    }
    Write-Host "EXE built successfully!" -ForegroundColor Green
}

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