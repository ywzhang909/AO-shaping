@echo off
setlocal enabledelayedexpansion

echo ========================================
echo DM Control DLL/EXE Build Script
echo ========================================

set "VSINSTALLDIR=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "VCVARS=%VSINSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat"
set "SRC_DIR=D:\Projects\TIFO\ao\SDKs\微驱动器\软件\c"

if not exist "%VCVARS%" (
    echo ERROR: Visual Studio not found!
    goto :pause_end
)

echo Setting up Visual Studio environment...
call "%VCVARS%" >nul 2>&1

cd /d "%SRC_DIR%"

echo.
echo Building dm_control.dll...
cl /LD /I. /Fe:dm_control.dll dm_control.c /Ws2_32.lib /DM_CONTROL_EXPORTS /EHsc /MD /W4

if errorlevel 1 (
    echo DLL build FAILED!
    goto :pause_end
)

echo.
echo Building dm_test.exe...
cl /Fe:dm_test.exe main.c dm_control.lib /Ws2_32.lib /EHsc /MD /W4

if errorlevel 1 (
    echo EXE build FAILED!
    goto :pause_end
)

echo.
echo ========================================
echo Build Complete!
echo ========================================

:pause_end
pause