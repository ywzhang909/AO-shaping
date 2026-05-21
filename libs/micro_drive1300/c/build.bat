@echo off
setlocal enabledelayedexpansion

echo ========================================
echo DM Control DLL/EXE Build Script
echo ========================================

set "SRC_DIR=%~dp0"
cd /d "%SRC_DIR%"

REM Try clang-cl first (via scoop)
set "CLANG_DIR=%USERPROFILE%\scoop\apps\llvm\current\bin"
if exist "%CLANG_DIR%\clang-cl.exe" (
    set "PATH=%CLANG_DIR%;%PATH%"
    echo Using clang-cl...
    echo.

    REM Clean old artifacts
    if exist dm_control.dll del /f dm_control.dll
    if exist dm_control.lib del /f dm_control.lib
    if exist dm_control.exp del /f dm_control.exp
    if exist dm_control.obj del /f dm_control.obj
    if exist dm_test.exe del /f dm_test.exe
    if exist main.obj del /f main.obj

    echo Building dm_control.dll...
    clang-cl /LD /I. /Fe:dm_control.dll dm_control.c ws2_32.lib /DM_CONTROL_EXPORTS /EHsc /MD /W4 /link /DEF:dm_control.def
    if errorlevel 1 (
        echo DLL build FAILED!
        goto :pause_end
    )

    echo.
    echo Building dm_test.exe...
    clang-cl /Fe:dm_test.exe main.c /I. dm_control.lib ws2_32.lib /EHsc /MD /W4
    if errorlevel 1 (
        echo EXE build FAILED!
        goto :pause_end
    )

    goto :build_done
)

REM Fallback to MSVC
set "VSINSTALLDIR=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "VCVARS=%VSINSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VCVARS%" (
    echo ERROR: Neither clang-cl nor Visual Studio found!
    goto :pause_end
)

echo Using MSVC cl.exe...
echo Setting up Visual Studio environment...
call "%VCVARS%" >nul 2>&1

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

:build_done
echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Output files:
for %%F in (dm_control.dll dm_control.lib dm_test.exe) do (
    if exist %%F echo   %%F
)

:pause_end
pause
