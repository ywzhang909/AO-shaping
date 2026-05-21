@echo off
setlocal enabledelayedexpansion

echo ========================================
echo DM Control DLL/EXE Build Script (Clang)
echo ========================================

set "CLANG_DIR=%USERPROFILE%\scoop\apps\llvm\current\bin"
set "SRC_DIR=%~dp0"

if not exist "%CLANG_DIR%\clang-cl.exe" (
    echo ERROR: clang-cl not found at %CLANG_DIR%!
    goto :pause_end
)

set "PATH=%CLANG_DIR%;%PATH%"

cd /d "%SRC_DIR%"

echo.
echo Using: clang-cl --version
clang-cl --version
echo.

REM Clean old artifacts
if exist dm_control.dll del /f dm_control.dll
if exist dm_control.lib del /f dm_control.lib
if exist dm_control.exp del /f dm_control.exp
if exist dm_control.obj del /f dm_control.obj
if exist dm_test.exe del /f dm_test.exe
if exist main.obj del /f main.obj

echo.
echo Building dm_control.dll with clang-cl...
clang-cl /LD /I. /Fe:dm_control.dll dm_control.c ws2_32.lib /DM_CONTROL_EXPORTS /EHsc /MD /W4 /link /DEF:dm_control.def

if errorlevel 1 (
    echo DLL build FAILED!
    goto :pause_end
)

echo.
echo Building dm_test.exe with clang-cl...
clang-cl /Fe:dm_test.exe main.c /I. dm_control.lib ws2_32.lib /EHsc /MD /W4

if errorlevel 1 (
    echo EXE build FAILED!
    goto :pause_end
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Output files:
for %%F in (dm_control.dll dm_control.lib dm_control.exp dm_test.exe) do (
    if exist %%F echo   %%F
)

:pause_end
pause
