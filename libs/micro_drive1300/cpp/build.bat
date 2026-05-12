@echo off
setlocal

echo ==========================================
echo   Building DM Control C++ DLL
echo ==========================================
echo.

cd /d "%~dp0"

:: Check for Visual Studio
if not exist "C:\Program Files\Microsoft Visual Studio" (
    if not exist "C:\Program Files (x86)\Microsoft Visual Studio" (
        echo [ERROR] Visual Studio not found
        echo Please install Visual Studio with C++ support
        pause
        exit /b 1
    )
)

:: Call Visual Studio vcvars
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if %ERRORLEVEL% neq 0 (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to initialize Visual Studio
        pause
        exit /b 1
    )
)

echo [INFO] Building DLL...

:: Build DLL
cl /LD /Iinclude src\dm_control.cpp src\controller.cpp src\thread_pool.cpp ^
    ws2_32.lib /DM_CONTROL_EXPORTS /EHsc /MD /W4 /Fe:dm_control.dll

if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAIL] DLL build failed
    pause
    exit /b 1
)

echo.
echo [OK] DLL built successfully

:: Build test executable
echo.
echo [INFO] Building test...

cl /Fe:dm_test.exe src\test_main.cpp dm_control.lib ws2_32.lib /EHsc /MD /W4

if %ERRORLEVEL% neq 0 (
    echo [WARN] Test build failed (continuing...)
) else (
    echo [OK] Test built successfully
)

echo.
echo ==========================================
echo   Build Complete
echo ==========================================
echo.
echo Output files:
dir /b *.dll *.lib *.exp 2>nul

echo.
pause