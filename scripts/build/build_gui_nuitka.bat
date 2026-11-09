@echo off
rem Build the AO-Shaping Qt console as an offline Windows standalone EXE (Nuitka).
rem Output: dist\gui\gui_entry.dist\AO-Shaping-Console.exe
rem Excluded on purpose: torch (training / fitting), cupy (GPU acceleration).
setlocal EnableExtensions

cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1

set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [build] vcvars64.bat not found: "%VCVARS%"
    exit /b 1
)
call "%VCVARS%" >nul
if errorlevel 1 (
    echo [build] failed to initialise MSVC environment
    exit /b 1
)

set "PYTHONPATH=%CD%\src"
set "PY=.venv\Scripts\python.exe"
set "DIST_ROOT=dist\gui"
set "DIST=%DIST_ROOT%\gui_entry.dist"

rem Daheng native DLLs are not part of the repository; copy them from the installed SDK.
if "%DAHENG_SDK_DLL%"=="" set "DAHENG_SDK_DLL=C:\Program Files\Daheng Imaging\GalaxySDK\APIDll\Win64"
if not exist "%DAHENG_SDK_DLL%\GxIAPI.dll" (
    echo [build] Daheng SDK DLLs not found: "%DAHENG_SDK_DLL%"
    echo [build] set DAHENG_SDK_DLL to the APIDll\Win64 folder of GalaxySDK
    exit /b 1
)
if not exist "%PY%" (
    echo [build] venv python not found: %PY%
    exit /b 1
)
if not exist "src\ao_shaping\drivers\ccd\_miicam_sdk\miicam.dll" (
    echo [build] MiiCam SDK missing: src\ao_shaping\drivers\ccd\_miicam_sdk\miicam.dll
    exit /b 1
)

if not exist "%DIST_ROOT%" mkdir "%DIST_ROOT%"

"%PY%" -m nuitka ^
    --standalone ^
    --assume-yes-for-downloads ^
    --enable-plugin=pyside6 ^
    --windows-console-mode=disable ^
    --windows-icon-from-ico=camera_rotate.ico ^
    --output-dir=%DIST_ROOT% ^
    --output-filename=AO-Shaping-Console.exe ^
    --file-reference-choice=runtime ^
    --include-package=ao_shaping ^
    --include-package-data=ao_shaping ^
    --include-package=gxipy ^
    --nofollow-import-to=torch ^
    --nofollow-import-to=cupy ^
    --include-data-file=libs\santec_slm\SLMFunc.dll=libs\santec_slm\SLMFunc.dll ^
    --include-data-file=libs\santec_slm\FTD3XX.dll=libs\santec_slm\FTD3XX.dll ^
    --include-data-file=src\ao_shaping\drivers\ccd\_miicam_sdk\miicam.dll=ao_shaping\drivers\ccd\_miicam_sdk\miicam.dll ^
    --include-data-dir=data\ccd_configs=data\ccd_configs ^
    --include-data-dir=data\slm_configs=data\slm_configs ^
    --include-data-dir=data\dm_configs=data\dm_configs ^
    --include-data-dir=data\micro_dm_configs=data\micro_dm_configs ^
    scripts\build\gui_entry.py
if errorlevel 1 (
    echo [build] Nuitka failed
    exit /b 1
)

rem Post-build: Santec/MiiCam DLLs are copied by Nuitka; Daheng native DLLs are added here.
if not exist "%DIST%\libs\daheng" mkdir "%DIST%\libs\daheng"
copy /y "%DAHENG_SDK_DLL%\*.dll" "%DIST%\libs\daheng\" >nul
if errorlevel 1 (
    echo [build] failed to copy Daheng DLLs
    exit /b 1
)

echo [build] done: %DIST%\AO-Shaping-Console.exe
endlocal
exit /b 0
