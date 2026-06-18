@echo off
cd /d "%~dp0src"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /LD /TP /I..\include /I. /DM_CONTROL_EXPORTS /EHsc /MD /W4 dm_control.cpp controller.cpp thread_pool.cpp /link ws2_32.lib /Fe:..\dm_control.dll