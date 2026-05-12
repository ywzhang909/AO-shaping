@echo off
cd /d "%~dp0src"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /LD /TP /I..\include dm_control.cpp controller.cpp thread_pool.cpp "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64\ws2_32.lib" /DM_CONTROL_EXPORTS /EHsc /MD /W4 /Fe:..\dm_control.dll
pause