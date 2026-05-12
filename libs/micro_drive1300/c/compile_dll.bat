@echo off
cd /d "D:\Projects\TIFO\ao\SDKs\微驱动器\软件\c"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /LD /I. /Fe:dm_control.dll dm_control.c /Ws2_32.lib /DM_CONTROL_EXPORTS /EHsc /MD /W4
if exist dm_control.dll echo DLL SUCCESS
if not exist dm_control.dll echo DLL FAILED