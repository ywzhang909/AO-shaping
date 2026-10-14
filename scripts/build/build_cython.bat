@echo off
cd /d "d:\workspace\AO-shaping\src\calculators"
python setup.py build_ext --build-lib ../ao_shaping/algorithm
echo Cython extensions built and placed in algorithm directory
pause
