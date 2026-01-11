cd d:\workspace\AO-shaping\src\calculators
uv run setup.py build_ext --build-lib ../ao_shaping/algorithm

nuitka --output-dir=dist --show-progress --standalone --nofollow-import-to=torch --include-data-file=.\libs\Drv_UDPST\x64\Release\Drv_UDPST.dll=Drv_UDPST.dll --include-data-file=./data/dm_adj.txt=./data/dm_adj.txt --windows-icon-from-ico=camera_rotate.ico .\src\ao_shaping\wf-less\DM_cam.py