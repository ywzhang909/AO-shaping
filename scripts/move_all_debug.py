from __future__ import annotations

import shutil
from pathlib import Path

src_root = Path(r'D:\Projects\TIFO\AO-shaping\report\shape_trials')
dst_root = Path(r'D:\Projects\TIFO\AO-shaping\data\debug')

dst_root.mkdir(parents=True, exist_ok=True)

copied = 0
skipped = 0

for exp_dir in src_root.iterdir():
    if not exp_dir.is_dir():
        continue
    
    debug_dir = exp_dir / 'debug'
    if not debug_dir.exists():
        continue
    
    for family_dir in debug_dir.iterdir():
        if not family_dir.is_dir():
            continue
        
        for ts_dir in family_dir.iterdir():
            if not ts_dir.is_dir():
                continue
            
            dst_family_dir = dst_root / family_dir.name
            dst_ts_dir = dst_family_dir / ts_dir.name
            
            if dst_ts_dir.exists():
                skipped += 1
                continue
            
            dst_ts_dir.mkdir(parents=True, exist_ok=True)
            
            for f in ts_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_ts_dir / f.name)
            
            copied += 1

print(f"Done. Copied: {copied}, Skipped: {skipped}")