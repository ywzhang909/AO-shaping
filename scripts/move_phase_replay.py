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
    
    # Check for phase_replay files directly in the directory
    pkl_file = exp_dir / 'phase_replay.pkl'
    json_file = exp_dir / 'phase_replay.json'
    png_file = exp_dir / 'phase_replay.png'
    
    if not pkl_file.exists():
        continue
    
    # Use experiment directory name as family name, and create a timestamp subdirectory
    family_name = f"phase_replay_{exp_dir.name}"
    # Use a fixed timestamp for the subdirectory
    ts_dir_name = "phase_replay"
    
    dst_family_dir = dst_root / family_name
    dst_ts_dir = dst_family_dir / ts_dir_name
    
    if dst_ts_dir.exists():
        print(f"SKIP (exists): {dst_ts_dir}")
        skipped += 1
        continue
    
    dst_ts_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files with phase_replay prefix
    if pkl_file.exists():
        shutil.copy2(pkl_file, dst_ts_dir / 'phase_replay.pkl')
    if json_file.exists():
        shutil.copy2(json_file, dst_ts_dir / 'phase_replay.json')
    if png_file.exists():
        shutil.copy2(png_file, dst_ts_dir / 'phase_replay.png')
    
    print(f"COPY: {family_name}/{ts_dir_name}")
    copied += 1

print(f"\nDone. Copied: {copied}, Skipped: {skipped}")