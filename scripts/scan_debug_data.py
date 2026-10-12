import pickle
import json
from pathlib import Path

debug_root = Path(r'D:\Projects\TIFO\AO-shaping\data\debug')

to_delete = []

# Scan all pickle files
for pkl_path in debug_root.rglob('*.pkl'):
    # Skip cache directories
    if '.cache' in str(pkl_path):
        continue
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"ERROR loading {pkl_path}: {e}")
        continue
    
    # Count iterations (epochs/rounds)
    if isinstance(data, dict):
        num_iters = len(data)
    elif data is None:
        num_iters = 0
    else:
        num_iters = 1
    
    # Check for simulation marker in sidecar JSON
    is_sim = False
    json_path = pkl_path.with_suffix('.json')
    if json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                config_str = meta.get('config', '')
                if 'cam_type' in config_str and 'sim' in config_str:
                    is_sim = True
        except:
            pass
    
    # Also check family name for sim_calib_abba
    family = pkl_path.parent.parent.name
    if 'sim_calib_abba' in family:
        is_sim = True
    
    if is_sim or num_iters < 2:
        to_delete.append((pkl_path.parent.parent, is_sim, num_iters))
        print(f"{'SIM' if is_sim else 'LOW_ITER'} | iters={num_iters:3d} | {pkl_path.parent.parent.name}")

print(f"\nTotal directories to delete: {len(to_delete)}")
for d, is_sim, iters in to_delete:
    print(f"  {'SIM' if is_sim else 'LOW_ITER'} | iters={iters:3d} | {d}")