import os

base = "data/debug/slm_transfer/cold_start_20261026_1004"
for d in ["plane", "random_no_abba", "local_continuation"]:
    p = os.path.join(base, d)
    files = os.listdir(p)
    has_iter = "iterations.pkl" in files
    caps = len([f for f in files if f.startswith("capture_")])
    print(f"{d}: iterations.pkl={has_iter} captures={caps}")
