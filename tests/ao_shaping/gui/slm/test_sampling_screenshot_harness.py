from __future__ import annotations
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from streamlit.testing.v1 import AppTest

def run_one(cfg):
    os.environ["SAMPLING_CFG"] = json.dumps(cfg) if False else __import__("json").dumps(cfg)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[4] / "scripts" / "slm" / "screenshot_sampling_assist.py"), default_timeout=60)
    at.run()
    print("EXC:", at.exception)
    # print markdown
    for m in at.markdown:
        print("MD:", m.value[:200])

if __name__ == "__main__":
    import json
    run_one({"main_side_mm": 10.0, "f_main_mm": 1000.0, "f_sample_mm": 50.0, "expansion": 4.0, "waist_mm": 1.0})
