"""Compatibility CLI; implementation lives in ao_shaping.tools.slm.transfer.capture_response."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ao_shaping.tools.slm.transfer.capture_response import *  # noqa: F403
from ao_shaping.tools.slm.transfer.capture_response import main

if __name__ == "__main__":
    main()
