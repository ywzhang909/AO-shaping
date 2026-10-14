"""``python -m ao_shaping.tools.slm.calibration`` entry point.

Dispatches to the geometry-calibration CLI by default, or to the defocus shift
calibration when the first argument is ``shift`` — mirroring the behaviour the
module carried before it was split into a package.
"""

from __future__ import annotations

import sys

from ao_shaping.tools.slm.calibration.calibration import main, main_shift_calib

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "shift":
        main_shift_calib.main(
            args=sys.argv[2:], prog_name="ao_shaping.tools.slm.calibration shift"
        )
    else:
        main.main(args=sys.argv[1:], prog_name="ao_shaping.tools.slm.calibration")
