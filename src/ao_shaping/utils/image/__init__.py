"""Image processing utilities: spot analysis, beam metrics, target masks,
resampling and camera helpers.

Moved from the flat ``ao_shaping.utils`` package (refactor D1-W3). Rendering/
visualization lives in ``ao_shaping.display`` (the former ``display.py`` was
moved there — see ``ao_shaping.display.image_display``); only per-function lazy
``matplotlib`` imports remain in the computation helpers here.
"""

from __future__ import annotations