"""Channel-number parsing shared by the two ``scripts/dm/md_img_*`` consumers.

``md_img_pipeline.py`` and ``md_img_diff_to_gif.py`` carried byte-identical
copies of ``_channel_of`` (plus its ``_CHANNEL_RE``); neither imports the
other, so the single copy lives here ("再迁复用"). The frame sort order in
both consumers *is* the channel order, so a drifted regex would silently
reorder GIF frames and per-channel report tables.
"""

from __future__ import annotations

import re
from pathlib import Path

# Filenames look like: <ip>-<channel>_cx<X>_cy<Y>.png  or  <ip>-<channel>.png
# (channel is the 3-digit number right after the last dash of the base name).
_CHANNEL_RE = re.compile(r"-(\d{3})")


def _channel_of(path: Path) -> int:
    m = _CHANNEL_RE.search(path.name)
    if m is None:
        raise ValueError(f"Cannot parse channel number from {path.name}")
    return int(m.group(1))
