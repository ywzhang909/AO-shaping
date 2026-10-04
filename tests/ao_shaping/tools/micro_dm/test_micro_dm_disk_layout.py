"""Pin the Micro-DM on-disk layout *before* refactoring its CLI (R-33).

``micro_dm_image_collect._save_frame`` writes the images that four
``scripts/md_img_*.py`` consumers and two ``ceramic_viewer`` GUIs read back
through :func:`ao_shaping.utils.io.file.find_cell_image`. Nothing in the repo
connects the writer to the reader, so a rename of the filename pattern would
break all six consumers while every unit test still passed.

The contract, pinned here:

* single-frame stem is exactly ``{ip}-{channel:03d}``
* multi-frame stem is ``{ip}-{channel:03d}-{frame_index:03d}``
* an optional ``.npy`` sibling shares the stem
* ``uint16`` frames are written with PIL mode ``I;16`` and ``uint8`` with ``L``,
  so the dynamic range is not clipped on the way to disk
* both forms live at ``{root}/{ip}/{stem}.png`` and are found by
  ``find_cell_image(root, ip_group, seq)``

The multi-frame case is deliberately asserted to resolve through the *prefix*
match rather than an exact name: ``find_cell_image`` uses ``startswith``, which
means several frames can match and it returns whichever ``iterdir()`` yields
first. That is a property of the reader, and the writer must not accidentally
break it by changing the stem's leading part.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ao_shaping.tools.micro_dm.micro_dm_image_collect import _save_frame
from ao_shaping.utils.io.file import find_cell_image

IP = "192.168.0.101"


def _ip_dir(tmp_path: Path) -> Path:
    d = tmp_path / IP
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_single_frame_stem_is_ip_channel(tmp_path):
    frame = np.zeros((4, 4), dtype=np.uint8)
    name = _save_frame(_ip_dir(tmp_path), IP, channel=7, frame=frame)
    assert name == f"{IP}-007.png"
    assert (_ip_dir(tmp_path) / name).is_file()


def test_multi_frame_stem_appends_the_index(tmp_path):
    frame = np.zeros((4, 4), dtype=np.uint8)
    name = _save_frame(_ip_dir(tmp_path), IP, channel=7, frame=frame, frame_index=12)
    assert name == f"{IP}-007-012.png"
    assert (_ip_dir(tmp_path) / name).is_file()


def test_channel_is_zero_padded_to_three_digits(tmp_path):
    frame = np.zeros((4, 4), dtype=np.uint8)
    for channel, expected in ((0, "000"), (5, "005"), (49, "049"), (123, "123")):
        name = _save_frame(_ip_dir(tmp_path), IP, channel=channel, frame=frame)
        assert name == f"{IP}-{expected}.png"


def test_npy_sibling_shares_the_stem(tmp_path):
    frame = np.arange(16, dtype=np.uint8).reshape(4, 4)
    name = _save_frame(_ip_dir(tmp_path), IP, channel=3, frame=frame, save_npy=True)
    stem = Path(name).stem
    npy = _ip_dir(tmp_path) / f"{stem}.npy"
    assert npy.is_file()
    assert np.array_equal(np.load(npy), frame)


def test_uint16_keeps_its_dynamic_range(tmp_path):
    """A uint16 frame must not be squashed into 8 bits on the way to disk."""
    frame = np.array([[0, 1000], [60000, 65535]], dtype=np.uint16)
    name = _save_frame(_ip_dir(tmp_path), IP, channel=1, frame=frame, save_npy=True)
    with Image.open(_ip_dir(tmp_path) / name) as img:
        assert img.mode == "I;16"
        assert np.array_equal(np.array(img), frame)


def test_uint8_uses_L_mode(tmp_path):
    frame = np.array([[0, 128], [255, 7]], dtype=np.uint8)
    name = _save_frame(_ip_dir(tmp_path), IP, channel=2, frame=frame)
    with Image.open(_ip_dir(tmp_path) / name) as img:
        assert img.mode == "L"
        assert np.array_equal(np.array(img), frame)


def test_single_frame_is_found_by_find_cell_image(tmp_path):
    frame = np.zeros((4, 4), dtype=np.uint8)
    _save_frame(_ip_dir(tmp_path), IP, channel=7, frame=frame)
    found = find_cell_image(tmp_path, ip_group=101, seq=7)
    assert found is not None
    assert found.name == f"{IP}-007.png"


def test_multi_frame_is_still_found_by_find_cell_image(tmp_path):
    """The reader matches on the ``{ip}-{seq:03d}`` prefix, so the leading part
    of the stem must stay byte-identical when a frame index is appended."""
    frame = np.zeros((4, 4), dtype=np.uint8)
    _save_frame(_ip_dir(tmp_path), IP, channel=7, frame=frame, frame_index=3)
    found = find_cell_image(tmp_path, ip_group=101, seq=7)
    assert found is not None
    assert found.name == f"{IP}-007-003.png"


def test_channel_zero_is_not_confused_with_channel_one(tmp_path):
    """seq=1 must not match ``...-010.png``; the 3-digit padding is load-bearing."""
    frame = np.zeros((4, 4), dtype=np.uint8)
    _save_frame(_ip_dir(tmp_path), IP, channel=10, frame=frame)
    assert find_cell_image(tmp_path, ip_group=101, seq=1) is None
    assert find_cell_image(tmp_path, ip_group=101, seq=10) is not None


def test_find_cell_image_returns_none_for_a_missing_group(tmp_path):
    assert find_cell_image(tmp_path, ip_group=126, seq=0) is None