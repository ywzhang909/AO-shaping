"""Shared matplotlib CJK-font setup for the ``scripts/`` family.

Seven scripts carried their own copy of "give matplotlib a CJK font so Chinese
labels do not render as boxes". The copies were **not** identical, so this module
deduplicates the *procedure* and keeps each site's font preference as data:

* the preference chain is the ``families`` parameter -- two distinct chains ship
  in this repo and both are reproduced verbatim below;
* the one copy that probed the installed font list keeps that probe, behind
  ``probe_installed``.

Why the chains are NOT merged into one superset: ``font.sans-serif`` is a
**fallback chain**, and these generators write figures that are tracked in git.
A family that a copy never listed still wins whenever it is installed ahead of
one that is, so merging silently re-renders committed PNGs with a different font
(verified on this checkout: ``findfont("中", font_list=...)`` resolves
``Noto Sans CJK SC`` for the four-family chain but ``SimHei`` for the
three-family chain the inverse-shaping generators used).

Contract (behaviour-preserving extraction, not a fix):

* rcParams only -- this module never imports ``pyplot``, so it cannot pre-empt
  the ``matplotlib.use("Agg")`` each call site keeps (repo convention: Agg
  before pyplot).
* ``axes.unicode_minus = False`` is unconditional, as it was in all seven copies.
* ``matplotlib`` is imported inside the function, so importing this module stays
  free for scripts that import matplotlib lazily.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["CJK_FONTS", "CJK_FONTS_INVERSE_SHAPING", "setup_cjk_font"]

#: Default preference: the three CJK families this repo's figures were built
#: with, terminated by matplotlib's own default. Replaces the whole chain, so no
#: installed-font probe happens -- matplotlib walks the list itself.
CJK_FONTS: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
)

#: The ``scripts/ml/generate_inverse_shaping*_report.py`` chain: the same
#: preference **without** ``Noto Sans CJK SC``. Kept separate on purpose -- the
#: figures under ``report/inverse_shaping_unet/`` were rendered with this chain,
#: and adding Noto would switch the resolved font on any host where Noto is
#: installed.
CJK_FONTS_INVERSE_SHAPING: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
)


def setup_cjk_font(
    families: Sequence[str] = CJK_FONTS,
    *,
    probe_installed: bool = False,
    fallback: str = "DejaVu Sans",
) -> str | None:
    """Point matplotlib's ``font.sans-serif`` at a CJK-capable family.

    Two modes, matching the two kinds of copy this replaces:

    * **default** (``probe_installed=False``): ``font.sans-serif`` is *replaced*
      by ``list(families)`` and matplotlib's own fallback walk decides. Nothing
      is probed, so a missing family is not an error.
    * **probe** (``probe_installed=True``): the first family of ``families`` that
      is actually installed wins, and ``font.sans-serif`` becomes
      ``[winner, fallback]``; when none is installed the existing chain is left
      untouched and a warning is logged. ``families`` must be a **tuple/list in
      priority order**, never a set -- set iteration order depends on string
      hashing, and two runs of the same generator then pick different fonts and
      emit byte-different PNGs for identical input.

    ``axes.unicode_minus = False`` is set in both modes and even when the probe
    finds nothing.

    Args:
        families: Font families, highest priority first.
        probe_installed: Resolve against the installed font list instead of
            handing the whole chain to matplotlib.
        fallback: Family appended after the winner in probe mode.

    Returns:
        The winning family in probe mode, else ``None``. Every call site in
        ``scripts/`` ignores the return value; it exists for callers that want
        to log which font was chosen.
    """
    import matplotlib

    chosen: str | None = None
    if probe_installed:
        from matplotlib import font_manager

        installed = {font.name for font in font_manager.fontManager.ttflist}
        for name in families:
            if name in installed:
                matplotlib.rcParams["font.sans-serif"] = [name, fallback]
                chosen = name
                break
        else:
            from loguru import logger

            logger.info("warning: no CJK font found; Chinese labels may be garbled")
    else:
        matplotlib.rcParams["font.sans-serif"] = list(families)

    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen
