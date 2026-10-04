"""Debug-corpus data layer for GSNet offline training (SLM+CCD pickles).

Moved here from ``ao_shaping/runners/`` on 2026-10-05. These four modules
never were runners -- no Click command, no ``__main__``, no device lifecycle.
They turn the ``data/debug/slm_pib_*`` pickle corpus into a trainable
Dataset plus the offline transforms that go with it.

* :mod:`ml.gsnet_debug.offline` -- numpy transforms (pupil phase / far field /
  Zernike) and the record index. Pure functions over ``ao_shaping.utils.*``.
* :mod:`ml.gsnet_debug.cache` -- mmap-able ``.gsnet_cache`` sidecars next to
  each pickle.
* :mod:`ml.gsnet_debug.dataset` -- the lazy ``Dataset`` and grouped sampler.
* :mod:`ml.gsnet_debug.train` -- the offline training entry point that ties
  them together. This is what ``runners/slm/gsnet_runner.py`` calls; the CLI
  itself stays in ``ao_shaping.runners`` because it owns hardware.

Naming: the ``gsnet_`` prefix was dropped since the package supplies it.

.. note::
   This is a sibling of :mod:`ml.gsnet`, NOT a part of it. ``ml.gsnet`` is the
   FourierGSNet network itself (unrolled GS layers, U-Net refinement);
   ``ml.gsnet_debug`` is the corpus plumbing that feeds it. They were kept
   apart deliberately -- both packages have a ``dataset`` and a ``train``
   module, and merging them would collide on names while describing two
   different things.

Import-cycle note: :mod:`ml.hwdataset` imports pure helpers from
:mod:`ml.gsnet_debug.offline`, and :mod:`ao_shaping.optimizer.wfless` reads
these cache directories. That dependency previously pointed at
``ao_shaping.runners`` -- i.e. the ML package imported the hardware-orchestration
package for pure math. It now points at a package that belongs to ML, which is
the direction the layering was supposed to go.
"""

from __future__ import annotations

__all__ = ["cache", "dataset", "offline", "train"]