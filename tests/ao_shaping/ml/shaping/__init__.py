"""Test package for :mod:`ml.shaping`.

Why this ``__init__.py`` exists
-------------------------------
Without it, pytest derives a test module's dotted name from its **basename
alone**, so two same-named files in sibling directories collide:

.. code-block:: text

    tests/ao_shaping/ml/gsnet/test_dataset.py    -> module "test_dataset"
    tests/ao_shaping/ml/shaping/test_dataset.py  -> module "test_dataset"   # same!

and collection aborts with::

    import file mismatch:
    imported module 'test_dataset' has this __file__ attribute: ...
    which is not the same as the test file we want to collect: ...

That is not hypothetical here -- it is exactly what happened when this package
was added, and it is a **whole-suite** failure rather than a local one, because
``pyproject.toml`` sets ``testpaths = ["tests"]`` so a plain ``pytest`` collects
the entire tree. The collision surfaced only when the parent directory was
collected; running the subdirectory alone passed and hid it.

Adding this file makes pytest walk up from the test file while ``__init__.py``
exists, so the module is imported as ``shaping.test_dataset`` and the basename
is no longer the identity. ``tests/ao_shaping/ml/`` deliberately has **no**
``__init__.py``, which makes it the insertion base.

This follows existing repo precedent -- ``tests/ao_shaping/display/``,
``drivers/``, ``drivers/wfs/``, ``model/`` and ``runners/`` are already packages
for the same reason.
"""