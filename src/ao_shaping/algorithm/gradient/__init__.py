"""Gradient optimizers (Adam-family, acceleration).

Submodules are importable directly by their full path, e.g.::

    from ao_shaping.algorithm.gradient.adam import Adam, AdamW, AdaMOD
    from ao_shaping.algorithm.gradient.acceleration import _njit

This ``__init__`` intentionally does NOT re-import the submodules: a docstring-only
package init avoids a circular-import cascade (submodules import each other and the
top-level ``ao_shaping.algorithm`` facade, so eagerly importing them here would run
the top-level ``__init__`` mid-initialisation). All public names are exposed through
the top-level ``ao_shaping.algorithm`` facade instead.
"""
