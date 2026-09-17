"""Signal-processing / beam-shaping algorithms.

Includes GS / differentiable shaping / phase-wrap / wavefront utilities,
the control-law loop, and the shared iterative-optimizer base class.

Submodules are importable directly by their full path, e.g.::

    from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton
    from ao_shaping.algorithm.signal_processing.differentiable_beam import DifferentiableBeamOptimizer
    from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer

This ``__init__`` intentionally does NOT re-import the submodules: a docstring-only
package init avoids a circular-import cascade (the submodules import the top-level
``ao_shaping.algorithm`` facade and each other, so eagerly importing them here would
run the top-level ``__init__`` mid-initialisation). All public names are exposed
through the top-level ``ao_shaping.algorithm`` facade.
"""
