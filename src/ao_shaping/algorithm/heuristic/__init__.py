"""Heuristic / black-box optimizers (GA, PSO, SA, HC, Random, CEM, DE).

Submodules are importable directly by their full path, e.g.::

    from ao_shaping.algorithm.heuristic.ga import GeneticAlgorithm, GAParams
    from ao_shaping.algorithm.heuristic.heuristic_base import HeuristicOptimizer

This ``__init__`` intentionally does NOT re-import the submodules: a docstring-only
package init avoids a circular-import cascade (the submodules import the shared
``heuristic_base`` and the top-level ``ao_shaping.algorithm`` facade, so eagerly
importing them here would run the top-level ``__init__`` mid-initialisation). All
public names are exposed through the top-level ``ao_shaping.algorithm`` facade.
"""
