"""Lock the final sim library/demo split verified for the issue #90 relocation.

Contract (verified 2026-10-25/26):
- ``src/ao_shaping/algorithm/signal_processing/sim.py`` is an import-only
  library: no ``__main__`` / ``argparse`` / ``plt.show``. The demo body
  was extracted to ``scripts/signal_processing/demo_wavefront_sim.py``
  by ``14b1cda`` and that script imports four symbols from the library,
  so relocating the module itself would break the import.
- The root ``AGENTS.md`` signal_processing row must state this final
  split (``14b1cda``) instead of the stale "migrate to scripts/" wording.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "src/ao_shaping/algorithm/signal_processing/sim.py"
DEMO = ROOT / "scripts/signal_processing/demo_wavefront_sim.py"
RECON = ROOT / "scripts/signal_processing/demo_wavefront_reconstruction.py"
AGENTS = ROOT / "AGENTS.md"

DOCUMENTED_SYMBOLS = (
    "VectorAdaptiveOptics",
    "VectorBeamPropagator",
    "calculate_stokes_rgb",
    "create_gaussian_field",
)


def test_sim_library_is_entrypoint_free() -> None:
    src = SIM.read_text(encoding="utf-8")
    for needle in ("__main__", "argparse", "plt.show"):
        assert needle not in src, f"sim.py must stay import-only; found {needle!r}"


def test_demo_imports_exactly_the_documented_symbols() -> None:
    demo = DEMO.read_text(encoding="utf-8")
    match = re.search(
        r"from ao_shaping\.algorithm\.signal_processing\.sim import \(([^)]*)\)",
        demo,
    )
    assert match is not None, "demo must import the four symbols from the library"
    imported = {t.strip().rstrip(",") for t in match.group(1).split("\n") if t.strip()}
    assert imported == set(DOCUMENTED_SYMBOLS), sorted(imported)
    sim_src = SIM.read_text(encoding="utf-8")
    for symbol in DOCUMENTED_SYMBOLS:
        assert re.search(rf"\b{re.escape(symbol)}\b", sim_src), f"{symbol} missing in sim.py"


def test_reconstruction_demo_companion_exists() -> None:
    assert RECON.is_file()


def test_agents_annotation_states_final_split() -> None:
    agents = AGENTS.read_text(encoding="utf-8")
    assert (
        "已于 `14b1cda` 迁至 `scripts/signal_processing/demo_wavefront_sim.py`" in agents
    ), "sim annotation must record the completed 14b1cda extraction"
    for stale in ("反模式表已注记其应迁", "本条只落册不动码"):
        assert stale not in agents, f"stale relocation wording still present: {stale!r}"
