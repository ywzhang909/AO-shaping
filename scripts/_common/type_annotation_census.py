"""Type-annotation census for ``src/ao_shaping`` (issue #178).

AST census of how much of the code base is actually annotated, so the
``~62.6%`` claim from issue #178 becomes a *measured, ratcheted* number instead
of an impression. Methodology (pinned, so numbers mean something across time):

* every ``def`` / ``async def`` in ``src/ao_shaping/**/*.py`` counts once,
  including nested functions and methods (lambdas have no annotations by law);
* a *parameter* counts when it is not ``self``/``cls`` and not ``*args`` /
  ``**kwargs`` name-wise -- those are style-optional in this repo;
* annotated == ``arg.annotation is not None`` for parameters,
  ``fn.returns is not None`` for return types;
* the headline ratio is ``(annotated_params + annotated_returns) /
  (total_params + total_defs)``, i.e. parameters and return slots weighted
  equally per slot.

Usage::

    oxpy python3 scripts/_common/type_annotation_census.py            # summary
    oxpy python3 scripts/_common/type_annotation_census.py --json -   # machine
    oxpy python3 scripts/_common/type_annotation_census.py --check    # ratchet
    oxpy python3 scripts/_common/type_annotation_census.py --update-baseline

The ratchet is one-way like the module-size gate: dropping below the pinned
floor (minus ``--tolerance`` percentage points, default 0.2) exits 1; a rise
only changes the world when ``--update-baseline`` re-pins it deliberately.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "ao_shaping"
BASELINE = ROOT / "scripts" / "_common" / "type_annotation_baseline.json"


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    a = fn.args
    args = [x for x in (*a.posonlyargs, *a.args) if x.arg not in ("self", "cls")]
    kwonly = list(a.kwonlyargs)
    return [*args, *kwonly]


def census() -> dict:
    """Count annotation slots across the whole ``src/ao_shaping`` tree."""
    defs = params_total = params_ann = returns_ann = 0
    by_area: dict[str, list[int]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        area = path.relative_to(SRC).parts[0] if len(path.relative_to(SRC).parts) > 1 else "(top)"
        acc = by_area.setdefault(area, [0, 0, 0, 0])
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defs += 1
            acc[0] += 1
            ps = _params(node)
            params_total += len(ps)
            acc[1] += len(ps)
            pa = sum(1 for x in ps if x.annotation is not None)
            params_ann += pa
            acc[2] += pa
            if node.returns is not None:
                returns_ann += 1
                acc[3] += 1
    slots = params_total + defs
    pct = round(100.0 * (params_ann + returns_ann) / slots, 2) if slots else 0.0
    areas = {
        k: {
            "defs": v[0],
            "params": v[1],
            "returns_annotated": v[3],
            "slots": v[0] + v[1],
            "pct": round(100.0 * (v[2] + v[3]) / (v[0] + v[1]), 2) if (v[0] + v[1]) else 0.0,
        }
        for k, v in sorted(by_area.items())
    }
    return {
        "defs": defs,
        "params_total": params_total,
        "params_annotated": params_ann,
        "returns_annotated": returns_ann,
        "slots": slots,
        "pct_annotated": pct,
        "by_area": areas,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", metavar="PATH", help="dump full census as JSON ('-' for stdout)")
    ap.add_argument("--check", action="store_true", help="fail if below the pinned baseline")
    ap.add_argument("--update-baseline", action="store_true", help="re-pin the floor to now")
    ap.add_argument("--tolerance", type=float, default=0.2, help="allowed drop in percentage points")
    args = ap.parse_args(argv)

    result = census()

    if args.update_baseline:
        BASELINE.write_text(json.dumps({"pct_annotated": result["pct_annotated"],
                                        "defs": result["defs"],
                                        "slots": result["slots"]},
                                       indent=1) + "\n", encoding="utf-8")
        print(f"baseline pinned at {result['pct_annotated']}% ({BASELINE.name})")
        return 0

    if args.json:
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        print(payload if args.json == "-" else (Path(args.json).write_text(payload, encoding="utf-8") or "ok"))

    print(f"defs={result['defs']}  param_slots={result['params_total']}  "
          f"annotated_params={result['params_annotated']}  annotated_returns={result['returns_annotated']}")
    print(f"annotation coverage: {result['pct_annotated']}%")
    for area, v in result["by_area"].items():
        print(f"  {area:<12} {v['pct']:>6}%  ({v['defs']} defs / {v['slots']} slots)")

    if args.check:
        if not BASELINE.exists():
            print(f"MISSING baseline: {BASELINE}", file=sys.stderr)
            return 1
        floor = json.loads(BASELINE.read_text(encoding="utf-8"))["pct_annotated"]
        if result["pct_annotated"] < floor - args.tolerance:
            print(f"REGRESSION: {result['pct_annotated']}% < floor {floor}% "
                  f"(-{args.tolerance})", file=sys.stderr)
            return 1
        print(f"ratchet ok: {result['pct_annotated']}% >= floor {floor}% (tol {args.tolerance})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
