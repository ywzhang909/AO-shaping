# Runs and wall clock — `runs/sim2f_kl_ab/`

Two-arm comparison of the sim2f closed loop under two aberration ensembles:
`iid` (the default, what `runs/sim2f_fair` used) and `kolmogorov` (the OOPAO /
von Kármán spatial-correlation model, commit `38303f58`). Same corpus, same
checkpoint, same samples, same seed, same rounds, detector off — only the
ensemble the per-sample states are drawn from differs.

All commands run from the repository root with `$env:PYTHONPATH="src"`.
Python: `.venv\Scripts\python.exe` (GPU present; `ml.sim2f` auto-selects CUDA).
Machine: the same desktop as `runs/sim2f_fair/`. Wall clock measured with
PowerShell `Measure-Command` around the whole process.

Recorded at git `HEAD = 41cc7b7902649a3cc96bff9b764bd359dec9294f`.

| # | command | wall clock | exit |
|---|---|---|---|
| 1 | `python runs/sim2f_kl_ab/run_two_arms.py` | **107.0 s** process (`103.3 s` inside `main`, after imports) | 0 |
| 2 | `python runs/sim2f_kl_ab/run_two_arms.py --reuse` | **1.7 s** (regenerates the report from the existing loop JSONs, no compute) | 0 |
| 3 | `python -m ml.sim2f eval --checkpoint runs/sim2f_fair/train/model_best.pt --cache runs/sim2f_fair/corpus --split test --out runs/sim2f_kl_ab/eval_test.json` | **7.3 s** | 0 |

Command 1 issues **four** `loop.run_loop` calls internally; their clocks,
measured around `run_loop` alone (so excluding process start, `ml.sim2f` import
and CUDA init — which is why they do not sum to the driver total):

| internal run | output | wall clock |
|---|---|---|
| `iid` (seed 7) | `loop/iid/` | **29.9 s** |
| `kolmogorov` (seed 7) | `loop/kolmogorov/` | **25.1 s** |
| `iid` (seed 8, noise floor) | `loop/iid_seed8/` | **23.2 s** |
| `kolmogorov` (seed 8, noise floor) | `loop/kolmogorov_seed8/` | **23.6 s** |

Run-to-run spread on this machine for the whole driver was 107.0 / 109.6 s over
three runs (the loop itself is deterministic — `iid` reproduced the published
baseline bit for bit on every run); the spread is process/CUDA warm-up, not
compute.

**Why one loop run here is ~25 s while `runs/sim2f_fair` recorded 83.6 s.**
Both are the same 24-sample, 2-round, detector-off config on the same machine,
and the `iid` arm here reproduces that run's numbers **bit for bit** (35/35
columns, max |Δ| = 0), so this is a measurement-boundary difference, not a
config difference. The fair run measured `Measure-Command` around a cold
`python -m ml.sim2f loop ...` whose log was piped through `Tee-Object` (the same
note that documents its `synth` exit-code artifact); the first run here also
pays the cold start, which is why it is 29.9 s against 25.1 s for the identical
second run. For the comparison itself nothing depends on this: all four runs
share one process, one conditions set, one clock.

### Invocation form (and the PowerShell exit-code artifact)

Because loguru writes DEBUG/INFO to **stderr**, PowerShell turns a native
command's stderr into a `NativeCommandError` and reports exit 1 even when Python
returned 0 — the same artifact documented in `runs/sim2f_fair/COMMANDS.md`.
Every command here was therefore launched through `cmd /c` with a plain `>`
redirect, and the exit code read from `$LASTEXITCODE`:

```powershell
$env:PYTHONPATH="src"
cmd /c ".venv\Scripts\python.exe runs\sim2f_kl_ab\run_two_arms.py > runs/sim2f_kl_ab\run_two_arms.log 2>&1"
echo "EXIT=$LASTEXITCODE"     # -> 0
```

### Why `--out` on command 3

`python -m ml.sim2f eval` defaults to writing `eval.json` **next to the
checkpoint**, i.e. into `runs/sim2f_fair/train/`, which is read-only for this
task. `--out` only redirects where the report is written; the numbers are
unaffected, and the written file is **byte-identical** to
`runs/sim2f_fair/eval_test.json` (checked key by key over the whole tree, zero
differences) — which is the point of running it: the label/eval path does not
depend on the ensemble at all.

## Artefact → script map

| artefact | produced by |
|---|---|
| `run_two_arms.py` | written by hand (this task) |
| `loop/{iid,kolmogorov}/eval_loop.{json,csv}` | `run_two_arms.py` → `ml.sim2f.loop.run_loop` |
| `loop/{iid,kolmogorov}_seed8/eval_loop.{json,csv}` | same, noise-floor replicate (`LoopConfig.seed=8`) |
| `two_arms.json` | `run_two_arms.py` (comparison, pairing, paired test, verdict) |
| `two_arms.md` | `run_two_arms.py` (rendered from `two_arms.json`) |
| `run_two_arms.log` | full stdout+stderr of command 1 |
| `eval_test.json`, `eval_test.log` | `python -m ml.sim2f eval` (command 3) |

## Nothing is committed, nothing tracked was edited

`runs/` is git-untracked by design; this whole directory is new and untracked.
`two_arms.json` → `setup.tracked_sources_dirty_snapshot` is a **record** of the
source trees' `git status` at run time, not something this driver produced. The
working tree was already dirty on arrival (modified tracked files from other
lanes, including `src/ml/sim2f/bench.py`, whose mtime — 11:03 — precedes this
session). This driver edits no tracked file; the `iid` arm reproducing the
published baseline bit for bit confirms those pre-existing edits do not reach
these numbers.
