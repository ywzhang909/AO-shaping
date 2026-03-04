# AGENTS.md - Agent Coding Guidelines for AO-shaping

**Generated:** 2026-03-04
**Project:** AO-shaping - Adaptive Optics Shaping

## OVERVIEW

Python project for Adaptive Optics, controlling hardware (SLM, DM, WFS, CCD) with PyTorch deep learning. Uses `uv` for dependency management.

## STRUCTURE

```
AO-shaping/
├── src/ao_shaping/
│   ├── drivers/        # Hardware SDKs (slm, dm, wfs, ccd, tm)
│   ├── algorithm/     # Optimization algorithms
│   ├── wf/           # Wavefront control
│   ├── wfless/       # Wavefrontless methods
│   ├── utils/        # Utilities
│   └── display/      # GUI/visualization
├── tests/ao_shaping/ # Mirrors src structure
├── scripts/           # Standalone apps (streamlit)
└── pyproject.toml    # uv + pytest config
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| SLM driver | `src/ao_shaping/drivers/slm/` |
| DM driver | `src/ao_shaping/drivers/dm/` |
| Wavefront algorithms | `src/ao_shaping/wf/` |
| Wavefrontless algorithms | `src/ao_shaping/wfless/` |
| Display utilities | `src/ao_shaping/display/` |

## CONVENTIONS (Project-Specific)

- **Import order**: stdlib → third-party → local (explicit relative)
- **Logging**: Use `loguru.logger` only (no stdlib logging)
- **Hardware drivers**: Must implement `open()`, `close()`, context manager
- **Error handling**: Custom exceptions per driver (e.g., `SantecSLM200Error`)
- **NumPy arrays**: Always specify `dtype` explicitly

## ANTI-PATTERNS (THIS PROJECT)

- ❌ No TODO/FIXME markers in code
- ❌ Don't use `as any` or `@ts-ignore` (no TypeScript here, but same principle for type safety)
- ❌ Don't mix Chinese/English in same file - pick one
- ❌ Don't use bare `except:` - catch specific exceptions

## TEST CONVENTIONS

- Test files mirror: `tests/ao_shaping/<module>/test_<driver>.py`
- Hardware-dependent tests use `@pytest.mark.skip` when device unavailable
- Fixtures: `slm()`, `open_slm()`, `dm()`, etc.

## COMMANDS

```bash
# Install
uv sync

# Test
pytest                              # all
pytest tests/ao_shaping/drivers/    # drivers only
pytest -k "test_slm"               # pattern match

# Coverage
pytest --cov=ao_shaping --cov-report=html

# Streamlit UI
streamlit run scripts/streamlit_visualizer.py
```
