# Contributing to surrogate-lab

surrogate-lab is a pre-1.0 project. The core schema, config, and training interfaces are still
settling, so please open an issue before starting large changes — small fixes and tests are
always welcome without one.

## Setup

```bash
git clone https://github.com/NicholasTJL/surrogate-lab.git
cd surrogate-lab
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .
mypy src
pytest --cov=surrogate_lab
```

All three must pass. New behavior needs a test; bug fixes need a regression test.

## Scope

The current milestone is `v0.1.0` (see [docs/vision.md](docs/vision.md) for scope and
non-goals). Features outside that scope are welcome as discussion issues but may be deferred.

## Good first issues

Issues labeled `good first issue` are self-contained and don't require familiarity with the
training pipeline internals. `help wanted` issues are open for anyone.

## Code style

- Type hints on all public functions and classes.
- No bare `except:` — catch specific exceptions and raise a project exception type
  (`DataError`, `ConfigError`, `UnknownModelError`, etc.) with a message that names the
  offending file, column, or model.
- Keep the public API in `surrogate_lab.core`, `surrogate_lab.reporting`, and
  `surrogate_lab.cli` stable; internal helpers can change freely.
- Any new uncertainty or confidence feature must document its assumptions and limitations as
  clearly as the existing residual-based prediction interval does — no overclaiming what a
  number means.
