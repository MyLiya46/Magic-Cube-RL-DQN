# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11+ Rubik’s Cube demo driven by a 12-action DQN policy.

```text
magic_cube/core/  Cube state, moves, and geometry
magic_cube/rl/    Gymnasium environment, observations, rules, and inference
magic_cube/ui/    PySide6/OpenGL views and the main window
scripts/          Training and evaluation entry points
tests/            pytest unit and UI tests
models/           Model documentation, checkpoints, and curriculum progress
```

`magic_cube/app.py` is the application entry point; `__main__.py` supports
`python -m magic_cube`.

## Build, Test, and Development Commands

- `uv sync` installs the locked runtime and development dependencies.
- `uv run pytest` runs the complete test suite quietly.
- `uv run pytest tests/test_moves.py` runs one focused test module.
- `uv run python -m magic_cube` or `uv run magic-cube` starts the GUI.
- `uv run python scripts/train_rl.py --depths 1,2,3` trains the curriculum and
  writes checkpoints under `models/`.
- `uv run python scripts/evaluate_rl.py --checkpoint models/cube_solver.zip`
  evaluates a saved policy.
- `uv build` creates distributable package artifacts.

## Coding Style & Naming Conventions

Use four-space indentation, Python type hints, and focused modules. Name
functions, variables, and files in `snake_case`, classes in `PascalCase`, and
constants in `UPPER_SNAKE_CASE`. Keep public behavior documented through clear
types and small tests. No formatter or linter is configured; preserve the
existing import style and run pytest before submitting changes.

## Testing Guidelines

Tests use pytest and follow `tests/test_*.py` with `test_*` functions. Add or
update tests for every behavior change, especially move invariants, the RL
observation/action contract, training schedules, and UI state transitions.
The UI tests configure Qt for offscreen execution, so they should run in CI
without a display. No coverage threshold is currently enforced.

## Commit & Pull Request Guidelines

Existing commits use short subjects such as `model weights`, `logs`, and
`Init: Init repo`; keep new subjects concise and action-oriented, with an
optional scope (for example, `rl: validate checkpoint shape`). A pull request
should explain the behavioral change, list validation commands and results,
link a relevant issue when one exists, and include screenshots or a short
recording for UI changes. For model or curriculum updates, include the exact
training/evaluation command and relevant success or redundancy metrics.

## Configuration & Model Artifacts

Do not commit secrets or machine-specific configuration. The GUI expects
`models/cube_solver.zip`, with a 363-element observation and `Discrete(12)`
action space. Treat checkpoint and progress-file changes as intentional,
reviewable artifacts because they are large or generated.
