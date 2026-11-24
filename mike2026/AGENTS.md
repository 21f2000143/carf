# Repository Guidelines

## Project Structure & Module Organization

- `src/` contains modeling, evaluation, experiment orchestration, and revision-specific data/imaging modules.
- `scripts/` provides data downloads, preparation, analysis, result validation, and figure generation.
- `configs/` holds `synthetic.yaml`, `quick.yaml`, and `full.yaml` experiment settings.
- `tests/` contains regression tests for modeling, preparation, evaluation, imaging, and manuscript assets.
- `data/` stores raw and processed inputs; `results/`, `results_quick/`, and `results_synthetic/` hold experiment outputs. Execution logs go in `logs/`.
- `paper/` contains the LaTeX manuscript, generated tables/macros, and figures. Consult `ANALYSIS_PLAN.md`, `DATASETS.md`, and `revision2/` for research scope.

## Build, Test, and Development Commands

Run commands from the repository root:

- `conda env create -f environment.yml` followed by `conda activate mike-pcr`: create the Python 3.11 environment.
- `python -m pip install -r requirements-revision2.txt`: install base dependencies plus imaging support for revision work.
- `make smoke`: generate synthetic inputs and run the synthetic experiment.
- `python -m src.experiments --config configs/quick.yaml`: run a reduced experiment after downloading and preparing data.
- `make all`: download, prepare, run the full analysis, and generate manuscript assets.
- `make figures`: regenerate manuscript assets from results.
- `make paper`: compile `paper/main.tex`; requires a LaTeX installation with `latexmk`.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Follow surrounding code conventions and keep experiment settings in YAML. No formatter or linter configuration is present; avoid unrelated formatting changes.

## Testing Guidelines

Run `python -m unittest discover -s tests -v`. Use `unittest.TestCase`, files named `test_*.py`, and methods named `test_*`. Add focused regression tests for changed behavior; no numeric coverage threshold is configured.

After experiments, run `python scripts/validate_results.py --config configs/synthetic.yaml` (or the corresponding full configuration). Record meaningful execution evidence in `SMOKE_TEST.txt` when updating reported results.

## Commit & Pull Request Guidelines

The available history uses concise imperative subjects, such as “Prepare anonymous ten-page MIKE 2026 breast cancer manuscript.” Follow that style. PR descriptions should explain the change, identify relevant issues, and include validation commands/results. Include figure previews for visual manuscript changes.

## Research Reproducibility

Keep feature selection and tuning inside training folds; never use external cohorts for model selection. Inspect metadata audits before interpreting results. Preserve official Springer template files and regenerate manuscript assets through scripts. Avoid committing local environments, caches, or incidental download artifacts.
