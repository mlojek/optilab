# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install       # uv sync — install all dependencies
make format        # ruff format + pyprojectsort
make check         # format, then ruff lint + ty type check + pyprojectsort --check
make test          # uv run pytest
make build_wheel   # uv build
```

Run a single test file or test by name:
```bash
uv run pytest tests/data_classes/test_point.py
uv run pytest -k "test_rank"
```

## Architecture

Optilab is a black-box optimization framework. Results are saved as Python pickles and analysed via the CLI (`optilab <pickle_path> [options]`).

### Data layer — `data_classes/`

All data classes are **Pydantic `BaseModel`** subclasses (not standard dataclasses). Always use keyword arguments when constructing them.

- **`Point(x, y, is_evaluated)`** — a single search-space point. `x: np.ndarray | None` is coerced to `float64` via a field validator. Equality is based on `x` only.
- **`PointList(points)`** — a list of Points with list-like dunder methods (`__iter__`, `__len__`, `__getitem__`/slice, `rank()`, `best()`, etc.). Used both as an optimization log and as a surrogate training set.
- **`Bounds(lower, upper)`** — search space bounds with `reflect`/`wrap`/`project` handlers and random sampling helpers.
- **`OptimizationRun`** — the serialised result of one experiment: metadata + logs (`list[PointList]`).

### Function layer — `functions/`

- **`ObjectiveFunction`** — base class. `__call__` validates dimensionality and increments `num_calls`; subclasses must call `super().__call__()` and add `assert point.x is not None` before using `point.x`.
- Concrete functions live in `unimodal/`, `multimodal/`, and `benchmarks/` (opfunu wrappers).
- **`SurrogateObjectiveFunction`** (`functions/surrogate/`) — extends `ObjectiveFunction` with a `train(PointList)` method and `is_ready` flag. Implementations: KNN (FAISS), locally-weighted polynomial regression, MLP, XGBoost, plain polynomial regression.

### Optimizer layer — `optimizers/`

- **`Optimizer`** — base class with `optimize()` (single run, returns `PointList`) and `run_optimization()` (runs in parallel via `multiprocessing.Pool`, returns `OptimizationRun`).
- All concrete optimizers wrap CMA-ES from the `cma` library. The hierarchy is: `Optimizer → CmaEs → IpopCmaEs`, with surrogate variants layering a metamodel on top.

### Metamodel layer — `metamodels/`

Metamodels sit between the optimizer and the objective function. They manage a training set and decide which candidate points to evaluate with the expensive objective function vs. the cheap surrogate.

- **`ApproximateRankingMetamodel`** — the main metamodel. Adaptively evaluates only enough candidates to stabilise the top-`mu` ranking, growing or shrinking the evaluation budget each generation.
- **`TopHalfMetamodel`** — evaluates only the top half of surrogate-ranked candidates.
- **`IepolationSurrogate`** — uses convex hull interpolation to decide whether to trust the surrogate.

### Surrogate ↔ Metamodel wiring

Optimizers that use surrogates (e.g. `LmmCmaEs`) construct a `SurrogateObjectiveFunction` and an `ApproximateRankingMetamodel`, then call `metamodel.adapt(candidates)` each generation instead of evaluating candidates directly. The metamodel's `train_set` accumulates evaluated points and retrains the surrogate after each evaluation batch.

### Plotting & utils — `plotting/`, `utils/`

Plotting functions (`plot_convergence_curve`, `plot_ecdf_curves`, `plot_box_plot`) accept `savepath: str | Path | None`. Statistical utilities (`mann_whitney_u_test_grid`, `aggregate_pvalues`, `aggregate_stats`) operate on lists of `OptimizationRun` results loaded from pickles.

### CLI — `cli.py` / `__main__.py`

`__main__.py` is only argument parsing; all logic lives in `OptilabCLI` in `cli.py`. The class holds accumulator DataFrames across files and delegates per-file work to `_analyze_file`, `_plot`, `_report_stats`, `_test_y`, `_test_evals`, and `_finalize`.

## Type checking notes

- `ty` is the type checker (`uvx ty check src tests`). FAISS stub signatures are wrong (extra `n` parameter); suppress with bare `# type: ignore` on those lines only.
- `PointList.__iter__` overrides `BaseModel.__iter__` with an incompatible return type; suppressed with `# type: ignore` on that line.
- pandas `columns=list(...)` calls require `# type: ignore` due to stub incompatibility.
