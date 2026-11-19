# CI helper scripts

Scripts used by `.github/workflows/cobalt_ci.yml`. Maintainer: Yi-Cheng Teng (GitHub: yichengt900).

## 1D answer report

For every pull request, the CI runs the 1D COBALT case (BATS column, 48 h, from CEFI-regional-MOM6 `main`) twice: once
with the PR's ocean_BGC (job `run-obgc_1d-ci`) and once with its base commit (job `base-1d`). The job `answer-report`
then compares the two runs and writes the result to the job summary page. A warning is shown on the PR's checks page
if the answers changed or the two runs can't be compared.

The report has four lines:

1. **Answers:** the data of every variable present in both runs is compared bitwise. This covers the restart files
   (all tracers), `ocean.stats` and `ocean.stats.nc`, and the diag files. A restart variable or file that the PR
   removed also counts.
2. **Outputs added or removed:** files or variables that only one run has, for example a new diagnostic.
3. **Parameters / diag list:** changes in `*_parameter_doc.*` and `available_diags.*`, for example a new parameter.
4. **Metadata only:** attribute changes, for example units.

It also checks that both runs used the same CEFI commit, submodule pins and compiler and library versions (`env.txt`).
If they didn't, the comparison is marked not valid.

**What it doesn't cover:** code that the 1D case doesn't run. That includes 3D configurations, land points, other
processor layouts, options that are off by default, and long runs. Those still need the Gaea tests.

**Report-only:** the report never fails the build. Today the text check of `ocean.stats` against
`.github/ref/OM4.single_column.COBALT.p4/docker-linux-gnu/ocean.stats` still decides whether the 1D answers may change.

## Files

| File | Purpose |
|---|---|
| `compare_1d.py` | compares two output folders and writes the report (`--help` for options) |
| `collect_1d_outputs.sh` | copies the outputs of a 1D run into one folder and writes `env.txt` |
| `check_all.sh` | runs every check for these scripts |
| `tests/test_compare_1d.py` | unit tests for `compare_1d.py` |
| `pyproject.toml`, `uv.lock` | pinned Python packages (netCDF4, numpy; ruff, pytest, actionlint, shellcheck for checks) |

## Run it yourself

```console
# Compare two output folders (for example two RESTART folders from your own runs)
uv run --project .github/scripts python .github/scripts/compare_1d.py BASE_DIR PR_DIR

# Run all checks (lint, format, unit tests, workflow and shell checks)
.github/scripts/check_all.sh
```

Without uv, any Python 3.11+ with netCDF4 and numpy works:
`python .github/scripts/compare_1d.py BASE_DIR PR_DIR` and
`PYTHON=python .github/scripts/check_all.sh` (unit tests only).

## When you change these scripts

- Add a unit test for every bug you fix, before the fix.
- Run `.github/scripts/check_all.sh`.
- After changing Python packages, run `uv lock --project .github/scripts`.
