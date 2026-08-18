# Reproducible Python environment

The repository pins Python to `>=3.12,<3.13` and keeps the resolver output in
the persistent `uv.lock`. The environment itself and every dependency/model
cache are intentionally ephemeral: they live only under `/tmp` or
`/home/codex/.cache`.

## Frozen packages

Build and resolution:

- setuptools 80.9.0 is the exact build backend and constrains Torch's
  transitive requirement; it is not declared as a DCDN runtime dependency.

Runtime:

- PyTorch 2.8.0 from the explicit official CUDA 12.8 wheel index;
- NumPy 2.1.2;
- PyYAML 6.0.3;
- psutil 7.1.0.

Development:

- pytest 9.1.1;
- Hypothesis 6.161.6;
- Ruff 0.12.12.

The optional `data` extra pins PyArrow 25.0.0, Polars 1.43.0, SciPy 1.18.0,
scikit-learn 1.9.0, and implicit 0.7.3. It is locked now but should be installed
only when a frozen public-data preprocessing step needs it. No baseline
repository or dataset is installed by the environment bootstrap.

Polars 1.43.0 depends on the yanked `polars-runtime-32==1.43.0` artifact. uv
rejects a yanked transitive artifact unless it is requested exactly, so the
data extra records that runtime pin directly. This retains the requested
Polars version; it does not substitute a different release. The initial lock
attempt failed as unsatisfiable on that exact transitive artifact; the direct
pin is therefore required rather than incidental. For this Linux x86_64
environment, `uv.lock` records the runtime wheel SHA-256 as
`3ecc8feaf04de5989a29245885921db612ef1ce9065e5cb6ec37495acfa55bba`
(and separately records hashes for every other locked platform artifact).

## Reproduction commands

From the repository root:

```bash
source scripts/env.sh
uv sync --frozen
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
python scripts/preflight.py --eta-hours 1
pytest
ruff check scripts tests
```

Install the data stack only for preprocessing:

```bash
source scripts/env.sh
uv sync --frozen --extra data
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
python scripts/preflight.py --require-data-extra --eta-hours 1
```

`uv lock --check --offline` verifies that `pyproject.toml` and `uv.lock` agree
without contacting a package index. The PyTorch source is constrained with an
explicit `tool.uv.sources` entry, so a CUDA wheel cannot silently resolve from
an unrelated index.

## Local-only safety boundary

`scripts/cost_guard.py` evaluates the immutable local ledger epoch, frozen
hourly rate, storage reserve, warning/hard limits, effective stop, and final
eight-hour reserve. `scripts/storage_guard.py` measures project-owned durable
bytes and temporary-overlay pressure. `scripts/preflight.py` combines those
checks with pinned versions, cache placement, telemetry settings, and the
two-A100/CUDA runtime contract.

These scripts make no network request, inspect no RunPod credential or billing
state, and perform no Pod, network-volume, or process lifecycle action. A
passing guard authorizes only the separately requested local command; it does
not broaden the experiment protocol or unlock test data.
