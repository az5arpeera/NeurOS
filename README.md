# NeurOS

**NeurOS (neural operating system)** is a hardware-agnostic software layer that lets any EEG headset and any application decode user intent **without a calibration session for new users**.

**Deadline:** YC Winter 2027 application, **1 Nov 2026**.

| Where | What |
|---|---|
| [`knowledge-base/`](knowledge-base/README.md) | Start here: research, datasets, decisions, the [MVP architecture](knowledge-base/architecture/mvp-architecture.md) (the coding guideline) and the [plan](knowledge-base/plan/road-to-nov-1.md) |
| [`src/neuros/`](src/neuros/__init__.py) | The Python package (architecture §9). So far: the data layer (registry, harmonisation, validation, montage, vocabulary, splits) |
| [`configs/`](configs/) | The data config (the signed-off signal contract) and the **locked test set** |
| [`tests/`](tests/) | Unit and property tests on synthetic EEG; `data`-marked tests run against downloaded datasets |

## Developer quickstart

Needs Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) (`python -m pip install uv`).

```bash
uv sync --extra dev                       # create .venv from the lockfile
uv run pytest                             # unit + property tests (no network)
uv run neuros datasets --check            # registry, cross-checked with MOABB's metadata
uv run neuros montage                     # C-core (26 channels) and headset coverage
uv run neuros split                       # verify the locked test set (D-026)
uv run neuros --config configs/data.yaml download --dataset BNCI2014_001
uv run neuros --config configs/data.yaml validate --dataset BNCI2014_001
```

- Raw data goes to `data/raw/` with a SHA-256 manifest in `data/manifests/`; reports go to `results/`. All three folders are git-ignored. Set `NEUROS_DATA_DIR` to put the data elsewhere (e.g. on the cluster).
- Run the downloaded-data tests with `NEUROS_DATA_TESTS=1 uv run pytest -m data`.
- Optional: `uv run pre-commit install` runs format, lint and types before each commit.

## House rules for code

The founder's rule, verbatim: *"clean, heavily commented, production-grade, and include thorough error handling and data boundary checks."* In practice (architecture §3):

1. **Train–serve parity:** one causal preprocessing implementation for training and live data.
2. **Person-level isolation:** splits by person; locked test people never touch training or tuning.
3. **Typed, validated boundaries:** every array carries its metadata; failures raise typed `NeurOS*Error`s with context.
4. **Fail safe, never silent:** the live engine turns any invalid input into `HOLD` with a reason code.
5. **Reproducible numbers:** results carry git SHA, config hash, data hash and seed.
