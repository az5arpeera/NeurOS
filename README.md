# NeurOS

**NeurOS (neural operating system)** is a hardware-agnostic software layer that lets any EEG headset and any application decode user intent **without a calibration session for new users**.

**Deadline:** YC Winter 2027 application, **1 Nov 2026**.

| Where | What |
|---|---|
| [`knowledge-base/`](knowledge-base/README.md) | Start here: research, datasets, decisions, the [MVP architecture](knowledge-base/architecture/mvp-architecture.md) (the coding guideline) and the [plan](knowledge-base/plan/road-to-nov-1.md) |
| [`src/neuros/`](src/neuros/__init__.py) | The Python package (architecture §9): the data layer (registry, harmonisation, validation, montage, vocabulary, splits), the causal signal chain (`signal/`), Euclidean Alignment (`align/`), the EEGNet encoder (`models/`) and its trainer (`train/`), experiments (`eval/`), and the **live engine** (`engine/`: sources, quality gate, online alignment, policy, model bundle, WebSocket server, simulator) with its Python client (`client.py`) |
| [`clients/js/`](clients/js/) | The JavaScript SDK (`neuros-client.js`) and the live demo page (`demo.html`, served by the engine) |
| [`configs/`](configs/) | The data config (the signed-off signal contract), one YAML per experiment (`e0.yaml`, ...) and the **locked test set** |
| [`tests/`](tests/) | Unit and property tests on synthetic EEG; `data`-marked tests run against downloaded datasets |

## Developer quickstart

Needs Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) (`python -m pip install uv`).

```bash
uv sync --extra dev --extra ml            # create .venv from the lockfile (ml = PyTorch, CPU)
uv run pytest                             # unit + property tests (no network)
uv run neuros datasets --check            # registry, cross-checked with MOABB's metadata
uv run neuros montage                     # C-core (26 channels) and headset coverage
uv run neuros split                       # verify the locked test set (D-026)
uv run neuros --config configs/data.yaml download --dataset BNCI2014_001
uv run neuros --config configs/data.yaml validate --dataset BNCI2014_001
uv run neuros filters                     # the causal filter's delay and gains at every native rate
uv run neuros --config configs/data.yaml baseline --dataset BNCI2014_001   # Riemannian baselines
uv run neuros --config configs/data.yaml loso --experiment configs/e0.yaml  # E0: EEGNet, LOSO
```

- Raw data goes to `data/raw/` with a SHA-256 manifest in `data/manifests/`; reports go to `results/`. All three folders are git-ignored. Set `NEUROS_DATA_DIR` to put the data elsewhere (e.g. on the cluster).
- Run the downloaded-data tests with `NEUROS_DATA_TESTS=1 uv run pytest -m data`.
- `neuros loso` trains one fold per person in parallel processes (default: CPUs ÷ 2 workers × 2 threads). E0 takes about 50 minutes per alignment on the dev laptop. Add `--max-epochs 2 --subjects 1 2 3` for a quick smoke test.
- Offline windows start after the class has left the screen (D-030). `neuros datasets` shows each dataset's window and flags the ones where the class stays on screen; `baseline --window START END` fixes a window for diagnostics such as the cue-only control (`--window 0 0.5`).
- Optional: `uv run pre-commit install` runs format, lint and types before each commit.

## The live engine (headset → intents)

Needs the engine extra: `uv sync --inexact --extra engine` (pylsl, websockets, brainflow; `--inexact` keeps the dev/ml packages).

```bash
# 1. A model bundle: the shipping model + what the engine needs (GPU recommended)
uv run neuros --config configs/data.yaml bundle --experiment configs/e2.yaml --out bundles/mvp-v0 --device cuda

# 2a. Rehearse with a virtual headset: a dataset person the bundle never trained on, in real time
uv run neuros --config configs/data.yaml simulate --bundle bundles/mvp-v0 \
    --dataset GuttmannFlury2025 --subject 3 --speed 1 --serve --wait-for-app
#     then open http://127.0.0.1:8765/ (the demo page)

# 2b. The real headset (OpenBCI Ganglion over BrainFlow; channels named by its profile)
uv run neuros --config configs/data.yaml run --bundle bundles/mvp-v0 \
    --brainflow GANGLION_BOARD --serial-port COM3 --headset ganglion
#     or any LSL EEG stream: --lsl [NAME]
```

Apps receive `neuros.intent.v0` events (architecture §7.3) on `ws://127.0.0.1:8765`; the engine binds to this machine only.

```js
import { NeurOSClient } from "./clients/js/neuros-client.js";
new NeurOSClient("ws://127.0.0.1:8765").on("intent", (e) => console.log(e.command)).connect();
```

```python
from neuros.client import intents
async for event in intents():            # inside an async function
    print(event["type"], event.get("command"))
```

- Decisions arrive 8 times a second. `HOLD` comes with a reason (`warmup`, `low_confidence`, `unrecognized`, `flat`, `saturated`, `motion`, `signal_gap`); the engine never guesses.
- `neuros simulate` refuses locked-test people, and people who trained the bundle (their accuracy would be inflated) unless `--allow-seen`. `--out FILE` writes the replay's score against the ground truth.
- `neuros simulate --lsl-out NAME` publishes a session as an LSL stream for rehearsing `neuros run --lsl NAME`. LSL opens network sockets (Windows may ask for firewall permission); the LSL test runs only with `NEUROS_LSL_TESTS=1`.

## House rules for code

The founder's rule, verbatim: *"clean, heavily commented, production-grade, and include thorough error handling and data boundary checks."* In practice (architecture §3):

1. **Train–serve parity:** one causal preprocessing implementation for training and live data.
2. **Person-level isolation:** splits by person; locked test people never touch training or tuning.
3. **Typed, validated boundaries:** every array carries its metadata; failures raise typed `NeurOS*Error`s with context.
4. **Fail safe, never silent:** the live engine turns any invalid input into `HOLD` with a reason code.
5. **Reproducible numbers:** results carry git SHA, config hash, data hash and seed.
