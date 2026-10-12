"""``neuros``: the developer command line (architecture §7.5).

Day-1 commands (data layer):

    neuros datasets [--check]                     registry table; --check cross-checks MOABB
    neuros montage                                C-core + headset coverage
    neuros split [--write] [--split-version N]    the locked test set (D-026; v2 since D-033)
    neuros download --dataset KEY [--subjects …]  fetch raw data + integrity manifest; datasets
                    [--accept-licence]            whose licence must be accepted need the flag
    neuros validate --dataset KEY [--subjects …]  harmonise + validate, write a JSON report

Day-2 commands (signal chain, alignment, baselines):

    neuros filters                                the causal filter at every native rate in scope
    neuros baseline --dataset KEY [...]           Riemannian baselines (LOSO + session transfer)
    neuros audit --dataset KEY [...]              cue-contamination audit (D-030): when and where
                                                  on the head the class is readable

Day-3 command (deep model; needs PyTorch: ``uv sync --extra dev --extra ml``):

    neuros loso --experiment configs/e0.yaml      EEGNet leave-one-person-out (E0)

E1 (pooled training across datasets; on AWS with ``--device cuda``, D-032):

    neuros e1 --experiment configs/e1.yaml        pooled vs single-dataset vs leave-one-dataset-out,
                                                  with the cue-only control (D-030)

Sprint experiments (each answers a decision with a bar set first, knowledge-base/plan/):

    neuros idle [--bundle bundles/mvp-v0]         E7: false commands at rest and during arithmetic

Every command that loads data refuses people that ``neuros download`` didn't fetch: MOABB would
otherwise download them unasked.

Exit codes: 0 = success, 1 = checks found problems, 2 = a NeurOS error (bad input/config).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neuros import __version__
from neuros.config.schemas import DataConfig, load_data_config
from neuros.errors import NeurOSError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from neuros.config.experiment import LosoExperimentConfig, PooledExperimentConfig
    from neuros.data.cache import RecordingCache
    from neuros.data.recording import Recording
    from neuros.data.registry import DatasetSpec, TaskEpoch
    from neuros.eval.pooled import PooledWindows
    from neuros.signal.windows import WindowSet

log = logging.getLogger("neuros")

# The current locked test set: v3 (D-038) = v2 (D-033: v1 of D-026 plus the cue-free datasets)
# plus Liu2024. tests/test_splits.py checks this is the current version's file.
DEFAULT_SPLIT_PATH = Path("configs/splits/locked_test_v3.yaml")
# A5's datasets: the trainable cue-free ones (knowledge-base/plan/a5-frequency-alignment.md).
A5_DATASETS = ("Yang2025", "BNCI2014_001")


# ---------------------------------------------------------------------- helpers
def _config(args: argparse.Namespace) -> DataConfig:
    return load_data_config(args.config)


def _table(rows: list[list[str]], header: list[str]) -> str:
    """Render a fixed-width text table (header, rule, rows)."""
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows, strict=True)]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()

    return "\n".join([line(header), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def _subjects(args: argparse.Namespace, all_subjects: Sequence[int]) -> list[int]:
    return list(args.subjects) if args.subjects else list(all_subjects)


def _finished(provenance: dict[str, Any]) -> dict[str, Any]:
    """A start-of-run provenance block plus the time the result was written."""
    import datetime as dt

    return {**provenance, "finished_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}


# ---------------------------------------------------------------------- commands
def cmd_datasets(args: argparse.Namespace) -> int:
    from neuros.data.registry import DATASETS, check_against_moabb, check_registry, task_epoch
    from neuros.data.vocabulary import default_vocabulary

    vocab = default_vocabulary()
    contract = _config(args).signal
    rows = []
    for spec in DATASETS.values():
        labels = sorted(set(vocab.event_maps.get(spec.key, {}).values()))
        # The offline window each dataset gets under the contract (D-030), and whether the
        # class is still on screen during it.
        window = "-"
        if spec.class_on_screen_s is not None:
            try:
                epoch = task_epoch(spec, contract)
                window = f"[{epoch.start_s:g}, {epoch.end_s:g}]" + ("" if epoch.cue_free else " !")
            except NeurOSError:
                window = "INVALID"  # explained by --check
        rows.append(
            [
                spec.key,
                spec.role,
                spec.use,
                str(spec.n_subjects),
                str(len(spec.usable_eeg_channels)),
                f"{spec.native_sfreq:g}",
                ",".join(labels),
                window,
                ",".join(spec.unverified) or "-",
            ]
        )
    header = ["dataset", "role", "use", "people", "EEG ch", "Hz", "labels", "window s"]
    print(_table(rows, [*header, "unverified"]))
    print("\n'!' = the class is on screen during the window: report the cue-only control (D-030)")
    print(
        "use: train = may train a model we ship (and test); test = testing only; "
        "diagnostic = never trains (D-033)"
    )
    if not args.check:
        return 0
    problems: list[str] = []
    try:
        check_registry(vocab, _config(args).signal)
    except NeurOSError as exc:
        problems.append(str(exc))
    for spec in DATASETS.values():
        if spec.moabb_class is not None:
            problems.extend(check_against_moabb(spec, vocab))
    print("\nregistry check:", "OK" if not problems else "")
    for problem in problems:
        print("  -", problem)
    return 1 if problems else 0


def cmd_montage(args: argparse.Namespace) -> int:
    from neuros.data.montage import PACKAGED_PROFILES, compute_c_core, load_c_core, load_profile
    from neuros.data.registry import DATASETS, keys_with_role

    computed = compute_c_core([DATASETS[k] for k in keys_with_role("core")])
    committed = load_c_core()
    print(f"C-core ({len(committed)} channels): {', '.join(committed)}")
    status = 0
    if computed != committed:
        print(f"MISMATCH: recomputed C-core is {computed}")
        status = 1
    rows = []
    for name in PACKAGED_PROFILES:
        profile = load_profile(name)
        present, _ = profile.coverage(committed)
        rows.append([f"headset:{name}", str(len(present)), ",".join(present)])
    for spec in DATASETS.values():
        # Usable channels only: a publisher-reported fault (excluded_channels) is never an input.
        present = tuple(ch for ch in committed if ch in spec.usable_eeg_channels)
        rows.append([f"dataset:{spec.key}", str(len(present)), ",".join(present)])
    print()
    print(_table(rows, ["layout", "in C-core", "channels present"]))
    return status


def cmd_split(args: argparse.Namespace) -> int:
    from neuros.eval.splits import (
        LOCKED_TEST_DECISIONS,
        LOCKED_TEST_FILES,
        LOCKED_TEST_VERSION,
        load_locked_test_set,
        make_locked_test_set,
        save_locked_test_set,
    )

    version = LOCKED_TEST_VERSION if args.split_version is None else args.split_version
    computed = make_locked_test_set(version=version)  # raises for an unknown version
    path = Path(args.path) if args.path else Path(LOCKED_TEST_FILES[version])
    print(f"locked test set v{version} ({LOCKED_TEST_DECISIONS[version]})")
    if args.write:
        # Each version after the first extends the one before; the save checks it's unchanged.
        extends = None
        if version > 1:
            extends = Path(args.extends) if args.extends else Path(LOCKED_TEST_FILES[version - 1])
        save_locked_test_set(computed, path, extends=extends)
        print(f"wrote {path} (sha256 {computed.digest[:16]}…)")
    rows = [[k, str(len(v)), " ".join(map(str, v))] for k, v in computed.test.items()]
    print(_table(rows, ["dataset", "test people", "subjects"]))
    print(f"total: {sum(len(v) for v in computed.test.values())} people")
    if path.is_file():
        stored = load_locked_test_set(path)  # also checks the version it extends
        same = dict(stored.test) == dict(computed.test) and stored.digest == computed.digest
        print(f"{path}: integrity OK; {'matches' if same else 'DIFFERS FROM'} the recomputation")
        return 0 if same else 1
    print(f"{path}: not written yet (use --write once)")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from neuros.data.download import download, verify_manifest
    from neuros.data.registry import get_spec

    config = _config(args)
    spec = get_spec(args.dataset)
    if args.verify:
        problems = verify_manifest(spec, config.data_dir)
        print("manifest intact" if not problems else "\n".join(problems))
        return 1 if problems else 0
    manifest = download(
        spec, _subjects(args, spec.subjects), config.data_dir, accept_licence=args.accept_licence
    )
    print(
        f"{spec.key}: {manifest['n_files']} files, {manifest['total_bytes'] / 1e9:.2f} GB "
        f"recorded in {config.data_dir / 'manifests' / (spec.key + '.json')}"
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from neuros.data.download import loading_dataset, prepare_mne_environment, require_downloaded
    from neuros.data.harmonize import harmonize_moabb_subject
    from neuros.data.registry import get_spec
    from neuros.data.validate import validate_recording
    from neuros.data.vocabulary import default_vocabulary
    from neuros.provenance import stamp

    config = _config(args)
    spec = get_spec(args.dataset)
    provenance = stamp(config, dataset=spec.key)  # at the start: the code that runs
    vocab = default_vocabulary()
    prepare_mne_environment(config.data_dir)
    # EEGMAT isn't a MOABB dataset: it has its own loader (neuros.data.eegmat).
    dataset = loading_dataset(spec, config.data_dir) if spec.moabb_class is not None else None
    results: list[dict[str, object]] = []
    rows: list[list[str]] = []
    pool = spec.subjects if args.include_excluded else spec.usable_subjects
    if spec.excluded_subjects and not args.include_excluded and not args.subjects:
        print(
            f"skipping excluded subjects {sorted(spec.excluded_subjects)} "
            "(see the registry for reasons; --include-excluded checks them anyway)\n"
        )
    # Loading would make MOABB download anything missing, unasked: refuse instead.
    require_downloaded(spec, _subjects(args, pool), config.data_dir)
    for subject in _subjects(args, pool):
        try:
            if dataset is None:
                from neuros.data.eegmat import harmonize_eegmat_subject

                recordings = harmonize_eegmat_subject(spec, subject, vocab, config.data_dir)
            else:
                recordings = harmonize_moabb_subject(spec, subject, vocab, dataset=dataset)
        except NeurOSError as exc:
            results.append(
                {"dataset": spec.key, "subject": subject, "ok": False, "load_error": str(exc)}
            )
            rows.append([str(subject), "-", "-", "LOAD ERROR", "-", "-", "-", "-", str(exc)[:80]])
            continue
        for rec in recordings:
            report = validate_recording(rec, spec, vocab, config.signal, config.validation)
            results.append(report.to_dict())
            rows.append(
                [
                    str(subject),
                    rec.session,
                    rec.run,
                    "ok" if report.ok else "FAIL",
                    f"{rec.sfreq:g}",
                    f"{rec.duration_s:.0f}",
                    " ".join(f"{k}={v}" for k, v in sorted(report.label_counts.items())),
                    f"{report.windows_flagged}/{report.windows_total}",
                    "; ".join(i.code for i in report.issues) or "-",
                ]
            )
    print(
        _table(
            rows, ["subj", "session", "run", "status", "Hz", "sec", "events", "flagged", "issues"]
        )
    )
    failures = [r for r in results if not r.get("ok")]
    out = config.results_dir / "validation" / f"{spec.key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"provenance": _finished(provenance), "recordings": results}, indent=1),
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n{len(results)} recordings, {len(failures)} failing; report: {out}")
    return 1 if failures else 0


def _native_rates() -> dict[float, list[str]]:
    """Every native sampling rate in scope (registry datasets + headset profiles) → users."""
    from neuros.data.montage import PACKAGED_PROFILES, load_profile
    from neuros.data.registry import DATASETS

    rates: dict[float, list[str]] = {}
    for spec in DATASETS.values():
        rates.setdefault(spec.native_sfreq, []).append(spec.key)
    for name in PACKAGED_PROFILES:
        rates.setdefault(load_profile(name).native_sfreq, []).append(f"headset:{name}")
    return dict(sorted(rates.items()))


def _format_db(gain_db: float) -> str:
    """Gain for display: a notch centre is a true zero (-inf dB), shown as '<-100'."""
    if gain_db < -100:
        return "<-100"
    return f"{round(gain_db) + 0:+d}"  # "+ 0" turns -0 into 0


def cmd_filters(args: argparse.Namespace) -> int:
    import numpy as np

    from neuros.signal.filters import (
        TRANSIENT_S,
        design_filter,
        frequency_response,
        group_delay_s,
        settle_time_s,
    )

    contract = _config(args).signal
    rows = []
    status = 0
    for rate, users in _native_rates().items():
        design = design_filter(rate, contract)
        probes = [f for f in (10.0, 50.0, 60.0, contract.sfreq / 2) if f < rate / 2]
        gains = 20 * np.log10(np.maximum(np.abs(frequency_response(design, probes)), 1e-12))
        gain_text = " ".join(f"{f:g}Hz:{_format_db(g)}" for f, g in zip(probes, gains, strict=True))
        settle = settle_time_s(design)
        if settle >= TRANSIENT_S:
            status = 1
        rows.append(
            [
                f"{rate:g}",
                f"{group_delay_s(design, 10.0) * 1e3:.1f}",
                f"{group_delay_s(design, 6.0) * 1e3:.1f}",
                f"{settle:.2f}",
                gain_text,
                ",".join(users),
            ]
        )
    print(
        f"band {contract.band_hz[0]:g}-{contract.band_hz[1]:g} Hz, Butterworth order "
        f"{contract.filter_order}, notches {list(contract.notch_hz)} (Q={contract.notch_q:g}); "
        f"windows within {TRANSIENT_S} s of a (re)start are flagged\n"
    )
    print(
        _table(
            rows,
            ["Hz", "delay@10Hz ms", "delay@6Hz ms", "settle99 s", "gain dB", "used by"],
        )
    )
    if status:
        print(f"\nPROBLEM: a settling time reaches TRANSIENT_S ({TRANSIENT_S} s)")
    return status


def _dev_subjects(
    spec: DatasetSpec, subjects_arg: Sequence[int] | None, split_path: str, purpose: str
) -> list[int]:
    """The subjects a development run may use: those asked for, or every development person.

    Person-level isolation (principle 2): the locked test set is loaded and checked first, so a
    locked test person can never enter a development run, even when named explicitly.

    Raises:
        NeurOSSplitError: if any requested subject is in the locked test set.
    """
    from neuros.data.recording import person_id
    from neuros.eval.splits import load_locked_test_set

    locked = load_locked_test_set(Path(split_path))
    subjects = list(subjects_arg) if subjects_arg else list(locked.dev_subjects(spec.key))
    locked.assert_disjoint([person_id(spec.key, s) for s in subjects], purpose)
    return subjects


def _recording_cache(config: DataConfig, args: argparse.Namespace) -> RecordingCache | None:
    """The preprocessed-recording cache under ``cache_dir`` (§4.5), unless ``--no-cache``."""
    from neuros.data.cache import RecordingCache

    return None if getattr(args, "no_cache", False) else RecordingCache(config.cache_dir)


def _warn_if_dirty(provenance: dict[str, Any]) -> None:
    """Say so when uncommitted code runs: the result can't be reproduced from git."""
    if provenance.get("git_dirty_code") == "true":
        print("warning: uncommitted code changes; this result won't be reproducible from git")


def _dev_windows(
    config: DataConfig,
    dataset_key: str,
    subjects_arg: Sequence[int] | None,
    labels: Sequence[str],
    channels: str,
    split_path: str,
    purpose: str,
    cache: RecordingCache | None = None,
) -> tuple[WindowSet, list[dict[str, object]], list[int], TaskEpoch]:
    """Task windows of a dataset's development people, printed per subject as they load.

    Person-level isolation (principle 2): the locked test set is loaded and checked first, so
    a locked test person can never enter a development run, even when named explicitly. With
    ``cache``, preprocessed recordings are read from (or written to) the disk cache.

    Returns:
        ``(windows, per-subject epoch summaries, subjects, the epoch used)``.
    """
    from neuros.data.download import loading_dataset, prepare_mne_environment, require_downloaded
    from neuros.data.montage import load_c_core, region_channels
    from neuros.data.prepare import subject_epochs
    from neuros.data.registry import get_spec, task_epoch
    from neuros.data.vocabulary import default_vocabulary
    from neuros.errors import NeurOSConfigError
    from neuros.signal.windows import WindowSet

    spec = get_spec(dataset_key)
    # Resolved first: a dataset that can't host the epoch fails before anything loads.
    epoch = task_epoch(spec, config.signal)
    print(f"{spec.key}: windows {epoch.describe()}", flush=True)
    vocab = default_vocabulary()
    subjects = _dev_subjects(spec, subjects_arg, split_path, purpose)
    require_downloaded(spec, subjects, config.data_dir)  # never let MOABB download unasked
    prepare_mne_environment(config.data_dir)
    dataset = loading_dataset(spec, config.data_dir)

    sets = []
    summaries: list[dict[str, object]] = []
    for subject in subjects:
        windows, summary = subject_epochs(
            spec, subject, vocab, config, labels, dataset=dataset, cache=cache
        )
        sets.append(windows)
        summaries.append(summary.to_dict())
        counts = " ".join(f"{k}={v}" for k, v in sorted(summary.label_counts.items()))
        print(
            f"{summary.person}: {summary.n_recordings} recordings, {counts}, "
            f"flagged {summary.flagged_extreme} extreme / {summary.flagged_transient} transient, "
            f"skipped {summary.skipped_outside}, cache {summary.cache}",
            flush=True,
        )
    windows = WindowSet.concat(sets)
    if channels == "c-core":
        windows = windows.pick_channels([ch for ch in load_c_core() if ch in windows.ch_names])
    elif channels == "motor":
        # D-031: the intent head reads the motor strip (the audit's "motor" region).
        windows = windows.pick_channels(list(region_channels("motor", windows.ch_names)))
    elif channels != "all":
        raise NeurOSConfigError(f"unknown channel set {channels!r} (all, c-core, motor)")
    return windows, summaries, subjects, epoch


def _window_overrides(window: Sequence[float]) -> dict[str, object]:
    """Signal overrides that move the offline epoch to ``[start, end]`` s from the cue.

    The window length follows the epoch, so one epoch stays exactly one model window. The
    caller validates the result as a whole contract (``apply_signal_overrides``): start < end,
    finite values, a whole number of samples, and so on.

    Raises:
        NeurOSConfigError: if ``window`` isn't exactly two numbers.
    """
    from neuros.errors import NeurOSConfigError

    if len(window) != 2:
        raise NeurOSConfigError(f"--window takes START END in seconds, got {list(window)}")
    start, end = (float(x) for x in window)
    # Rounded so 3.5 - 1.5 can't come out as 1.9999999999999998 and miss a whole sample.
    return {"epoch_s": [start, end], "window_s": round(end - start, 9)}


def _baseline_output_name(spec_key: str, args: argparse.Namespace, epoch: TaskEpoch) -> str:
    """Result file name: names every setting that changes the numbers, so runs never clobber.

    Every name carries its epoch: ``_aftercue`` for the contract's rule (D-030), or the fixed
    window. Files without one are from before D-030 (4 Oct) and used [0.5, 2.5] s; nothing
    writes such a name any more, so they're never overwritten.
    """
    name = f"{spec_key}_{'-'.join(args.labels)}_{args.channels}_{args.reference}"
    if args.alignment_unit == "session":
        name += "_session"
    if epoch.rule == "after_cue":
        name += "_aftercue"
    else:
        name += f"_window{epoch.start_s:g}-{epoch.end_s:g}s"
    if getattr(args, "band", None):
        low, high = args.band
        name += f"_band{low:g}-{high:g}"
    if args.subjects:
        name += "_subj" + "-".join(str(s) for s in args.subjects)
    return name + ".json"


def cmd_baseline(args: argparse.Namespace) -> int:
    from neuros.config.experiment import apply_signal_overrides
    from neuros.data.registry import get_spec
    from neuros.errors import NeurOSConfigError
    from neuros.eval.baselines import run_riemannian_baselines
    from neuros.provenance import stamp

    if args.workers < 1:
        raise NeurOSConfigError("--workers must be >= 1")
    config = _config(args)
    overrides: dict[str, object] = {}
    if args.window is not None:
        # A diagnostic window (e.g. one that starts after the on-screen cue has gone).
        overrides.update(_window_overrides(args.window))
    if args.band is not None:
        # E0b's band comparison (4-30 vs 8-30 vs 4-40 Hz); validated with the whole contract.
        overrides["band_hz"] = [float(edge) for edge in args.band]
    if overrides:
        # Applied before the stamp so the recorded config hash describes the windows used.
        config = config.model_copy(
            update={"signal": apply_signal_overrides(config.signal, overrides)}
        )
    spec = get_spec(args.dataset)
    spec_key = spec.key
    provenance = stamp(config, dataset=spec_key)  # at the start: the code that runs
    _warn_if_dirty(provenance)
    windows, summaries, subjects, epoch = _dev_windows(
        config,
        spec_key,
        args.subjects,
        args.labels,
        args.channels,
        args.split,
        "a baseline run",
        cache=_recording_cache(config, args),
    )
    print(
        f"\n{len(windows)} windows, {len(windows.people)} people, {len(windows.ch_names)} "
        f"channels ({args.channels}), {windows.n_samples} samples per window "
        f"{epoch.describe()}, reference {args.reference}, alignment per {args.alignment_unit}\n"
    )
    result = run_riemannian_baselines(
        windows,
        reference=args.reference,
        unit=args.alignment_unit,
        workers=args.workers,
        progress=print,
    )

    rows = []
    for entry in result["results"]:
        acc = entry["summary_balanced_accuracy"]
        rows.append(
            [
                entry["name"],
                entry["protocol"],
                f"{acc['mean'] * 100:.1f} ± {acc['std'] * 100:.1f}",
                f"[{acc['ci95_low'] * 100:.1f}, {acc['ci95_high'] * 100:.1f}]",
                f"{acc['median'] * 100:.1f}",
                f"{acc['share_usable'] * 100:.0f}%",
                f"{entry['summary_kappa']['mean']:.3f}",
            ]
        )
    print()
    print(
        _table(
            rows,
            ["baseline", "protocol", "bal. acc %", "95% CI", "median", ">=70%", "kappa"],
        )
    )
    for comparison in result["comparisons"]:
        print(
            f"{comparison['a']} vs {comparison['b']}: {comparison['mean_diff'] * 100:+.1f} "
            f"points, Wilcoxon p = {comparison['p_value']:.3g} (n = {comparison['n_people']})"
        )

    out = config.results_dir / "baselines" / _baseline_output_name(spec_key, args, epoch)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": _finished(provenance),
        "protocol": {
            "dataset": spec_key,
            "labels": list(args.labels),
            "channels": list(windows.ch_names),
            "channel_set": args.channels,
            "reference": args.reference,
            "alignment_unit": args.alignment_unit,
            "band_hz": list(config.signal.band_hz),
            **epoch.to_dict(),
            "class_on_screen_s": spec.class_on_screen_s,
            "cue_margin_s": config.signal.cue_margin_s,
            "window_s": config.signal.window_s,
            "sfreq": config.signal.sfreq,
            "subjects": subjects,
            "split_file": str(args.split),
        },
        "subjects": summaries,
        **result,
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")
    print(f"\nresults: {out}")
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    """The calibration curve (Q28, 10 Oct): a person's own calibrated decoder vs data used.

    Same windows as ``neuros baseline`` (development people, D-030's cue-controlled epoch, the
    same channels, reference and band), so the curve sits next to E0b's and E1's numbers.
    """
    from neuros.config.experiment import apply_signal_overrides
    from neuros.data.registry import get_spec
    from neuros.eval.calibration import calibration_curve, minutes_label
    from neuros.provenance import stamp

    config = _config(args)
    if args.band is not None:
        config = config.model_copy(
            update={
                "signal": apply_signal_overrides(
                    config.signal, {"band_hz": [float(edge) for edge in args.band]}
                )
            }
        )
    spec = get_spec(args.dataset)
    provenance = stamp(config, dataset=spec.key)  # at the start: the code that runs
    _warn_if_dirty(provenance)
    windows, summaries, subjects, epoch = _dev_windows(
        config,
        spec.key,
        args.subjects,
        args.labels,
        args.channels,
        args.split,
        "a calibration-curve run",
        cache=_recording_cache(config, args),
    )
    print(
        f"\n{len(windows)} windows, {len(windows.people)} people, {len(windows.ch_names)} "
        f"channels ({args.channels}), {epoch.describe()}, reference {args.reference}, "
        f"calibrated per {args.block}\n"
    )
    result = calibration_curve(
        windows,
        labels=args.labels,
        sizes=args.sizes,
        block=args.block,
        reference=args.reference,
        progress=print,
    )
    rows = []
    for point in result["curve"]:
        acc = point["summary_balanced_accuracy"]
        rows.append(
            [
                str(point["n_per_class"]),
                minutes_label(point["minutes_median"]),
                str(point["n_people"]),
                f"{acc['mean'] * 100:.1f} ± {acc['std'] * 100:.1f}",
                f"[{acc['ci95_low'] * 100:.1f}, {acc['ci95_high'] * 100:.1f}]",
                f"{acc['share_usable'] * 100:.0f}%",
            ]
        )
    print()
    print(
        _table(
            rows,
            ["per class", "calibration", "people", "bal. acc %", "95% CI", ">=70%"],
        )
    )
    name = f"{spec.key}_{'-'.join(args.labels)}_{args.channels}_{args.reference}_{args.block}"
    if args.band is not None:
        name += f"_band{args.band[0]:g}-{args.band[1]:g}"
    if args.subjects:
        name += "_subj" + "-".join(str(s) for s in args.subjects)
    out = config.results_dir / "calibration" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": _finished(provenance),
        "protocol": {
            "dataset": spec.key,
            "labels": list(args.labels),
            "channels": list(windows.ch_names),
            "channel_set": args.channels,
            "band_hz": list(config.signal.band_hz),
            **epoch.to_dict(),
            "class_on_screen_s": spec.class_on_screen_s,
            "cue_margin_s": config.signal.cue_margin_s,
            "window_s": config.signal.window_s,
            "sfreq": config.signal.sfreq,
            "subjects": subjects,
            "split_file": str(args.split),
        },
        "subjects": summaries,
        **result,
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")
    print(f"\nresults: {out}")
    return 0


def cmd_smr(args: argparse.Namespace) -> int:
    """Resting SMR strength per development person (Q31: can the first minute predict who
    NeurOS will work for?). Unlabeled data only; the pass bar is set in the open questions."""
    from neuros.data.download import loading_dataset, require_downloaded
    from neuros.data.prepare import prepare_subject
    from neuros.data.recording import person_id
    from neuros.data.registry import get_spec
    from neuros.data.vocabulary import default_vocabulary
    from neuros.errors import NeurOSConfigError
    from neuros.eval.smr import FLOOR_BANDS, PEAK_BAND, SMR_CHANNELS, smr_strength
    from neuros.eval.warmup import profile_channels
    from neuros.provenance import stamp

    if not (args.seconds > 0):
        raise NeurOSConfigError(f"--seconds must be > 0, got {args.seconds}")
    config = _config(args)
    provenance = stamp(config)  # at the start: the code that runs
    _warn_if_dirty(provenance)
    cache = _recording_cache(config, args)
    vocab = default_vocabulary()
    layout = _e1_layout("motor")
    people: dict[str, Any] = {}
    for key in args.datasets:
        spec = get_spec(key)
        profile = profile_channels(layout, spec)
        subjects = _dev_subjects(spec, None, args.split, "an SMR-strength run")
        require_downloaded(spec, subjects, config.data_dir)
        dataset = loading_dataset(spec, config.data_dir)
        for subject in subjects:
            if cache is None:
                recordings = prepare_subject(spec, subject, vocab, config, dataset=dataset)
            else:
                recordings, _ = cache.load_or_prepare(
                    spec, subject, vocab, config, prepare=prepare_subject, dataset=dataset
                )
            person = person_id(spec.key, subject)
            first = smr_strength(recordings, spec, profile, seconds=args.seconds)
            whole = smr_strength(recordings, spec, profile, seconds=None)
            people[person] = {"dataset": spec.key, "first": first, "whole_session": whole}
            print(
                f"{person}: {first['smr_db']:.2f} dB in the first {first['seconds_used']:.0f} s, "
                f"{whole['smr_db']:.2f} dB over the first session",
                flush=True,
            )
    out = config.results_dir / "smr" / f"smr_{'-'.join(args.datasets)}_{args.seconds:g}s.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": _finished(provenance),
        "protocol": {
            "datasets": list(args.datasets),
            "seconds": args.seconds,
            "channels": list(SMR_CHANNELS),
            "floor_bands_hz": [list(b) for b in FLOOR_BANDS],
            "peak_band_hz": list(PEAK_BAND),
            "band_hz": list(config.signal.band_hz),
            "split_file": str(args.split),
        },
        "people": people,
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")
    print(f"\nresults: {out}")
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    """Train the deployable model on every development person of an E1-style experiment's
    training datasets and write a model bundle for the engine (architecture §6.5)."""
    from neuros.config.experiment import apply_signal_overrides, load_pooled_experiment_config
    from neuros.data.registry import assert_may_train, get_spec
    from neuros.engine.bundle import train_bundle
    from neuros.errors import NeurOSConfigError
    from neuros.provenance import config_hash, stamp

    experiment = load_pooled_experiment_config(args.experiment)
    out = Path(args.out)
    if (out / "bundle.json").exists():
        raise NeurOSConfigError(f"{out} already holds a bundle; choose another --out")
    if experiment.training_windows is not None and experiment.training_windows.extra_labels:
        raise NeurOSConfigError("bundles are trained on the experiment's labels only")
    # Licences (D-033): a bundle is a model we may ship, so only datasets allowed to train one.
    assert_may_train(experiment.train_datasets, f"model bundle {out}")
    config = _config(args)
    config = config.model_copy(
        update={"signal": apply_signal_overrides(config.signal, experiment.signal)}
    )
    layout = _e1_layout(experiment.channels)
    derived = {
        key: get_spec(key).derived_channels
        for key in experiment.train_datasets
        if get_spec(key).derived_channels
    }
    provenance = stamp(
        config, experiment=experiment.name, experiment_sha256=config_hash(experiment)
    )
    _warn_if_dirty(provenance)
    cache = _recording_cache(config, args)
    task_windows, _ = _e1_windows(
        config, experiment.train_datasets, experiment.labels, args.split, None, cache
    )
    print(f"training the bundle on {sum(len(w.people) for w in task_windows.values())} people")
    bundle = train_bundle(
        task_windows,
        layout,
        classes=experiment.labels,
        model_config=experiment.model,
        train_config=experiment.train,
        contract=config.signal,
        reference=experiment.reference,
        alignment=experiment.alignment,
        unit=experiment.alignment_unit,
        derived=derived,
        validation_fraction=experiment.validation_fraction,
        seed=experiment.seeds[0],
        device=args.device,
        provenance=_finished(provenance) | {"datasets": list(experiment.train_datasets)},
        progress=print,
    )
    bundle.save(out)
    training = bundle.provenance["training"]
    print(
        f"\nbundle: {out} (end epoch {training.get('end_epoch')}, input scale "
        f"{bundle.input_scale:.4g}, {len(bundle.people_covariances)} people in the prior)"
    )
    return 0


def _bundle_people(bundle: Any) -> int | None:
    """How many people a bundle learned from: its training people plus the validation people
    that chose its stopping epoch (mvp-v0: 306 + 33 = 339)."""
    people = len(bundle.provenance.get("train_people", [])) + len(
        bundle.provenance.get("validation_people", [])
    )
    return people or None


def cmd_simulate(args: argparse.Namespace) -> int:
    """Replay a dataset person's raw session through the real engine (§7.4): a virtual
    headset. With ``--serve``, apps receive the events live on ws://127.0.0.1:PORT."""
    import asyncio

    from neuros.data.download import loading_dataset
    from neuros.data.harmonize import harmonize_moabb_subject
    from neuros.data.recording import person_id
    from neuros.data.registry import get_spec, task_epoch
    from neuros.data.vocabulary import default_vocabulary
    from neuros.engine.bundle import ModelBundle
    from neuros.engine.calibration import (
        DEFAULT_PER_CLASS,
        REQUESTS,
        TRIAL_PHASES,
        CalibrationProtocol,
        CalibrationRecipe,
        Calibrator,
    )
    from neuros.engine.events import CalibrationEvent, StatusEvent
    from neuros.engine.pipeline import Engine
    from neuros.engine.policy import PolicySettings
    from neuros.engine.runner import replay, score_replay
    from neuros.engine.server import EventHub, start_server
    from neuros.engine.simulator import Simulator
    from neuros.errors import NeurOSConfigError
    from neuros.eval.calibration import natural_key

    bundle = ModelBundle.load(Path(args.bundle))
    config = _config(args)
    spec = get_spec(args.dataset)
    person = person_id(spec.key, args.subject)
    _dev_subjects(spec, [args.subject], args.split, "a simulation")  # never a locked test person
    seen = set(bundle.provenance.get("train_people", [])) | set(
        bundle.provenance.get("validation_people", [])
    )
    if person in seen and not args.allow_seen:
        raise NeurOSConfigError(
            f"{person} trained this bundle: a replay would overstate accuracy. Pick a person it "
            "never saw (e.g. a test-only dataset), or pass --allow-seen for plumbing checks"
        )
    recordings = harmonize_moabb_subject(
        spec, args.subject, default_vocabulary(), dataset=loading_dataset(spec, config.data_dir)
    )
    sessions = sorted({r.session for r in recordings}, key=natural_key)
    session = args.session if args.session is not None else sessions[0]
    runs = sorted((r for r in recordings if r.session == session), key=lambda r: natural_key(r.run))
    if not runs:
        raise NeurOSConfigError(f"{person} has no session {session!r} (sessions: {sessions})")
    commands = {"left_hand": "LEFT", "right_hand": "RIGHT"}
    policy = PolicySettings(
        commands={c: commands.get(c, c.upper()) for c in bundle.classes},
        threshold=args.threshold,
    )
    epoch = task_epoch(spec, bundle.contract)

    def new_engine() -> Engine:
        return Engine(
            bundle.decoder(),
            bundle.engine_config(runs[0].ch_names, runs[0].sfreq, policy=policy),
        )

    engine = new_engine()
    simulators = []
    t0 = 0.0
    for recording in runs:  # a 5-s pause between runs, as between a real session's blocks
        simulator = Simulator(recording, t0=t0, noise_uv=args.noise_uv, seed=args.subject)
        simulators.append(simulator)
        t0 += simulator.duration_s + 5.0
    print(
        f"{person} session {session}: {len(runs)} run(s), {t0 / 60:.1f} min; headset profile "
        f"{len(engine.profile)} channels {list(engine.profile)}",
        flush=True,
    )
    if args.lsl_out:  # a virtual headset for `neuros run --lsl NAME` (rehearsal)
        from neuros.engine.sources import publish_lsl

        speed = args.speed or 1.0
        print(f"publishing LSL stream {args.lsl_out!r} at {speed:g}x real time (Ctrl+C stops)")
        try:
            publish_lsl(simulators, name=args.lsl_out, speed=speed)
        except KeyboardInterrupt:
            print("stopped")
        return 0

    markers = [m for s in simulators for m in s.markers]
    # The guided calibration (D-040): on a replay it uses the recording's own trials (the person
    # followed the dataset's cues, not ours), from the request on, with the dataset's window.
    recipe = (
        CalibrationRecipe.from_experiment(args.recipe) if args.calibrate or args.serve else None
    )

    def calibrator_for(target: Engine) -> Calibrator | None:
        if recipe is None:
            return None
        return Calibrator(
            target,
            bundle.model,
            recipe=recipe,
            commands=policy.commands,
            protocol=CalibrationProtocol(per_class=args.calibrate or DEFAULT_PER_CLASS),
            markers=markers,
            recorded_window_offset_s=epoch.start_s,
            recorded_task_s=spec.task_s,
            seed=args.subject,
        )

    def show(event: Any) -> None:
        if isinstance(event, StatusEvent):
            print(f"[{event.t:8.1f} s] state {event.previous} -> {event.state} ({event.reason})")
        elif isinstance(event, CalibrationEvent) and event.phase not in TRIAL_PHASES:
            details = f" {dict(event.details)}" if event.details else ""
            print(f"[{event.t:8.1f} s] calibration {event.phase}: {event.message}{details}")

    async def run_arm(
        target: Engine, calibrator: Calibrator | None, *, serve: bool, speed: float | None
    ) -> list[Any]:
        hub = None
        server = None
        telemetry = None
        if serve:
            info = None
            if args.telemetry:  # the stage view: what the engine sees, alongside its decisions
                from neuros.engine.telemetry import Telemetry, stage_info

                telemetry = Telemetry(target)
                info = stage_info(
                    target,
                    source=f"replay of {person} (dataset recording)",
                    model=Path(args.bundle).name,
                    train_people=_bundle_people(bundle),
                )
            hub = EventHub(
                list(policy.commands.values()),
                requests=REQUESTS if calibrator else (),
                info=info,
            )
            server = await start_server(hub, port=args.port)
            print(
                f"serving intents on ws://127.0.0.1:{args.port}; demo page: http://127.0.0.1:{args.port}/"
                + (f"; stage view: http://127.0.0.1:{args.port}/stage" if telemetry else "")
            )
            if args.wait_for_app:
                print("waiting for an app to connect ...", flush=True)
                while hub.n_clients == 0:
                    await asyncio.sleep(0.2)
        try:
            return await replay(
                simulators,
                target,
                hub=hub,
                speed=speed,
                on_event=show,
                calibrator=calibrator,
                calibrate_at_start=bool(args.calibrate),
                telemetry=telemetry,
            )
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()

    def score_of(events: list[Any], scored: list[Any]) -> Any:
        return score_replay(
            events,
            scored,
            policy.commands,
            window_s=bundle.contract.window_s,
            task_start_s=epoch.start_s,
            task_end_s=spec.task_s,
            tolerance_s=bundle.contract.hop_s / 2,  # the decision nearest a one-window epoch
        )

    where = "cue-free task windows" if epoch.cue_free else "task windows (class on screen)"

    def report(label: str, score: Any) -> None:
        print(
            f"\n{label}: {score.n_decisions} decisions, {score.n_commands} commands; holds "
            f"{dict(score.holds)}"
            f"\n  {where}: {score.task_decisions} decisions, {score.task_commands} commands "
            f"({score.commit_rate * 100:.0f}% committed), correct "
            f"{score.task_accuracy * 100:.1f}% (balanced {score.balanced_task_accuracy * 100:.1f}%)"
            f"; leaning {score.balanced_argmax_accuracy * 100:.1f}% (balanced)"
            f"\n  latency: median {score.latency_median_ms:.2f} ms, "
            f"p99 {score.latency_p99_ms:.2f} ms"
        )

    extra: dict[str, Any] = {}
    if args.calibrate:
        # Two arms on the same recording: zero calibration, then quick calibration from the
        # session's first trials. Both are scored on the trials after the calibration ended.
        print("zero-calibration arm (as fast as possible) ...", flush=True)
        zero_events = asyncio.run(run_arm(engine, None, serve=False, speed=None))
        engine = new_engine()
        calibrator = calibrator_for(engine)
        print("quick-calibration arm ...", flush=True)
        events = asyncio.run(run_arm(engine, calibrator, serve=args.serve, speed=args.speed))
        result = calibrator.result if calibrator is not None else None
        scored_from = result.ended_t if result is not None else math.inf
        scored = [m for m in markers if m.t >= scored_from]
        score = score_of(events, scored)
        zero = score_of(zero_events, scored)
        report("zero calibration", zero)
        report("quick calibration", score)
        extra = {
            "zero_calibration": zero.to_dict(),
            "calibration": {
                "ok": bool(result is not None and result.ok),
                "message": result.message if result is not None else "never finished",
                "usable": result.usable if result is not None else {},
                "fit_on_trials": result.fit_on_trials if result is not None else None,
                "started_t": result.started_t if result is not None else None,
                "ended_t": result.ended_t if result is not None else None,
                "per_class": args.calibrate,
                "n_scored_trials": len(scored),
                "recipe": str(args.recipe),
            },
        }
    else:
        events = asyncio.run(
            run_arm(engine, calibrator_for(engine), serve=args.serve, speed=args.speed)
        )
        score = score_of(events, markers)
        report("replay", score)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "person": person,
                    "session": session,
                    "bundle": str(args.bundle),
                    "bundle_provenance": {
                        k: v
                        for k, v in bundle.provenance.items()
                        if k not in ("train_people", "validation_people")
                    },
                    "threshold": args.threshold,
                    "noise_uv": args.noise_uv,
                    "score": score.to_dict(),
                }
                | extra,
                indent=1,
            ),
            encoding="utf-8",
            newline="\n",
        )
        print(f"results: {out}")
    return 0


def _source_channels(stream_names: Sequence[str], profile_name: str | None) -> tuple[str, ...]:
    """The stream's channels as 10-20 names: mapped through a headset profile's device labels
    (``--headset``), or as published when no profile is given.

    Raises:
        NeurOSConfigError: if a profile is given and a stream label isn't one of its devices.
    """
    from neuros.data.montage import load_profile
    from neuros.errors import NeurOSConfigError

    if profile_name is None:
        return tuple(stream_names)
    profile = load_profile(profile_name)
    mapping = {c.device: c.canonical for c in profile.channels}
    unknown = [name for name in stream_names if name not in mapping]
    if unknown:
        raise NeurOSConfigError(
            f"stream channels {unknown} aren't devices of profile {profile.name!r} "
            f"({list(mapping)}): update the profile's device labels"
        )
    return tuple(mapping[name] for name in stream_names)


def cmd_run(args: argparse.Namespace) -> int:
    """The live engine (§7.5 `neuros run`): headset in (LSL or BrainFlow), intents out on a
    local WebSocket, until Ctrl+C (or ``--seconds``)."""
    import asyncio

    from neuros.data.montage import load_profile
    from neuros.engine.bundle import ModelBundle
    from neuros.engine.calibration import (
        REQUESTS,
        TRIAL_PHASES,
        CalibrationProtocol,
        CalibrationRecipe,
        Calibrator,
    )
    from neuros.engine.events import CalibrationEvent, StatusEvent
    from neuros.engine.pipeline import Engine
    from neuros.engine.policy import PolicySettings
    from neuros.engine.runner import run_live
    from neuros.engine.server import EventHub, start_server
    from neuros.engine.sources import BrainFlowSource, LslSource, brainflow_api, describe_source
    from neuros.errors import NeurOSConfigError

    bundle = ModelBundle.load(Path(args.bundle))
    if (args.lsl is None) == (args.brainflow is None):
        raise NeurOSConfigError("choose one source: --lsl [NAME] or --brainflow BOARD")
    if args.brainflow is not None:
        BoardIds = brainflow_api().BoardIds

        try:
            board = int(args.brainflow)
        except ValueError:
            if not hasattr(BoardIds, args.brainflow):
                raise NeurOSConfigError(f"unknown BrainFlow board {args.brainflow!r}") from None
            board = getattr(BoardIds, args.brainflow).value
        names = None
        if args.headset is not None:  # BrainFlow streams rows in the profile's device order
            names = list(load_profile(args.headset).canonical_channels)
        source: Any = BrainFlowSource(board, serial_port=args.serial_port, ch_names=names)
        channels = source.ch_names
    else:
        source = LslSource(args.lsl or None, units=args.units)
        channels = _source_channels(source.ch_names, args.headset)
    print(describe_source(source), "->", list(channels), flush=True)
    commands = {"left_hand": "LEFT", "right_hand": "RIGHT"}
    engine = Engine(
        bundle.decoder(),
        bundle.engine_config(
            channels,
            source.sfreq,
            policy=PolicySettings(
                commands={c: commands.get(c, c.upper()) for c in bundle.classes},
                threshold=args.threshold,
            ),
        ),
    )
    print(f"model channels on this headset: {list(engine.profile)}", flush=True)
    # A headset without C3 and C4 (the EPOC X, for one) still streams and decodes, but left vs
    # right imagery shows over the hand areas: say so up front, so a session that stays at
    # chance isn't mistaken for a broken engine.
    missing_hands = [ch for ch in ("C3", "C4") if ch not in engine.profile]
    if missing_hands:
        print(
            f"warning: no electrode at {missing_hands} (the hand areas): left vs right "
            "commands aren't expected to work on this headset",
            flush=True,
        )
    # The optional guided calibration (D-040): an app's "calibrate" request starts it; the
    # engine schedules the cued trials on the headset's clock.
    calibrator = Calibrator(
        engine,
        bundle.model,
        recipe=CalibrationRecipe.from_experiment(args.recipe),
        commands=engine.config.policy.commands,
        protocol=CalibrationProtocol(per_class=args.calibration_trials),
    )

    def show(event: Any) -> None:
        if isinstance(event, StatusEvent):
            print(f"[{event.t:10.1f}] state {event.previous} -> {event.state} ({event.reason})")
        elif isinstance(event, CalibrationEvent) and event.phase not in TRIAL_PHASES:
            details = f" {dict(event.details)}" if event.details else ""
            print(f"[{event.t:10.1f}] calibration {event.phase}: {event.message}{details}")

    recorder = None
    if args.record is not None:  # the session's evidence: raw EEG, events, the app's markers
        from neuros.engine.recorder import SessionRecorder

        recorder = SessionRecorder(
            Path(args.record),
            channels=channels,
            sfreq=source.sfreq,
            source=describe_source(source),
            details={
                "bundle": str(args.bundle),
                "bundle_provenance": {
                    k: v
                    for k, v in bundle.provenance.items()
                    if k not in ("train_people", "validation_people")
                },
                "headset": args.headset,
                "threshold": args.threshold,
            },
        )
        print(f"recording the session to {args.record}", flush=True)

    telemetry = None
    info = None
    if args.telemetry:  # the stage view: what the engine sees, alongside its decisions
        from neuros.engine.telemetry import Telemetry, stage_info

        telemetry = Telemetry(engine)
        info = stage_info(
            engine,
            source=describe_source(source) + (f" ({args.headset})" if args.headset else ""),
            model=Path(args.bundle).name,
            train_people=_bundle_people(bundle),
        )

    async def main() -> None:
        requests = (*REQUESTS, "mark") if recorder is not None else REQUESTS
        hub = EventHub(list(engine.config.policy.commands.values()), requests=requests, info=info)
        server = await start_server(hub, port=args.port)
        print(
            f"intents on ws://127.0.0.1:{args.port}; demo page: http://127.0.0.1:{args.port}/"
            + (f"; stage view: http://127.0.0.1:{args.port}/stage" if telemetry else "")
        )
        try:
            await run_live(
                source,
                engine,
                hub=hub,
                seconds=args.seconds,
                on_event=show,
                calibrator=calibrator,
                recorder=recorder,
                telemetry=telemetry,
            )
        finally:
            server.close()
            await server.wait_closed()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if recorder is not None:  # also when the server never started (closing is idempotent)
            recorder.close()
            print(f"session saved: {args.record} ({recorder.n_samples} samples)")
    return 0


def cmd_session(args: argparse.Namespace) -> int:
    """Summarise a recorded headset session (``neuros run --record``): what was saved, and the
    door game recomputed from the saved markers and intents, per calibration level."""
    from neuros.engine.recorder import door_scores, load_session

    session = load_session(Path(args.folder))
    meta = session.meta
    seconds = session.data.shape[1] / session.sfreq
    print(
        f"{args.folder}: {meta.get('source')}; {len(session.channels)} channels "
        f"{list(session.channels)} at {session.sfreq:g} Hz; {seconds / 60:.1f} min; "
        f"{len(session.events)} events; {len(session.markers())} door markers"
    )
    print(f"started {meta.get('started_utc')}, ended {meta.get('ended_utc') or 'not cleanly'}")
    scores = door_scores(session)
    if not scores:
        print("no door markers: nothing to score")
        return 0
    rows = [
        [
            s.level,
            str(s.doors),
            str(s.right),
            str(s.wrong),
            str(s.held),
            f"{s.accuracy * 100:.1f}%",
            f"{s.answered_fraction * 100:.0f}%",
            f"{s.p_value():.3g}",
        ]
        for s in scores.values()
    ]
    header = ["level", "doors", "right", "wrong", "held", "right of answered", "answered", "p"]
    print(_table(rows, header))
    print("p: one-sided binomial test of the answered doors against 50%.")
    return 0


def _default_fold_workers() -> int:
    """Parallel folds for the Riemannian baselines: half the CPUs, at most 12.

    Each worker holds its own copy of the covariance matrices (≈ 125 MB for PhysionetMI at 64
    channels) plus its fitting buffers, so the cap keeps memory well inside a 16 GB laptop.
    """
    import os

    return max(1, min(12, (os.cpu_count() or 2) // 2))


def _audit_output_name(spec_key: str, args: argparse.Namespace, task: TaskEpoch) -> str:
    """Result file name for an audit: every setting that changes the numbers, as for baselines."""
    name = f"{spec_key}_{'-'.join(args.labels)}_{args.reference}_{args.alignment_unit}"
    if task.rule == "after_cue":
        name += "_aftercue"
    else:
        name += f"_window{task.start_s:g}-{task.end_s:g}s"
    if args.subjects:
        name += "_subj" + "-".join(str(s) for s in args.subjects)
    return name + ".json"


def cmd_audit(args: argparse.Namespace) -> int:
    from neuros.data.download import loading_dataset, prepare_mne_environment, require_downloaded
    from neuros.data.montage import SCALP_REGIONS, region_channels
    from neuros.data.prepare import prepare_subject
    from neuros.data.recording import person_id
    from neuros.data.registry import get_spec
    from neuros.data.vocabulary import default_vocabulary
    from neuros.errors import NeurOSConfigError
    from neuros.eval.audit import REGION_ORDER, audit_windows, cut_audit_windows, run_audit
    from neuros.provenance import stamp
    from neuros.signal.windows import WindowSet

    if args.workers < 1:
        raise NeurOSConfigError("--workers must be >= 1")
    config = _config(args)
    spec = get_spec(args.dataset)
    # Resolved before anything loads: a dataset without class cues, or one whose task can't host
    # a window, fails here.
    windows_spec = audit_windows(spec, config.signal)
    vocab = default_vocabulary()
    unknown = sorted(set(args.labels) - set(vocab.labels))
    if unknown or len(set(args.labels)) < 2:
        raise NeurOSConfigError(f"--labels needs two or more vocabulary labels; unknown: {unknown}")
    provenance = stamp(config, dataset=spec.key)  # at the start: the code that runs
    _warn_if_dirty(provenance)
    subjects = _dev_subjects(spec, args.subjects, args.split, "a cue audit")
    require_downloaded(spec, subjects, config.data_dir)  # never let MOABB download unasked
    for window in windows_spec:
        print(f"{window.name}: {window.epoch.describe()}", flush=True)
    prepare_mne_environment(config.data_dir)
    dataset = loading_dataset(spec, config.data_dir)
    cache = _recording_cache(config, args)

    parts: dict[str, list[WindowSet]] = {w.name: [] for w in windows_spec}
    per_subject: list[dict[str, object]] = []
    for subject in subjects:
        cache_state = "off"
        if cache is None:
            recordings = prepare_subject(spec, subject, vocab, config, dataset=dataset)
        else:
            recordings, hit = cache.load_or_prepare(
                spec, subject, vocab, config, prepare=prepare_subject, dataset=dataset
            )
            cache_state = "hit" if hit else "miss"
        cut, skipped = cut_audit_windows(recordings, windows_spec, config.validation, args.labels)
        counts = {name: len(windows) for name, windows in cut.items()}
        for name, windows in cut.items():
            parts[name].append(windows)
        person = person_id(spec.key, subject)
        per_subject.append(
            {"person": person, "windows": counts, "skipped_outside": skipped, "cache": cache_state}
        )
        print(
            f"{person}: "
            + ", ".join(f"{name} {counts[name]} (skipped {skipped[name]})" for name in counts)
            + f", cache {cache_state}",
            flush=True,
        )
    window_sets = {name: WindowSet.concat(sets) for name, sets in parts.items()}
    ch_names = next(iter(window_sets.values())).ch_names
    region_sizes = {"all": len(ch_names)} | {
        region: len(region_channels(region, ch_names)) for region in SCALP_REGIONS
    }
    print(
        f"\n{len(subjects)} people, {len(ch_names)} channels; per region: "
        + ", ".join(f"{region} {n}" for region, n in region_sizes.items())
        + f"; {args.workers} fold workers\n",
        flush=True,
    )

    task = windows_spec[-1].epoch
    out = config.results_dir / "audit" / _audit_output_name(spec.key, args, task)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "protocol": {
            "dataset": spec.key,
            "labels": list(args.labels),
            "channels": list(ch_names),
            "region_channels": {
                region: list(region_channels(region, ch_names)) for region in SCALP_REGIONS
            },
            "class_on_screen_s": spec.class_on_screen_s,
            "cue_margin_s": config.signal.cue_margin_s,
            "windows": {w.name: w.epoch.to_dict() for w in windows_spec},
            "sfreq": config.signal.sfreq,
            "subjects": subjects,
            "split_file": str(args.split),
            "workers": args.workers,
        },
        "subjects": per_subject,
    }

    def save_progress(cells: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
        # Rewritten after every cell, so an interrupted audit keeps the cells it finished.
        partial = {
            "provenance": provenance,
            "complete": False,
            **header,
            "cells": cells,
            "skipped_cells": skipped,
        }
        out.write_text(json.dumps(partial, indent=1), encoding="utf-8", newline="\n")

    result = run_audit(
        window_sets,
        {w.name: w.epoch for w in windows_spec},
        reference=args.reference,
        unit=args.alignment_unit,
        workers=args.workers,
        on_cell=save_progress,
        progress=print,
    )

    by_cell = {(c["window"], c["region"]): c for c in result["cells"]}
    rows = []
    for window in windows_spec:
        row = [
            window.name,
            f"[{window.epoch.start_s:g}, {window.epoch.end_s:g}]",
            "yes" if window.epoch.cue_free else "no",
        ]
        for region in REGION_ORDER:
            cell = by_cell.get((window.name, region))
            row.append(
                "-" if cell is None else f"{cell['summary_balanced_accuracy']['mean'] * 100:.1f}"
            )
        rows.append(row)
    print(
        f"\nbalanced accuracy %, {result['method']} leave-one-person-out ('-' = too few channels)"
    )
    print(_table(rows, ["window", "epoch s", "cue-free", *REGION_ORDER]))
    print()
    for finding in result["findings"]:
        print(f"[{finding['level']}] {finding['message']}")

    payload = {"provenance": _finished(provenance), "complete": True, **header, **result}
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")
    print(f"\nresults: {out}")
    # Exit 1 ("checks found problems") on a leak; a cue-dominated dataset is a finding, not a bug.
    return 1 if any(f["level"] == "leak" for f in result["findings"]) else 0


def _require_torch() -> None:
    """Fail early, with the install command, if the optional PyTorch dependency is missing."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        from neuros.errors import NeurOSConfigError

        raise NeurOSConfigError(
            "this command needs PyTorch: run `uv sync --extra dev --extra ml`"
        ) from exc


def _loso_output_name(experiment: LosoExperimentConfig, args: argparse.Namespace) -> str:
    """Result file name: names every setting that changes the numbers, so runs never clobber."""
    seeds = experiment.seeds
    parts = [
        experiment.dataset,
        "-".join(experiment.labels),
        experiment.channels,
        experiment.reference,
        "-".join(experiment.alignments),
        f"s{seeds[0]}" + (f"x{len(seeds)}" if len(seeds) > 1 else ""),
    ]
    if args.max_epochs:
        parts.append(f"ep{args.max_epochs}")
    if args.subjects:
        parts.append("subj" + "-".join(str(s) for s in args.subjects))
    return "_".join(parts) + ".json"


def cmd_loso(args: argparse.Namespace) -> int:
    import os

    from pydantic import ValidationError

    from neuros.config.experiment import (
        LosoExperimentConfig,
        apply_signal_overrides,
        load_experiment_config,
    )
    from neuros.data.registry import get_spec
    from neuros.errors import NeurOSConfigError
    from neuros.eval.deep import run_loso_experiment
    from neuros.provenance import config_hash, stamp

    experiment = load_experiment_config(args.experiment)
    # Command-line overrides are re-validated as a whole config and land in the result file.
    updates = experiment.model_dump()
    if args.alignments:
        updates["alignments"] = args.alignments
    if args.seeds:
        updates["seeds"] = args.seeds
    if args.max_epochs:
        updates["train"]["max_epochs"] = args.max_epochs
    try:
        experiment = LosoExperimentConfig.model_validate(updates)
    except ValidationError as exc:
        raise NeurOSConfigError(f"invalid command-line override:\n{exc}") from exc
    if args.threads < 1 or (args.workers is not None and args.workers < 1):
        raise NeurOSConfigError("--threads and --workers must be >= 1")
    _require_torch()
    from neuros.train.device import resolve_device

    # Fails now, before any data loads, if CUDA was asked for and isn't there (D-032).
    first_device = resolve_device(args.device)
    deterministic = not args.nondeterministic

    config = _config(args)
    config = config.model_copy(
        update={"signal": apply_signal_overrides(config.signal, experiment.signal)}
    )
    spec = get_spec(experiment.dataset)
    spec_key = spec.key
    # Provenance at the start: worker processes import the code on disk when they start, so
    # the checkout must not change during a run, and the stamp must describe it as it was.
    provenance = stamp(
        config,
        dataset=spec_key,
        experiment_file=str(args.experiment),
        experiment_sha256=config_hash(experiment),
        threads_per_worker=str(args.threads),
        device=args.device,
        deterministic=str(deterministic).lower(),
    )
    _warn_if_dirty(provenance)
    windows, summaries, subjects, epoch = _dev_windows(
        config,
        spec_key,
        args.subjects,
        experiment.labels,
        experiment.channels,
        args.split,
        f"experiment {experiment.name}",
        cache=_recording_cache(config, args),
    )
    workers = args.workers or max(
        1, min(len(windows.people), (os.cpu_count() or 2) // args.threads)
    )
    print(
        f"\n{experiment.name}: {len(windows)} windows, {len(windows.people)} people, "
        f"{len(windows.ch_names)} channels ({experiment.channels}), {windows.n_samples} samples "
        f"per window {epoch.describe()}, reference {experiment.reference}, device "
        f"{args.device} (first fold on {first_device})\n",
        flush=True,
    )
    result = run_loso_experiment(
        windows,
        experiment,
        workers=workers,
        threads=args.threads,
        progress=print,
        device=args.device,
        deterministic=deterministic,
    )

    rows = []
    for run in result["runs"]:
        acc = run["summary_balanced_accuracy"]
        rows.append(
            [
                run["name"],
                str(run["seed"]),
                f"{acc['mean'] * 100:.1f} ± {acc['std'] * 100:.1f}",
                f"[{acc['ci95_low'] * 100:.1f}, {acc['ci95_high'] * 100:.1f}]",
                f"{acc['median'] * 100:.1f}",
                f"{acc['share_usable'] * 100:.0f}%",
                f"{run['summary_kappa']['mean']:.3f}",
                f"{run['minutes']:.1f}",
            ]
        )
    print()
    print(
        _table(
            rows,
            ["model", "seed", "bal. acc %", "95% CI", "median", ">=70%", "kappa", "minutes"],
        )
    )
    for comparison in result["comparisons"]:
        print(
            f"{comparison['a']} vs {comparison['b']} (seed {comparison['seed']}): "
            f"{comparison['mean_diff'] * 100:+.1f} points, Wilcoxon p = "
            f"{comparison['p_value']:.3g} (n = {comparison['n_people']})"
        )
    for spread in result["seed_summaries"]:
        acc = spread["summary_balanced_accuracy"]
        print(
            f"{spread['name']} over {len(spread['seeds'])} seeds: {acc['mean'] * 100:.1f}% "
            f"(seed SD of the mean {spread['seed_std_of_mean'] * 100:.1f} points)"
        )

    out = (
        Path(args.out)
        if args.out
        else config.results_dir / experiment.name / _loso_output_name(experiment, args)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": {**_finished(provenance), "workers": str(workers)},
        "protocol": {
            "dataset": spec_key,
            "labels": list(experiment.labels),
            "channels": list(windows.ch_names),
            "channel_set": experiment.channels,
            "reference": experiment.reference,
            **epoch.to_dict(),
            "class_on_screen_s": spec.class_on_screen_s,
            "cue_margin_s": config.signal.cue_margin_s,
            "window_s": config.signal.window_s,
            "sfreq": config.signal.sfreq,
            "subjects": subjects,
            "split_file": str(args.split),
        },
        "experiment": experiment.model_dump(mode="json"),
        "subjects": summaries,
        **result,
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")
    print(f"\nresults: {out}")
    return 0


# ---------------------------------------------------------------------- parser
def _e1_layout(channels: str) -> tuple[str, ...]:
    """The model's input layout: C-core (§4.3), or its motor-strip rows (D-031)."""
    from neuros.data.montage import load_c_core, region_channels

    c_core = load_c_core()
    return c_core if channels == "c-core" else region_channels("motor", c_core)


def _e1_windows(
    config: DataConfig,
    keys: Sequence[str],
    labels: Sequence[str],
    split_path: str,
    max_subjects: int | None,
    cache: RecordingCache | None,
) -> tuple[dict[str, WindowSet], dict[str, Any]]:
    """Each dataset's development windows under ``config``'s epoch rule, plus a load report.

    Raises:
        NeurOSDataError: if a dataset (or person) isn't downloaded: E1 never triggers a download.
    """
    from neuros.data.download import loading_dataset, prepare_mne_environment, require_downloaded
    from neuros.data.prepare import subject_epochs
    from neuros.data.registry import get_spec, task_epoch
    from neuros.data.vocabulary import default_vocabulary
    from neuros.signal.windows import WindowSet

    vocab = default_vocabulary()
    prepare_mne_environment(config.data_dir)
    per_dataset: dict[str, WindowSet] = {}
    loaded: dict[str, Any] = {}
    for key in keys:
        spec = get_spec(key)
        epoch = task_epoch(spec, config.signal)
        subjects = _dev_subjects(spec, None, split_path, "experiment E1")
        if max_subjects is not None:
            subjects = subjects[:max_subjects]
        require_downloaded(spec, subjects, config.data_dir)
        dataset = loading_dataset(spec, config.data_dir)
        sets, summaries = [], []
        for subject in subjects:
            windows, summary = subject_epochs(
                spec, subject, vocab, config, labels, dataset=dataset, cache=cache
            )
            sets.append(windows)
            summaries.append(summary.to_dict())
        per_dataset[key] = WindowSet.concat(sets)
        loaded[key] = {**epoch.to_dict(), "subjects": subjects, "per_subject": summaries}
        print(
            f"{key}: {len(subjects)} people, {len(per_dataset[key])} windows, {epoch.describe()}",
            flush=True,
        )
    return per_dataset, loaded


def _memory_note(stage: str) -> None:
    """Print this process's resident memory at a stage of a long run (Linux; silent elsewhere).

    E1/E2 runs share a 30-GB machine between the main process and its training workers; on
    9 Oct the D-039 runs ran it out of memory twice. These lines show where memory goes.
    """
    try:
        status = Path("/proc/self/status").read_text(encoding="ascii")
    except OSError:
        return
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            gigabytes = int(line.split()[1]) / 1048576  # the kernel reports kB
            print(f"memory: {gigabytes:.1f} GB resident ({stage})", flush=True)
            return


def _release_freed_memory() -> None:
    """Hand memory the process has freed back to the operating system (glibc's ``malloc_trim``).

    Data preparation frees gigabytes of arrays, but glibc may keep freed heap memory for reuse,
    so the main process can still hold it when the training workers start (9 Oct: 10.7 GB,
    then the kernel's out-of-memory killer). A no-op off Linux or without glibc.
    """
    import ctypes
    import gc
    import platform

    gc.collect()
    if platform.system() != "Linux":  # (not sys.platform: type checkers fold that per OS)
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # another C library: nothing to trim
        return


def _e1_training_windows(
    experiment: PooledExperimentConfig,
    config: DataConfig,
    task_windows: Mapping[str, WindowSet],
    layout: Sequence[str],
    derived: Mapping[str, Mapping[str, Sequence[str]]],
    cache: RecordingCache | None,
) -> tuple[PooledWindows, dict[str, Any]]:
    """D-039's training set for every trainable dataset (crops, extra classes), assembled.

    The people are exactly those of ``task_windows`` (the development windows already loaded),
    and each person's windows take the EA references of their own test windows. Windows are
    cut and placed one person at a time, keeping only the placed result: holding every
    dataset's raw training windows and their copies at once ran the 30-GB machine out of
    memory (9 Oct). A dataset that gains neither crops nor classes is placed as tested. Newly
    cut windows go through the artifact rule (``screen_amplitude``); the report counts drops.

    Returns:
        ``(the assembled training windows, a report per dataset)``.

    Raises:
        NeurOSConfigError: if the experiment has no training_windows section, or an extra
            label isn't in the vocabulary.
        NeurOSDataError: see ``subject_training_windows`` and ``assemble``.
    """
    import numpy as np

    from neuros.data.download import loading_dataset
    from neuros.data.registry import get_spec, task_epoch, training_crop_starts
    from neuros.data.training_windows import subject_training_windows
    from neuros.data.vocabulary import default_vocabulary
    from neuros.errors import NeurOSConfigError
    from neuros.eval.pooled import PooledWindows, assemble, screen_amplitude

    settings = experiment.training_windows
    if settings is None:
        raise NeurOSConfigError("the experiment has no training_windows section")
    vocab = default_vocabulary()
    unknown = sorted(set(settings.extra_labels) - set(vocab.labels))
    if unknown:
        raise NeurOSConfigError(f"extra_labels {unknown} are not vocabulary labels")
    placement = {
        "reference": experiment.reference,
        "alignment": experiment.alignment,
        "unit": experiment.alignment_unit,
    }
    parts: list[PooledWindows] = []
    report: dict[str, Any] = {}
    print("training windows (D-039):", flush=True)
    for key in experiment.train_datasets:
        spec = get_spec(key)
        epoch = task_epoch(spec, config.signal)
        starts = (
            training_crop_starts(spec, config.signal, settings.crop_stride_s)
            if settings.crop_stride_s is not None
            else (epoch.start_s,)
        )
        own = set(vocab.event_maps[key].values())
        labels = [label for label in experiment.train_labels if label in own]
        extra = [label for label in labels if label not in experiment.labels]
        windows = task_windows[key]
        mapping = {key: derived[key]} if key in derived else None
        if len(starts) == 1 and not extra:  # nothing to add: placed exactly as tested
            part, _ = assemble(
                {key: windows},
                layout,
                **placement,  # type: ignore[arg-type]
                derived=mapping,
                primary={key: np.ones(len(windows), dtype=bool)},
            )
            parts.append(part)
            report[key] = {
                "crop_starts_s": list(starts),
                "labels": labels,
                "reused": True,
                "n_people": len(windows.people),
                "n_trials": len(windows),
                "n_windows": len(windows),
            }
            print(f"{key}: as tested ({len(windows)} windows)", flush=True)
            continue
        dataset = loading_dataset(spec, config.data_dir)
        summaries: list[dict[str, Any]] = []
        for subject in dict.fromkeys(windows.subjects.tolist()):
            cut, first, summary = subject_training_windows(
                spec, subject, vocab, config, labels, starts, dataset=dataset, cache=cache
            )
            part, _ = assemble(
                {key: cut},
                layout,
                **placement,  # type: ignore[arg-type]
                derived=mapping,
                references_from={key: windows.for_people({summary["person"]})},
                primary={key: first},
            )
            # Newly cut signal can hold glitches the test windows never met (10 Oct: two
            # crops at 7,000x a person's median RMS broke 4 of 5 models): drop them.
            part, dropped = screen_amplitude(part)  # one dataset: rows in ``first``'s order
            summary["n_artifacts_dropped"] = int(dropped.sum())
            summary["n_primary_dropped"] = int((dropped & np.asarray(first, dtype=bool)).sum())
            parts.append(part)
            summaries.append(summary)
            del cut  # only the placed windows are kept
        counts: dict[str, int] = {}
        for summary in summaries:
            for label, count in summary["label_counts"].items():
                counts[label] = counts.get(label, 0) + count
        n_dropped = sum(s["n_artifacts_dropped"] for s in summaries)
        report[key] = {
            "crop_starts_s": list(starts),
            "labels": labels,
            "reused": False,
            "n_people": len(summaries),
            "n_trials": sum(s["n_trials"] for s in summaries),
            # Windows cut (label_counts counts these), then dropped by the artifact rule.
            "n_windows_cut": sum(s["n_windows"] for s in summaries),
            "n_artifacts_dropped": n_dropped,
            "n_primary_dropped": sum(s["n_primary_dropped"] for s in summaries),
            "n_windows": sum(s["n_windows"] for s in summaries) - n_dropped,
            "label_counts": counts,
            "per_subject": summaries,
        }
        print(
            f"{key}: {report[key]['n_trials']} trials -> {report[key]['n_windows']} windows "
            f"(crops at {', '.join(f'{s:g}' for s in starts)} s; classes {', '.join(labels)}; "
            f"{n_dropped} dropped as artifacts)",
            flush=True,
        )
    return PooledWindows.concat(parts), report


def _recording_loader(
    config: DataConfig, cache: RecordingCache | None
) -> Callable[[str, int], list[Recording]]:
    """``(dataset, subject) → that person's preprocessed recordings``, from the cache (MOABB is
    only touched on a cache miss, once per dataset)."""
    from neuros.data.download import loading_dataset
    from neuros.data.prepare import prepare_subject
    from neuros.data.registry import get_spec
    from neuros.data.vocabulary import default_vocabulary

    vocab = default_vocabulary()
    moabb_objects: dict[str, Any] = {}

    def load(key: str, subject: int) -> list[Recording]:
        spec = get_spec(key)
        if key not in moabb_objects:
            moabb_objects[key] = loading_dataset(spec, config.data_dir)
        if cache is None:
            return prepare_subject(spec, subject, vocab, config, dataset=moabb_objects[key])
        recordings, _ = cache.load_or_prepare(
            spec, subject, vocab, config, prepare=prepare_subject, dataset=moabb_objects[key]
        )
        return recordings

    return load


def _e1_finetune_sets(
    experiment: PooledExperimentConfig,
    config: DataConfig,
    task_windows: Mapping[str, WindowSet],
    layout: Sequence[str],
    people: Mapping[str, Sequence[str]],
    cache: RecordingCache | None,
    work: Path,
    folders: dict[str, Path],
) -> dict[str, Any]:
    """Q30: build and save the fine-tuning sets, add them to ``folders``, return the report.

    Raises:
        NeurOSConfigError: if the experiment has no finetune or warm-up section.
    """
    from neuros.data.registry import get_spec
    from neuros.errors import NeurOSConfigError
    from neuros.eval.finetune import build_finetune_sets
    from neuros.eval.splits import make_dev_folds

    if experiment.finetune is None or experiment.warmup is None:
        raise NeurOSConfigError("fine-tuning needs the finetune and warmup sections")
    folds = make_dev_folds(
        {d: [int(p.split(":", 1)[1]) for p in members] for d, members in people.items()},
        n_folds=experiment.folds,
        seed=experiment.seeds[0],
    )
    print("fine-tune sets (Q30):", flush=True)
    sets, report = build_finetune_sets(
        task_windows,
        layout,
        specs={key: get_spec(key) for key in task_windows},
        folds=folds,
        trainable=experiment.train_datasets,
        labels=experiment.labels,
        warmup=experiment.warmup,
        finetune=experiment.finetune,
        load_recordings=_recording_loader(config, cache),
        max_ptp_uv=config.validation.ptp_flag_uv,
    )
    for kind, windows in sets.items():
        windows.save(work / kind)
        folders[kind] = work / kind
    return report


def _e1_warmup_sets(
    experiment: PooledExperimentConfig,
    config: DataConfig,
    task_windows: Mapping[str, WindowSet],
    layout: Sequence[str],
    people: Mapping[str, Sequence[str]],
    cache: RecordingCache | None,
    work: Path,
    folders: dict[str, Path],
) -> dict[str, Any]:
    """E2 (§8.5): build and save the warm-up sets, add them to ``folders``, return the report.

    The folds are the ones ``plan_tasks`` draws for the experiment's single seed, so each test
    person's prior comes from the people their pooled model trains on.

    Raises:
        NeurOSConfigError: if the experiment has no warm-up section.
    """
    from neuros.data.download import loading_dataset
    from neuros.data.prepare import prepare_subject
    from neuros.data.registry import get_spec
    from neuros.data.vocabulary import default_vocabulary
    from neuros.errors import NeurOSConfigError
    from neuros.eval.splits import make_dev_folds
    from neuros.eval.warmup import build_warmup_sets, warmup_kinds

    warmup = experiment.warmup
    if warmup is None:
        raise NeurOSConfigError("the experiment has no warmup section")
    vocab = default_vocabulary()
    moabb_objects: dict[str, Any] = {}

    def load_recordings(key: str, subject: int) -> list[Recording]:
        spec = get_spec(key)
        if key not in moabb_objects:  # MOABB is only touched on a cache miss
            moabb_objects[key] = loading_dataset(spec, config.data_dir)
        if cache is None:
            return prepare_subject(spec, subject, vocab, config, dataset=moabb_objects[key])
        recordings, _ = cache.load_or_prepare(
            spec, subject, vocab, config, prepare=prepare_subject, dataset=moabb_objects[key]
        )
        return recordings

    folds = make_dev_folds(
        {d: [int(p.split(":", 1)[1]) for p in members] for d, members in people.items()},
        n_folds=experiment.folds,
        seed=experiment.seeds[0],
    )
    print("E2 warm-up sets:", flush=True)
    sets, report = build_warmup_sets(
        task_windows,
        layout,
        specs={key: get_spec(key) for key in task_windows},
        folds=folds,
        trainable=experiment.train_datasets,
        settings=warmup,
        load_recordings=load_recordings,
        max_ptp_uv=config.validation.ptp_flag_uv,
    )
    for kind, windows in sets.items():
        windows.save(work / kind)
        folders[kind] = work / kind
    report["kinds"] = [kind for kind, _, _ in warmup_kinds(warmup)]
    return report


def cmd_e1(args: argparse.Namespace) -> int:
    import datetime as dt

    from pydantic import ValidationError

    from neuros.config.experiment import (
        PooledExperimentConfig,
        apply_signal_overrides,
        load_pooled_experiment_config,
    )
    from neuros.data.registry import assert_may_train, get_spec, task_epoch
    from neuros.errors import NeurOSConfigError
    from neuros.eval.finetune import FinetuneSettings
    from neuros.eval.pooled import (
        MIN_LAYOUT_CHANNELS,
        TaskSettings,
        assemble,
        layout_shortfalls,
        plan_tasks,
        report,
        run_tasks,
    )
    from neuros.provenance import config_hash, stamp
    from neuros.train.device import resolve_device

    experiment = load_pooled_experiment_config(args.experiment)
    updates = experiment.model_dump()
    if args.seeds:
        updates["seeds"] = args.seeds
    if args.max_epochs:
        updates["train"]["max_epochs"] = args.max_epochs
    if args.conditions:
        updates["conditions"] = args.conditions
    try:
        experiment = PooledExperimentConfig.model_validate(updates)
    except ValidationError as exc:
        raise NeurOSConfigError(f"invalid command-line override:\n{exc}") from exc
    if args.threads < 1 or args.workers < 1:
        raise NeurOSConfigError("--threads and --workers must be >= 1")
    if args.max_subjects is not None and args.max_subjects < experiment.folds:
        raise NeurOSConfigError(f"--max-subjects must be >= the {experiment.folds} folds")
    # Before anything loads: who may train a model we ship (D-033), and the device (D-032).
    assert_may_train(experiment.train_datasets, f"experiment {experiment.name}")
    _require_torch()
    first_device = resolve_device(args.device)
    deterministic = not args.nondeterministic

    config = _config(args)
    config = config.model_copy(
        update={"signal": apply_signal_overrides(config.signal, experiment.signal)}
    )
    start, end = experiment.cue_only_epoch_s
    cue_config = config.model_copy(
        update={
            "signal": apply_signal_overrides(
                config.signal, {"epoch_s": [start, end], "window_s": round(end - start, 9)}
            )
        }
    )
    for key in experiment.datasets:  # every dataset must host both windows before any loading
        task_epoch(get_spec(key), config.signal)
        if "cue_only" in experiment.conditions:
            task_epoch(get_spec(key), cue_config.signal)
    layout = _e1_layout(experiment.channels)
    # Every dataset must share enough of the layout's channels, counting its derived channels
    # (D-037): checked from the registry here, not at assembly after hours of loading (6 Oct:
    # E1 stopped at assembly because Shin2017A shared only Cz with the motor strip).
    short = layout_shortfalls(
        {key: get_spec(key).model_channels for key in experiment.datasets}, layout
    )
    derived = {
        key: get_spec(key).derived_channels
        for key in experiment.datasets
        if get_spec(key).derived_channels
    }
    if short:
        details = "; ".join(f"{key} has {have or 'none'}" for key, have in short.items())
        raise NeurOSConfigError(
            f"these datasets can't be placed in the {experiment.channels!r} layout (at least "
            f"{MIN_LAYOUT_CHANNELS} of its {len(layout)} channels are needed): {details}"
        )
    provenance = stamp(
        config,
        experiment_file=str(args.experiment),
        experiment_sha256=config_hash(experiment),
        device=args.device,
        deterministic=str(deterministic).lower(),
    )
    _warn_if_dirty(provenance)
    cache = _recording_cache(config, args)
    print(
        f"{experiment.name}: layout {len(layout)} channels ({experiment.channels}), device "
        f"{args.device} ({first_device} first)",
        flush=True,
    )

    run_id = f"{experiment.name}-{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    work = config.cache_dir / "pooled" / run_id
    task_windows, loaded = _e1_windows(
        config, experiment.datasets, experiment.labels, args.split, args.max_subjects, cache
    )
    pooled, assembly = assemble(
        task_windows,
        layout,
        reference=experiment.reference,
        alignment=experiment.alignment,
        unit=experiment.alignment_unit,
        derived=derived,
    )
    pooled.save(work / "task")
    people = pooled.people_by_dataset()
    del pooled  # saved: workers map it, and the steps below need the memory (9 Oct)
    _memory_note("task windows assembled")
    cue_free = {key: bool(loaded[key]["cue_free"]) for key in experiment.datasets}
    folders = {"task": work / "task"}
    warmup_report: dict[str, Any] | None = None
    if experiment.warmup is not None:  # E2: built from the unaligned task windows
        warmup_report = _e1_warmup_sets(
            experiment, config, task_windows, layout, people, cache, work, folders
        )
        _memory_note("warm-up sets saved")
    finetune_report: dict[str, Any] | None = None
    if experiment.finetune is not None:  # Q30: calibration and test windows per size
        finetune_report = _e1_finetune_sets(
            experiment, config, task_windows, layout, people, cache, work, folders
        )
        _memory_note("fine-tune sets saved")
    training_report: dict[str, Any] | None = None
    if experiment.training_windows is not None:  # D-039: aligned by the task windows' references
        train_pooled, training_report = _e1_training_windows(
            experiment, config, task_windows, layout, derived, cache
        )
        train_pooled.save(work / "train")
        folders["train"] = work / "train"
        del train_pooled
        _memory_note("training windows saved")
    del task_windows  # workers map the saved arrays
    if "cue_only" in experiment.conditions:
        print("cue-only control windows:", flush=True)
        cue_windows, _ = _e1_windows(
            cue_config, experiment.datasets, experiment.labels, args.split, args.max_subjects, cache
        )
        cue_pooled, _ = assemble(
            cue_windows,
            layout,
            reference=experiment.reference,
            alignment=experiment.alignment,
            unit=experiment.alignment_unit,
            derived=derived,
        )
        cue_pooled.save(work / "cue_only")
        folders["cue_only"] = work / "cue_only"
        del cue_windows, cue_pooled
    # The workers need the memory the data preparation used: hand freed memory back first.
    _release_freed_memory()
    _memory_note("before training")

    out = Path(args.out) if args.out else config.results_dir / experiment.name / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = TaskSettings(
        classes=tuple(experiment.labels),
        model=experiment.model,
        train=experiment.train,
        device=args.device,
        deterministic=deterministic,
        threads=args.threads,
        extra_kinds=tuple(warmup_report["kinds"]) if warmup_report else (),
        train_classes=experiment.train_labels if training_report is not None else (),
        finetune=(
            FinetuneSettings.from_config(experiment.finetune)
            if experiment.finetune is not None
            else None
        ),
    )
    payload: dict[str, Any] = {
        "complete": False,
        "provenance": provenance,
        "experiment": experiment.model_dump(mode="json"),
        "protocol": {
            "layout": list(layout),
            "datasets": {
                key: {k: v for k, v in loaded[key].items() if k != "per_subject"}
                for key in experiment.datasets
            },
            "cue_only_epoch_s": list(experiment.cue_only_epoch_s),
            "split_file": str(args.split),
            "work_folder": str(work),
        },
        "assembly": assembly,
        "subjects": {key: loaded[key]["per_subject"] for key in experiment.datasets},
        "tasks": [],
    }
    if warmup_report is not None:
        payload["warmup"] = warmup_report
    if training_report is not None:
        payload["training_windows"] = training_report
    if finetune_report is not None:
        payload["finetune"] = finetune_report

    def save() -> None:
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8", newline="\n")
        tmp.replace(out)

    def on_task(outcome: dict[str, Any]) -> None:
        payload["tasks"].append(outcome)
        save()

    trainable = list(experiment.train_datasets)
    for seed in experiment.seeds:
        tasks = plan_tasks(
            people,
            trainable,
            conditions=experiment.conditions,
            folds=experiment.folds,
            validation_fraction=experiment.validation_fraction,
            seed=seed,
            train_data="train" if training_report is not None else None,
        )
        print(f"seed {seed}: {len(tasks)} models on {args.workers} worker(s)", flush=True)
        save()
        outcomes = run_tasks(
            tasks, folders, settings, workers=args.workers, progress=print, on_task=on_task
        )
        payload.setdefault("reports", {})[str(seed)] = report(outcomes, people, cue_free)
        if experiment.warmup is not None:
            from neuros.eval.warmup import summarize_warmup

            payload["warmup"]["summary"] = summarize_warmup(outcomes, experiment.warmup)
    payload["complete"] = True
    payload["provenance"] = _finished(provenance)
    save()

    def pct(entry: dict[str, Any], condition: str) -> str:
        return f"{entry[condition]['mean'] * 100:.1f}" if condition in entry else "-"

    rows = []
    for seed, seed_report in payload["reports"].items():
        for key, entry in seed_report["datasets"].items():
            delta = entry.get("pooled_vs_single")
            rows.append(
                [
                    seed,
                    key,
                    "yes" if entry["cue_free"] else "NO",
                    str(entry["n_people"]),
                    pct(entry, "pooled"),
                    pct(entry, "single"),
                    pct(entry, "lodo"),
                    pct(entry, "cue_only"),
                    f"{delta['mean_diff'] * 100:+.1f} (p={delta['p_value']:.2g})" if delta else "-",
                ]
            )
    print()
    print(
        _table(
            rows,
            [
                "seed",
                "dataset",
                "cue-free",
                "people",
                "pooled",
                "single",
                "LODO",
                "cue-only",
                "pooled-single",
            ],
        )
    )
    if experiment.warmup is not None:
        _print_warmup(payload["warmup"]["summary"])
    print(f"\nresults: {out}")
    return 0


def _print_warmup(summary: dict[str, Any]) -> None:
    """E2's curve per dataset: mean balanced accuracy per set, and S (§8.6)."""
    for dataset, entry in summary.items():
        rows = []
        for kind, result in entry["sets"].items():
            gap = result.get("vs_full")
            rows.append(
                [
                    kind,
                    f"{result['mean'] * 100:.1f}",
                    f"{gap['mean_diff'] * 100:+.1f} (p={gap['p_value']:.2g})" if gap else "-",
                ]
            )
        print(f"\nE2 warm-up, {dataset} ({entry['n_people']} people):")
        print(_table(rows, ["set", "bal. acc %", "vs full-session"]))
        s_seconds = entry["s_seconds"]
        within = entry["s_within_points"]
        if s_seconds is None:
            print(f"S: no stream budget came within {within:g} points of full-session")
        else:
            print(f"S = {s_seconds:g} s (stream within {within:g} points of full-session)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuros", description="NeurOS developer tools")
    parser.add_argument("--version", action="version", version=f"neuros {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--config", default=None, help="data config YAML (default: built-in)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("datasets", help="list the dataset registry")
    p.add_argument("--check", action="store_true", help="cross-check against MOABB metadata")
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("montage", help="show C-core and headset coverage")
    p.set_defaults(func=cmd_montage)

    p = sub.add_parser("split", help="show or write the locked test set")
    p.add_argument("--write", action="store_true", help="write the split file (only if absent)")
    p.add_argument(
        "--split-version", type=int, default=None, help="which version (default: the current one)"
    )
    p.add_argument("--path", default=None, help="split file (default: the version's own file)")
    p.add_argument(
        "--extends",
        default=None,
        help="with --write: the previous version's file (default: its own)",
    )
    p.set_defaults(func=cmd_split)

    for name, func, text in (
        ("download", cmd_download, "download raw data and record the integrity manifest"),
        ("validate", cmd_validate, "harmonise and validate recordings"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--dataset", required=True)
        p.add_argument("--subjects", type=int, nargs="+", default=None)
        if name == "download":
            p.add_argument(
                "--verify", action="store_true", help="re-hash files against the manifest"
            )
            p.add_argument(
                "--accept-licence",
                action="store_true",
                help="you, the person running this, accept the dataset's licence (needed for "
                "datasets whose download requires it, e.g. Shin2017A: GPL-3.0)",
            )
        else:
            p.add_argument(
                "--include-excluded",
                action="store_true",
                help="also validate subjects the registry excludes for data quality",
            )
        p.set_defaults(func=func)

    p = sub.add_parser("filters", help="the causal filter's delay and gains at every rate")
    p.set_defaults(func=cmd_filters)

    p = sub.add_parser("baseline", help="Riemannian baselines (LOSO + session transfer)")
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="default: the dataset's development people (usable, not in the locked test set)",
    )
    p.add_argument("--labels", nargs="+", default=["left_hand", "right_hand"])
    p.add_argument("--channels", choices=["all", "c-core", "motor"], default="all")
    p.add_argument("--reference", choices=["as_recorded", "car"], default="as_recorded")
    p.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="band-pass edges in Hz instead of the contract's 4-30 (E0b)",
    )
    p.add_argument(
        "--alignment-unit",
        choices=["person", "session"],
        default="person",
        help="one EA/RA reference per person (the Day-2 default) or per recording session (E0)",
    )
    p.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="a fixed epoch [START, END] s from the cue instead of the contract's after-cue "
        "rule (D-030): e.g. 0 0.5 for the cue-only control, -0.5 0 for a pre-cue null check",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=_default_fold_workers(),
        help="parallel leave-one-person-out folds (identical results; default: half the CPUs, "
        "at most 12)",
    )
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute every recording instead of using the preprocessed cache (§4.5)",
    )
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser(
        "calibration",
        help="calibration curve: a person's own calibrated decoder vs windows used (Q28)",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="default: the dataset's development people (usable, not in the locked test set)",
    )
    p.add_argument("--labels", nargs="+", default=["left_hand", "right_hand"])
    p.add_argument("--channels", choices=["all", "c-core", "motor"], default="motor")
    p.add_argument("--reference", choices=["as_recorded", "car"], default="car")
    p.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="band-pass edges in Hz instead of the contract's default (E1 used 8 30)",
    )
    p.add_argument(
        "--block",
        choices=["session", "person"],
        default="session",
        help="calibrate and test within each session, or within each person (sessions "
        "concatenated in order; for datasets with very short sessions)",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40],
        help="windows per class to calibrate on (the whole calibration pool is always added)",
    )
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute every recording instead of using the preprocessed cache (§4.5)",
    )
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser(
        "smr",
        help="resting sensorimotor-rhythm strength per person, from unlabeled data (Q31)",
    )
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="how much of the first session's start to use (the engine's first minute)",
    )
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute every recording instead of using the preprocessed cache (§4.5)",
    )
    p.set_defaults(func=cmd_smr)

    p = sub.add_parser(
        "bundle", help="train the deployable model and write a model bundle for the engine"
    )
    p.add_argument("--experiment", required=True, help="an E1-style experiment YAML")
    p.add_argument("--out", required=True, help="the bundle folder (never overwritten)")
    p.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N")
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute every recording instead of using the preprocessed cache (§4.5)",
    )
    p.set_defaults(func=cmd_bundle)

    p = sub.add_parser(
        "simulate", help="replay a dataset person's session through the engine (virtual headset)"
    )
    p.add_argument("--bundle", required=True, help="a model bundle folder (`neuros bundle`)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--session", default=None, help="default: the person's first session")
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="1.0 = real time (demo); default: as fast as possible",
    )
    p.add_argument("--threshold", type=float, default=0.6, help="the policy's θ (0.5-1)")
    p.add_argument("--noise-uv", type=float, default=0.0, help="added white noise (µV)")
    p.add_argument("--serve", action="store_true", help="broadcast events on a local WebSocket")
    p.add_argument(
        "--telemetry",
        action="store_true",
        help="with --serve: also send what the engine sees (for the stage view, /stage)",
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--wait-for-app", action="store_true", help="with --serve: start once an app connects"
    )
    p.add_argument(
        "--allow-seen",
        action="store_true",
        help="allow a person who trained the bundle (plumbing checks only: accuracy is inflated)",
    )
    p.add_argument("--out", default=None, help="write the replay's score as JSON")
    p.add_argument(
        "--calibrate",
        type=int,
        nargs="?",
        const=10,
        default=None,
        metavar="N",
        help="test the guided calibration (D-040): a zero-calibration arm, then quick "
        "calibration on the session's first N trials per class (default 10), both scored on "
        "the trials after it",
    )
    p.add_argument(
        "--recipe",
        default="configs/e2_finetune.yaml",
        help="the fine-tuning recipe: an experiment YAML with a finetune section (Q30's)",
    )
    p.add_argument(
        "--lsl-out",
        default=None,
        metavar="NAME",
        help="publish the session as an LSL stream (a virtual headset for `neuros run --lsl`) "
        "instead of running the engine here",
    )
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("run", help="the live engine: headset in, intents out (local WebSocket)")
    p.add_argument("--bundle", required=True, help="a model bundle folder (`neuros bundle`)")
    p.add_argument(
        "--lsl", nargs="?", const="", default=None, metavar="NAME",
        help="read an LSL EEG stream (by name; without a name, the first EEG stream)",
    )  # fmt: skip
    p.add_argument(
        "--brainflow", default=None, metavar="BOARD",
        help="read a BrainFlow board, e.g. GANGLION_BOARD or SYNTHETIC_BOARD",
    )  # fmt: skip
    p.add_argument("--serial-port", default="", help="the board's port (BrainFlow)")
    p.add_argument(
        "--headset", default=None, help="a headset profile mapping its channels (e.g. ganglion)"
    )
    p.add_argument("--units", default=None, help="the stream's units if it doesn't declare them")
    p.add_argument("--threshold", type=float, default=0.6, help="the policy's θ (0.5-1)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seconds", type=float, default=None, help="stop after this long")
    p.add_argument(
        "--record",
        default=None,
        metavar="DIR",
        help="save the session (raw EEG, every event, the app's markers) to a new folder, "
        "e.g. sessions/2026-10-12-founder-a (local only, git-ignored)",
    )
    p.add_argument(
        "--calibration-trials",
        type=int,
        default=10,
        help="trials per class of the optional guided calibration (D-040; an app starts it)",
    )
    p.add_argument(
        "--recipe",
        default="configs/e2_finetune.yaml",
        help="the fine-tuning recipe: an experiment YAML with a finetune section (Q30's)",
    )
    p.add_argument(
        "--telemetry",
        action="store_true",
        help="also send what the engine sees, for the stage view (http://127.0.0.1:PORT/stage)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "session", help="summarise a recorded headset session and score its door game"
    )
    p.add_argument("folder", help="a folder written by `neuros run --record`")
    p.set_defaults(func=cmd_session)

    p = sub.add_parser("audit", help="cue-contamination audit of a cued dataset (D-030)")
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="default: the dataset's development people (usable, not in the locked test set)",
    )
    p.add_argument("--labels", nargs="+", default=["left_hand", "right_hand"])
    p.add_argument("--reference", choices=["as_recorded", "car"], default="as_recorded")
    p.add_argument(
        "--alignment-unit",
        choices=["person", "session"],
        default="session",
        help="one EA reference per recording session (as in E0) or per person",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=_default_fold_workers(),
        help="parallel leave-one-person-out folds (identical results; default: half the CPUs, "
        "at most 12, since each worker holds a copy of the covariances)",
    )
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute every recording instead of using the preprocessed cache (§4.5)",
    )
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("loso", help="train + test EEGNet leave-one-person-out (E0)")
    p.add_argument("--experiment", required=True, help="experiment YAML, e.g. configs/e0.yaml")
    p.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="default: the dataset's development people (usable, not in the locked test set)",
    )
    p.add_argument("--alignments", nargs="+", choices=["none", "ea"], default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--max-epochs", type=int, default=None, help="override (e.g. a smoke test)")
    p.add_argument("--threads", type=int, default=2, help="PyTorch threads per worker")
    p.add_argument(
        "--workers", type=int, default=None, help="parallel folds (default: CPUs / threads)"
    )
    p.add_argument("--out", default=None, help="result file (default: results/<name>/...)")
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute every recording instead of using the preprocessed cache (§4.5)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="cpu (default), cuda, cuda:N or auto; cuda spreads folds over the visible GPUs",
    )
    p.add_argument(
        "--nondeterministic",
        action="store_true",
        help="CUDA only: allow faster nondeterministic kernels (results vary in the last bits)",
    )
    p.set_defaults(func=cmd_loso)

    p = sub.add_parser(
        "e1", help="E1: pooled vs single-dataset vs leave-one-dataset-out, with cue-only controls"
    )
    p.add_argument("--experiment", required=True, help="E1 YAML, e.g. configs/e1.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=["pooled", "single", "lodo", "cue_only"],
        default=None,
        help="run only these model conditions",
    )
    p.add_argument("--max-epochs", type=int, default=None, help="override (e.g. a smoke test)")
    p.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="only the first N development people of each dataset (smoke tests)",
    )
    p.add_argument("--workers", type=int, default=1, help="models trained in parallel")
    p.add_argument("--threads", type=int, default=2, help="PyTorch threads per worker")
    p.add_argument("--device", default="cpu", help="cpu (default), cuda, cuda:N or auto")
    p.add_argument(
        "--nondeterministic",
        action="store_true",
        help="CUDA only: allow faster nondeterministic kernels (results vary in the last bits)",
    )
    p.add_argument("--out", default=None, help="result file (default: results/<name>/<run>.json)")
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute every recording instead of using the preprocessed cache (§4.5)",
    )
    p.set_defaults(func=cmd_e1)

    p = sub.add_parser("e6", help="E6: rest vs mental effort across people (EEGMAT)")
    p.add_argument("--experiment", default="configs/e6.yaml", help="E6 YAML")
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.add_argument("--out", default=None, help="result file (default: results/e6/<run>.json)")
    p.add_argument(
        "--max-people", type=int, default=None, help="only the first N people (smoke tests)"
    )
    p.set_defaults(func=cmd_e6)

    p = sub.add_parser("idle", help="E7: false commands at rest and during mental arithmetic")
    p.add_argument("--bundle", default="bundles/mvp-v0", help="the shipping bundle folder")
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.add_argument("--out", default=None, help="result file (default: results/e7/<run>.json)")
    p.add_argument(
        "--imagery-dataset", default="BNCI2014_004", help="real-imagery context (never seen)"
    )
    p.add_argument("--skip-imagery", action="store_true", help="skip the imagery context")
    p.add_argument(
        "--max-people", type=int, default=None, help="only the first N people (smoke tests)"
    )
    p.set_defaults(func=cmd_idle)

    for name, func, text in (
        ("freq-align", cmd_freq_align, "A5: per-person frequency alignment vs fixed bands"),
        (
            "four-electrodes",
            cmd_four_electrodes,
            "A4: the best four electrodes (commands + states)",
        ),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
        p.add_argument("--out", default=None, help="result file (default: results/<id>/...)")
        p.add_argument(
            "--max-people", type=int, default=None, help="only the first N people (smoke tests)"
        )
        p.add_argument(
            "--no-cache",
            action="store_true",
            help="recompute every recording instead of using the preprocessed cache (§4.5)",
        )
        if func is cmd_freq_align:
            p.add_argument(
                "--datasets", nargs="+", default=list(A5_DATASETS), help="the plan's two by default"
            )
        if func is cmd_four_electrodes:
            p.add_argument(
                "--state-experiment", default="configs/e6.yaml", help="E6's settings for states"
            )
        p.set_defaults(func=func)
    return parser


def cmd_e6(args: argparse.Namespace) -> int:
    """E6: rest vs mental effort across people (knowledge-base/plan/e6-state-channel.md)."""
    import datetime as dt

    from neuros.eval.state import load_state_config, run_e6
    from neuros.provenance import file_sha256, stamp

    config = _config(args)
    experiment_path = Path(args.experiment)
    split_path = Path(args.split)
    experiment = load_state_config(experiment_path)
    provenance = stamp(  # at the start: the code that runs
        experiment,
        experiment=experiment.name,
        experiment_sha256=file_sha256(experiment_path),
        split=str(split_path),
        split_sha256=file_sha256(split_path),
    )
    body = run_e6(config, experiment, split_path=split_path, max_people=args.max_people)
    complete = args.max_people is None
    run = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else config.results_dir / "e6" / f"e6-{run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"provenance": _finished(provenance), "complete": complete, **body}, indent=1),
        encoding="utf-8",
        newline="\n",
    )
    checks = body["checks"]
    print(
        f"checks: theta/alpha present {checks['theta_alpha_present']} (median ratio "
        f"{checks['spectrum_ratio_low_over_high']['median']:.1f}); rest/task parietal alpha "
        f"{checks['eyes_alpha_rest_over_task']['median']:.2f} (eyes probably closed: "
        f"{checks['eyes_probably_closed_at_rest']})"
    )
    if body["verdict"].get("outcome") == "STOPPED":
        print(f"STOPPED: {body['verdict']['reason']}; result: {out}")
        return 1
    rows = []
    for key, arms in body["main"].items():
        model, condition = key.split("/")
        for arm, summary in arms.items():
            drift = body["drift_control"][model][arm] if condition == "baseline" else None
            rows.append(
                [
                    key,
                    arm,
                    f"{summary['mean'] * 100:.1f}",
                    f"{summary['people_at_or_above']}/{summary['n_people']}",
                    f"{summary['wilcoxon_p_vs_50']:.2g}",
                    f"{summary['mean_auroc']:.3f}" if summary["mean_auroc"] is not None else "-",
                    f"{drift['mean'] * 100:.1f}" if drift else "-",
                ]
            )
    header = ["model/condition", "arm", "mean %", ">= 60%", "p vs 50%", "AUROC", "drift %"]
    print(_table(rows, header))
    late = body["late_rest_control"]
    print(
        f"primary model: {body['primary_model']}; late-rest control (report only): "
        + ", ".join(f"arm {arm} {late[arm]['mean'] * 100:.1f}%" for arm in late if arm != "model")
    )
    for name, judged in body["verdict"].items():
        print(f"VERDICT {name}: {judged['outcome']} {judged['checks']}")
    print(f"result: {out}{'' if complete else ' (smoke run: incomplete)'}")
    return 0


def _people_for(
    spec: DatasetSpec, split: str, purpose: str, max_people: int | None
) -> list[int] | None:
    """The first ``max_people`` development subjects (smoke runs), or None for all of them."""
    if max_people is None:
        return None
    return _dev_subjects(spec, None, split, purpose)[:max_people]


def _timestamped(config: DataConfig, name: str, out: str | None) -> Path:
    """``--out`` if given, else ``results/<name>/<name>-<UTC time>.json``."""
    import datetime as dt

    if out:
        return Path(out)
    run = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return config.results_dir / name / f"{name}-{run}.json"


def cmd_freq_align(args: argparse.Namespace) -> int:
    """A5: per-person frequency alignment vs fixed bands (knowledge-base/plan/a5-...)."""
    import numpy as np
    from scipy.stats import spearmanr

    from neuros.data.download import loading_dataset
    from neuros.data.prepare import prepare_subject
    from neuros.data.recording import person_id
    from neuros.data.registry import get_spec
    from neuros.data.vocabulary import default_vocabulary
    from neuros.eval import frequency
    from neuros.eval.smr import first_session_rows, welch_psd
    from neuros.eval.warmup import profile_channels
    from neuros.provenance import stamp

    config = _config(args)
    provenance = stamp(config, experiment="a5")  # at the start: the code that runs
    _warn_if_dirty(provenance)
    cache = _recording_cache(config, args)
    vocab = default_vocabulary()
    layout = _e1_layout("motor")
    labels = ["left_hand", "right_hand"]
    per_dataset: dict[str, dict[str, dict[str, float]]] = {}
    peaks: dict[str, dict[str, Any]] = {}
    for key in args.datasets:
        spec = get_spec(key)
        profile = profile_channels(layout, spec)
        subjects_arg = _people_for(spec, args.split, "A5", args.max_people)
        windows, _, subjects, _ = _dev_windows(
            config, key, subjects_arg, labels, "motor", args.split, "A5 (frequency)", cache
        )
        dataset = loading_dataset(spec, config.data_dir)
        for subject in subjects:
            if cache is None:
                recordings = prepare_subject(spec, subject, vocab, config, dataset=dataset)
            else:
                recordings, _ = cache.load_or_prepare(
                    spec, subject, vocab, config, prepare=prepare_subject, dataset=dataset
                )
            rows, sfreq, _ = first_session_rows(recordings, spec, profile, seconds=60.0)
            freqs, power = welch_psd(rows, sfreq)
            peaks[person_id(key, subject)] = frequency.person_frequency(freqs, power)
        used = {p: float(v["used_hz"]) for p, v in peaks.items() if p.startswith(key + ":")}
        per_dataset[key] = frequency.compare(
            windows.data, windows.sfreq, windows.labels, windows.person_ids, used
        )
        for condition, scores in per_dataset[key].items():
            print(f"{key} {condition}: {np.mean(list(scores.values())) * 100:.1f}%", flush=True)
    verdict = frequency.verdict(per_dataset)
    gains = {
        p: per_dataset[p.split(":")[0]]["aligned"][p] - per_dataset[p.split(":")[0]]["fixed"][p]
        for p in peaks
    }
    distance = [abs(float(peaks[p]["used_hz"]) - frequency.REFERENCE_HZ) for p in gains]
    result = spearmanr(distance, list(gains.values()))
    body = {
        "provenance": _finished(provenance),
        "complete": args.max_people is None and list(args.datasets) == list(A5_DATASETS),
        "plan": "knowledge-base/plan/a5-frequency-alignment.md",
        "protocol": {
            "band_edges_x_peak": list(frequency.BAND_EDGES),
            "peak_search_hz": list(frequency.PEAK_SEARCH_HZ),
            "peak_clip_hz": list(frequency.PEAK_CLIP_HZ),
            "min_peak_db": frequency.MIN_PEAK_DB,
            "peak_data": "first 60 s of the first session, C3/C4 (Q31's data)",
            "band_hz": list(config.signal.band_hz),
        },
        "peaks": peaks,
        "accuracy": per_dataset,
        "gain_vs_peak_distance_spearman": {
            "rho": float(result.statistic),
            "p_value": float(result.pvalue),
        },
        "verdict": verdict,
    }
    out = _timestamped(config, "a5", args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="\n")
    clear = sum(bool(v["clear"]) for v in peaks.values())
    used_all = np.array([float(v["used_hz"]) for v in peaks.values()])
    print(
        f"peaks: {clear}/{len(peaks)} clear; used {used_all.min():.1f}-{used_all.max():.1f} Hz "
        f"(median {np.median(used_all):.1f})"
    )
    print(
        f"VERDICT: {verdict['outcome']} gain {verdict['mean_gain_points']:+.2f} points "
        f"(p = {verdict['wilcoxon_p']:.2g}; by dataset {verdict['gain_by_dataset_points']}) "
        f"{verdict['checks']}"
    )
    print(f"result: {out}{'' if body['complete'] else ' (smoke run: incomplete)'}")
    return 0


def cmd_idle(args: argparse.Namespace) -> int:
    """E7: does the engine stay quiet when nobody is trying? (knowledge-base/plan/e7-...)"""
    import numpy as np

    from neuros.data.download import loading_dataset
    from neuros.data.eegmat import harmonize_eegmat_subject
    from neuros.data.harmonize import harmonize_moabb_subject
    from neuros.data.recording import person_id
    from neuros.data.registry import get_spec, task_epoch
    from neuros.data.validate import validate_recording
    from neuros.data.vocabulary import default_vocabulary
    from neuros.engine.bundle import ModelBundle
    from neuros.engine.pipeline import Engine
    from neuros.engine.policy import PolicySettings
    from neuros.engine.runner import score_replay
    from neuros.errors import NeurOSConfigError
    from neuros.eval import idle
    from neuros.eval.calibration import natural_key
    from neuros.provenance import stamp

    config = _config(args)
    provenance = stamp(config, experiment="e7", bundle=str(args.bundle))  # at the start
    _warn_if_dirty(provenance)
    bundle = ModelBundle.load(Path(args.bundle))
    seen = set(bundle.provenance.get("train_people", [])) | set(
        bundle.provenance.get("validation_people", [])
    )
    vocabulary = default_vocabulary()
    window_s, hop_s = bundle.contract.window_s, bundle.contract.hop_s

    def new_engine(channels: Sequence[str], sfreq: float, policy: PolicySettings) -> Engine:
        return Engine(bundle.decoder(), bundle.engine_config(channels, sfreq, policy=policy))

    def people(spec_key: str) -> list[int]:
        spec = get_spec(spec_key)
        subjects = _dev_subjects(spec, None, args.split, "E7")[: args.max_people or None]
        trained = [s for s in subjects if person_id(spec.key, s) in seen]
        if trained:  # a replay of a training person would understate false commands
            raise NeurOSConfigError(f"{spec.key} people {trained} trained this bundle")
        return subjects

    # 1. Idle: EEGMAT rest, a 5-s pause, mental arithmetic, on C3/Cz/C4.
    eegmat = get_spec("EEGMAT")
    counts: dict[float, list[dict[str, idle.SegmentCount]]] = {t: [] for t in idle.THRESHOLDS}
    per_person: dict[str, Any] = {}
    for subject in people(eegmat.key):
        recordings = harmonize_eegmat_subject(eegmat, subject, vocabulary, config.data_dir)
        for recording in recordings:  # data boundary checks before anything streams
            validate_recording(
                recording, eegmat, vocabulary, config.signal, config.validation
            ).raise_if_errors()
        person = recordings[0].person_id
        per_person[person] = {}
        for threshold in idle.THRESHOLDS:
            policy = idle.policy_for(bundle.classes, threshold)
            events, simulators, engine = idle.replay_recordings(
                new_engine, recordings, policy, channels=idle.IDLE_CHANNELS, seed=subject
            )
            segments = idle.count_segments(
                events,
                idle.replay_spans([s.duration_s for s in simulators]),
                window_s=window_s,
                settle_s=engine.config.settling_s,
                commands=list(policy.commands.values()),
            )
            counts[threshold].append(segments)
            per_person[person][f"{threshold:g}"] = {
                name: count.to_dict(hop_s) for name, count in segments.items()
            }
        at_bar = per_person[person][f"{idle.BAR_THRESHOLD:g}"]
        print(
            f"{person}: committed {at_bar['rest']['commit_rate'] * 100:.1f}% at rest, "
            f"{at_bar['arithmetic']['commit_rate'] * 100:.1f}% during arithmetic "
            f"(theta {idle.BAR_THRESHOLD})",
            flush=True,
        )
    summary = {
        f"{t:g}": {name: idle.pooled([c[name] for c in counts[t]], hop_s) for name in idle.SEGMENTS}
        for t in idle.THRESHOLDS
    }
    at_bar = summary[f"{idle.BAR_THRESHOLD:g}"]
    outcome = idle.verdict(at_bar["rest"]["commit_rate"], at_bar["arithmetic"]["commit_rate"])

    # 2. Context (not the bar): the same engine during real imagery, BNCI2014-004's first
    # sessions (as the 10 Oct replay ran), at each threshold.
    imagery: dict[str, Any] = {}
    if not args.skip_imagery:
        spec = get_spec(args.imagery_dataset)
        epoch = task_epoch(spec, bundle.contract)
        # Pooled over people per threshold: scored task decisions, commands, right commands,
        # and each person's balanced committed accuracy (people who committed at all).
        decided = dict.fromkeys(idle.THRESHOLDS, 0)
        committed = dict.fromkeys(idle.THRESHOLDS, 0)
        right = dict.fromkeys(idle.THRESHOLDS, 0)
        balanced: dict[float, list[float]] = {t: [] for t in idle.THRESHOLDS}
        for subject in people(spec.key):
            recordings = harmonize_moabb_subject(
                spec, subject, vocabulary, dataset=loading_dataset(spec, config.data_dir)
            )
            first = sorted({r.session for r in recordings}, key=natural_key)[0]
            runs = sorted(
                (r for r in recordings if r.session == first), key=lambda r: natural_key(r.run)
            )
            for threshold in idle.THRESHOLDS:
                policy = idle.policy_for(bundle.classes, threshold)
                events, simulators, _ = idle.replay_recordings(
                    new_engine, runs, policy, seed=subject
                )
                score = score_replay(
                    events,
                    [m for s in simulators for m in s.markers],
                    policy.commands,
                    window_s=window_s,
                    task_start_s=epoch.start_s,
                    task_end_s=spec.task_s,
                    tolerance_s=hop_s / 2,
                )
                decided[threshold] += score.task_decisions
                committed[threshold] += score.task_commands
                right[threshold] += score.task_correct
                if score.task_commands:
                    balanced[threshold].append(score.balanced_task_accuracy)
            print(f"{spec.key} subject {subject}: replayed", flush=True)
        imagery = {
            "dataset": spec.key,
            "channels": list(bundle.profile(runs[0].ch_names)),
            "session": "first",
            "class_on_screen": not epoch.cue_free,
            "per_threshold": {
                f"{t:g}": {
                    "task_decisions": decided[t],
                    "commit_rate": committed[t] / decided[t] if decided[t] else None,
                    "committed_right_pooled": right[t] / committed[t] if committed[t] else None,
                    "committed_right_balanced_mean": (
                        float(np.mean(balanced[t])) if balanced[t] else None
                    ),
                    "people_with_commands": len(balanced[t]),
                }
                for t in idle.THRESHOLDS
            },
        }

    complete = args.max_people is None
    out = _timestamped(config, "e7", args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "provenance": _finished(provenance),
        "complete": complete,
        "plan": "knowledge-base/plan/e7-quiet-when-idle.md",
        "channels": list(idle.IDLE_CHANNELS),
        "gap_s": idle.GAP_S,
        "thresholds": list(idle.THRESHOLDS),
        "summary": summary,
        "verdict": outcome if complete else {"outcome": "SMOKE", "note": "not all people"},
        "imagery_context": imagery,
        "per_person": per_person,
    }
    out.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="\n")
    rows = []
    for t in idle.THRESHOLDS:
        key = f"{t:g}"
        context = imagery.get("per_threshold", {}).get(key, {})
        rows.append(
            [
                key,
                f"{summary[key]['rest']['commit_rate'] * 100:.1f}%",
                f"{summary[key]['arithmetic']['commit_rate'] * 100:.1f}%",
                f"{summary[key]['rest']['onsets_per_minute_median']:.2f}",
                f"{summary[key]['arithmetic']['onsets_per_minute_median']:.2f}",
                _pct(context.get("commit_rate")),
                _pct(context.get("committed_right_pooled")),
            ]
        )
    print()
    print(
        _table(
            rows,
            [
                "theta",
                "rest commits",
                "arithmetic commits",
                "rest onsets/min",
                "arith. onsets/min",
                "imagery commits",
                "imagery right",
            ],
        )
    )
    print(f"\nverdict: {body['verdict']['outcome']} ({outcome['bar']}); result: {out}")
    return 0


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def cmd_four_electrodes(args: argparse.Namespace) -> int:
    """A4: the best four electrodes for commands and states (knowledge-base/plan/a4-...)."""
    from neuros.data.registry import get_spec
    from neuros.eval import four_electrodes as a4
    from neuros.eval.state import load_state_config, load_state_people
    from neuros.provenance import stamp

    config = _config(args)
    provenance = stamp(config, experiment="a4")  # at the start: the code that runs
    _warn_if_dirty(provenance)
    cache = _recording_cache(config, args)
    state_config = load_state_config(Path(args.state_experiment))
    people, _, _ = load_state_people(
        config, state_config, split_path=Path(args.split), max_people=args.max_people
    )
    yang = get_spec("Yang2025")
    windows, _, _, _ = _dev_windows(
        config,
        yang.key,
        _people_for(yang, args.split, "A4", args.max_people),
        ["left_hand", "right_hand"],
        "all",
        args.split,
        "A4 (four electrodes)",
        cache,
    )
    states_sel, states_conf = a4.split_halves([p.person for p in people], "EEGMAT")
    commands_sel, commands_conf = a4.split_halves(sorted(set(windows.person_ids)), "Yang2025")
    by_name = {p.person: p for p in people}
    montages = a4.candidate_montages()
    selection: dict[tuple[str, ...], dict[str, float]] = {}
    for index, montage in enumerate(montages, start=1):
        states = a4.states_scores([by_name[p] for p in states_sel], montage, state_config)
        commands = a4.commands_scores(windows, commands_sel, montage)
        selection[montage] = {"states": a4.mean(states), "commands": a4.mean(commands)}
        if index % 10 == 0 or index == len(montages):
            print(f"selection half: {index}/{len(montages)} montages scored", flush=True)
    today = tuple(sorted(a4.TODAY))
    winner = a4.pick(selection, today)
    confirmation = {
        "states": {
            "winner": a4.states_scores([by_name[p] for p in states_conf], winner, state_config),
            "today": a4.states_scores([by_name[p] for p in states_conf], today, state_config),
        },
        "commands": {
            "winner": a4.commands_scores(windows, commands_conf, winner),
            "today": a4.commands_scores(windows, commands_conf, today),
        },
    }
    verdict = a4.verdict(winner, today, confirmation["states"], confirmation["commands"])

    def top(measure: str) -> list[dict[str, Any]]:
        ranked = sorted(selection.items(), key=lambda kv: kv[1][measure], reverse=True)[:10]
        return [{"montage": list(m), **{k: round(v, 4) for k, v in s.items()}} for m, s in ranked]

    body = {
        "provenance": _finished(provenance),
        "complete": args.max_people is None,
        "plan": "knowledge-base/plan/a4-four-electrodes.md",
        "halves": {
            "EEGMAT": {"selection": states_sel, "confirmation": states_conf},
            "Yang2025": {"selection": commands_sel, "confirmation": commands_conf},
        },
        "selection": [{"montage": list(m), **s} for m, s in selection.items()],
        "top_states": top("states"),
        "top_commands": top("commands"),
        "today": {"montage": list(today), **selection[today]},
        "winner": {"montage": list(winner), **selection[winner]},
        "confirmation": confirmation,
        "verdict": verdict,
    }
    out = _timestamped(config, "a4", args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="\n")
    print(
        f"selection half: today {', '.join(today)}: states {selection[today]['states'] * 100:.1f}%"
        f", commands {selection[today]['commands'] * 100:.1f}%; pick {', '.join(winner)}: states "
        f"{selection[winner]['states'] * 100:.1f}%, commands "
        f"{selection[winner]['commands'] * 100:.1f}%"
    )
    print(f"VERDICT: {verdict['outcome']} {verdict['checks']}")
    if "states" in verdict:
        print(
            f"confirmation half: states {verdict['states']['mean_diff'] * 100:+.1f} points "
            f"(p = {verdict['states']['p_value']:.2g}), commands "
            f"{verdict['commands']['mean_diff'] * 100:+.1f} points"
        )
    print(f"result: {out}{'' if body['complete'] else ' (smoke run: incomplete)'}")
    return 0


def _utf8_output() -> None:
    """Write UTF-8 when output is redirected (Windows defaults to cp1252 and can't print µ, →).

    Interactive consoles already handle Unicode, so only non-TTY streams are reconfigured.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        isatty = getattr(stream, "isatty", None)
        if callable(reconfigure) and callable(isatty):
            try:
                if not isatty():
                    reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):  # pragma: no cover - exotic stream
                pass


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_output()
    # MOABB 1.7 asks MNE for the 'standard_1005' montage, which MNE 1.13 deprecates. It's
    # MOABB's internal call and harmless for us; hide it so real warnings stay visible.
    warnings.filterwarnings(
        "ignore", message="Montage name 'standard_1005' is deprecated", category=FutureWarning
    )
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except NeurOSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
