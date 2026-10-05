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

Exit codes: 0 = success, 1 = checks found problems, 2 = a NeurOS error (bad input/config).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neuros import __version__
from neuros.config.schemas import DataConfig, load_data_config
from neuros.errors import NeurOSError

if TYPE_CHECKING:
    from neuros.config.experiment import LosoExperimentConfig
    from neuros.data.cache import RecordingCache
    from neuros.data.registry import DatasetSpec, TaskEpoch
    from neuros.signal.windows import WindowSet

log = logging.getLogger("neuros")

# The current locked test set (D-033: v2 extends D-026's v1 with the cue-free datasets).
DEFAULT_SPLIT_PATH = Path("configs/splits/locked_test_v2.yaml")


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
    from neuros.data.download import prepare_mne_environment
    from neuros.data.harmonize import harmonize_moabb_subject
    from neuros.data.registry import get_spec, moabb_dataset
    from neuros.data.validate import validate_recording
    from neuros.data.vocabulary import default_vocabulary
    from neuros.provenance import stamp

    config = _config(args)
    spec = get_spec(args.dataset)
    provenance = stamp(config, dataset=spec.key)  # at the start: the code that runs
    vocab = default_vocabulary()
    prepare_mne_environment(config.data_dir)
    dataset = moabb_dataset(spec)
    results: list[dict[str, object]] = []
    rows: list[list[str]] = []
    pool = spec.subjects if args.include_excluded else spec.usable_subjects
    if spec.excluded_subjects and not args.include_excluded and not args.subjects:
        print(
            f"skipping excluded subjects {sorted(spec.excluded_subjects)} "
            "(see the registry for reasons; --include-excluded checks them anyway)\n"
        )
    for subject in _subjects(args, pool):
        try:
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
    from neuros.data.download import prepare_mne_environment
    from neuros.data.montage import load_c_core, region_channels
    from neuros.data.prepare import subject_epochs
    from neuros.data.registry import get_spec, moabb_dataset, task_epoch
    from neuros.data.vocabulary import default_vocabulary
    from neuros.errors import NeurOSConfigError
    from neuros.signal.windows import WindowSet

    spec = get_spec(dataset_key)
    # Resolved first: a dataset that can't host the epoch fails before anything loads.
    epoch = task_epoch(spec, config.signal)
    print(f"{spec.key}: windows {epoch.describe()}", flush=True)
    vocab = default_vocabulary()
    subjects = _dev_subjects(spec, subjects_arg, split_path, purpose)
    prepare_mne_environment(config.data_dir)
    dataset = moabb_dataset(spec)

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
    from neuros.data.download import prepare_mne_environment
    from neuros.data.montage import SCALP_REGIONS, region_channels
    from neuros.data.prepare import prepare_subject
    from neuros.data.recording import person_id
    from neuros.data.registry import get_spec, moabb_dataset
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
    for window in windows_spec:
        print(f"{window.name}: {window.epoch.describe()}", flush=True)
    prepare_mne_environment(config.data_dir)
    dataset = moabb_dataset(spec)
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
    return parser


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
