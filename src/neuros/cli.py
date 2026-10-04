"""``neuros``: the developer command line (architecture §7.5).

Day-1 commands (data layer):

    neuros datasets [--check]                     registry table; --check cross-checks MOABB
    neuros montage                                C-core + headset coverage
    neuros split [--write]                        the locked test set (D-026)
    neuros download --dataset KEY [--subjects …]  fetch raw data + integrity manifest
    neuros validate --dataset KEY [--subjects …]  harmonise + validate, write a JSON report

Day-2 commands (signal chain, alignment, baselines):

    neuros filters                                the causal filter at every native rate in scope
    neuros baseline --dataset KEY [...]           Riemannian baselines (LOSO + session transfer)

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

from neuros import __version__
from neuros.config.schemas import DataConfig, load_data_config
from neuros.errors import NeurOSError

log = logging.getLogger("neuros")

DEFAULT_SPLIT_PATH = Path("configs/splits/locked_test_v1.yaml")


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


# ---------------------------------------------------------------------- commands
def cmd_datasets(args: argparse.Namespace) -> int:
    from neuros.data.registry import DATASETS, check_against_moabb, check_registry
    from neuros.data.vocabulary import default_vocabulary

    vocab = default_vocabulary()
    rows = []
    for spec in DATASETS.values():
        labels = sorted(set(vocab.event_maps.get(spec.key, {}).values()))
        rows.append(
            [
                spec.key,
                spec.role,
                str(spec.n_subjects),
                str(len(spec.eeg_channels)),
                f"{spec.native_sfreq:g}",
                ",".join(labels),
                ",".join(spec.unverified) or "-",
            ]
        )
    print(_table(rows, ["dataset", "role", "people", "EEG ch", "Hz", "labels", "unverified"]))
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
        present = tuple(ch for ch in committed if ch in spec.eeg_channels)
        rows.append([f"dataset:{spec.key}", str(len(present)), ",".join(present)])
    print()
    print(_table(rows, ["layout", "in C-core", "channels present"]))
    return status


def cmd_split(args: argparse.Namespace) -> int:
    from neuros.eval.splits import load_locked_test_set, make_locked_test_set, save_locked_test_set

    computed = make_locked_test_set()
    path = Path(args.path)
    if args.write:
        save_locked_test_set(computed, path)
        print(f"wrote {path} (sha256 {computed.digest[:16]}…)")
    rows = [[k, str(len(v)), " ".join(map(str, v))] for k, v in computed.test.items()]
    print(_table(rows, ["dataset", "test people", "subjects"]))
    print(f"total: {sum(len(v) for v in computed.test.values())} people")
    if path.is_file():
        stored = load_locked_test_set(path)
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
    manifest = download(spec, _subjects(args, spec.subjects), config.data_dir)
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
        json.dumps(
            {"provenance": stamp(config, dataset=spec.key), "recordings": results}, indent=1
        ),
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


def cmd_baseline(args: argparse.Namespace) -> int:
    from neuros.data.download import prepare_mne_environment
    from neuros.data.montage import load_c_core
    from neuros.data.prepare import subject_epochs
    from neuros.data.recording import person_id
    from neuros.data.registry import get_spec, moabb_dataset
    from neuros.data.vocabulary import default_vocabulary
    from neuros.eval.baselines import run_riemannian_baselines
    from neuros.eval.splits import load_locked_test_set
    from neuros.provenance import stamp
    from neuros.signal.windows import WindowSet

    config = _config(args)
    spec = get_spec(args.dataset)
    vocab = default_vocabulary()
    # Person-level isolation (principle 2): locked test people never enter a development run.
    locked = load_locked_test_set(Path(args.split))
    subjects = list(args.subjects) if args.subjects else list(locked.dev_subjects(spec.key))
    locked.assert_disjoint([person_id(spec.key, s) for s in subjects], "a baseline run")
    prepare_mne_environment(config.data_dir)
    dataset = moabb_dataset(spec)

    sets = []
    summaries = []
    for subject in subjects:
        windows, summary = subject_epochs(
            spec, subject, vocab, config, args.labels, dataset=dataset
        )
        sets.append(windows)
        summaries.append(summary.to_dict())
        counts = " ".join(f"{k}={v}" for k, v in sorted(summary.label_counts.items()))
        print(
            f"{summary.person}: {summary.n_recordings} recordings, {counts}, "
            f"flagged {summary.flagged_extreme} extreme / {summary.flagged_transient} transient, "
            f"skipped {summary.skipped_outside}"
        )
    windows = WindowSet.concat(sets)
    if args.channels == "c-core":
        windows = windows.pick_channels([ch for ch in load_c_core() if ch in windows.ch_names])
    print(
        f"\n{len(windows)} windows, {len(windows.people)} people, {len(windows.ch_names)} "
        f"channels ({args.channels}), reference {args.reference}\n"
    )
    result = run_riemannian_baselines(windows, reference=args.reference, progress=print)

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

    out = (
        config.results_dir
        / "baselines"
        / f"{spec.key}_{'-'.join(args.labels)}_{args.channels}_{args.reference}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": stamp(config, dataset=spec.key),
        "protocol": {
            "dataset": spec.key,
            "labels": list(args.labels),
            "channels": list(windows.ch_names),
            "channel_set": args.channels,
            "reference": args.reference,
            "epoch_s": list(config.signal.epoch_s),
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
    p.add_argument("--path", default=str(DEFAULT_SPLIT_PATH))
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
    p.add_argument("--channels", choices=["all", "c-core"], default="all")
    p.add_argument("--reference", choices=["as_recorded", "car"], default="as_recorded")
    p.add_argument("--split", default=str(DEFAULT_SPLIT_PATH), help="locked test set file")
    p.set_defaults(func=cmd_baseline)
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
