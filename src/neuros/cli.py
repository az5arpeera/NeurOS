"""``neuros``: the developer command line (architecture §7.5).

Day-1 commands (data layer):

    neuros datasets [--check]                     registry table; --check cross-checks MOABB
    neuros montage                                C-core + headset coverage
    neuros split [--write]                        the locked test set (D-026)
    neuros download --dataset KEY [--subjects …]  fetch raw data + integrity manifest
    neuros validate --dataset KEY [--subjects …]  harmonise + validate, write a JSON report

Exit codes: 0 = success, 1 = checks found problems, 2 = a NeurOS error (bad input/config).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
    for subject in _subjects(args, spec.subjects):
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
        p.set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
