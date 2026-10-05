"""Regenerate ``budget-guard.yaml`` with ``budget_guard.py`` inlined (CloudFormation ZipFile).

The Lambda's code lives in ``budget_guard.py`` (tested), and CloudFormation wants it inline in
the template so the whole guard deploys from one uploaded file. This script keeps the two in step:

    python infra/aws/build_template.py          # rewrite budget-guard.yaml
    python infra/aws/build_template.py --check  # exit 1 if the template is out of date

``tests/test_budget_guard.py`` runs the check, so an edit to one without the other fails CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "budget_guard.py"
TEMPLATE = HERE / "budget-guard.yaml"
MARKER = "        ZipFile: |\n"  # the inline code goes right after this line
INDENT = " " * 10  # two levels deeper than ``ZipFile:``


def render(template_text: str, code: str) -> str:
    """``template_text`` with the block after ``ZipFile: |`` replaced by ``code``.

    Raises:
        ValueError: if the marker is missing or appears more than once.
    """
    if template_text.count(MARKER) != 1:
        raise ValueError(f"expected exactly one {MARKER.strip()!r} line in the template")
    head, rest = template_text.split(MARKER)
    # The inline block ends at the first line indented less than INDENT (or the end).
    lines = rest.splitlines(keepends=True)
    end = next(
        (i for i, line in enumerate(lines) if line.strip() and not line.startswith(INDENT)),
        len(lines),
    )
    inlined = "".join(INDENT + line if line.strip() else "\n" for line in code.splitlines(True))
    tail = "".join(lines[end:])
    return head + MARKER + inlined.rstrip("\n") + "\n" + ("\n" + tail.lstrip("\n") if tail else "")


def main(argv: list[str]) -> int:
    code = SOURCE.read_text(encoding="utf-8")
    current = TEMPLATE.read_text(encoding="utf-8")
    wanted = render(current, code)
    if "--check" in argv:
        if wanted != current:
            print(f"{TEMPLATE.name} is out of date: run python infra/aws/build_template.py")
            return 1
        print(f"{TEMPLATE.name} matches {SOURCE.name}")
        return 0
    TEMPLATE.write_text(wanted, encoding="utf-8", newline="\n")
    print(f"wrote {TEMPLATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
