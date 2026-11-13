"""Check the manual against the real software.

    python docs/manual_source/verify_manual.py

1. Every `rakshpulse ...` command printed in the manual (terminal blocks and inline code) parses with the real CLI
   parser (``rakshpulse.cli.build_parser``): option names, choices and types are checked.
2. The generated PDF exists, has 15 chapters and at least 12 figures, and its text contains the required honesty
   statements and none of the forbidden claims (pin numbers, validated-for-MAX30102, tested firmware).
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import content  # noqa: E402

from rakshpulse.cli import build_parser  # noqa: E402

PDF = HERE.parent / "RAKSHPULSE_HARDWARE_INTEGRATION_MANUAL.pdf"


class Recorder:
    """Stand-in for the PDF builder that only collects text."""

    def __init__(self):
        self.texts: list[str] = []
        self.terminal_lines: list[str] = []

    def __getattr__(self, name):
        def collect(*args, **kw):
            for a in list(args) + list(kw.values()):
                self._walk(a)

        return collect

    def _walk(self, a):
        if isinstance(a, str):
            self.texts.append(a)
        elif isinstance(a, (list, tuple)):
            for x in a:
                self._walk(x)

    def terminal(self, code, shell="Terminal", note=None):
        self.terminal_lines += [ln[2:] for ln in code.strip("\n").split("\n") if ln.startswith("> ")]


def commands(rec: Recorder) -> list[str]:
    found = [ln for ln in rec.terminal_lines if ln.startswith("rakshpulse ")]
    for t in rec.texts:
        found += re.findall(r"`(rakshpulse [^`]+)`", t)
        if t.startswith("rakshpulse ") and "`" not in t:  # command-reference examples
            found.append(t)
    return sorted(set(found))


def check_commands() -> int:
    rec = Recorder()
    content.build(rec)
    parser = build_parser()
    bad = 0
    cmds = commands(rec)
    subcommands = set(parser._subparsers._group_actions[0].choices)
    for c in cmds:
        if re.match(r"^rakshpulse [\w-]+: ", c):  # a heading, not a command
            continue
        try:
            argv = shlex.split(c)[1:]
        except ValueError as e:
            bad += 1
            print(f"  {'QUOTING':<11} {c} ({e})")
            continue
        argv = [re.sub(r"<fs[^>]*>", "100", re.sub(r"<PORT>", "COM5", a)) for a in argv]  # templates
        if len(argv) == 1:  # a mention of a command in running text: only the name is checked
            status = "ok (name)" if argv[0] in subcommands else "PARSE ERROR"
        else:
            try:
                with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    parser.parse_args(argv)
                status = "ok"
            except SystemExit as e:
                status = "ok (help)" if e.code == 0 else "PARSE ERROR"
        if status == "PARSE ERROR":
            bad += 1
        print(f"  {status:<11} {c}")
    print(f"{len(cmds)} commands checked, {bad} failed")
    return bad


def check_pdf() -> int:
    from pypdf import PdfReader

    r = PdfReader(str(PDF))
    text = "\n".join(p.extract_text() for p in r.pages)
    flat = re.sub(r"\s+", " ", text)
    problems = []
    chapters = len([o for o in r.outline if not isinstance(o, list)])
    figures = len(set(re.findall(r"Figure (\d+)\.", flat)))
    print(f"pages {len(r.pages)}, chapters (bookmarks) {chapters}, figures {figures}")
    if chapters != 15:
        problems.append("expected 15 chapters")
    if figures < 12:
        problems.append("fewer than 12 figures")
    required = [
        "not a medical device",
        "not been compiled or run",
        "demonstration",
        "not a blood-pressure measurement",
        "No model is validated for MAX30102",
        "no pin numbers",
    ]
    for s in required:
        if s.lower() not in flat.lower():
            problems.append(f"missing statement: {s!r}")
    forbidden = [
        r"\bGPIO ?\d+",
        r"\bpin \d+\b",
        r"\bD[12]\b.*\bSDA\b",
        r"firmware (has been|was) tested",
        r"(?<!not )(?<!No )(?<!never )validated for MAX30102",
        r"\bis validated\b",
    ]
    for pat in forbidden:
        for m in re.finditer(pat, flat, flags=re.I):
            ctx = flat[max(0, m.start() - 60) : m.end() + 40]
            if re.search(r"\b(no|not|never)\b", flat[max(0, m.start() - 30) : m.start()], flags=re.I):
                continue  # a negated statement, e.g. "No RakshPulse model is validated for MAX30102"
            problems.append(f"forbidden pattern {pat!r}: ...{ctx}...")
    for p in problems:
        print("  PROBLEM:", p)
    print("pdf checks:", "PASS" if not problems else f"{len(problems)} problem(s)")
    return len(problems)


if __name__ == "__main__":
    n = check_commands() + check_pdf()
    sys.exit(1 if n else 0)
