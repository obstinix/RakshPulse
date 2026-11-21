"""Check the manual against the real software and repository.

    python docs/manual/verify_manual.py

1. Every `rakshpulse ...` command printed in the manual (typed text, captured outputs and inline code) parses with
   the real CLI parser (``rakshpulse.cli.build_parser``): option names, choices and types are checked.
2. Every repository path the manual names (``hardware/...``, ``docs/...``, ``reports/...`` ...) exists, and every
   figure and captured output it uses exists.
3. The generated PDF exists, has 15 chapters (bookmarks) and at least 20 figures, contains the required honesty
   statements and labels, and none of the forbidden claims (invented pin numbers, a validated MAX30102 model,
   tested firmware, an ESP32 that shows SBP/DBP).
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import content  # noqa: E402

from rakshpulse.cli import build_parser  # noqa: E402

PDF = HERE.parent / "RAKSHPULSE_COMPLETE_HARDWARE_AND_OPERATOR_MANUAL.pdf"
SYNTHETIC_LABEL = "ILLUSTRATIVE SYNTHETIC EXAMPLE — NOT A REAL BLOOD-PRESSURE MEASUREMENT"
TEMPLATES = {r"<PORT>": "COM5", r"<fs[^>]*>": "100", r"<SBP>|<S>": "120", r"<DBP>|<D>": "80"}
PATH_RE = re.compile(r"\b((?:hardware|docs|reports|models|src|tests|scripts|examples|notebooks)/[\w./-]+\.\w+)")


class Recorder:
    """Stand-in for the PDF builder that only collects text, commands and the files it would embed."""

    def __init__(self):
        self.texts: list[str] = []
        self.terminal_lines: list[str] = []
        self.files: list[Path] = []

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

    def output(self, name, shell="Terminal", note=None, max_lines=None):
        path = HERE / "outputs" / f"{name}.txt"
        self.files.append(path)
        if path.exists():
            self.terminal(path.read_text(encoding="utf-8"))

    def image(self, name, title, caption, width=1.0):
        self.files.append((HERE / "figures" / name).resolve())
        self.texts += [title, caption]


def recorded() -> Recorder:
    rec = Recorder()
    content.build(rec)
    return rec


def commands(rec: Recorder) -> list[str]:
    found = [ln for ln in rec.terminal_lines if ln.startswith("rakshpulse ")]
    for t in rec.texts:
        found += re.findall(r"`(rakshpulse [^`]+)`", t)
        if t.startswith("rakshpulse ") and "`" not in t:  # command-reference examples
            found.append(t)
    return sorted(set(found))


def check_commands(rec: Recorder) -> int:
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
        for pat, val in TEMPLATES.items():
            argv = [re.sub(pat, val, a) for a in argv]
        if argv and argv[-1] in ("…", "..."):  # an abbreviated mention: only the command name is checked
            argv = argv[:1]
        if len(argv) == 1 and not argv[0].startswith("-"):  # a mention in running text: only the name is checked
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


def check_paths(rec: Recorder) -> int:
    bad = 0
    named = sorted({m.rstrip(".") for t in rec.texts for m in PATH_RE.findall(t)})
    for rel in named:
        ok = (ROOT / rel).exists()
        bad += not ok
        print(f"  {'ok' if ok else 'MISSING':<8} {rel}")
    files = sorted(set(rec.files))
    for f in files:
        if not f.exists():
            bad += 1
            print(f"  MISSING  {f}")
    print(f"{len(named)} repository paths and {len(files)} figure/output files checked, {bad} missing")
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
    if figures < 20:
        problems.append("fewer than 20 figures")
    required = [
        "not a medical device",
        "could not compile or run the sketch",
        "Do not guess 3.3 V or 5 V",
        "demonstration",
        "not validated for MAX30102",
        "Why Your ESP32 Is Not Showing SYS and DIA — What Is Currently Missing?",
        "Exact sensor model not confirmed from the available source documentation",
        "Why can't the MAX30102 directly display blood pressure?",
        "What Will I Actually See",
    ]
    for s in required:
        if s.lower() not in flat.lower():
            problems.append(f"missing statement: {s!r}")
    n_label = flat.count(SYNTHETIC_LABEL)
    print(f"synthetic-example labels: {n_label}")
    if n_label < 3:
        problems.append(f"the synthetic-example label appears {n_label} times (expected at least 3)")
    forbidden = [
        r"\bGPIO ?\d+",
        r"\bpin \d+\b",
        r"\bD[12]\b.{0,20}\bSDA\b",
        r"firmware (has been|was) tested",
        r"(?<!not )(?<!No )(?<!never )validated for MAX30102",
        r"\bis validated\b",
        r"ESP32 (shows|displays) (SBP|SYS|systolic)",
    ]
    for pat in forbidden:
        for m in re.finditer(pat, flat, flags=re.I):
            ctx = flat[max(0, m.start() - 60) : m.end() + 40]
            if re.search(r"\b(no|not|never)\b", flat[max(0, m.start() - 30) : m.start()], flags=re.I):
                continue  # a negated statement, e.g. "No RakshPulse model is validated for MAX30102"
            if pat.startswith(r"\bGPIO") and "[12]" in flat[m.end() : m.end() + 40]:
                continue  # the cited Arduino-ESP32 'ESP32C3 Dev Module' default (SDA = GPIO 8, SCL = GPIO 9) [12]
            problems.append(f"forbidden pattern {pat!r}: ...{ctx}...")
    for p in problems:
        print("  PROBLEM:", p)
    print("pdf checks:", "PASS" if not problems else f"{len(problems)} problem(s)")
    return len(problems)


if __name__ == "__main__":
    rec = recorded()
    n = check_commands(rec) + check_paths(rec) + check_pdf()
    print("verify_manual:", "PASS" if not n else f"FAIL ({n})")
    sys.exit(1 if n else 0)
