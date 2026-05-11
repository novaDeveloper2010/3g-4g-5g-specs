#!/usr/bin/env python3
"""Run a Claude Code sanity check against the 3G/4G/5G PDF specs.

This script:
  1. extracts text from the PDFs (prefers pdftotext, falls back to pypdf)
  2. searches for a small set of high-value candidate terms
  3. sends the evidence to Claude Code (Opus 4.7) for a structured sanity review
  4. prints and optionally saves a JSON report

Typical usage:
  python3 sanity_check_with_claude.py \
    --pdfs 3g/TS_102_223_CAT_V7_17_0.pdf 4g/TS_102_223_CAT_V14_2_0.pdf 5g/TS_102_223_CAT_V18_2_0.pdf \
    --out sanity_report.json

You can also pass your own terms with --term.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_TERMS = [
    "PROVIDE LOCAL INFORMATION (NMR(UTRAN/E-UTRAN/Satellite E-UTRAN/NGRAN/Satellite NG-RAN))",
    "PROVIDE LOCAL INFORMATION, Slice(s) information",
    "PROVIDE LOCAL INFORMATION, Rejected Slice(s) Information",
    "Support of Extended information for PLI",
    "Support of chaining of PLI/Envelope commands",
    "5G ProSe usage information reporting",
    "PROVIDE LOCAL INFORMATION (NG-RAN/Satellite NG-RAN Timing Advance Information)",
    "Event: Network Rejection for NGRAN",
    "Event: Network Rejection for Satellite NG-RAN",
    "Event: Slices Status Change",
]


@dataclass
class Match:
    pdf: str
    line: int
    context: str


@dataclass
class Check:
    term: str
    matches: List[Match]


PDF_LINE_RE = re.compile(r"(?mi)^.*$")


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pdftotext or pypdf."""
    if shutil.which("pdftotext"):
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout

    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency may not exist
        raise RuntimeError(
            f"Could not extract text from {pdf_path}. Install pdftotext or pypdf."
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def find_context_lines(text: str, pattern: str, window: int = 120) -> List[tuple[int, str]]:
    """Find approximate matches even when PDF text wraps across lines."""
    normalized_text = re.sub(r"\s+", " ", text)
    normalized_pattern = re.sub(r"\s+", " ", pattern).strip()
    if not normalized_pattern:
        return []

    lower_text = normalized_text.lower()
    lower_pattern = normalized_pattern.lower()
    hits: List[tuple[int, str]] = []
    start = 0
    while True:
        idx = lower_text.find(lower_pattern, start)
        if idx == -1:
            break
        snippet_start = max(0, idx - window)
        snippet_end = min(len(normalized_text), idx + len(normalized_pattern) + window)
        snippet = normalized_text[snippet_start:snippet_end].strip()
        line_no = text[: min(len(text), max(0, idx))].count("\n") + 1
        hits.append((line_no, snippet))
        start = idx + len(lower_pattern)
    return hits


def collect_evidence(pdfs: Sequence[Path], terms: Sequence[str]) -> Dict[str, object]:
    pdf_text: Dict[str, str] = {}
    evidence: List[Check] = []

    for pdf in pdfs:
        pdf_text[str(pdf)] = extract_pdf_text(pdf)

    for term in terms:
        matches: List[Match] = []
        for pdf in pdfs:
            snippets = find_context_lines(pdf_text[str(pdf)], term)
            for line_no, snippet in snippets[:5]:
                matches.append(Match(pdf=str(pdf), line=line_no, context=snippet))
        evidence.append(Check(term=term, matches=matches))

    return {
        "pdfs": [str(p) for p in pdfs],
        "terms": list(terms),
        "evidence": [asdict(item) for item in evidence],
    }


def run_claude_review(evidence: Dict[str, object], model: str) -> dict:
    prompt = (
        "You are auditing ETSI USAT/CAT PDF extracts for a sanity check. "
        "For each term, classify it using only the provided evidence into one of: "
        "present_in_5g_only, shared_across_generations, not_supported_by_evidence, needs_manual_review. "
        "Return strict JSON with this shape:\n"
        '{"results":[{"term":"...","verdict":"...","confidence":"low|medium|high","reason":"...","pdfs":["..."],"notes":"..."}],"summary":"..."}\n\n'
        "Evidence JSON follows:\n"
        f"{json.dumps(evidence, indent=2, ensure_ascii=False)}"
    )

    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
        "--max-turns",
        "1",
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Claude Code failed with exit code %s\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (proc.returncode, proc.stdout, proc.stderr)
        )

    raw = proc.stdout.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Claude sometimes wraps JSON in fences. Strip them and try once more.
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claude Code PDF sanity checker")
    parser.add_argument(
        "--pdfs",
        nargs="+",
        type=Path,
        required=True,
        help="PDF files to inspect",
    )
    parser.add_argument(
        "--term",
        action="append",
        default=[],
        help="Additional search term to include (may be repeated)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to save the JSON report",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Claude model name (default: claude-opus-4-7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract evidence and print it without calling Claude",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pdfs = [p.expanduser().resolve() for p in args.pdfs]
    missing = [str(p) for p in pdfs if not p.exists()]
    if missing:
        eprint("Missing PDF(s):")
        for item in missing:
            eprint(f"  - {item}")
        return 2

    terms = list(DEFAULT_TERMS)
    for extra in args.term:
        if extra not in terms:
            terms.append(extra)

    evidence = collect_evidence(pdfs, terms)

    if args.dry_run:
        report = {"mode": "dry-run", "evidence": evidence}
    else:
        report = run_claude_review(evidence, args.model)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)

    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        eprint(f"Wrote report to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
