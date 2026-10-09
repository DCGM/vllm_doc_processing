"""Bounded text summary of earlier scans, sent with the next scan's image (see docs/PROMPTS.md).

The summary is derived deterministically from the stored observations, which it never modifies;
it is provisional (observations may be wrong) and is rebuilt from scratch for every scan.

Bounding rule: every quoted value is cut to ``VALUE_CHARS``, every list to a fixed number of items
(``BIBLIO_VALUES_PER_FIELD``, ``NUMBERED_SCANS``, ``IRREGULARITIES``, ``FAILED_SCANS``) and the
previous-scan lines to ``recent_scans``. If the text is still longer than ``max_chars``, the oldest
previous-scan lines are dropped first, then whole lines from the end, so the result never exceeds
``max_chars``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import BiblioField, PrintedNumber, ScanObservation, ScanRecord

CONTEXT_VERSION = "1"
"""Bump on any change of the context text format; recorded in ``run.prompt_versions['context']``."""

VALUE_CHARS = 80
BIBLIO_VALUES_PER_FIELD = 3
NUMBERED_SCANS = 5
IRREGULARITIES = 2
FAILED_SCANS = 5
SINGLE_VALUED = {
    BiblioField.TITLE,
    BiblioField.SUBTITLE,
    BiblioField.PART_NAME,
    BiblioField.PART_NUMBER,
    BiblioField.EDITION,
    BiblioField.PUBLICATION_DATE,
}

RECENT_HEADER = "Previous scans:"


def build_context(previous: Sequence[ScanRecord], *, recent_scans: int, max_chars: int) -> str | None:
    """Summary of ``previous`` (all earlier scans, in scan order); None if there are none."""
    if not previous:
        return None
    observed = [(s.scan_index + 1, s.observation) for s in previous if s.observation is not None]
    lines = [f'Scans 1-{previous[-1].scan_index + 1} so far; "scan N" is a scan position, not a page number.']
    lines += _bibliography(observed)
    lines += _numbering(observed)
    lines += _chapter(observed)
    lines += _toc(observed)
    lines += _open_questions(previous, observed)
    recent = [_scan_line(s) for s in previous[-recent_scans:]] if recent_scans else []
    return _fit(lines, recent, max_chars)


def _fit(lines: list[str], recent: list[str], max_chars: int) -> str:
    def text() -> str:
        return "\n".join(lines + ([RECENT_HEADER, *recent] if recent else []))

    while recent and len(text()) > max_chars:
        recent.pop(0)
    out = text()
    if len(out) > max_chars:
        out = out[: out.rfind("\n", 0, max_chars + 1)] if "\n" in out[: max_chars + 1] else out[:max_chars]
    return out


def _q(text: str) -> str:
    text = " ".join(text.split())
    return '"' + (text if len(text) <= VALUE_CHARS else text[: VALUE_CHARS - 3] + "...") + '"'


def _bibliography(observed: list[tuple[int, ScanObservation]]) -> list[str]:
    values: dict[BiblioField, dict[str, list[int]]] = {}
    for pos, obs in observed:
        for cand in obs.bibliographic_candidates:
            scans = values.setdefault(cand.field, {}).setdefault(cand.value, [])
            if pos not in scans:
                scans.append(pos)
    if not values:
        return ["Bibliographic data: none seen yet."]
    lines = ["Bibliographic data seen (field: value (scans)):"]
    for field in BiblioField:  # fixed order
        if field not in values:
            continue
        items = list(values[field].items())
        shown = "; ".join(f"{_q(v)} ({_positions(p)})" for v, p in items[:BIBLIO_VALUES_PER_FIELD])
        more = f"; +{len(items) - BIBLIO_VALUES_PER_FIELD} more" if len(items) > BIBLIO_VALUES_PER_FIELD else ""
        lines.append(f"- {field.value}: {shown}{more}")
    return lines


def _positions(positions: list[int]) -> str:
    shown = ", ".join(map(str, positions[:3]))
    return f"scan {shown}" + (", ..." if len(positions) > 3 else "")


def _number(n: PrintedNumber) -> str:
    label = n.normalized or n.raw
    return f"{label} ({n.side})" if n.side else label


def _numbering(observed: list[tuple[int, ScanObservation]]) -> list[str]:
    numbered = [(pos, obs.printed_numbers) for pos, obs in observed if obs.printed_numbers]
    if not numbered:
        return ["Printed page numbers: none seen yet."]
    recent = "; ".join(f"scan {pos}: {', '.join(map(_number, nums))}" for pos, nums in numbered[-NUMBERED_SCANS:])
    lines = [f"Printed page numbers, latest: {recent}."]
    if numbered[-1][0] != observed[-1][0]:
        lines.append(f"No printed page number on the observed scans after scan {numbered[-1][0]}.")
    return lines


def _irregularities(observed: list[tuple[int, ScanObservation]]) -> list[str]:
    """Places where a numeral system's values do not increase (restart, repeat or decrease)."""
    last: dict[str, tuple[int, int]] = {}
    found: list[str] = []
    for pos, obs in observed:
        for n in obs.printed_numbers:
            if n.numeric_value is None or n.numeral_system is None:
                continue
            prev = last.get(n.numeral_system)
            if prev is not None and n.numeric_value <= prev[1]:
                found.append(f"page number {n.numeric_value} on scan {pos} after {prev[1]} on scan {prev[0]}")
            last[n.numeral_system] = (pos, n.numeric_value)
    return found[-IRREGULARITIES:]


def _chapter(observed: list[tuple[int, ScanObservation]]) -> list[str]:
    """Latest heading per level; a heading closes deeper levels. Unknown level counts as level 1."""
    stack: dict[int, str] = {}
    for pos, obs in observed:
        for h in obs.headings:
            level = h.level or 1
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = f"level {h.level or '?'} {_q(h.text)} (scan {pos})"
    if not stack:
        return ["Current chapter: no heading seen yet."]
    return ["Current chapter (latest headings): " + "; ".join(stack[k] for k in sorted(stack)) + "."]


def _toc(observed: list[tuple[int, ScanObservation]]) -> list[str]:
    toc = [(pos, len(obs.toc_entries)) for pos, obs in observed if obs.toc_entries]
    if not toc:
        return []
    return [
        f"Table of contents entries seen on scan {', '.join(str(p) for p, _ in toc[:10])}"
        f"{', ...' if len(toc) > 10 else ''} ({sum(n for _, n in toc)} entries)."
    ]


def _open_questions(previous: Sequence[ScanRecord], observed: list[tuple[int, ScanObservation]]) -> list[str]:
    questions: list[str] = []
    failed = [s.scan_index + 1 for s in previous if s.observation is None]
    if failed:
        shown = ", ".join(map(str, failed[-FAILED_SCANS:]))
        questions.append(f"scan {shown} not observed (request failed){' and earlier' if len(failed) > FAILED_SCANS else ''}")
    values: dict[BiblioField, set[str]] = {}
    for _, obs in observed:
        for c in obs.bibliographic_candidates:
            values.setdefault(c.field, set()).add(c.value)
    competing = [f.value for f in BiblioField if f in SINGLE_VALUED and len(values.get(f, ())) > 1]
    if competing:
        questions.append(f"several forms seen for {', '.join(competing)}")
    questions += _irregularities(observed)
    return [f"Unresolved: {'; '.join(questions)}."] if questions else []


def _scan_line(scan: ScanRecord) -> str:
    obs = scan.observation
    head = f"- scan {scan.scan_index + 1}: "
    if obs is None:
        return head + "not observed"
    parts = [obs.page_type.value if obs.page_type else "page type unknown", obs.side or "side unknown"]
    if obs.subpages:
        parts[-1] += " (" + ", ".join(f"{s.side} {s.page_type.value if s.page_type else '?'}" for s in obs.subpages) + ")"
    parts.append("page " + ", ".join(map(_number, obs.printed_numbers)) if obs.printed_numbers else "no page number")
    parts += [f"heading {_q(h.text)}" for h in obs.headings[:2]]
    if len(obs.headings) > 2:
        parts.append(f"+{len(obs.headings) - 2} headings")
    if obs.toc_entries:
        parts.append(f"{len(obs.toc_entries)} TOC entries")
    return head + "; ".join(parts)
