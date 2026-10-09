"""Document-wide reconciliation after all scans are observed: ``scans[].observation`` -> ``resolved``.

Work is split between code and one text-only LLM request (see docs/PROMPTS.md):

* deterministic: page type and side per scan (copied from the observation), page-number sequences
  (observed labels, labels inferred between two agreeing printed numbers, conflicts, gaps), mapping
  TOC page references to scans, chapter hierarchy and bounds, and every invariant check;
* LLM: the merged bibliography, the chapter list (which TOC entry and heading belong together, titles,
  levels) and free-text doubts. Its answer refers to scans by position and is checked against the
  observations: unknown scans, values not grounded in any observation and references not printed in a
  TOC are dropped with a warning, and values that differ from what was observed are logged in ``changes``.

Observations are never modified. Inputs longer than ``reconcile_max_chars`` fail before any request
(no chunking yet).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import Field

from .context import SINGLE_VALUED
from .llm import LLMClient
from .models import (
    AnnotatedBook,
    BiblioField,
    Bibliography,
    Claim,
    LeafSide,
    PageLabel,
    PageType,
    PrintedNumber,
    ReconciliationChange,
    ReconciliationWarning,
    ResolvedBook,
    ResolvedScan,
    ResolvedSubpage,
    ScanObservation,
    ScanRecord,
    ScanSide,
    StrictModel,
    StructureNode,
)
from .prompts import PROMPT_VERSIONS, RECONCILE_SYSTEM

log = logging.getLogger(__name__)


class ReconcileError(Exception):
    """Reconciliation cannot run (e.g. the input exceeds ``reconcile_max_chars``); safe to show."""


# --- LLM response contract (scan numbers are 1-based positions, as in the input text) ------------


class BiblioValue(StrictModel):
    field: BiblioField
    value: str = Field(min_length=1)
    source_scans: list[int]
    notes: str | None


class Chapter(StrictModel):
    title: str = Field(min_length=1)
    level: int = Field(ge=1)
    toc_scans: list[int]
    printed_page_reference: str | None
    heading_scan: int | None
    notes: str | None


class Issue(StrictModel):
    code: str = Field(min_length=1)
    message: str
    scans: list[int]


class ReconcileResponse(StrictModel):
    bibliography: list[BiblioValue]
    chapters: list[Chapter]
    issues: list[Issue]


# --- Entry point -------------------------------------------------------------------------------


def reconcile_book(client: LLMClient, book: AnnotatedBook) -> AnnotatedBook:
    """Return a copy of ``book`` with ``resolved`` set and the reconcile call(s) added to ``run``.

    Raises ``ReconcileError`` (no request made) if the input is too long and ``LLMError`` (carrying
    every attempt's ``CallRecord``) if the request fails after retries; ``book`` is left unchanged.
    """
    config = client.config
    book = book.model_copy(deep=True)
    pagination = _pagination(book.scans)
    text = reconcile_input(book.scans, pagination)
    if len(text) > config.reconcile_max_chars:
        raise ReconcileError(
            f"reconciliation input has {len(text)} characters, more than reconcile_max_chars="
            f"{config.reconcile_max_chars}; raise the limit if the postprocess model's context can hold it "
            "(splitting the book into several requests is not implemented)"
        )
    book.run.postprocess_model = config.effective_postprocess_model
    book.run.prompt_versions["reconcile"] = PROMPT_VERSIONS["reconcile"]
    result = client.request(
        ReconcileResponse,
        stage="reconcile",
        model=config.effective_postprocess_model,
        system=RECONCILE_SYSTEM,
        user=text,
        max_output_tokens=config.reconcile_max_output_tokens,
    )
    book.run.calls += result.calls
    book.run.refresh_totals()
    book.resolved = _resolve(book.scans, pagination, result.value)
    book.run.finished_at = datetime.now(UTC)
    r = book.resolved
    log.info(
        "reconcile input_chars=%d chapters=%d inferred_labels=%d warnings=%d changes=%d",
        len(text),
        len(r.structure),
        sum(label.origin == "inferred" for s in r.scans for label in s.page_labels),
        len(r.warnings),
        len(r.changes),
    )
    return book


def _resolve(scans: list[ScanRecord], pagination: Pagination, response: ReconcileResponse) -> ResolvedBook:
    out = ResolvedBook(warnings=list(pagination.warnings))
    out.scans = [_resolved_scan(s, pagination.labels[s.scan_index]) for s in scans]
    refs = _RefChecker(scans, out.warnings)
    out.bibliography = _bibliography(scans, response.bibliography, refs, out)
    out.structure = _structure(scans, out.scans, response.chapters, refs, out)
    for issue in response.issues:
        out.warnings.append(
            ReconciliationWarning(
                code=issue.code,
                message=issue.message,
                detected_by="llm",
                scan_ids=refs.ids(issue.scans, f"issue {issue.code!r}"),
            )
        )
    return out


def _resolved_scan(scan: ScanRecord, labels: list[PageLabel]) -> ResolvedScan:
    obs = scan.observation
    if obs is None:
        return ResolvedScan(scan_id=scan.scan_id, page_labels=labels)
    src = [scan.scan_id]
    return ResolvedScan(
        scan_id=scan.scan_id,
        page_type=Claim[PageType](value=obs.page_type, source_scan_ids=src, confidence=obs.page_type_confidence)
        if obs.page_type
        else None,
        side=Claim[ScanSide](value=obs.side, source_scan_ids=src, confidence=obs.side_confidence) if obs.side else None,
        subpages=[
            ResolvedSubpage(
                side=sub.side,
                page_type=Claim[PageType](value=sub.page_type, source_scan_ids=src, confidence=sub.confidence)
                if sub.page_type
                else None,
            )
            for sub in obs.subpages
        ],
        page_labels=labels,
    )


# --- Page numbering ----------------------------------------------------------------------------


@dataclass
class _Anchor:
    k: int  # page ordinal: pages counted over all scans, a spread counts as two
    scan: int  # scan_index
    side: LeafSide | None
    system: str
    value: int
    label: str

    @property
    def offset(self) -> int:
        return self.value - self.k


@dataclass
class Pagination:
    labels: dict[int, list[PageLabel]]
    """Observed and inferred labels per scan_index."""
    runs: list[str] = field(default_factory=list)
    warnings: list[ReconciliationWarning] = field(default_factory=list)


def _pagination(scans: list[ScanRecord]) -> Pagination:
    """Observed labels, plus labels inferred on unnumbered pages strictly between two printed numbers of
    the same numeral system whose difference equals the number of pages between them. Nothing is
    extrapolated before the first or after the last printed number; disagreements become warnings."""
    labels: dict[int, list[PageLabel]] = {s.scan_index: [] for s in scans}
    pages: list[tuple[int, LeafSide | None, list[PrintedNumber]]] = []  # (scan_index, side, numbers)
    for s in scans:
        obs = s.observation
        for n in obs.printed_numbers if obs else []:
            labels[s.scan_index].append(
                PageLabel(
                    side=n.side,
                    label=n.normalized or n.raw,
                    numeric_value=n.numeric_value,
                    numeral_system=n.numeral_system,
                    source_scan_ids=[s.scan_id],
                    confidence=n.confidence,
                )
            )
        if obs is not None and obs.side == "both":
            unsided = [n for n in obs.printed_numbers if n.side is None]  # page unknown: blocks both pages
            pages += [(s.scan_index, side, [n for n in obs.printed_numbers if n.side == side] + unsided)
                      for side in ("left", "right")]
        else:
            side = obs.side if obs is not None and obs.side in ("left", "right") else None
            pages.append((s.scan_index, side, list(obs.printed_numbers) if obs else []))

    anchors = [
        _Anchor(k, scan, side, nums[0].numeral_system, nums[0].numeric_value, nums[0].normalized or nums[0].raw)
        for k, (scan, side, nums) in enumerate(pages)
        if len(nums) == 1 and nums[0].numeric_value is not None and nums[0].numeral_system in ("arabic", "roman")
    ]
    result = Pagination(labels)
    pos = {s.scan_index: s.scan_index + 1 for s in scans}
    ids = {s.scan_index: s.scan_id for s in scans}

    def warn(code: str, message: str, scan_indexes: list[int]) -> None:
        result.warnings.append(
            ReconciliationWarning(
                code=code,
                message=message,
                detected_by="check",
                field_path=f"resolved.scans[{scan_indexes[len(scan_indexes) // 2]}].page_labels",
                scan_ids=list(dict.fromkeys(ids[i] for i in scan_indexes)),
            )
        )

    outliers = set()
    for p, a, n in zip(anchors, anchors[1:], anchors[2:], strict=False):
        if p.system == a.system == n.system and p.offset == n.offset != a.offset:
            outliers.add(a.k)
            expected = _format(a.k + p.offset, a.system, p.label)
            warn(
                "page_number_conflict",
                f"scan {pos[a.scan]} shows page number {a.label}, but the numbers on scan {pos[p.scan]} "
                f"({p.label}) and scan {pos[n.scan]} ({n.label}) imply {expected}; kept as observed",
                [p.scan, a.scan, n.scan],
            )
    good = [a for a in anchors if a.k not in outliers]
    run = good[:1]
    for p, n in zip(good, good[1:], strict=False):
        between, step = n.k - p.k, n.value - p.value
        consistent = p.system == n.system and step == between
        if consistent:
            for k in range(p.k + 1, n.k):
                scan, side, nums = pages[k]
                if not nums:
                    labels[scan].append(
                        PageLabel(
                            side=side,
                            label=_format(k + p.offset, p.system, p.label),
                            numeric_value=k + p.offset,
                            numeral_system=p.system,
                            origin="inferred",
                            source_scan_ids=list(dict.fromkeys([ids[p.scan], ids[n.scan]])),
                        )
                    )
            run.append(n)
            continue
        result.runs.append(_run_line(run, pos))
        run = [n]
        where = f"from {p.label} (scan {pos[p.scan]}) to {n.label} (scan {pos[n.scan]})"
        if p.system != n.system:
            pass  # e.g. roman front matter followed by arabic body
        elif step <= 0:
            warn("page_number_sequence_break", f"page numbers do not increase {where}", [p.scan, n.scan])
        elif step > between:
            warn(
                "page_number_gap",
                f"page numbers jump {where} with {between - 1} page(s) in between: scans may be missing "
                "or a number misread; no labels inferred",
                [p.scan, n.scan],
            )
        # 0 < step < between: unnumbered extra leaves (e.g. plates); nothing inferred, nothing to flag
    if run:
        result.runs.append(_run_line(run, pos))
    return result


def _run_line(run: list[_Anchor], pos: dict[int, int]) -> str:
    first, last = run[0], run[-1]
    if first is last:
        return f"- scan {pos[first.scan]}: {first.system} {first.label}"
    return f"- scans {pos[first.scan]}-{pos[last.scan]}: {first.system} {first.label}-{last.label}"


_ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
          (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def _to_roman(value: int) -> str:
    out = ""
    for v, s in _ROMAN:
        while value >= v:
            out, value = out + s, value - v
    return out


def _roman_value(text: str) -> int | None:
    text, value, i = text.upper(), 0, 0
    for v, s in _ROMAN:
        while text.startswith(s, i):
            value, i = value + v, i + len(s)
    return value if text and i == len(text) and _to_roman(value) == text else None


def _format(value: int, system: str, like: str) -> str:
    if system != "roman":
        return str(value)
    roman = _to_roman(value) or str(value)
    return roman.lower() if like.islower() else roman


def _parse_reference(ref: str) -> tuple[str, int] | None:
    """Leading page number of a printed TOC reference ("17", "[xii]", "17-20"); None if there is none."""
    m = re.match(r"\W*([0-9]+|[ivxlcdm]+|[IVXLCDM]+)\b", ref)
    if not m:
        return None
    token = m.group(1)
    if token.isdigit():
        return "arabic", int(token)
    value = _roman_value(token)
    return ("roman", value) if value else None


# --- Input text --------------------------------------------------------------------------------


def _q(text: str) -> str:
    return json.dumps(" ".join(text.split()), ensure_ascii=False)


def reconcile_input(scans: list[ScanRecord], pagination: Pagination) -> str:
    """Compact text given to the LLM: numbering summary plus every scan with headings, TOC entries,
    bibliographic candidates or a change of page type. Values are never shortened."""
    lines = [f'Book with {len(scans)} scans; "scan N" is the position in scanning order, not a page number.']
    failed = [str(s.scan_index + 1) for s in scans if s.observation is None]
    if failed:
        lines.append(f"Not observed (request failed): scan {', '.join(failed)}.")
    lines += ["", "Printed page numbering (computed from the observations):"]
    lines += pagination.runs or ["- no page numbers observed"]
    lines += [f"- {w.code}: {w.message}" for w in pagination.warnings]
    lines += ["", "Scans with headings, table-of-contents entries, bibliographic data or a change of page type "
              "(all other scans are not listed):"]
    previous_type: object = None
    for s in scans:
        obs = s.observation
        if obs is None:
            continue
        if obs.headings or obs.toc_entries or obs.bibliographic_candidates or obs.page_type != previous_type:
            lines += _scan_lines(s.scan_index + 1, obs)
        previous_type = obs.page_type
    return "\n".join(lines)


def _scan_lines(position: int, obs: ScanObservation) -> list[str]:
    parts = [obs.page_type.value if obs.page_type else "page type unknown", obs.side or "side unknown"]
    numbers = [f"{n.normalized or n.raw}{f' ({n.side})' if n.side else ''}" for n in obs.printed_numbers]
    parts.append("page " + ", ".join(numbers) if numbers else "no page number")
    lines = [f"scan {position}: " + "; ".join(parts)]
    lines += [f"  heading level {h.level or '?'}: {_q(h.text)}" for h in obs.headings]
    lines += [
        f"  TOC entry level {e.level or '?'}: {_q(e.title)} -> "
        f"{_q(e.printed_page_reference) if e.printed_page_reference else 'no page reference'}"
        for e in obs.toc_entries
    ]
    lines += [f"  {c.field.value}: {_q(c.value)}" for c in obs.bibliographic_candidates]
    return lines


# --- Checking the LLM answer -------------------------------------------------------------------


def _norm(text: str) -> str:
    """Comparison key: case-, whitespace- and trailing-punctuation-insensitive."""
    return " ".join(re.sub(r"[.,;:]+(\s|$)", r"\1", text).split()).casefold()


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


class _RefChecker:
    """Maps 1-based scan positions from the LLM to scan IDs; out-of-range positions become warnings."""

    def __init__(self, scans: list[ScanRecord], warnings: list[ReconciliationWarning]):
        self.scans, self.warnings = scans, warnings
        self._by_id = {s.scan_id: s for s in scans}

    def ids(self, positions: list[int], what: str) -> list[str]:
        bad = [p for p in positions if not 1 <= p <= len(self.scans)]
        if bad:
            self.warnings.append(
                ReconciliationWarning(
                    code="invalid_scan_reference",
                    message=f"{what}: scan position(s) {bad} outside 1-{len(self.scans)} ignored",
                    detected_by="check",
                )
            )
        return list(dict.fromkeys(self.scans[p - 1].scan_id for p in positions if 1 <= p <= len(self.scans)))

    def scan(self, scan_id: str) -> ScanRecord:
        return self._by_id[scan_id]


def _bibliography(
    scans: list[ScanRecord], values: list[BiblioValue], refs: _RefChecker, out: ResolvedBook
) -> Bibliography:
    observed = [(s, c) for s in scans if s.observation for c in s.observation.bibliographic_candidates]
    for f in SINGLE_VALUED:
        forms = list(dict.fromkeys(_norm(c.value) for _, c in observed if c.field == f))
        if len(forms) > 1:
            out.warnings.append(
                ReconciliationWarning(
                    code="competing_observed_values",
                    message=f"{f.value} observed in {len(forms)} different forms",
                    detected_by="check",
                    field_path=f"resolved.bibliography.{f.value}",
                    scan_ids=list(dict.fromkeys(s.scan_id for s, c in observed if c.field == f)),
                )
            )
    biblio = Bibliography()
    for v in values:
        path = f"resolved.bibliography.{v.field.value}"
        exact = [s.scan_id for s, c in observed if c.field == v.field and _same(c.value, v.value)]
        similar = [s.scan_id for s, c in observed if _norm(c.value) == _norm(v.value)]
        cited = [i for i in refs.ids(v.source_scans, path) if refs.scan(i).observation
                 and refs.scan(i).observation.bibliographic_candidates]
        sources = list(dict.fromkeys(exact + similar)) or cited
        if not sources:
            out.warnings.append(
                ReconciliationWarning(
                    code="ungrounded_value",
                    message=f"{v.field.value} {v.value!r} proposed by the LLM is not supported by any "
                    "bibliographic observation; dropped",
                    detected_by="check",
                    field_path=path,
                )
            )
            continue
        claim = Claim[str](value=v.value, origin="observed" if exact else "inferred", source_scan_ids=sources,
                           notes=v.notes)
        current = getattr(biblio, v.field.value)
        if v.field in SINGLE_VALUED:
            if current is not None:
                out.warnings.append(
                    ReconciliationWarning(
                        code="competing_values",
                        message=f"LLM gave several values for {v.field.value}; kept {current.value!r}, "
                        f"not {v.value!r}",
                        detected_by="check",
                        field_path=path,
                        scan_ids=sources,
                    )
                )
                continue
            setattr(biblio, v.field.value, claim)
        else:
            if any(_norm(c.value) == _norm(v.value) for c in current):
                continue
            path = f"{path}[{len(current)}]"
            current.append(claim)
        if not exact:
            old = list(dict.fromkeys(c.value for _, c in observed if c.field == v.field))
            out.changes.append(
                ReconciliationChange(
                    field_path=path,
                    old_value=old or None,
                    new_value=v.value,
                    reason=v.notes or f"not observed in this form as {v.field.value}",
                    source_scan_ids=sources,
                )
            )
    return biblio


def _structure(
    scans: list[ScanRecord],
    resolved_scans: list[ResolvedScan],
    chapters: list[Chapter],
    refs: _RefChecker,
    out: ResolvedBook,
) -> list[StructureNode]:
    index = {s.scan_id: s.scan_index for s in scans}
    pages: dict[tuple[str, int], list[str]] = {}
    for r in resolved_scans:
        for label in r.page_labels:
            if label.numeral_system and label.numeric_value is not None:
                ids = pages.setdefault((label.numeral_system, label.numeric_value), [])
                if r.scan_id not in ids:
                    ids.append(r.scan_id)

    def warn(code: str, message: str, path: str, scan_ids: list[str]) -> None:
        out.warnings.append(
            ReconciliationWarning(code=code, message=message, detected_by="check", field_path=path, scan_ids=scan_ids)
        )

    nodes: list[StructureNode] = []
    for i, ch in enumerate(chapters):
        path, name = f"resolved.structure[{len(nodes)}]", f"chapter {ch.title!r}"
        toc_ids = refs.ids(ch.toc_scans, path + ".toc_scan_ids")
        heading_ids = refs.ids([ch.heading_scan] if ch.heading_scan is not None else [], path + ".heading_scan_ids")
        toc_obs = [e for sid in toc_ids if (o := refs.scan(sid).observation) for e in o.toc_entries]
        head_obs = [h for sid in heading_ids if (o := refs.scan(sid).observation) for h in o.headings]
        if not toc_obs and toc_ids:
            warn("toc_scan_without_entries", f"{name}: TOC scan has no observed TOC entries; ignored", path, toc_ids)
            toc_ids = []
        if not head_obs and heading_ids:
            warn("heading_not_observed", f"{name}: no heading observed on the given scan; ignored", path, heading_ids)
            heading_ids = []
        ref = ch.printed_page_reference
        if ref is not None and not any(e.printed_page_reference and _same(e.printed_page_reference, ref)
                                       for e in toc_obs):
            warn("ungrounded_reference", f"{name}: page reference {ref!r} not printed in its TOC entries; dropped",
                 path, toc_ids)
            ref = None
        if not toc_ids and not heading_ids:
            warn("ungrounded_chapter", f"{name}: neither a TOC entry nor a heading cited; dropped", path, [])
            continue

        target: str | None = None
        parsed = _parse_reference(ref) if ref else None
        if ref is not None:
            candidates = pages.get(parsed, []) if parsed else []
            if len(candidates) == 1:
                target = candidates[0]
            elif len(candidates) > 1:
                warn("ambiguous_toc_reference", f"{name}: page {ref!r} matches several scans; unresolved", path,
                     candidates)
            elif not heading_ids:
                warn("unresolved_toc_reference", f"{name}: no scan with page {ref!r}; start unresolved", path,
                     toc_ids)
        start = heading_ids[0] if heading_ids else target
        if heading_ids and target and target != heading_ids[0]:
            warn("toc_heading_mismatch",
                 f"{name}: heading on scan {index[heading_ids[0]] + 1}, TOC page {ref!r} is on scan "
                 f"{index[target] + 1}; start taken from the heading", path, [heading_ids[0], target])

        texts = [e.title for e in toc_obs] + [h.text for h in head_obs]
        title_observed = any(_same(t, ch.title) for t in texts)
        title_sources = list(dict.fromkeys(toc_ids + heading_ids))
        if not title_observed:
            out.changes.append(
                ReconciliationChange(
                    field_path=path + ".title",
                    old_value=None,
                    new_value=ch.title,
                    reason=ch.notes or "chapter title not observed in this form",
                    source_scan_ids=title_sources,
                )
            )
        parent = next((n for n in reversed(nodes) if n.level < ch.level), None)
        nodes.append(
            StructureNode(
                id=f"ch{i + 1}",
                parent_id=parent.id if parent else None,
                level=ch.level,
                title=Claim[str](value=ch.title, origin="observed" if title_observed else "inferred",
                                 source_scan_ids=title_sources),
                printed_page_reference=ref,
                toc_scan_ids=toc_ids,
                heading_scan_ids=heading_ids,
                start_scan_id=start,
                origin="observed",
                notes=ch.notes,
            )
        )

    _bounds(nodes, scans, index, warn)
    return nodes


def _bounds(nodes: list[StructureNode], scans: list[ScanRecord], index: dict[str, int], warn) -> None:
    """End = the scan before the next chapter of the same or a higher level (that scan itself if it is a
    spread or the same scan); unknown when that start is unknown or there is no such chapter."""
    seen: dict[tuple[str, str | None], str] = {}
    last: StructureNode | None = None
    for i, node in enumerate(nodes):
        path = f"resolved.structure[{i}]"
        if node.start_scan_id is None:
            continue
        key = (_norm(node.title.value) if node.title else "", node.start_scan_id)
        if key in seen:
            warn("duplicate_chapter", f"chapter {node.title.value!r} listed twice ({seen[key]}, {node.id})",
                 path, [node.start_scan_id])
        seen[key] = node.id
        if last is not None and index[node.start_scan_id] < index[last.start_scan_id]:
            warn("contradictory_order",
                 f"chapter {node.id} starts on scan {index[node.start_scan_id] + 1}, before the preceding "
                 f"chapter {last.id} (scan {index[last.start_scan_id] + 1})", path,
                 [last.start_scan_id, node.start_scan_id])
        last = node
    for i, node in enumerate(nodes):
        if node.start_scan_id is None:
            continue
        nxt = next((n for n in nodes[i + 1:] if n.level <= node.level), None)
        if nxt is None or nxt.start_scan_id is None:
            continue
        start, end = index[node.start_scan_id], index[nxt.start_scan_id]
        if end < start:
            continue
        obs = scans[end].observation
        if end > start and not (obs and obs.side == "both"):
            end -= 1
        node.end_scan_id = scans[end].scan_id
