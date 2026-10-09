"""Document-wide reconciliation after all scans are observed: ``scans[].observation`` -> ``resolved``.

Work is split between code and one text-only LLM request (see docs/PROMPTS.md):

* LLM: corrections of clearly wrong page types and sides, the merged bibliography, the chapter list
  (which TOC entry and heading belong together, titles, levels) and free-text doubts;
* deterministic: everything else -- page labels in Czech NDK notation (``pagination``), mapping TOC
  page references to scans, chapter hierarchy and bounds, and every check of the LLM answer against
  the observations: unknown scans, bibliographic values and chapter titles that match no observed candidate,
  TOC entry or heading, and references not printed in the matching TOC entry are dropped with a warning;
  resolved values that differ from the observations are logged in ``changes``.

Observations are never modified. Inputs longer than ``reconcile_max_chars`` fail before any request
(no chunking yet).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from .context import SINGLE_VALUED
from .llm import LLMClient
from .models import (
    AnnotatedBook,
    BiblioField,
    Bibliography,
    Claim,
    PageType,
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
from .pagination import Pagination, paginate, roman_value
from .prompts import PROMPT_VERSIONS, RECONCILE_SYSTEM

log = logging.getLogger(__name__)


class ReconcileError(Exception):
    """Reconciliation cannot run (e.g. the input exceeds ``reconcile_max_chars``); safe to show."""


# --- LLM response contract (scan numbers are 1-based positions, as in the input text) ------------


class ScanCorrection(StrictModel):
    scan: int
    page_type: PageType | None = Field(description="New page type, or null to keep it.")
    side: Literal["keep", "left", "right", "both", "none"] = Field(description="'none' = not applicable/unknown.")
    reason: str = Field(min_length=1)


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
    scan_corrections: list[ScanCorrection]
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
    text = reconcile_input(book.scans, _paginate(book.scans, [_observed_scan(s) for s in book.scans]))
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
    book.resolved = _resolve(book.scans, result.value)
    book.run.finished_at = datetime.now(UTC)
    r = book.resolved
    log.info(
        "reconcile input_chars=%d corrections=%d chapters=%d unlabelled_scans=%d warnings=%d changes=%d",
        len(text),
        sum(c.field_path.startswith("resolved.scans[") for c in r.changes),
        len(r.structure),
        sum(s.page_number is None for s in r.scans),
        len(r.warnings),
        len(r.changes),
    )
    return book


def _resolve(scans: list[ScanRecord], response: ReconcileResponse) -> ResolvedBook:
    out = ResolvedBook()
    refs = _RefChecker(scans, out.warnings)
    out.scans = [_observed_scan(s) for s in scans]
    _apply_corrections(scans, out, response.scan_corrections, refs)
    pagination = _paginate(scans, out.scans)
    for s, r in zip(scans, out.scans, strict=True):
        r.page_labels = pagination.labels[s.scan_index]
        r.page_number = pagination.page_number(s.scan_index)
    out.warnings += pagination.warnings
    out.bibliography = _bibliography(scans, response.bibliography, out)
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


def _paginate(scans: list[ScanRecord], resolved: list[ResolvedScan]) -> Pagination:
    return paginate(
        scans,
        [r.page_type.value if r.page_type else None for r in resolved],
        [r.side.value if r.side else None for r in resolved],
        [{sp.side: sp.page_type.value if sp.page_type else None for sp in r.subpages} for r in resolved],
    )


def _observed_scan(scan: ScanRecord) -> ResolvedScan:
    """Page type, side and subpages as observed (labels are added later)."""
    obs = scan.observation
    if obs is None:
        return ResolvedScan(scan_id=scan.scan_id)
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
    )


def _apply_corrections(
    scans: list[ScanRecord], out: ResolvedBook, corrections: list[ScanCorrection], refs: _RefChecker
) -> None:
    """Apply the LLM's page type/side corrections to observed scans (first correction per scan wins)."""
    done: set[int] = set()
    for c in corrections:
        ids = refs.ids([c.scan], "scan correction")
        if not ids or c.scan in done:
            continue
        done.add(c.scan)
        i, scan_id = c.scan - 1, ids[0]
        r, path = out.scans[i], f"resolved.scans[{c.scan - 1}]"
        if scans[i].observation is None:
            out.warnings.append(
                ReconciliationWarning(
                    code="correction_of_unobserved_scan",
                    message=f"scan {c.scan} was not observed; correction ignored",
                    detected_by="check",
                    field_path=path,
                    scan_ids=ids,
                )
            )
            continue
        old_type = r.page_type.value if r.page_type else None
        if c.page_type is not None and c.page_type != old_type:
            r.page_type = Claim[PageType](value=c.page_type, origin="inferred", source_scan_ids=ids, notes=c.reason)
            out.changes.append(ReconciliationChange(field_path=path + ".page_type", old_value=old_type,
                                                    new_value=c.page_type.value, reason=c.reason,
                                                    source_scan_ids=ids))
        old_side = r.side.value if r.side else None
        new_side = None if c.side == "none" else c.side
        if c.side != "keep" and new_side != old_side:
            r.side = Claim[ScanSide](value=new_side, origin="inferred", source_scan_ids=ids, notes=c.reason) \
                if new_side else None
            if new_side != "both":
                r.subpages = []
            out.changes.append(ReconciliationChange(field_path=path + ".side", old_value=old_side,
                                                    new_value=new_side, reason=c.reason, source_scan_ids=ids))


# --- Input text --------------------------------------------------------------------------------


def _q(text: str) -> str:
    return json.dumps(" ".join(text.split()), ensure_ascii=False)


def reconcile_input(scans: list[ScanRecord], pagination: Pagination) -> str:
    """Compact text given to the LLM: numbering summary, then one line per scan (type, side, printed
    numbers, computed NDK label) with headings, TOC entries and bibliographic data indented below.
    Values are never shortened."""
    lines = [f'Book with {len(scans)} scans; "scan N" is the position in scanning order, not a page number.']
    lines += ["", "Printed page numbering (computed from the observations):"]
    lines += pagination.runs or ["- no page numbers observed"]
    lines += [f"- {w.code}: {w.message}" for w in pagination.warnings]
    lines += ["", "Scans (page type and side with self-reported confidence and reason; leaf kind; printed page "
              "numbers; computed NDK page label):"]
    for s in scans:
        label = pagination.page_number(s.scan_index) or "unresolved"
        if s.observation is None:
            lines.append(f"scan {s.scan_index + 1}: not observed (request failed); label {label}")
        else:
            lines += _scan_lines(s.scan_index + 1, s.observation, label)
    return "\n".join(lines)


def _scan_lines(position: int, obs: ScanObservation, label: str) -> list[str]:
    side = _evidence(obs.side or "side unknown", obs.side_confidence if obs.side else None, obs.side_reason)
    if obs.subpages:
        side += " (" + ", ".join(f"{s.side} {s.page_type.value if s.page_type else '?'}" for s in obs.subpages) + ")"
    page_type = obs.page_type.value if obs.page_type else "page type unknown"
    parts = [_evidence(page_type, obs.page_type_confidence, obs.page_type_reason), side]
    if obs.leaf or obs.leaf_reason:
        parts.append(_evidence(f"leaf {obs.leaf or 'unknown'}", None, obs.leaf_reason))
    numbers = [
        f"{n.raw}{f' ({n.side})' if n.side else ''}{f' at {n.position}' if n.position else ''}"
        for n in obs.printed_numbers
    ]
    parts.append("printed " + ", ".join(numbers) if numbers else "no printed number")
    parts.append(f"label {label}")
    lines = [f"scan {position}: " + "; ".join(parts)]
    lines += [f"  heading level {h.level or '?'}: {_q(h.text)}" for h in obs.headings]
    lines += [
        f"  TOC entry level {e.level or '?'}: {_q(e.title)} -> "
        f"{_q(e.printed_page_reference) if e.printed_page_reference else 'no page reference'}"
        for e in obs.toc_entries
    ]
    lines += [f"  {c.field.value}: {_q(c.value)}" for c in obs.bibliographic_candidates]
    if obs.notes:
        lines.append(f"  note: {_q(obs.notes)}")
    return lines


def _evidence(value: str, confidence: float | None, reason: str | None) -> str:
    """'titlePage (0.9, "full title and author")'; only the parts that are known."""
    extra = [f"{confidence:g}"] if confidence is not None else []
    extra += [_q(reason)] if reason else []
    return f"{value} ({', '.join(extra)})" if extra else value


def _parse_reference(ref: str) -> tuple[str, int] | None:
    """Leading page number of a printed TOC reference ("17", "[xii]", "17-20"); None if there is none."""
    m = re.match(r"\W*([0-9]+|[ivxlcdm]+|[IVXLCDM]+)\b", ref)
    if not m:
        return None
    token = m.group(1)
    if token.isdigit():
        return "arabic", int(token)
    value = roman_value(token)
    return ("roman", value) if value else None


# --- Checking the LLM answer -------------------------------------------------------------------


def _norm(text: str) -> str:
    """Comparison key: case-, whitespace- and trailing-punctuation-insensitive."""
    return " ".join(re.sub(r"[.,;:]+(\s|$)", r"\1", text).split()).casefold()


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def _words(text: str) -> list[str]:
    """Lower-case words without accents ("V Praze," -> ["v", "praze"])."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.findall(r"\w+", plain.casefold())


def _word_match(a: str, b: str) -> bool:
    """Same word up to an initial ("k" ~ "karel") or a Czech inflected ending ("praha" ~ "praze")."""
    if a == b:
        return True
    if min(len(a), len(b)) == 1:
        return a[0] == b[0]
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    return common >= max(3, min(len(a), len(b)) - 2)


def _matches(title: str, text: str) -> bool:
    """A TOC entry or heading ``text`` belongs to a chapter ``title`` if one contains the other's words
    ("Úvod" ~ "KAPITOLA I. Úvod")."""
    return _covered(title, [text]) or _covered(text, [title])


def _covered(value: str, observed: list[str]) -> bool:
    """Every word of ``value`` occurs, up to initials and inflection, in the ``observed`` texts."""
    seen = [w for text in observed for w in _words(text)]
    words = _words(value)
    return bool(words) and all(any(_word_match(w, o) for o in seen) for w in words)


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
    scans: list[ScanRecord], values: list[BiblioValue], out: ResolvedBook
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
        # The same text in any field (field moved), or a variant of a candidate of this field (initials,
        # inflection, a shortened form); citing a scan is not enough.
        similar = [s.scan_id for s, c in observed if _norm(c.value) == _norm(v.value)]
        related = [s.scan_id for s, c in observed if c.field == v.field and _covered(v.value, [c.value])]
        sources = list(dict.fromkeys(exact + similar + related))
        if not sources:
            out.warnings.append(
                ReconciliationWarning(
                    code="ungrounded_value",
                    message=f"{v.field.value} {v.value!r} proposed by the LLM matches no observed "
                    f"{v.field.value} candidate (or the same text in another field); dropped",
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
        # Only the TOC entries and headings on the cited scans whose text matches the title count.
        toc_obs = [e for sid in toc_ids if (o := refs.scan(sid).observation) for e in o.toc_entries
                   if _matches(ch.title, e.title)]
        head_obs = [h for sid in heading_ids if (o := refs.scan(sid).observation) for h in o.headings
                    if _matches(ch.title, h.text)]
        if not toc_obs and toc_ids:
            warn("unmatched_toc_entry", f"{name}: no TOC entry on the cited scan(s) matches the title; TOC ignored",
                 path, toc_ids)
            toc_ids = []
        if not head_obs and heading_ids:
            warn("unmatched_heading", f"{name}: no heading on the cited scan matches the title; heading ignored",
                 path, heading_ids)
            heading_ids = []
        ref = ch.printed_page_reference
        if ref is not None and not any(e.printed_page_reference and _same(e.printed_page_reference, ref)
                                       for e in toc_obs):
            warn("ungrounded_reference", f"{name}: page reference {ref!r} not printed in its TOC entries; dropped",
                 path, toc_ids)
            ref = None
        if not toc_ids and not heading_ids:
            warn("ungrounded_chapter", f"{name}: no matching TOC entry or heading cited; dropped", path, [])
            continue
        texts = [e.title for e in toc_obs] + [h.text for h in head_obs]
        if not _covered(ch.title, texts):
            warn("ungrounded_title", f"{name}: words not in its TOC entry or heading {texts}; dropped", path,
                 toc_ids + heading_ids)
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

    _bounds(nodes, resolved_scans, index, warn)
    return nodes


def _bounds(nodes: list[StructureNode], scans: list[ResolvedScan], index: dict[str, int], warn) -> None:
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
        side = scans[end].side
        if end > start and not (side and side.value == "both"):
            end -= 1
        node.end_scan_id = scans[end].scan_id
