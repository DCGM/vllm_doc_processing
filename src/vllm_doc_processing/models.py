"""Versioned Pydantic contract for one annotated book (see docs/OUTPUT_SCHEMA.md).

The document has two strictly separated layers:

* ``scans[].observation`` -- what the vision model reported for one image, kept
  unchanged after it was recorded;
* ``resolved`` -- the document-level reconciled view (bibliography, per-scan
  page labels, chapter structure) where every claim carries its origin and the
  scan IDs it is based on.

Unknown values are ``None`` or empty lists, never guesses.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "0.3"

Confidence = Annotated[float | None, Field(ge=0.0, le=1.0)]
"""Self-reported or heuristic score in [0, 1]; not a calibrated probability."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Vocabularies -----------------------------------------------------------


class PageType(StrEnum):
    """NDK page types (Pravidla pro popis monografií 2.4, table 1.2.2), values exactly as in NDK METS/MODS.

    MetaKat uses an older, PascalCase variant; see docs/OUTPUT_SCHEMA.md for the mapping.
    """

    FRONT_JACKET = "frontJacket"
    COVER = "cover"
    FRONT_COVER = "frontCover"
    BACK_COVER = "backCover"
    FRONT_END_SHEET = "frontEndSheet"
    BACK_END_SHEET = "backEndSheet"
    FRONT_END_PAPER = "frontEndPaper"
    BACK_END_PAPER = "backEndPaper"
    TITLE_PAGE = "titlePage"
    PREFACE = "preface"
    INTRODUCTION = "introduction"
    NORMAL_PAGE = "normalPage"
    BLANK = "blank"
    ILLUSTRATION = "illustration"
    MAP = "map"
    TABLE = "table"
    ADVERTISEMENT = "advertisement"
    IMPRESSUM = "impressum"
    COLOPHON = "colophon"
    FRONTISPIECE = "frontispiece"
    IMPRIMATUR = "imprimatur"
    DEDICATION = "dedication"
    ERRATA = "errata"
    SHEET_MUSIC = "sheetMusic"
    APPENDIX = "appendix"
    BIBLIOGRAPHY = "bibliography"
    AFTERWORD = "afterword"
    CONCLUSION = "conclusion"
    TABLE_OF_CONTENTS = "tableOfContents"
    INDEX = "index"
    LIST_OF_ILLUSTRATIONS = "listOfIllustrations"
    LIST_OF_MAPS = "listOfMaps"
    LIST_OF_TABLES = "listOfTables"
    EDGE = "edge"
    SPINE = "spine"
    JACKET = "jacket"
    FLYLEAF = "flyleaf"


ScanSide = Literal["left", "right", "both"]
"""Physical side of a whole scan; ``both`` is a two-page spread. Unknown = None."""

LeafSide = Literal["left", "right"]
"""Position of one page within a scan; never ``both``. Unknown/not applicable = None."""

NumeralSystem = Literal["arabic", "roman", "other"]

NumberPosition = Literal["top_left", "top_center", "top_right", "bottom_left", "bottom_center", "bottom_right", "other"]

Leaf = Literal["book_block", "plate", "binding", "loose"]
"""Physical kind of the photographed sheet; decides whether it belongs to the page count (NDK 1.1.4)."""

Origin = Literal["observed", "inferred", "catalogued"]
"""``catalogued`` is reserved for future external metadata."""


class BiblioField(StrEnum):
    """Book subset of MetaKat ``BiblioType``; values equal ``Bibliography`` field names."""

    TITLE = "title"
    SUBTITLE = "subtitle"
    PART_NAME = "part_name"
    PART_NUMBER = "part_number"
    SERIES_NAME = "series_name"
    SERIES_NUMBER = "series_number"
    EDITION = "edition"
    PUBLISHER = "publisher"
    PUBLICATION_PLACE = "publication_place"
    PUBLICATION_DATE = "publication_date"
    MANUFACTURE_PUBLISHER = "manufacture_publisher"
    MANUFACTURE_PLACE = "manufacture_place"
    AUTHOR = "author"
    EDITOR = "editor"
    TRANSLATOR = "translator"
    ILLUSTRATOR = "illustrator"
    PHOTOGRAPHER = "photographer"


# --- Per-scan observation (vision model output) ------------------------------


class PrintedNumber(StrictModel):
    """A page number as printed on paper; independent of ``scan_index``."""

    side: LeafSide | None = None
    raw: str = Field(min_length=1, description="Exactly as printed, e.g. '[12]', 'xii'.")
    normalized: str | None = Field(default=None, description="E.g. '12', 'XII'.")
    numeric_value: int | None = Field(default=None, ge=0)
    numeral_system: NumeralSystem | None = None
    position: NumberPosition | None = Field(default=None, description="Where on its page the number is printed.")
    confidence: Confidence = None
    notes: str | None = None


class SubpageObservation(StrictModel):
    """Optional classification of one page of a spread."""

    side: LeafSide
    page_type: PageType | None = None
    confidence: Confidence = None
    leaf: Leaf | None = Field(default=None, description="Leaf kind of this page if it differs from the scan's.")


class Heading(StrictModel):
    text: str = Field(min_length=1)
    level: int | None = Field(default=None, ge=1, description="1 = top level.")
    side: LeafSide | None = None
    confidence: Confidence = None


class TocEntry(StrictModel):
    title: str = Field(min_length=1)
    printed_page_reference: str | None = Field(default=None, description="Page reference as printed.")
    level: int | None = Field(default=None, ge=1)
    confidence: Confidence = None


class BiblioCandidate(StrictModel):
    field: BiblioField
    value: str = Field(min_length=1)
    confidence: Confidence = None
    notes: str | None = None


class ScanObservation(StrictModel):
    """Everything the vision model reported for one scan."""

    page_type: PageType | None = None
    page_type_confidence: Confidence = None
    page_type_reason: str | None = Field(default=None, description="Visible evidence for the page type, briefly.")
    side: ScanSide | None = None
    side_confidence: Confidence = None
    side_reason: str | None = Field(default=None, description="Visible evidence for the side, briefly.")
    leaf: Leaf | None = None
    leaf_reason: str | None = Field(default=None, description="Visible evidence for the leaf kind, briefly.")
    subpages: list[SubpageObservation] = Field(default_factory=list)
    printed_numbers: list[PrintedNumber] = Field(default_factory=list)
    headings: list[Heading] = Field(default_factory=list)
    toc_entries: list[TocEntry] = Field(default_factory=list)
    bibliographic_candidates: list[BiblioCandidate] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_subpages(self) -> ScanObservation:
        sides = [s.side for s in self.subpages]
        if len(sides) != len(set(sides)):
            raise ValueError("subpages must have distinct sides")
        if self.subpages and self.side != "both":
            raise ValueError("subpages are only allowed on a scan with side='both'")
        return self


class OcrInput(StrictModel):
    """The OCR sidecar supplied with one scan: metadata only, the OCR text itself is never stored."""

    status: Literal["ok", "missing"] = Field(description="'missing': no sidecar, the scan was sent as image only.")
    filename: str | None = Field(default=None, description="Sidecar file name in the OCR directory.")
    format: Literal["txt", "alto"] | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", description="Hash of the sidecar bytes.")
    size_bytes: int | None = Field(default=None, ge=0)
    chars: int | None = Field(default=None, ge=0, description="Length of the normalized OCR text.")
    sent_chars: int | None = Field(default=None, ge=0, description="Length of the OCR text sent (<= ocr_max_chars).")
    truncated: bool = Field(default=False, description="The middle of the text was omitted to fit ocr_max_chars.")

    @model_validator(mode="after")
    def _check_status(self) -> OcrInput:
        details = (self.filename, self.format, self.sha256, self.size_bytes, self.chars, self.sent_chars)
        if self.status == "ok" and None in details:
            raise ValueError("an 'ok' OCR input needs filename, format, sha256, size_bytes, chars and sent_chars")
        if self.status == "missing" and (any(d is not None for d in details) or self.truncated):
            raise ValueError("a 'missing' OCR input has no file details")
        return self


class ScanRecord(StrictModel):
    scan_id: str = Field(min_length=1, description="Name from the order file (file name without extension).")
    scan_index: int = Field(ge=0, description="Zero-based position in the order file; not a page number.")
    filename: str = Field(min_length=1, description="Original file name, with extension, in the input directory.")
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    ocr: OcrInput | None = Field(default=None, description="None = the run used no OCR directory.")
    observation: ScanObservation | None = Field(default=None, description="None = not (successfully) observed.")
    observation_call_id: str | None = None


# --- Resolved (reconciled) document view -------------------------------------

T = TypeVar("T")


class Claim(StrictModel, Generic[T]):
    """A resolved value with provenance."""

    value: T
    origin: Origin = "observed"
    source_scan_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = None
    notes: str | None = None


class Bibliography(StrictModel):
    """Book subset of MetaKat ``MetakatTitle`` + ``MetakatVolume``; field names match ``BiblioField``."""

    title: Claim[str] | None = None
    subtitle: Claim[str] | None = None
    part_name: Claim[str] | None = None
    part_number: Claim[str] | None = None
    edition: Claim[str] | None = None
    publication_date: Claim[str] | None = None
    series_name: list[Claim[str]] = Field(default_factory=list)
    series_number: list[Claim[str]] = Field(default_factory=list)
    publisher: list[Claim[str]] = Field(default_factory=list)
    publication_place: list[Claim[str]] = Field(default_factory=list)
    manufacture_publisher: list[Claim[str]] = Field(default_factory=list)
    manufacture_place: list[Claim[str]] = Field(default_factory=list)
    author: list[Claim[str]] = Field(default_factory=list)
    editor: list[Claim[str]] = Field(default_factory=list)
    translator: list[Claim[str]] = Field(default_factory=list)
    illustrator: list[Claim[str]] = Field(default_factory=list)
    photographer: list[Claim[str]] = Field(default_factory=list)

    def claims(self) -> list[Claim[str]]:
        out: list[Claim[str]] = []
        for name in BiblioField:
            value = getattr(self, name.value)
            out.extend(value if isinstance(value, list) else [value] if value else [])
        return out


class PageLabel(StrictModel):
    """Resolved page number of one page of a scan in Czech NDK notation (see docs/OUTPUT_SCHEMA.md).

    Printed numbers are plain (``12``, ``XII``); numbers that are not printed are in brackets: counted
    pages ``[12]``, pages outside the count ``[12a]`` (lettered after the preceding page number).
    """

    side: LeafSide | None = None
    label: str = Field(min_length=1, description="NDK notation, e.g. '12', 'XII', '[13]', '[1a]', '[26b]'.")
    numeric_value: int | None = Field(default=None, ge=0)
    numeral_system: NumeralSystem | None = None
    origin: Origin = "observed"
    source_scan_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = None
    notes: str | None = None


class ResolvedSubpage(StrictModel):
    """Resolved type and leaf kind of one page of a spread."""

    side: LeafSide
    page_type: Claim[PageType] | None = None
    leaf: Claim[Leaf] | None = Field(default=None, description="Decides the page count of this page.")


class ResolvedScan(StrictModel):
    scan_id: str
    page_type: Claim[PageType] | None = None
    side: Claim[ScanSide] | None = None
    leaf: Claim[Leaf] | None = Field(
        default=None, description="Leaf kind of a single page; on a spread the subpages' leaf kinds are used."
    )
    subpages: list[ResolvedSubpage] = Field(default_factory=list)
    page_labels: list[PageLabel] = Field(default_factory=list)
    page_number: str | None = Field(
        default=None,
        description="NDK page label of the whole scan (METS ORDERLABEL, MetaKat pageNumber): the labels of its "
        "pages joined by ',', e.g. '5', '[1a]', '[4],5'; None if any page is unresolved.",
    )

    @model_validator(mode="after")
    def _check_subpages(self) -> ResolvedScan:
        sides = [s.side for s in self.subpages]
        if len(sides) != len(set(sides)):
            raise ValueError("subpages must have distinct sides")
        if self.subpages and (self.side is None or self.side.value != "both"):
            raise ValueError("subpages are only allowed on a scan resolved as side='both'")
        if self.page_number is not None and self.page_number != ",".join(p.label for p in self.page_labels):
            raise ValueError("page_number must equal the page labels joined by ','")
        return self


class StructureNode(StrictModel):
    """Chapter/section in a flat list; hierarchy via ``parent_id`` and ``level``."""

    id: str = Field(min_length=1)
    parent_id: str | None = None
    level: int = Field(default=1, ge=1, description="1 = top level.")
    title: Claim[str] | None = None
    subtitle: Claim[str] | None = None
    part_number: Claim[str] | None = None
    printed_page_reference: str | None = Field(default=None, description="Target page as printed in the TOC.")
    toc_scan_ids: list[str] = Field(default_factory=list)
    heading_scan_ids: list[str] = Field(default_factory=list)
    start_scan_id: str | None = None
    end_scan_id: str | None = None
    origin: Origin = "observed"
    confidence: Confidence = None
    notes: str | None = None


class ReconciliationWarning(StrictModel):
    code: str = Field(min_length=1, description="Short slug, e.g. 'page_number_conflict'.")
    message: str
    detected_by: Literal["llm", "check"]
    field_path: str | None = None
    scan_ids: list[str] = Field(default_factory=list)


class ReconciliationChange(StrictModel):
    """Audit entry: the resolved value differs from what was observed."""

    field_path: str
    old_value: JsonValue = None
    new_value: JsonValue = None
    reason: str
    source_scan_ids: list[str] = Field(default_factory=list)


class ResolvedBook(StrictModel):
    bibliography: Bibliography = Field(default_factory=Bibliography)
    scans: list[ResolvedScan] = Field(default_factory=list)
    structure: list[StructureNode] = Field(default_factory=list)
    warnings: list[ReconciliationWarning] = Field(default_factory=list)
    changes: list[ReconciliationChange] = Field(default_factory=list)


# --- Run provenance -----------------------------------------------------------

Stage = Literal["observe", "reconcile", "escalate", "revisit"]


class CallRecord(StrictModel):
    """One paid API request attempt (retries are separate records)."""

    call_id: str = Field(min_length=1)
    stage: Stage
    scan_id: str | None = None
    provider: str
    model: str
    attempt: int = Field(default=1, ge=1)
    status: Literal["ok", "invalid_response", "error"]
    started_at: datetime | None = None
    latency_s: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0, description="As reported by the provider, if any.")
    response_id: str | None = Field(default=None, description="Provider's response/generation ID.")
    served_model: str | None = Field(default=None, description="Model as reported in the response (may be a snapshot).")
    upstream_provider: str | None = Field(default=None, description="Serving provider behind OpenRouter, if reported.")
    error: str | None = None


class UsageTotals(StrictModel):
    requests: int = 0
    failed_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = Field(default=None, description="Sum of reported costs; None if none reported.")
    cost_complete: bool = Field(default=True, description="False if any request lacked a reported cost.")

    @classmethod
    def from_calls(cls, calls: list[CallRecord]) -> UsageTotals:
        costs = [c.cost_usd for c in calls if c.cost_usd is not None]
        return cls(
            requests=len(calls),
            failed_requests=sum(c.status != "ok" for c in calls),
            prompt_tokens=sum(c.prompt_tokens or 0 for c in calls),
            completion_tokens=sum(c.completion_tokens or 0 for c in calls),
            cost_usd=sum(costs) if costs else None,
            cost_complete=len(costs) == len(calls),
        )


class RunInfo(StrictModel):
    """``totals`` must equal ``UsageTotals.from_calls(calls)``; call ``refresh_totals()`` after adding calls."""

    tool_version: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    provider: str | None = None
    base_url: str | None = None
    vision_model: str | None = None
    postprocess_model: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict, description="Non-secret request/config settings.")
    calls: list[CallRecord] = Field(default_factory=list)
    totals: UsageTotals = Field(default_factory=UsageTotals)
    warnings: list[str] = Field(default_factory=list)

    def refresh_totals(self) -> None:
        self.totals = UsageTotals.from_calls(self.calls)

    @model_validator(mode="after")
    def _check_totals(self) -> RunInfo:
        expected, got = UsageTotals.from_calls(self.calls), self.totals
        if None in (expected.cost_usd, got.cost_usd):
            cost_ok = expected.cost_usd == got.cost_usd
        else:
            cost_ok = math.isclose(expected.cost_usd, got.cost_usd, rel_tol=1e-9)
        if not cost_ok or got.model_dump(exclude={"cost_usd"}) != expected.model_dump(exclude={"cost_usd"}):
            raise ValueError(f"run.totals inconsistent with run.calls; expected {expected.model_dump()}")
        return self


# --- Top level ---------------------------------------------------------------


class SourceInfo(StrictModel):
    input_directory: str | None = None
    order_file: str | None = Field(default=None, description="File listing scan names in physical order.")
    ocr_directory: str | None = Field(default=None, description="Directory of OCR sidecars; None = image-only run.")
    scan_count: int = Field(ge=0)


class AnnotatedBook(StrictModel):
    schema_version: Literal["0.3"] = SCHEMA_VERSION
    book_id: str = Field(min_length=1)
    source: SourceInfo
    scans: list[ScanRecord]
    resolved: ResolvedBook | None = Field(default=None, description="None until reconciliation has run.")
    run: RunInfo = Field(default_factory=RunInfo)

    @model_validator(mode="after")
    def _check_invariants(self) -> AnnotatedBook:
        if self.source.scan_count != len(self.scans):
            raise ValueError("source.scan_count does not match number of scans")
        if [s.scan_index for s in self.scans] != list(range(len(self.scans))):
            raise ValueError("scans must be sorted with contiguous scan_index starting at 0")
        _require_unique([s.scan_id for s in self.scans], "scan_id")
        _require_unique([s.filename for s in self.scans], "filename")
        if any((s.ocr is not None) != (self.source.ocr_directory is not None) for s in self.scans):
            raise ValueError("every scan has OCR input metadata exactly when source.ocr_directory is set")
        call_ids = [c.call_id for c in self.run.calls]
        _require_unique(call_ids, "call_id")

        index = {s.scan_id: s.scan_index for s in self.scans}
        refs = [c.scan_id for c in self.run.calls if c.scan_id]
        calls = {c.call_id: c for c in self.run.calls}
        for s in self.scans:
            _check_observation_call(s, calls)
        if self.resolved is not None:
            if [r.scan_id for r in self.resolved.scans] != [s.scan_id for s in self.scans]:
                raise ValueError("resolved.scans must contain exactly one record per input scan, in scan order")
            refs += _resolved_scan_refs(self.resolved)
            _check_structure(self.resolved.structure, index)
        unknown = sorted(set(refs) - index.keys())
        if unknown:
            raise ValueError(f"references to unknown scan IDs: {unknown}")
        return self


def _require_unique(values: list[str], what: str) -> None:
    if len(values) != len(set(values)):
        dupes = sorted({v for v in values if values.count(v) > 1})
        raise ValueError(f"duplicate {what}: {dupes}")


def _check_observation_call(scan: ScanRecord, calls: dict[str, CallRecord]) -> None:
    """The producing call must exist, belong to this scan, be an observe/escalate call and have succeeded."""
    if scan.observation_call_id is None:
        return
    where = f"scan {scan.scan_id!r}: observation_call_id {scan.observation_call_id!r}"
    if scan.observation is None:
        raise ValueError(f"{where} set without an observation")
    call = calls.get(scan.observation_call_id)
    if call is None:
        raise ValueError(f"{where} is unknown")
    if call.scan_id != scan.scan_id or call.stage not in ("observe", "escalate") or call.status != "ok":
        raise ValueError(f"{where} must be a successful observe/escalate call for this scan")


def _resolved_scan_refs(resolved: ResolvedBook) -> list[str]:
    refs: list[str] = []
    for claim in resolved.bibliography.claims():
        refs += claim.source_scan_ids
    for scan in resolved.scans:
        refs.append(scan.scan_id)
        subclaims = [c for sub in scan.subpages for c in (sub.page_type, sub.leaf)]
        for claim in (scan.page_type, scan.side, scan.leaf, *subclaims):
            if claim:
                refs += claim.source_scan_ids
        for label in scan.page_labels:
            refs += label.source_scan_ids
    for node in resolved.structure:
        refs += node.toc_scan_ids + node.heading_scan_ids
        refs += [i for i in (node.start_scan_id, node.end_scan_id) if i]
        for claim in (node.title, node.subtitle, node.part_number):
            if claim:
                refs += claim.source_scan_ids
    for warning in resolved.warnings:
        refs += warning.scan_ids
    for change in resolved.changes:
        refs += change.source_scan_ids
    return refs


def _check_structure(nodes: list[StructureNode], scan_index: dict[str, int]) -> None:
    """Unique IDs, parents listed before children with a lower level, ordered ranges."""
    _require_unique([n.id for n in nodes], "structure id")
    levels: dict[str, int] = {}
    for node in nodes:
        if node.parent_id is not None:
            if node.parent_id not in levels:
                raise ValueError(f"structure node {node.id!r}: parent must be listed before the child")
            if node.level <= levels[node.parent_id]:
                raise ValueError(f"structure node {node.id!r}: level must exceed parent level")
        levels[node.id] = node.level
        start, end = node.start_scan_id, node.end_scan_id
        if start in scan_index and end in scan_index and scan_index[start] > scan_index[end]:
            raise ValueError(f"structure node {node.id!r}: start_scan_id is after end_scan_id")


def dump_json(book: AnnotatedBook) -> str:
    """Re-validate (catches in-place edits after construction) and serialize; non-ASCII stays UTF-8."""
    return AnnotatedBook.model_validate(book.model_dump()).model_dump_json(indent=2) + "\n"


def load_json(text: str | bytes) -> AnnotatedBook:
    return AnnotatedBook.model_validate_json(text)
