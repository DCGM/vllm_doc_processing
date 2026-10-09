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

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "0.1"

Confidence = Annotated[float | None, Field(ge=0.0, le=1.0)]
"""Self-reported or heuristic score in [0, 1]; not a calibrated probability."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Vocabularies -----------------------------------------------------------


class PageType(StrEnum):
    """MetaKat ``PageType`` vocabulary, copied verbatim for comparability."""

    ABSTRACT = "Abstract"
    ADVERTISEMENT = "Advertisement"
    APPENDIX = "Appendix"
    BACK_COVER = "BackCover"
    BACK_END_PAPER = "BackEndPaper"
    BACK_END_SHEET = "BackEndSheet"
    BIBLIOGRAPHY = "Bibliography"
    BLANK = "Blank"
    CALIBRATION_TABLE = "CalibrationTable"
    COVER = "Cover"
    CUSTOM_INCLUDE = "CustomInclude"
    DEDICATION = "Dedication"
    EDGE = "Edge"
    ERRATA = "Errata"
    FLY_LEAF = "FlyLeaf"
    FRAGMENTS_OF_BOOKBINDING = "FragmentsOfBookbinding"
    FRONT_COVER = "FrontCover"
    FRONT_END_PAPER = "FrontEndPaper"
    FRONT_END_SHEET = "FrontEndSheet"
    FRONT_JACKET = "FrontJacket"
    FRONTISPIECE = "Frontispiece"
    ILLUSTRATION = "Illustration"
    IMPRESSUM = "Impressum"
    IMPRIMATUR = "Imprimatur"
    INDEX = "Index"
    JACKET = "Jacket"
    LIST_OF_ILLUSTRATIONS = "ListOfIllustrations"
    LIST_OF_MAPS = "ListOfMaps"
    LIST_OF_TABLES = "ListOfTables"
    MAP = "Map"
    NORMAL_PAGE = "NormalPage"
    OBITUARY = "Obituary"
    PREFACE = "Preface"
    SHEET_MUSIC = "SheetMusic"
    SPINE = "Spine"
    TABLE = "Table"
    TABLE_OF_CONTENTS = "TableOfContents"
    TITLE_PAGE = "TitlePage"


ScanSide = Literal["left", "right", "both"]
"""Physical side of a whole scan; ``both`` is a two-page spread. Unknown = None."""

LeafSide = Literal["left", "right"]
"""Position of one page within a scan; never ``both``. Unknown/not applicable = None."""

NumeralSystem = Literal["arabic", "roman", "other"]

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
    confidence: Confidence = None
    notes: str | None = None


class SubpageObservation(StrictModel):
    """Optional classification of one page of a spread."""

    side: LeafSide
    page_type: PageType | None = None
    confidence: Confidence = None


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
    side: ScanSide | None = None
    side_confidence: Confidence = None
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


class ScanRecord(StrictModel):
    scan_id: str = Field(min_length=1, description="Name from the order file (file name without extension).")
    scan_index: int = Field(ge=0, description="Zero-based position in the order file; not a page number.")
    filename: str = Field(min_length=1, description="Original file name, with extension, in the input directory.")
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
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
    """Resolved printed page number of one page of a scan (observed or inferred)."""

    side: LeafSide | None = None
    label: str = Field(min_length=1, description="Normalized label, e.g. '12', 'XII'.")
    numeric_value: int | None = Field(default=None, ge=0)
    numeral_system: NumeralSystem | None = None
    origin: Origin = "observed"
    source_scan_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = None
    notes: str | None = None


class ResolvedScan(StrictModel):
    scan_id: str
    page_type: Claim[PageType] | None = None
    side: Claim[ScanSide] | None = None
    page_labels: list[PageLabel] = Field(default_factory=list)


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


# --- Top level ---------------------------------------------------------------


class SourceInfo(StrictModel):
    input_directory: str | None = None
    order_file: str | None = Field(default=None, description="File listing scan names in physical order.")
    scan_count: int = Field(ge=0)


class AnnotatedBook(StrictModel):
    schema_version: Literal["0.1"] = SCHEMA_VERSION
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
        call_ids = [c.call_id for c in self.run.calls]
        _require_unique(call_ids, "call_id")

        index = {s.scan_id: s.scan_index for s in self.scans}
        refs = [c.scan_id for c in self.run.calls if c.scan_id]
        for s in self.scans:
            if s.observation_call_id is not None and s.observation_call_id not in call_ids:
                raise ValueError(f"unknown observation_call_id {s.observation_call_id!r}")
        if self.resolved is not None:
            refs += _resolved_scan_refs(self.resolved)
            _require_unique([s.scan_id for s in self.resolved.scans], "resolved scan_id")
            _check_structure(self.resolved.structure, index)
        unknown = sorted(set(refs) - index.keys())
        if unknown:
            raise ValueError(f"references to unknown scan IDs: {unknown}")
        return self


def _require_unique(values: list[str], what: str) -> None:
    if len(values) != len(set(values)):
        dupes = sorted({v for v in values if values.count(v) > 1})
        raise ValueError(f"duplicate {what}: {dupes}")


def _resolved_scan_refs(resolved: ResolvedBook) -> list[str]:
    refs: list[str] = []
    for claim in resolved.bibliography.claims():
        refs += claim.source_scan_ids
    for scan in resolved.scans:
        refs.append(scan.scan_id)
        for claim in (scan.page_type, scan.side):
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
    """Serialize with indentation; non-ASCII text is kept as UTF-8."""
    return book.model_dump_json(indent=2) + "\n"


def load_json(text: str | bytes) -> AnnotatedBook:
    return AnnotatedBook.model_validate_json(text)
