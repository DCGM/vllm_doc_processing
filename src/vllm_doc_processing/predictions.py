"""Import predictions of this tool, MetaKat and Kramerius into one comparable form (docs/EVALUATION.md).

Every value is mapped explicitly into this project's vocabulary. A value without an equivalent
becomes ``Incomparable`` and is never counted as equal to anything. A field missing from
``PredictedScan.values`` is not provided by the system, which is different from a ``None`` value
(the system predicted "unknown"/"none").
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from .models import SCHEMA_VERSION, AnnotatedBook, BiblioField, PageType
from .ocr import ocr_summary
from .pagination import roman_value, to_roman

SCAN_FIELDS = ("page_type", "side", "leaf", "printed_numbers_exact", "printed_numbers_normalized", "page_number")

# Scan fields each source/layer outputs by contract; a scan without a value for them (e.g. a failed
# observation) counts as ``no_prediction``, it does not make the field "not provided".
OBSERVED_FIELDS = ("page_type", "side", "leaf", "printed_numbers_exact", "printed_numbers_normalized")
RESOLVED_FIELDS = ("page_type", "side", "leaf", "page_number")
METAKAT_FIELDS = ("page_type", "side", "page_number")
KRAMERIUS_FIELDS = ("page_type", "page_number")

_PAGE_TYPES = {t.value.casefold(): t.value for t in PageType}

METAKAT_BIBLIO = {
    "title": "title",
    "subTitle": "subtitle",
    "partName": "part_name",
    "partNumber": "part_number",
    "seriesName": "series_name",
    "seriesNumber": "series_number",
    "edition": "edition",
    "publisher": "publisher",
    "placeTerm": "publication_place",
    "dateIssued": "publication_date",
    "manufacturePublisher": "manufacture_publisher",
    "manufacturePlaceTerm": "manufacture_place",
    "author": "author",
    "editor": "editor",
    "translator": "translator",
    "illustrator": "illustrator",
    "photographer": "photographer",
}


class PredictionError(ValueError):
    pass


@dataclass(frozen=True)
class Incomparable:
    """A predicted value with no equivalent in this project's vocabulary (e.g. MetaKat ``single_page``)."""

    raw: str


@dataclass
class PredictedScan:
    scan_id: str
    scan_index: int
    image_sha256: str | None
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictedChapter:
    title: str | None
    level: int | None
    start_scan_id: str | None
    printed_page_reference: str | None


@dataclass
class Prediction:
    """One book as predicted by one system at one layer."""

    system: str
    layer: Literal["observed", "resolved", "reference"]
    role: Literal["system", "comparator"]
    path: str
    sha256: str
    book_id: str
    scans: dict[str, PredictedScan]
    fields: tuple[str, ...]
    """Scan fields this system outputs at this layer."""
    bibliography: dict[str, list[str]] | None = None
    structure: list[PredictedChapter] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.system} ({self.layer})"


# --- Normalization ------------------------------------------------------------


def page_type(value: str | None) -> str | Incomparable | None:
    """NDK page type for an NDK or MetaKat (PascalCase) spelling; case-insensitive (Kramerius mixes both)."""
    if value is None:
        return None
    return _PAGE_TYPES.get(value.strip().casefold(), Incomparable(value))


def metakat_side(value: str | None) -> str | Incomparable | None:
    if value in (None, "left", "right"):
        return value
    return Incomparable(value)  # 'single_page' has no equivalent in left|right|both


def normalize_number(raw: str) -> str:
    """'[12].' -> '12', 'xii' -> 'XII'; other text is case-folded."""
    text = raw.strip().strip("[]()-–—.,· ").strip()
    if text.isdigit():
        return str(int(text))
    value = roman_value(text)
    return to_roman(value) if value else text.casefold()


def normalize_text(text: str) -> str:
    """For bibliography and titles: NFC, case-folded, whitespace collapsed, outer punctuation removed."""
    text = " ".join(unicodedata.normalize("NFC", text).casefold().split())
    return text.strip(" .,:;/-–—")


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", "", label)


def library_label(label: str | None) -> str | None:
    """Kramerius/MetaKat label in NDK notation: older records write unprinted numbers as '(12)' for '[12]'."""
    if not label:
        return None
    return ",".join(re.sub(r"^\((.+)\)$", r"[\1]", part.strip()) for part in label.split(","))


# --- Loading ------------------------------------------------------------------


READABLE_SCHEMA_VERSIONS = ("0.2", SCHEMA_VERSION)
"""0.3 only added optional OCR fields, so 0.2 files (image-only runs) are read as they are."""


def load_prediction(path: Path, system: str | None = None) -> list[Prediction]:
    """Detect the format and import. An annotated book yields its observed and (if present) resolved layer."""
    data_bytes = path.read_bytes()
    sha = hashlib.sha256(data_bytes).hexdigest()
    try:
        data = json.loads(data_bytes)
    except json.JSONDecodeError as exc:
        raise PredictionError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PredictionError(f"{path}: expected a JSON object")
    if "schema_version" in data and "scans" in data:
        return from_annotated_book(data, str(path), sha, system or "vllm-doc")
    if "elements" in data and "batch_id" in data:
        return [from_metakat(data, str(path), sha, system or "metakat")]
    if "pages" in data and "document" in data:
        return [from_kramerius(data, str(path), sha, system or "kramerius")]
    raise PredictionError(f"{path}: unknown format (expected an annotated book, MetakatIO or *.kramerius.json)")


def from_annotated_book(data: dict, path: str, sha: str, system: str) -> list[Prediction]:
    version = data.get("schema_version")
    if version not in READABLE_SCHEMA_VERSIONS:
        raise PredictionError(
            f"{path}: schema_version {version!r} is not supported (expected one of {READABLE_SCHEMA_VERSIONS})"
        )
    try:
        book = AnnotatedBook.model_validate(data | {"schema_version": SCHEMA_VERSION})
    except ValidationError as exc:
        raise PredictionError(f"{path}: invalid annotated book: {exc}") from exc

    observed: dict[str, PredictedScan] = {}
    for s in book.scans:
        values: dict[str, Any] = {}
        if (o := s.observation) is not None:
            raw = [n.raw for n in sorted(o.printed_numbers, key=lambda n: {"left": 0, None: 1, "right": 2}[n.side])]
            values = {
                "page_type": o.page_type and o.page_type.value,
                "side": o.side,
                "leaf": o.leaf,
                "printed_numbers_exact": raw or None,
                "printed_numbers_normalized": raw or None,
            }
        observed[s.scan_id] = PredictedScan(s.scan_id, s.scan_index, s.image_sha256, values)
    provenance = _provenance(book) | {"schema_version": version}
    common = dict(system=system, role="system", path=path, sha256=sha, book_id=book.book_id, provenance=provenance)
    out = [Prediction(layer="observed", scans=observed, fields=OBSERVED_FIELDS, **common)]
    if (r := book.resolved) is not None:
        resolved = {}
        for s, rs in zip(book.scans, r.scans, strict=True):
            values = {
                "page_type": rs.page_type and rs.page_type.value.value,
                "side": rs.side and rs.side.value,
                "leaf": rs.leaf and rs.leaf.value,
                "page_number": rs.page_number,
            }
            resolved[s.scan_id] = PredictedScan(s.scan_id, s.scan_index, s.image_sha256, values)
        biblio = {}
        for name in BiblioField:
            value = getattr(r.bibliography, name.value)
            biblio[name.value] = [c.value for c in (value if isinstance(value, list) else [value] if value else [])]
        structure = [
            PredictedChapter(n.title and n.title.value, n.level, n.start_scan_id, n.printed_page_reference)
            for n in r.structure
        ]
        out.append(
            Prediction(
                layer="resolved", scans=resolved, fields=RESOLVED_FIELDS, bibliography=biblio, structure=structure,
                **common,
            )
        )
    return out


def _provenance(book: AnnotatedBook) -> dict[str, Any]:
    run = book.run
    params = run.parameters
    ocr: dict[str, Any] | str = "image-only"
    if book.source.ocr_directory is not None:
        ocr = {"format": params.get("ocr_format"), "max_chars": params.get("ocr_max_chars"), **ocr_summary(book.scans)}
    duration = (run.finished_at - run.started_at).total_seconds() if run.started_at and run.finished_at else None
    requests = run.totals.requests
    return {
        "book_id": book.book_id,
        "schema_version": book.schema_version,
        "tool_version": run.tool_version,
        "provider": run.provider,
        "base_url": run.base_url,
        "vision_model": run.vision_model,
        "postprocess_model": run.postprocess_model,
        "served_models": sorted({c.served_model for c in run.calls if c.served_model}),
        "prompt_versions": dict(sorted(run.prompt_versions.items())),
        "image_policy": {k: params.get(k) for k in ("image_max_side", "image_format", "image_detail")},
        "context": {k: params.get(k) for k in ("use_context", "context_recent_scans", "context_max_chars")},
        "ocr_mode": ocr or "image-only",
        "request_params": params.get("request_params"),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "duration_s": duration,
        "latency_s_total": round(sum(c.latency_s or 0 for c in run.calls), 3),
        "totals": run.totals.model_dump(),
        "retry_attempts": sum(c.attempt > 1 for c in run.calls),
        "error_rate": run.totals.failed_requests / requests if requests else None,
        "scans": len(book.scans),
        "unobserved_scans": sum(s.observation is None for s in book.scans),
        "reconciled": book.resolved is not None,
        "reconcile_warnings": len(book.resolved.warnings) if book.resolved else None,
    }


def _first(value: Any) -> Any:
    """Value of a MetaKat ``(value, confidence[, detection_id])`` tuple."""
    return value[0] if value else None


def from_metakat(data: dict, path: str, sha: str, system: str) -> Prediction:
    """MetakatIO (book subset): pages, title/volume bibliography and chapters.

    Scan IDs are the stems of ``page_to_image_mapping`` file names, else the page UUIDs. Chapter
    ``pageIndex*`` values refer to ``MetakatPage.pageIndex`` (``batch_index`` if pages have none).
    """
    try:
        elements = data["elements"]
        pages = sorted((e for e in elements if e["type"] == "page"), key=lambda e: e["batch_index"])
        images = data.get("page_to_image_mapping") or {}
        scans: dict[str, PredictedScan] = {}
        by_index: dict[int, str] = {}
        use_page_index = bool(pages) and all(p.get("pageIndex") is not None for p in pages)
        for p in pages:
            scan_id = Path(images[p["id"]]).stem if p["id"] in images else p["id"]
            values = {
                "page_type": page_type(_first(p.get("pageType"))),
                "side": metakat_side(_first(p.get("side"))),
                "page_number": library_label(_first(p.get("pageNumber"))),
            }
            index = p["pageIndex"] if use_page_index else p["batch_index"]
            scans[scan_id] = PredictedScan(scan_id, index, None, values)
            by_index[index] = scan_id

        volumes = [e for e in elements if e["type"] == "volume"]
        titles = [e for e in elements if e["type"] == "title"]
        if len(volumes) > 1 or len(titles) > 1:
            raise PredictionError(f"{path}: expected at most one title and one volume element (one book)")
        biblio: dict[str, list[str]] = {}
        for key, name in METAKAT_BIBLIO.items():
            source = next((e for e in volumes + titles if e.get(key)), None)
            value = source.get(key) if source else None
            items = value if value and isinstance(value[0], list) else [value] if value else []
            biblio[name] = [v[0] for v in items if v and v[0]]

        chapters = [e for e in elements if e["type"] == "chapter"]
        parents = {c["id"]: c.get("parent_id") for c in chapters}

        def level(chapter_id: str) -> int:
            n = 1
            while parents.get(chapter_id) in parents:
                chapter_id, n = parents[chapter_id], n + 1
                if n > len(parents):
                    raise PredictionError(f"{path}: chapter parent_id cycle at {chapter_id!r}")
            return n

        structure = [
            PredictedChapter(
                _first(c.get("title")),
                level(c["id"]),
                by_index.get(c["pageIndexStart"]) if c.get("pageIndexStart") is not None else None,
                _first(c.get("pageNumber")),
            )
            for c in chapters
        ]
        book_id = (volumes or titles or [{"id": data["batch_id"]}])[0]["id"]
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise PredictionError(f"{path}: invalid MetakatIO: {exc!r}") from exc
    return Prediction(system, "reference", "comparator", path, sha, str(book_id), scans, METAKAT_FIELDS, biblio, structure)


def from_kramerius(data: dict, path: str, sha: str, system: str) -> Prediction:
    """``*.kramerius.json`` of scripts/kramerius_order.py: per-page page type and NDK page label only."""
    try:
        scans = {
            p["scan_id"]: PredictedScan(
                p["scan_id"],
                p["scan_index"],
                None,
                {"page_type": page_type(p.get("page_type")), "page_number": library_label(p.get("page_number"))},
            )
            for p in data["pages"]
        }
        book_id = str(data["document"]["pid"]).removeprefix("uuid:")
    except (KeyError, TypeError) as exc:
        raise PredictionError(f"{path}: invalid Kramerius file: {exc!r}") from exc
    return Prediction(system, "reference", "comparator", path, sha, book_id, scans, KRAMERIUS_FIELDS)
