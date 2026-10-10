"""Human-verified reference annotations (gold standard) for evaluation (see docs/EVALUATION.md).

One gold file describes one book: the scan manifest (IDs, order, image hashes, OCR sidecars) and,
per field, a label with an explicit review status. A field that was not reviewed is never scored,
so an empty template is a valid gold file that scores nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Generic, Literal, TypeVar

from pydantic import Field, model_validator

from .images import Inventory
from .models import BiblioField, Leaf, PageType, ScanSide, StrictModel

GOLD_VERSION = "1"

Status = Literal["verified", "absent", "not_applicable", "ambiguous", "not_reviewed"]
"""``verified``: ``value`` is correct; ``absent``: verified that there is no value (e.g. no printed
number); ``not_applicable``: the field has no meaning here (e.g. side of a spine), expected null;
``ambiguous``: any of ``alternatives`` is acceptable (null allowed), none = unscorable;
``not_reviewed``: never scored."""

T = TypeVar("T")


class GoldValue(StrictModel, Generic[T]):
    status: Status = "not_reviewed"
    value: T | None = None
    alternatives: list[T | None] = Field(default_factory=list, description="Acceptable values when ambiguous.")
    reviewer: str | None = None
    notes: str | None = Field(default=None, description="Brief evidence or reason for the decision.")

    @model_validator(mode="after")
    def _check_status(self) -> GoldValue[T]:
        if self.status == "verified":
            if self.value is None or self.value == []:
                raise ValueError("a verified value must not be null or empty; use status 'absent'")
        elif self.value is not None:
            raise ValueError(f"status {self.status!r} must not carry a value")
        if self.alternatives and self.status != "ambiguous":
            raise ValueError("alternatives are only allowed with status 'ambiguous'")
        return self


class Sidecar(StrictModel):
    """An OCR file available for a scan (not committed; recorded for matched OCR experiments)."""

    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GoldScan(StrictModel):
    scan_id: str = Field(min_length=1, description="Name from the order file.")
    scan_index: int = Field(ge=0, description="Zero-based position in the book's order file.")
    filename: str | None = None
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="Labels are valid only for this image.")
    txt: Sidecar | None = None
    alto: Sidecar | None = None
    page_type: GoldValue[PageType] = Field(default_factory=GoldValue)
    side: GoldValue[ScanSide] = Field(default_factory=GoldValue)
    leaf: GoldValue[Leaf] = Field(default_factory=GoldValue)
    printed_numbers: GoldValue[list[str]] = Field(
        default_factory=GoldValue, description="Page numbers exactly as printed, left to right; not page labels."
    )
    page_number: GoldValue[str] = Field(
        default_factory=GoldValue, description="NDK page label of the scan, e.g. '5', '[1a]', '[4],5'."
    )
    notes: str | None = None


class GoldChapter(StrictModel):
    title: str = Field(min_length=1)
    level: int = Field(default=1, ge=1)
    start_scan_id: str | None = None
    printed_page_reference: str | None = Field(default=None, description="Target page as printed in the TOC.")


GoldBibliography = dict[BiblioField, GoldValue[list[str]]]
"""Missing fields are not reviewed. Every field is a list (single-valued fields hold one value)."""


class GoldBook(StrictModel):
    gold_version: Literal["1"] = GOLD_VERSION
    book_id: str = Field(min_length=1)
    dataset_version: str | None = None
    source: str | None = Field(default=None, description="Where the scans come from, e.g. a Kramerius URL.")
    scan_count: int = Field(ge=1, description="Number of scans of the whole book (order file length).")
    reviewers: list[str] = Field(default_factory=list)
    scans: list[GoldScan] = Field(description="All scans of the book or a selected subset, in scan order.")
    bibliography: GoldBibliography = Field(default_factory=dict)
    structure: GoldValue[list[GoldChapter]] = Field(
        default_factory=GoldValue, description="Chapter list of the whole book; only with all scans listed."
    )
    notes: str | None = None

    @property
    def complete(self) -> bool:
        return len(self.scans) == self.scan_count

    @model_validator(mode="after")
    def _check(self) -> GoldBook:
        ids = [s.scan_id for s in self.scans]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate scan_id")
        indices = [s.scan_index for s in self.scans]
        if indices != sorted(set(indices)) or (indices and indices[-1] >= self.scan_count):
            raise ValueError("scans must be sorted by unique scan_index below scan_count")
        if self.structure.status not in ("not_reviewed", "ambiguous") and not self.complete:
            raise ValueError("structure can only be reviewed when every scan of the book is listed")
        for chapters in [self.structure.value or [], *(a or [] for a in self.structure.alternatives)]:
            unknown = sorted({c.start_scan_id for c in chapters if c.start_scan_id} - set(ids))
            if unknown:
                raise ValueError(f"chapter start_scan_id not among the scans: {unknown}")
        return self


def load_gold(path: Path) -> GoldBook:
    return GoldBook.model_validate_json(path.read_bytes())


def gold_template(
    inventory: Inventory, book_id: str, source: str | None, txt_dir: Path | None, alto_dir: Path | None
) -> GoldBook:
    """Manifest of every listed scan with all labels ``not_reviewed``; OCR sidecars ``<scan_id>.txt|.xml``."""
    scans = [
        GoldScan(
            scan_id=s.scan_id,
            scan_index=s.scan_index,
            filename=s.filename,
            image_sha256=s.image_sha256,
            txt=_sidecar(txt_dir, s.scan_id + ".txt"),
            alto=_sidecar(alto_dir, s.scan_id + ".xml"),
        )
        for s in inventory.scans
    ]
    bibliography = {f: GoldValue[list[str]]() for f in BiblioField}
    return GoldBook(
        book_id=book_id, source=source, scan_count=inventory.total_listed, scans=scans, bibliography=bibliography
    )


def _sidecar(directory: Path | None, name: str) -> Sidecar | None:
    if directory is None or not (directory / name).is_file():
        return None
    return Sidecar(filename=name, sha256=hashlib.sha256((directory / name).read_bytes()).hexdigest())
