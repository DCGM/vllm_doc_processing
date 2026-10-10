"""Optional pre-existing OCR sidecars (plain text or ALTO XML), one per scan, sent as bounded text with the image.

Sidecars are matched by the scan ID (``<scan_id>.txt`` or ``<scan_id>.xml``, extension case-insensitive)
in one directory. A scan whose sidecar is missing or unusable (ambiguous match, unreadable file, invalid
UTF-8, malformed or non-ALTO XML) is observed from the image only; the problem is recorded in its
``ocr`` metadata and reported, never replaced by other text.
The OCR files are only read; their text is never written to the output, checkpoint or logs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import OcrFormat
from .images import InputError, Inventory
from .models import OcrInput, ScanRecord

EXTENSIONS: dict[str, str] = {".txt": "txt", ".xml": "alto"}
TRUNCATION_MARK = "\n[... {omitted} characters of OCR text omitted ...]\n"


def attach_ocr(inventory: Inventory, ocr_dir: Path, ocr_format: OcrFormat, max_chars: int) -> Inventory:
    """The inventory with ``scan.ocr`` metadata for every scan and the bounded OCR text of usable sidecars.

    Only a missing OCR directory raises ``InputError``; a sidecar that cannot be used is recorded in
    its scan's metadata (``status="error"``) and that scan gets no OCR text.
    """
    if not ocr_dir.is_dir():
        raise InputError(f"OCR directory not found: {ocr_dir}")
    by_stem = find_sidecars(ocr_dir, {ext for ext, fmt in EXTENSIONS.items() if ocr_format in ("auto", fmt)})
    scans, texts = [], {}
    for scan in inventory.scans:
        info, text = _load_sidecar(by_stem.get(scan.scan_id, []), max_chars)
        scans.append(scan.model_copy(update={"ocr": info}))
        if text is not None:
            texts[scan.scan_id] = text
    return dataclasses.replace(inventory, scans=scans, ocr_dir=ocr_dir, ocr_texts=texts)


def find_sidecars(directory: Path, extensions: set[str]) -> dict[str, list[Path]]:
    """Sidecar candidates by file stem (= scan ID): files directly in ``directory`` whose lower-case
    extension is in ``extensions``, sorted. More than one candidate for a scan is an ambiguous match.
    Shared by processing and ``gold-template`` so both see the same OCR files."""
    by_stem: dict[str, list[Path]] = {}
    for entry in directory.iterdir():
        if entry.is_file() and not entry.name.startswith(".") and entry.suffix.lower() in extensions:
            by_stem.setdefault(entry.stem, []).append(entry)
    return {stem: sorted(paths) for stem, paths in by_stem.items()}


def _load_sidecar(candidates: list[Path], max_chars: int) -> tuple[OcrInput, str | None]:
    """Metadata and bounded text of the scan's only sidecar; no text if it is missing or unusable."""
    if not candidates:
        return OcrInput(status="missing"), None
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        return OcrInput(status="error", error=f"several OCR files match the scan: {names}"), None
    path = candidates[0]
    fmt = EXTENSIONS[path.suffix.lower()]
    try:
        data = path.read_bytes()
    except OSError as exc:
        error = f"cannot read: {exc.strerror or exc}"
        return OcrInput(status="error", filename=path.name, format=fmt, error=error), None
    file_info = dict(filename=path.name, format=fmt, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    try:
        text = read_alto(data) if fmt == "alto" else read_txt(data)
    except ValueError as exc:
        return OcrInput(status="error", error=str(exc), **file_info), None
    sent = bound_text(text, max_chars)
    return OcrInput(status="ok", chars=len(text), sent_chars=len(sent), truncated=sent != text, **file_info), sent


def read_txt(data: bytes) -> str:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"not valid UTF-8 text ({exc.reason} at byte {exc.start})") from None
    return normalize(text)


def read_alto(data: bytes) -> str:
    """Text of an ALTO document (any ALTO namespace version, or none) in document order.

    One line per ``TextLine`` (``String/@CONTENT`` joined by spaces, ``HYP`` appended to the last word),
    a blank line between ``TextBlock``s. Coordinates and styles are ignored.
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"malformed XML ({exc})") from None
    if _local(root.tag) != "alto":
        raise ValueError(f"not an ALTO document (root element <{_local(root.tag)}>)")
    blocks = []
    for block in root.iter():
        if _local(block.tag) != "TextBlock":
            continue
        lines = []
        for line in block.iter():
            if _local(line.tag) != "TextLine":
                continue
            words: list[str] = []
            for item in line:
                tag, content = _local(item.tag), item.get("CONTENT", "")
                if tag == "String" and content:
                    words.append(content)
                elif tag == "HYP" and words:
                    words[-1] += content
            lines.append(" ".join(words))
        blocks.append("\n".join(lines))
    return normalize("\n\n".join(blocks))


def normalize(text: str) -> str:
    """Unix line breaks, no trailing spaces, at most one blank line in a row, no leading/trailing blank lines."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")


def bound_text(text: str, max_chars: int) -> str:
    """``text`` if short enough, else its start and end (where page numbers and running heads are) cut at
    line breaks where possible, joined by a visible marker; the result is at most ``max_chars`` long."""
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(TRUNCATION_MARK.format(omitted=len(text)))
    head, tail = text[: budget * 2 // 3], text[len(text) - budget // 3 :]
    if "\n" in head:
        head = head[: head.rfind("\n")]
    if "\n" in tail:
        tail = tail[tail.find("\n") + 1 :]
    return head + TRUNCATION_MARK.format(omitted=len(text) - len(head) - len(tail)) + tail


def ocr_summary(scans: list[ScanRecord]) -> dict[str, int]:
    """Counts of scans by OCR use: TXT or ALTO text sent, no sidecar, unusable sidecar, shortened text."""
    infos = [s.ocr for s in scans if s.ocr is not None]
    return {
        "txt": sum(i.status == "ok" and i.format == "txt" for i in infos),
        "alto": sum(i.status == "ok" and i.format == "alto" for i in infos),
        "missing": sum(i.status == "missing" for i in infos),
        "error": sum(i.status == "error" for i in infos),
        "truncated": sum(i.truncated for i in infos),
    }


def ocr_problems(scans: list[ScanRecord]) -> list[str]:
    """One message per scan whose sidecar could not be used (it was sent as image only)."""
    return [
        f"OCR of scan {s.scan_id}{f' ({s.ocr.filename})' if s.ocr.filename else ''} not used, image only: {s.ocr.error}"
        for s in scans
        if s.ocr is not None and s.ocr.status == "error"
    ]


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""
