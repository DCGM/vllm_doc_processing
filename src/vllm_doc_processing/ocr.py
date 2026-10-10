"""Optional pre-existing OCR sidecars (plain text or ALTO XML), one per scan, sent as bounded text with the image.

Sidecars are matched by the scan ID (``<scan_id>.txt`` or ``<scan_id>.xml``, extension case-insensitive)
in one directory. A missing sidecar is allowed (the scan is observed from the image only); an ambiguous
match, an unreadable file, invalid UTF-8 or malformed/non-ALTO XML stops the run before any request.
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
from .models import OcrInput

EXTENSIONS: dict[str, str] = {".txt": "txt", ".xml": "alto"}
TRUNCATION_MARK = "\n[... {omitted} characters of OCR text omitted ...]\n"


def attach_ocr(inventory: Inventory, ocr_dir: Path, ocr_format: OcrFormat, max_chars: int) -> Inventory:
    """The inventory with ``scan.ocr`` metadata for every scan and the bounded OCR text of matched scans.

    Raises ``InputError`` listing every ambiguous, unreadable or malformed sidecar.
    """
    if not ocr_dir.is_dir():
        raise InputError(f"OCR directory not found: {ocr_dir}")
    allowed = {ext for ext, fmt in EXTENSIONS.items() if ocr_format in ("auto", fmt)}
    by_stem: dict[str, list[Path]] = {}
    for entry in ocr_dir.iterdir():
        if entry.is_file() and not entry.name.startswith(".") and entry.suffix.lower() in allowed:
            by_stem.setdefault(entry.stem, []).append(entry)

    scans, texts, problems = [], {}, []
    for scan in inventory.scans:
        candidates = sorted(by_stem.get(scan.scan_id, []))
        if len(candidates) > 1:
            problems.append(f"{scan.scan_id!r} matches several OCR files: {', '.join(p.name for p in candidates)}")
            continue
        if not candidates:
            scans.append(scan.model_copy(update={"ocr": OcrInput(status="missing")}))
            continue
        path = candidates[0]
        fmt = EXTENSIONS[path.suffix.lower()]
        try:
            data = path.read_bytes()
            text = read_alto(data) if fmt == "alto" else read_txt(data)
        except OSError as exc:
            problems.append(f"cannot read {path.name}: {exc.strerror or exc}")
            continue
        except ValueError as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        sent = bound_text(text, max_chars)
        texts[scan.scan_id] = sent
        info = OcrInput(
            status="ok",
            filename=path.name,
            format=fmt,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            chars=len(text),
            sent_chars=len(sent),
            truncated=sent != text,
        )
        scans.append(scan.model_copy(update={"ocr": info}))
    if problems:
        shown = problems[:20] + ([f"... and {len(problems) - 20} more"] if len(problems) > 20 else [])
        raise InputError(f"unusable OCR sidecars in {ocr_dir}: " + "; ".join(shown))
    return dataclasses.replace(inventory, scans=scans, ocr_dir=ocr_dir, ocr_texts=texts)


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


def ocr_summary(inventory: Inventory) -> dict[str, int]:
    """Counts of scans by OCR status and format; empty without an OCR directory."""
    if inventory.ocr_dir is None:
        return {}
    infos = [s.ocr for s in inventory.scans if s.ocr is not None]
    return {
        "txt": sum(i.format == "txt" for i in infos),
        "alto": sum(i.format == "alto" for i in infos),
        "missing": sum(i.status == "missing" for i in infos),
        "truncated": sum(i.truncated for i in infos),
    }


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""
