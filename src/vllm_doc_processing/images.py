"""Scan inventory (order file -> image files -> manifest) and in-memory image preparation.

Original files are only read, never modified. Scan IDs are the names listed in the
order file; ``scan_index`` is their zero-based position there.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ConfigError
from .models import ScanRecord

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
"""Accepted input extensions (case-insensitive)."""

UPLOAD_MIME_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
"""Pillow formats sent unchanged when no conversion is needed; anything else (e.g. TIFF) is re-encoded."""

JPEG_QUALITY = 90
EXIF_ORIENTATION = 0x0112

# Large 600 dpi spreads exceed Pillow's default decompression-bomb limit (~89 Mpx).
Image.MAX_IMAGE_PIXELS = 400_000_000

ImageFormat = Literal["jpeg", "png"]


class InputError(ConfigError):
    """Invalid book input (order file, missing/ambiguous/corrupt images)."""


def read_order_file(path: Path) -> list[str]:
    """Scan names in order; blank lines and surrounding whitespace are ignored."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read order file {path}: {exc}") from None
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        raise InputError(f"order file is empty: {path}")
    problems = []
    seen: set[str] = set()
    for lineno, name in enumerate(names, 1):
        if "/" in name or "\\" in name:
            problems.append(f"entry {lineno} {name!r} contains a path separator; list bare names")
        if name in seen:
            problems.append(f"duplicate name {name!r}")
        seen.add(name)
    if problems:
        raise InputError(f"invalid order file {path}: " + "; ".join(problems))
    return names


def match_files(
    directory: Path, names: list[str], extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS
) -> tuple[dict[str, Path], list[str]]:
    """Match each name to exactly one ``<name><ext>`` file in ``directory`` (non-recursive).

    Returns the matches and the sorted file names with an accepted extension that were not listed.
    Raises ``InputError`` listing every missing or ambiguous name.
    """
    by_stem: dict[str, list[Path]] = {}
    for entry in directory.iterdir():
        if entry.is_file() and not entry.name.startswith("."):
            by_stem.setdefault(entry.stem, []).append(entry)
    if not any(p.suffix.lower() in extensions for paths in by_stem.values() for p in paths):
        raise InputError(f"no files with extensions {', '.join(extensions)} in {directory}")

    matches: dict[str, Path] = {}
    problems: list[str] = []
    for name in names:
        candidates = by_stem.get(name, [])
        accepted = sorted(p for p in candidates if p.suffix.lower() in extensions)
        if len(accepted) == 1:
            matches[name] = accepted[0]
        elif accepted:
            problems.append(f"{name!r} matches several files: {', '.join(p.name for p in accepted)}")
        elif candidates:
            other = ", ".join(sorted(p.name for p in candidates))
            problems.append(f"{name!r} has no supported image, only: {other}")
        else:
            problems.append(f"{name!r} has no matching file")
    if problems:
        shown = problems[:20] + ([f"... and {len(problems) - 20} more"] if len(problems) > 20 else [])
        raise InputError(f"order file does not match {directory}: " + "; ".join(shown))

    listed = set(names)
    unlisted = sorted(
        p.name for stem, paths in by_stem.items() if stem not in listed for p in paths if p.suffix.lower() in extensions
    )
    return matches, unlisted


@dataclass(frozen=True)
class Inventory:
    book_dir: Path
    scans: list[ScanRecord]
    unlisted: list[str]
    """Supported image files in ``book_dir`` that the order file does not list (not processed)."""
    total_listed: int

    def path(self, scan: ScanRecord) -> Path:
        return self.book_dir / scan.filename


def build_inventory(book_dir: Path, order_file: Path, max_pages: int | None = None) -> Inventory:
    """Validate the whole order file, then hash and decode the first ``max_pages`` listed images."""
    names = read_order_file(order_file)
    matches, unlisted = match_files(book_dir, names)
    selected = names[:max_pages] if max_pages else names
    scans: list[ScanRecord] = []
    problems: list[str] = []
    for index, name in enumerate(selected):
        path = matches[name]
        try:
            data = path.read_bytes()
            width, height = _upright_size(_open(data, path))
        except InputError as exc:
            problems.append(str(exc))
            continue
        except OSError as exc:
            problems.append(f"cannot read {path.name}: {exc}")
            continue
        scans.append(
            ScanRecord(
                scan_id=name,
                scan_index=index,
                filename=path.name,
                image_sha256=hashlib.sha256(data).hexdigest(),
                width=width,
                height=height,
            )
        )
    if problems:
        raise InputError(f"unusable images in {book_dir}: " + "; ".join(problems))
    return Inventory(book_dir=book_dir, scans=scans, unlisted=unlisted, total_listed=len(names))


@dataclass(frozen=True)
class PreparedImage:
    """Upload-ready image bytes; carries no file name or path."""

    data: bytes
    mime_type: str
    width: int
    height: int
    reencoded: bool

    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode('ascii')}"


def prepare_image(path: Path, max_side: int | None, image_format: ImageFormat = "jpeg") -> PreparedImage:
    """Return the scan as upload-ready bytes, upright and with its longest side <= ``max_side``.

    JPEG/PNG/WebP files in RGB or grayscale that need no rotation or resizing are sent byte-for-byte;
    everything else (TIFF, CMYK, 16-bit, alpha, EXIF-rotated, oversized) is re-encoded in memory as
    ``image_format``. Images are never upscaled.
    """
    data = path.read_bytes()
    image = _open(data, path)
    oversized = max_side is not None and max(image.size) > max_side
    orientation = image.getexif().get(EXIF_ORIENTATION, 1)
    if image.format in UPLOAD_MIME_TYPES and image.mode in ("RGB", "L") and orientation == 1 and not oversized:
        return PreparedImage(data, UPLOAD_MIME_TYPES[image.format], *image.size, reencoded=False)

    image = _to_rgb_or_gray(ImageOps.exif_transpose(image))
    if max_side is not None:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    if image_format == "jpeg":
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    else:
        image.save(buffer, format="PNG")
    return PreparedImage(buffer.getvalue(), f"image/{image_format}", *image.size, reencoded=True)


def _open(data: bytes, path: Path) -> Image.Image:
    """Open and fully decode one single-frame image; errors name only the file, not its directory."""
    try:
        image = Image.open(io.BytesIO(data))
        # Camera JPEGs often open as MPO (main image + previews); its first frame is the image itself.
        if getattr(image, "n_frames", 1) > 1 and image.format != "MPO":
            raise InputError(f"{path.name}: multi-frame image ({image.n_frames} frames); expected one image per scan")
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InputError(f"{path.name}: cannot decode image ({exc.__class__.__name__}: {exc})") from None
    return image


def _upright_size(image: Image.Image) -> tuple[int, int]:
    """Size after applying the EXIF orientation (as the model sees the scan)."""
    width, height = image.size
    if image.getexif().get(EXIF_ORIENTATION, 1) in (5, 6, 7, 8):
        return height, width
    return width, height


def _to_rgb_or_gray(image: Image.Image) -> Image.Image:
    if image.mode in ("RGB", "L"):
        return image
    if image.mode == "1":
        return image.convert("L")
    if image.mode.startswith("I"):  # 16/32-bit grayscale scans; plain convert("L") would clip
        image = image.convert("I")
        scale = 255 / 65535 if image.getextrema()[1] > 255 else 1
        return image.point(lambda v: v * scale).convert("L")
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return image.convert("RGB")
