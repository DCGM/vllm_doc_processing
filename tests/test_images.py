import hashlib
import io

import pytest
from PIL import Image

from vllm_doc_processing.images import InputError, build_inventory, match_files, prepare_image, read_order_file

ORDER = ["f3c1", "07aa", "b9e2"]  # deliberately not sorted


@pytest.fixture
def book(tmp_path):
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    Image.new("RGB", (300, 400), "white").save(book_dir / "f3c1.JPG")
    Image.new("L", (500, 350), 200).save(book_dir / "07aa.png")
    Image.new("I;16", (200, 100), 40000).save(book_dir / "b9e2.tif")
    Image.new("RGB", (10, 10)).save(book_dir / "zz-unlisted.webp")
    (book_dir / "notes.txt").write_text("not an image")
    (book_dir / "order.txt").write_text("\n" + "\n".join(f"  {n} " for n in ORDER) + "\n\n")
    return book_dir


def test_inventory_follows_order_file(book):
    inv = build_inventory(book, book / "order.txt")
    assert [(s.scan_index, s.scan_id, s.filename) for s in inv.scans] == [
        (0, "f3c1", "f3c1.JPG"),
        (1, "07aa", "07aa.png"),
        (2, "b9e2", "b9e2.tif"),
    ]
    assert [(s.width, s.height) for s in inv.scans] == [(300, 400), (500, 350), (200, 100)]
    assert inv.scans[1].image_sha256 == hashlib.sha256((book / "07aa.png").read_bytes()).hexdigest()
    assert inv.unlisted == ["zz-unlisted.webp"]  # reported; non-image files are ignored
    assert [s.scan_id for s in build_inventory(book, book / "order.txt", max_pages=2).scans] == ORDER[:2]


@pytest.mark.parametrize(
    "order, extra_file, message",
    [
        ("\n  \n", None, "order file is empty"),
        ("f3c1\n07aa\nf3c1\n", None, "duplicate name 'f3c1'"),
        ("f3c1\nsub/07aa\n", None, "path separator"),
        ("f3c1\nmissing\n", None, "'missing' has no matching file"),
        ("f3c1\n", "f3c1.png", "'f3c1' matches several files: f3c1.JPG, f3c1.png"),
        ("f3c1\nscan9\n", "scan9.gif", "'scan9' has no supported image, only: scan9.gif"),
    ],
)
def test_order_file_errors(book, order, extra_file, message):
    (book / "order.txt").write_text(order)
    if extra_file:
        Image.new("RGB", (10, 10)).save(book / extra_file)
    with pytest.raises(InputError, match=message):
        build_inventory(book, book / "order.txt")


def test_corrupt_and_multiframe_images_are_reported(book):
    (book / "07aa.png").write_bytes(b"\x89PNG broken")
    frames = [Image.new("L", (10, 10)), Image.new("L", (10, 10))]
    frames[0].save(book / "b9e2.tif", save_all=True, append_images=frames[1:])
    with pytest.raises(InputError) as exc:
        build_inventory(book, book / "order.txt")
    message = str(exc.value)
    assert "07aa.png: cannot decode image" in message and "b9e2.tif: multi-frame" in message


def test_tiff_is_converted_in_memory(book):
    original = (book / "b9e2.tif").read_bytes()
    prepared = prepare_image(book / "b9e2.tif", max_side=2048)
    assert (prepared.mime_type, prepared.reencoded, prepared.width, prepared.height) == ("image/jpeg", True, 200, 100)
    decoded = Image.open(io.BytesIO(prepared.data))
    assert decoded.format == "JPEG" and decoded.mode == "L"
    assert 150 < decoded.getpixel((5, 5)) < 160  # 16-bit 40000/65535 scaled, not clipped to white
    assert (book / "b9e2.tif").read_bytes() == original
    assert prepared.data_url().startswith("data:image/jpeg;base64,")
    assert prepare_image(book / "b9e2.tif", None, "png").mime_type == "image/png"


def test_small_upright_files_pass_through_and_large_ones_are_downscaled(book):
    small = prepare_image(book / "07aa.png", max_side=2048)
    assert not small.reencoded and small.data == (book / "07aa.png").read_bytes()
    assert small.mime_type == "image/png"
    large = prepare_image(book / "07aa.png", max_side=256)
    assert large.reencoded and (large.width, large.height) == (256, 179)
    assert prepare_image(book / "07aa.png", max_side=None).data == small.data  # never upscaled


def test_exif_orientation_is_applied(book):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    Image.new("RGB", (300, 400)).save(book / "f3c1.JPG", exif=exif)
    scan = build_inventory(book, book / "order.txt").scans[0]
    assert (scan.width, scan.height) == (400, 300)
    prepared = prepare_image(book / "f3c1.JPG", max_side=2048)
    assert prepared.reencoded and (prepared.width, prepared.height) == (400, 300)


def test_directory_without_images(tmp_path):
    (tmp_path / "order.txt").write_text("a\n")
    Image.new("L", (8, 8)).save(tmp_path / "a.gif")
    with pytest.raises(InputError) as exc:
        build_inventory(tmp_path, tmp_path / "order.txt")
    assert "contains no files with extensions" in str(exc.value) and "only: a.gif" in str(exc.value)


@pytest.mark.parametrize("white, black_text", [(4095, 0), (65535, 0), (40000, 0)])
def test_high_bit_depth_gray_keeps_full_range(tmp_path, white, black_text):
    """12-bit data in a 16-bit TIFF (white = 4095) must not turn almost black."""
    image = Image.new("I;16", (20, 10), white)
    image.paste(black_text, (0, 0, 10, 10))
    image.save(tmp_path / "s.tif")
    decoded = Image.open(io.BytesIO(prepare_image(tmp_path / "s.tif", None, "png").data))
    expected_white = round(white * 255 / (4095 if white <= 4095 else 65535))
    assert abs(decoded.getpixel((15, 5)) - expected_white) <= 1 and decoded.getpixel((5, 5)) == 0


def test_match_files_supports_other_extensions(book):
    (book / "07aa.xml").write_text("<alto/>")
    matches, unlisted = match_files(book, ["07aa"], extensions=(".xml",))
    assert matches["07aa"].name == "07aa.xml" and unlisted == []


def test_read_order_file_strips_bom(tmp_path):
    path = tmp_path / "order.txt"
    path.write_text("﻿a\nb\n", encoding="utf-8")
    assert read_order_file(path) == ["a", "b"]
