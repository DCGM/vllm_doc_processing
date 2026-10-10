import json

import pytest
from test_llm import completion
from test_process import NAMES, RECONCILED, answers, checkpoint, output, run  # noqa: F401 (fixture)

from vllm_doc_processing.cli import EXIT_CONFIG
from vllm_doc_processing.images import InputError, build_inventory
from vllm_doc_processing.ocr import attach_ocr, bound_text, ocr_problems, read_alto
from vllm_doc_processing.predictions import load_prediction

ALTO = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="{ns}"><Layout><Page><PrintSpace>
  <TextBlock ID="b1">
    <TextLine><String CONTENT="KAPITOLA"/><SP/><String CONTENT="I."/></TextLine>
    <TextLine><String CONTENT="Příliš"/><SP/><String CONTENT="žluťoučký"/><SP/><String CONTENT="kůň"/><HYP CONTENT="-"/></TextLine>
    <TextLine><String CONTENT="úpěl"/></TextLine>
  </TextBlock>
  <ComposedBlock><TextBlock ID="b2"><TextLine><String CONTENT="12"/></TextLine></TextBlock></ComposedBlock>
</PrintSpace></Page></Layout></alto>"""
ALTO_TEXT = "KAPITOLA I.\nPříliš žluťoučký kůň-\núpěl\n\n12"


@pytest.mark.parametrize(
    "ns", ["http://www.loc.gov/standards/alto/ns-v2#", "http://www.loc.gov/standards/alto/ns-v4#", ""]
)
def test_alto_text_in_document_order_for_any_namespace(ns):
    data = ALTO.format(ns=ns).replace(' xmlns=""', "").encode()
    assert read_alto(data) == ALTO_TEXT


@pytest.mark.parametrize(
    ("data", "message"),
    [(b"<alto><Layout>", "malformed XML"), (b"<html><body/></html>", "not an ALTO document")],
)
def test_bad_alto_is_an_error(data, message):
    with pytest.raises(ValueError, match=message):
        read_alto(data)


def test_truncation_keeps_start_and_end_within_bound():
    text = "\n".join(f"line {i:04d} " + "x" * 40 for i in range(200))
    short = bound_text(text, 1000)
    assert len(short) <= 1000 and short.startswith("line 0000") and short.endswith(text.splitlines()[-1])
    assert "characters of OCR text omitted" in short
    assert bound_text("short", 1000) == "short"


@pytest.fixture
def book(tmp_path):
    from PIL import Image

    book_dir, ocr_dir = tmp_path / "book", tmp_path / "ocr"
    book_dir.mkdir(), ocr_dir.mkdir()
    (book_dir / "order.txt").write_text("a\nb\nc\n")
    for name in "abc":
        Image.new("L", (40, 60), 255).save(book_dir / f"{name}.png")
    return book_dir, ocr_dir


def test_sidecars_matched_by_scan_id_missing_and_unusable_recorded_per_scan(book):
    book_dir, ocr_dir = book
    (ocr_dir / "a.txt").write_text("Cesty po Šumavě\r\n\r\n\r\n\r\nPraha 1923  \n", encoding="utf-8")
    (ocr_dir / "b.XML").write_text(ALTO.format(ns="http://www.loc.gov/standards/alto/ns-v3#"), encoding="utf-8")
    inventory = build_inventory(book_dir, book_dir / "order.txt")
    ocr = attach_ocr(inventory, ocr_dir, "auto", 1000)
    assert ocr.ocr_texts == {"a": "Cesty po Šumavě\n\nPraha 1923", "b": ALTO_TEXT}
    a, b, c = (s.ocr for s in ocr.scans)
    assert (a.status, a.format, a.filename, a.truncated) == ("ok", "txt", "a.txt", False)
    assert (b.format, b.chars, b.sent_chars) == ("alto", len(ALTO_TEXT), len(ALTO_TEXT)) and len(b.sha256) == 64
    assert c.status == "missing" and c.filename is None
    assert all(s.ocr is None for s in inventory.scans)  # the original inventory is unchanged

    assert attach_ocr(inventory, ocr_dir, "txt", 1000).ocr_texts.keys() == {"a"}

    # Unusable sidecars never stop the book: the scan gets no OCR text and the reason is recorded.
    (ocr_dir / "a.xml").write_text("<alto/>")  # a.txt + a.xml: ambiguous in auto mode
    (ocr_dir / "b.XML").write_text("<alto><Layout>")  # malformed
    (ocr_dir / "c.txt").write_bytes("Šumava".encode("cp1250"))  # not UTF-8
    ocr = attach_ocr(inventory, ocr_dir, "auto", 1000)
    assert ocr.ocr_texts == {} and [s.ocr.status for s in ocr.scans] == ["error"] * 3
    a, b, c = (s.ocr for s in ocr.scans)
    assert a.error == "several OCR files match the scan: a.txt, a.xml" and a.filename is None
    assert b.error.startswith("malformed XML") and b.filename == "b.XML" and len(b.sha256) == 64
    assert c.error.startswith("not valid UTF-8") and c.chars is None
    assert ocr_problems(ocr.scans)[1].startswith("OCR of scan b (b.XML) not used, image only: malformed XML")
    with pytest.raises(InputError, match="OCR directory not found"):
        attach_ocr(inventory, ocr_dir / "nope", "auto", 1000)


def test_process_with_ocr_sends_own_text_only_and_records_metadata(run, tmp_path):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / f"{NAMES[0]}.txt").write_text("CESTY PO ŠUMAVĚ\nnapsal Jan Novák", encoding="utf-8")
    (ocr_dir / f"{NAMES[1]}.xml").write_text(ALTO.format(ns="http://www.loc.gov/standards/alto/ns-v4#"))
    (ocr_dir / f"{NAMES[2]}.txt").write_text("y" * 5000)
    flags = ["--ocr-dir", str(ocr_dir), "--max-pages", "4", "--config", str(tmp_path / "cfg.json")]
    (tmp_path / "cfg.json").write_text(json.dumps({"ocr_max_chars": 1000}))

    code, requests = run(answers(0, 4) + [completion(json.dumps(RECONCILED))], *flags)
    assert code == 0
    prompts = [r["messages"][1]["content"][-1]["text"] for r in requests[:4]]
    assert "<ocr>\nCESTY PO ŠUMAVĚ\nnapsal Jan Novák\n</ocr>" in prompts[0]
    assert ALTO_TEXT in prompts[1] and "CESTY" not in prompts[1]  # no OCR of earlier scans
    assert "characters of OCR text omitted" in prompts[2] and prompts[2].count("y") < 1000
    assert "<ocr>" not in prompts[3]  # missing sidecar: image only
    assert all(r["messages"][1]["content"][0]["type"] == "image_url" for r in requests[:4])

    book = output(tmp_path)
    assert book.source.ocr_directory == str(ocr_dir) and "ocr" in book.run.prompt_versions
    assert [s.ocr.format for s in book.scans] == ["txt", "alto", "txt", None]
    assert book.scans[2].ocr.truncated and book.scans[3].ocr.status == "missing"
    assert book.run.warnings == [
        "1 scan(s) had no OCR sidecar and were sent as image only",
        "OCR text of 1 scan(s) was shortened to ocr_max_chars",
    ]
    for path in ("book.json", "book.checkpoint.json"):  # raw OCR text is never stored
        assert "napsal" not in (tmp_path / "out" / path).read_text(encoding="utf-8")
    provenance = load_prediction(tmp_path / "out" / "book.json")[0].provenance
    assert provenance["ocr_mode"] == {
        "format": "auto", "max_chars": 1000, "txt": 2, "alto": 1, "missing": 1, "error": 0, "truncated": 1
    }

    # A changed, added or dropped sidecar, OCR setting or OCR on/off invalidates the checkpoint.
    (ocr_dir / f"{NAMES[3]}.txt").write_text("new")
    for extra in ([], ["--ocr-format", "txt"]):
        assert run([], "--resume", *flags, *extra) == (EXIT_CONFIG, [])
    (ocr_dir / f"{NAMES[3]}.txt").unlink()
    assert run([], "--resume", "--max-pages", "4") == (EXIT_CONFIG, [])
    assert run([completion(json.dumps(RECONCILED))], "--resume", *flags)[0] == 0


def test_corrupted_sidecar_is_reported_and_scan_processed_from_image(run, tmp_path):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / f"{NAMES[0]}.xml").write_text("<alto><Layout>")
    (ocr_dir / f"{NAMES[1]}.txt").write_text("Kapitola", encoding="utf-8")
    code, requests = run(answers(0, 2) + [completion(json.dumps(RECONCILED))], "--max-pages", "2", "--ocr-dir", str(ocr_dir))
    assert code == 0 and len(requests) == 3
    prompts = [r["messages"][1]["content"][-1]["text"] for r in requests[:2]]
    assert "<ocr>" not in prompts[0] and "<ocr>\nKapitola\n</ocr>" in prompts[1]
    book = output(tmp_path)
    assert all(s.observation for s in book.scans) and book.scans[0].ocr.status == "error"
    assert any(w.startswith(f"OCR of scan {NAMES[0]} ({NAMES[0]}.xml) not used") for w in book.run.warnings)


def test_image_only_run_has_no_ocr(run, tmp_path):
    code, requests = run(answers(0, 2) + [completion(json.dumps(RECONCILED))], "--max-pages", "2")
    assert code == 0 and not any("<ocr>" in r["messages"][1]["content"][-1]["text"] for r in requests)
    book = output(tmp_path)
    assert book.source.ocr_directory is None and all(s.ocr is None for s in book.scans)
    assert "ocr" not in book.run.prompt_versions and checkpoint(tmp_path).identity["ocr"] is None

    # Image-only outputs of schema 0.2 (before OCR support) are still evaluated.
    path = tmp_path / "out" / "book.json"
    path.write_text(path.read_text().replace('"schema_version": "0.3"', '"schema_version": "0.2"'))
    provenance = load_prediction(path)[0].provenance
    assert provenance["ocr_mode"] == "image-only" and provenance["schema_version"] == "0.2"
