import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from vllm_doc_processing.cli import EXIT_CONFIG, main
from vllm_doc_processing.evaluation import evaluate, to_markdown
from vllm_doc_processing.gold import GoldBook, GoldChapter, GoldValue, load_gold
from vllm_doc_processing.predictions import Incomparable, load_prediction, normalize_number, page_type

EXAMPLES = Path(__file__).parents[1] / "examples"
BOOK = EXAMPLES / "annotated_book.example.json"
GOLD = EXAMPLES / "gold.example.json"
P = "5b1f0c2e-7d4a-4e8b-9c3f-00000000000"
H = "0" * 63


def gold_book(scans, **extra) -> GoldBook:
    return GoldBook.model_validate({"book_id": "g", "scan_count": 5, "scans": scans, **extra})


def scan(i, **labels):
    return {"scan_id": P + str(i), "scan_index": i - 1, "image_sha256": H + str(i), **labels}


def run(gold, *preds, comparators=()):
    return evaluate([("gold.json", "-", gold)] if gold else [], list(preds), list(comparators))


def example_gold() -> GoldBook:
    return load_gold(GOLD)[0]


def test_gold_status_rules():
    with pytest.raises(ValidationError, match="absent"):
        gold_book([scan(1, page_type={"status": "verified"})])
    with pytest.raises(ValidationError, match="must not carry a value"):
        gold_book([scan(1, page_type={"status": "not_reviewed", "value": "blank"})])
    with pytest.raises(ValidationError, match="every scan"):
        gold_book([scan(1)], structure={"status": "verified", "value": [{"title": "A"}]})
    assert example_gold().complete


def test_null_vs_unreviewed_and_denominators():
    book = load_prediction(BOOK)  # observed: [] numbers on scans 1-3, ['v','vi'], ['1','2']
    gold = gold_book(
        [
            scan(1, printed_numbers={"status": "absent"}, side={"status": "not_applicable"}),  # pred None: correct
            scan(2, printed_numbers={"status": "verified", "value": ["3"]}),  # pred none: missing_value
            scan(3, printed_numbers={"status": "not_reviewed"}, side={"status": "ambiguous"}),  # never scored
            scan(4, printed_numbers={"status": "absent"}),  # pred ['v','vi']: spurious_value
            scan(5, printed_numbers={"status": "ambiguous", "alternatives": [["1", "2"], []]}),
        ]
    )
    observed = next(a for a in run(gold, *book)["accuracy"] if a["layer"] == "observed")
    numbers = observed["fields"]["printed_numbers_exact"]
    assert numbers["outcomes"]["correct"] == 2 and numbers["outcomes"]["missing_value"] == 1
    assert numbers["outcomes"]["spurious_value"] == 1
    assert (numbers["eligible"], numbers["scored"], numbers["accuracy"]) == (4, 4, 0.5)
    assert numbers["by_reference_status"]["not_reviewed"] == {"not_scored": 1}
    side = observed["fields"]["side"]
    assert (side["eligible"], side["scored"], side["correct"]) == (2, 1, 1)  # ambiguous w/o alternatives unscorable
    assert observed["fields"]["page_type"]["eligible"] == 0  # all not reviewed


def test_printed_numbers_exact_vs_normalized():
    assert normalize_number("[xii].") == "XII" and normalize_number("- 012 -") == "12"
    gold = gold_book([scan(4, printed_numbers={"status": "verified", "value": ["V", "VI"]})])
    fields = run(gold, *load_prediction(BOOK))["accuracy"][0]["fields"]
    assert fields["printed_numbers_exact"]["correct"] == 0
    assert fields["printed_numbers_normalized"]["correct"] == 1


def test_hash_mismatch_and_missing_scans_are_not_scored():
    other = gold_book(
        [
            scan(1, page_type={"status": "verified", "value": "frontCover"}),
            {**scan(2, page_type={"status": "verified", "value": "titlePage"}), "image_sha256": "f" * 64},
        ]
    )
    resolved = run(other, *load_prediction(BOOK))["accuracy"][1]
    assert resolved["scans"]["hash_mismatch"] == 1
    pt = resolved["fields"]["page_type"]
    assert (pt["eligible"], pt["scored"], pt["outcomes"]["hash_mismatch"]) == (2, 1, 1)
    assert resolved["errors"][0]["outcome"] == "hash_mismatch"


def metakat(tmp_path):
    def page(i, ptype, side, number):
        return {"type": "page", "id": f"p{i}", "batch_id": "b", "batch_index": i, "pageIndex": i + 10,
                "pageType": [ptype, 0.9], "side": [side, 0.8], "pageNumber": [number, 0.7, "d"]}

    data = {
        "batch_id": "b",
        "page_to_image_mapping": {f"p{i}": f"{P}{i + 1}.jpg" for i in range(5)},
        "elements": [
            page(0, "FrontCover", "single_page", "[Ia]"),
            page(1, "Abstract", "right", "(I)"),
            page(2, "Impressum", "left", None),
            page(3, "TableOfContents", "left", "V,VI"),
            page(4, "NormalPage", "right", "1,2"),
            {"type": "volume", "id": "v", "title": ["Cesty po Šumavě", 0.9, "d"],
             "author": [["Jan Žlutický", 0.9, "d"], [None, 0.1, "d"]], "publisher": [None, 0.0, "d"]},
            {"type": "chapter", "id": "c1", "parent_id": "v", "title": ["Na cestě", 0.9, "d"], "pageIndexStart": 14},
            {"type": "chapter", "id": "c2", "parent_id": "c1", "title": ["U Černého jezera", 0.9, "d"]},
        ],
    }
    path = tmp_path / "metakat.json"
    path.write_text(json.dumps(data, ensure_ascii=False))
    return path


def test_metakat_import_maps_explicitly(tmp_path):
    [mk] = load_prediction(metakat(tmp_path))
    assert mk.role == "comparator" and mk.book_id == "v"
    s = mk.scans
    assert s[P + "1"].values["page_type"] == "frontCover"
    assert s[P + "1"].values["side"] == Incomparable("single_page")
    assert s[P + "2"].values["page_type"] == Incomparable("Abstract")
    assert s[P + "2"].values["page_number"] == "[I]"  # older '(I)' notation for an unprinted number
    assert s[P + "2"].scan_index == 11  # pageIndex
    assert mk.bibliography["publisher"] == []  # null MetaKat values are dropped
    assert mk.bibliography["title"] == ["Cesty po Šumavě"] and mk.bibliography["author"] == ["Jan Žlutický"]
    assert [(c.title, c.level, c.start_scan_id) for c in mk.structure] == [
        ("Na cestě", 1, P + "5"),
        ("U Černého jezera", 2, None),
    ]
    assert page_type("TITLEPAGE") == "titlePage" and page_type("FlyLeaf") == "flyleaf"


def test_gold_accuracy_separate_from_agreement(tmp_path):
    kramerius = tmp_path / "k.kramerius.json"
    pages = [{"scan_index": i, "scan_id": P + str(i + 1), "page_number": n, "page_type": t}
             for i, (n, t) in enumerate([("[1]", "FrontCover"), ("[III]", None), ("[IV]", "colophon"),
                                          ("V,VI", "tableOfContents"), ("1,2", "NormalPage")])]
    kramerius.write_text(json.dumps({"source": "x", "document": {"pid": "uuid:k"}, "pages": pages}))
    preds = load_prediction(BOOK)
    comps = load_prediction(metakat(tmp_path)) + load_prediction(kramerius)
    report = run(example_gold(), *preds, comparators=comps)

    systems = {(a["system"], a["layer"]) for a in report["accuracy"]}
    assert systems == {("vllm-doc", "observed"), ("vllm-doc", "resolved"), ("metakat", "reference"),
                       ("kramerius", "reference")}
    mk = next(a for a in report["accuracy"] if a["system"] == "metakat")
    assert mk["fields"]["page_type"]["outcomes"]["incomparable"] == 1  # Abstract is never equal to titlePage
    assert mk["fields"]["side"]["outcomes"]["incomparable"] == 1  # single_page vs not_applicable is not "null"
    assert "leaf" in mk["fields_not_provided"] and "printed_numbers_exact" in mk["fields_not_provided"]

    agree = {(a["layer"], a["comparator"]): a for a in report["agreement"]}
    assert set(agree) == {(lay, c) for lay in ("observed", "resolved") for c in ("metakat", "kramerius")}
    kr = agree[("resolved", "kramerius")]
    assert (kr["fields"]["page_number"]["agreeing"], kr["fields"]["page_number"]["scored"]) == (4, 5)  # [Ia] vs [1]
    assert kr["fields"]["page_type"]["eligible"] == 4  # null Kramerius page type is no reference value
    assert "accuracy" not in kr["fields"]["page_type"]
    assert report["incomparable_values"]["metakat (reference)"] == {
        "page_type": {"Abstract": 1}, "side": {"single_page": 1}
    }
    assert report["provenance"]["vllm-doc"]["usage"]["requests"] == 7


def test_structure_scoring():
    structure = run(example_gold(), *load_prediction(BOOK))["accuracy"][1]["structure"][0]
    assert structure["title_recall"] == 1.0 and structure["start_scan_correct"] == 2
    assert structure["wrong_toc_references"] == [{"title": "Předmluva", "reference": "V", "predicted": "vi"}]


def test_cli_reports_are_deterministic(tmp_path):
    argv = ["evaluate", "--gold", str(GOLD), "--prediction", str(BOOK), "--comparator", str(metakat(tmp_path))]
    outputs = []
    for n in (1, 2):
        assert main([*argv, "--json", str(tmp_path / f"{n}.json"), "--markdown", str(tmp_path / f"{n}.md")]) == 0
        outputs.append(((tmp_path / f"{n}.json").read_text(), (tmp_path / f"{n}.md").read_text()))
    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0][0])
    assert report["inputs"]["gold"][0]["sha256"] != "-"
    assert "## Agreement with comparator systems (not accuracy)" in outputs[0][1]
    assert to_markdown(report) == outputs[0][1]
    assert main(["evaluate", "--gold", str(GOLD), "--prediction", str(GOLD)]) == EXIT_CONFIG  # unknown format
    assert main(["evaluate", "--prediction", str(BOOK)]) == EXIT_CONFIG  # nothing to compare with


def test_gold_template(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    (book / "order.txt").write_text("b\na\n")
    for name in ("a", "b"):
        Image.new("L", (40, 60), 255).save(book / f"{name}.png")
    ocr = tmp_path / "ocr"
    ocr.mkdir()
    (ocr / "a.txt").write_text("Kapitola první\n", encoding="utf-8")
    out = tmp_path / "new-dir" / "gold.json"
    argv = ["gold-template", "--input", str(book), "--output", str(out), "--txt-dir", str(ocr)]
    assert main(argv) == 0
    gold, sha = load_gold(out)
    assert [(s.scan_id, s.scan_index, bool(s.txt)) for s in gold.scans] == [("b", 0, False), ("a", 1, True)]
    assert all(s.page_type.status == "not_reviewed" for s in gold.scans) and gold.complete
    assert evaluate([(str(out), sha, gold)], [], [])["gold_status_counts"]["page_type"] == {"not_reviewed": 2}
    assert main(argv) == EXIT_CONFIG  # never overwrites annotations


def test_metakat_chapter_cycle_is_an_error(tmp_path):
    data = {"batch_id": "b", "elements": [
        {"type": "chapter", "id": "c1", "parent_id": "c2"}, {"type": "chapter", "id": "c2", "parent_id": "c1"}]}
    path = tmp_path / "mk.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cycle"):
        load_prediction(path)


def test_absent_structure_scored_correct():
    gold = example_gold().model_copy(update={"structure": GoldValue[list[GoldChapter]](status="absent")})
    book = load_prediction(BOOK)[1]
    book.structure = []
    assert run(gold, book)["accuracy"][0]["structure"][0]["no_chapters_correct"] is True


def test_document_level_needs_whole_unchanged_book():
    gold = example_gold()
    partial = load_prediction(BOOK)[1]
    del partial.scans[P + "5"]  # e.g. a --max-pages run
    acc = run(gold, partial)["accuracy"][0]
    assert acc["documents"][0]["reason"] == "partial prediction"
    assert acc["bibliography"]["title"]["outcomes"]["document_incompatible"] == 1
    assert acc["bibliography"]["title"]["scored"] == 0
    assert acc["structure"][0] == {
        "book_id": "example-book", "reference_status": "verified", "scored": False, "reason": "document_incompatible"
    }
    assert acc["fields"]["page_type"]["scored"] == 4  # scans present are still scored

    changed = gold.model_copy(deep=True)
    changed.scans[1].image_sha256 = "f" * 64
    acc = run(changed, load_prediction(BOOK)[1])["accuracy"][0]
    assert acc["documents"][0]["reason"] == "image hash mismatch"
    assert acc["bibliography"]["author"]["outcomes"]["document_incompatible"] == 1
    assert acc["structure"][0]["scored"] is False

    ok = run(gold, load_prediction(BOOK)[1])["accuracy"][0]
    assert ok["documents"][0]["compatible"] and ok["documents"][0]["hashes_verified"]
    assert ok["documents"][0]["hashes_checked"] == ok["documents"][0]["hashes_total"] == 5
    assert ok["structure"][0]["scored"] and ok["structure"][0]["hashes_verified"]


def test_document_level_without_hashes_is_reported_unverified(tmp_path):
    acc = run(example_gold(), *load_prediction(metakat(tmp_path)))["accuracy"][0]
    assert acc["documents"][0]["compatible"] and not acc["documents"][0]["hashes_verified"]
    assert acc["documents"][0]["hashes_checked"] == 0
    assert acc["documents"][0]["hashes_total"] == 5
    assert acc["bibliography"]["title"]["correct"] == 1


def test_partial_gold_manifest_does_not_verify_whole_book_hashes():
    # The prediction covers the complete book, but the gold subset contains hashes for just two scans.
    data = example_gold().model_dump(mode="json")
    data["scans"] = data["scans"][:2]
    data["structure"] = {"status": "not_reviewed"}  # whole-book structure cannot be gold-verified on a subset
    partial_gold = GoldBook.model_validate(data)
    acc = run(partial_gold, load_prediction(BOOK)[1])["accuracy"][0]

    doc = acc["documents"][0]
    assert doc["compatible"] is True  # unchanged: document-level scoring is still permitted
    assert doc["hashes_checked"] == 2
    assert doc["hashes_total"] == 5
    assert doc["hashes_verified"] is False
    assert acc["bibliography"]["title"]["correct"] == 1


def test_failed_observations_count_as_no_prediction(tmp_path):
    from vllm_doc_processing.models import AnnotatedBook, ScanRecord, SourceInfo, dump_json

    ids = [f"s{i}" for i in range(10)]
    hashes = [f"{i:064x}" for i in range(10)]
    book = AnnotatedBook(
        book_id="failed",
        source=SourceInfo(scan_count=10),
        scans=[ScanRecord(scan_id=s, scan_index=i, filename=f"{s}.jpg", image_sha256=h)
               for i, (s, h) in enumerate(zip(ids, hashes))],
    )
    path = tmp_path / "failed.json"
    path.write_text(dump_json(book))
    verified = {"status": "verified", "value": "normalPage"}
    gold = GoldBook.model_validate({"book_id": "g", "scan_count": 10, "scans": [
        {"scan_id": s, "scan_index": i, "image_sha256": h, "page_type": verified, "printed_numbers": {"status": "absent"}}
        for i, (s, h) in enumerate(zip(ids, hashes))]})
    [observed] = load_prediction(path)
    acc = run(gold, observed)["accuracy"][0]
    assert acc["fields_not_provided"] == ["page_number"]
    for field in ("page_type", "printed_numbers_exact"):
        stats = acc["fields"][field]
        assert (stats["eligible"], stats["scored"], stats["coverage"]) == (10, 0, 0.0)
        assert stats["outcomes"]["no_prediction"] == 10
