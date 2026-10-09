import json

import pytest
from test_llm import FakeClient, completion
from test_observe import BLANK

from vllm_doc_processing.config import build_config
from vllm_doc_processing.llm import LLMClient
from vllm_doc_processing.models import AnnotatedBook, ScanObservation, ScanRecord, SourceInfo, dump_json
from vllm_doc_processing.reconcile import ReconcileError, reconcile_book


def page(number=None, system="arabic", page_type="NormalPage", heading=None, level=1, toc=(), biblio=()):
    numbers = []
    if number is not None:
        label = str(number) if system == "arabic" else number
        value = number if system == "arabic" else {"IV": 4, "VI": 6}[number]
        numbers = [{"side": None, "raw": label, "normalized": label, "numeric_value": value,
                    "numeral_system": system, "confidence": 0.9, "notes": None}]
    return {
        **BLANK,
        "page_type": page_type,
        "side": None,
        "printed_numbers": numbers,
        "headings": [{"text": heading, "level": level, "side": None, "confidence": 0.9}] if heading else [],
        "toc_entries": [{"title": t, "printed_page_reference": r, "level": 1, "confidence": 0.9} for t, r in toc],
        "bibliographic_candidates": [{"field": f, "value": v, "confidence": 0.9, "notes": None} for f, v in biblio],
    }


# index: printed number (scan position = index + 1)
PAGES = [
    page(page_type="FrontCover", biblio=[("title", "CESTY PO ŠUMAVĚ")]),  # 0
    page(page_type="TitlePage", biblio=[("title", "Cesty po Šumavě"), ("author", "Karel Klostermann"),
                                        ("publication_place", "V Praze")]),  # 1
    page(page_type="Blank"),  # 2
    page("IV", "roman", "TableOfContents",
         toc=[("Úvod", "1"), ("Kapitola II. Na horách", "6"), ("Kapitola III. Domů", "40")]),  # 3
    page(page_type="Blank"),  # 4: inferred V
    page("VI", "roman", "Preface"),  # 5
    page(1, heading="KAPITOLA I. Úvod"),  # 6
    page(),  # 7: inferred 2
    page(3),  # 8
    page(31),  # 9: misread 4 -> conflict, kept
    page(5),  # 10
    page(6, heading="Kapitola II. Na horách"),  # 11
    page(heading="Na vrcholu", level=2),  # 12: 7 or 8? 6 -> 9 over one page = gap, not inferred
    page(9),  # 13
    page(10),  # 14
]

ANSWER = {
    "bibliography": [
        {"field": "title", "value": "Cesty po Šumavě", "source_scans": [1, 2], "notes": None},
        {"field": "title", "value": "Cesty", "source_scans": [2], "notes": None},
        {"field": "author", "value": "Karel Klostermann", "source_scans": [2], "notes": None},
        {"field": "publication_place", "value": "Praha", "source_scans": [2], "notes": "nominative form"},
        {"field": "publisher", "value": "Invented Press", "source_scans": [8], "notes": None},
    ],
    "chapters": [
        {"title": "Úvod", "level": 1, "toc_scans": [4], "printed_page_reference": "1", "heading_scan": 7,
         "notes": None},
        {"title": "Kapitola II. Na horách", "level": 1, "toc_scans": [4], "printed_page_reference": "6",
         "heading_scan": 12, "notes": None},
        {"title": "Na vrcholu", "level": 2, "toc_scans": [], "printed_page_reference": None, "heading_scan": 13,
         "notes": None},
        {"title": "Kapitola III. Domů", "level": 1, "toc_scans": [4], "printed_page_reference": "40",
         "heading_scan": None, "notes": None},
        {"title": "Doslov", "level": 1, "toc_scans": [], "printed_page_reference": None, "heading_scan": 99,
         "notes": None},
    ],
    "issues": [{"code": "title_variants", "message": "Cover title in capitals.", "scans": [1, 2]}],
}


def make_book(pages=PAGES):
    scans = [
        ScanRecord(scan_id=f"s{i}", scan_index=i, filename=f"s{i}.jpg", observation=ScanObservation.model_validate(p))
        for i, p in enumerate(pages)
    ]
    return AnnotatedBook(book_id="book", source=SourceInfo(scan_count=len(scans)), scans=scans)


def make_client(fake, **settings):
    config = build_config({"provider": "openrouter", "model": "vision/model", "postprocess_model": "text/model",
                           **settings}, {})
    return LLMClient(config, client=fake, sleep=lambda s: None)


def labels(book, i):
    return [(lab.label, lab.origin) for lab in book.resolved.scans[i].page_labels]


def warnings(book, code):
    return [w for w in book.resolved.warnings if w.code == code]


def test_reconcile_checks_llm_answer_and_flags_conflicts_without_editing_observations():
    book = make_book()
    fake = FakeClient(completion(json.dumps(ANSWER)))
    out = reconcile_book(make_client(fake), book)

    # One text-only request with its own output cap; observations untouched, input book unchanged.
    (request,) = fake.requests
    assert request["model"] == "text/model" and request["extra_body"]["max_tokens"] == 16000
    assert [part["type"] for part in request["messages"][1]["content"]] == ["text"]
    text = request["messages"][1]["content"][0]["text"]
    assert "- scans 4-6: roman IV-VI" in text and 'TOC entry level 1: "Kapitola III. Domů" -> "40"' in text
    assert "scan 8:" not in text  # ordinary page without headings: not listed
    assert [s.observation for s in out.scans] == [s.observation for s in book.scans] and book.resolved is None
    assert [c.stage for c in out.run.calls] == ["reconcile"] and out.run.postprocess_model == "text/model"
    assert out.run.prompt_versions["reconcile"].startswith("1-")
    assert AnnotatedBook.model_validate_json(dump_json(out)) == out

    # Pagination: roman front matter and arabic body coexist; inference only between agreeing numbers.
    assert labels(out, 4) == [("V", "inferred")] and labels(out, 7) == [("2", "inferred")]
    assert labels(out, 9) == [("31", "observed")]  # contradictory number flagged, not modified
    (conflict,) = warnings(out, "page_number_conflict")
    assert conflict.scan_ids == ["s8", "s9", "s10"] and "imply 4" in conflict.message
    assert labels(out, 12) == [] and warnings(out, "page_number_gap")[0].scan_ids == ["s11", "s13"]
    assert labels(out, 0) == labels(out, 2) == []  # nothing extrapolated before the first number
    assert out.resolved.scans[3].page_type.value == "TableOfContents"

    # Bibliography: grounded in observations, changes logged, invented and competing values not kept.
    bib = out.resolved.bibliography
    assert (bib.title.value, bib.title.origin, bib.title.source_scan_ids) == ("Cesty po Šumavě", "observed",
                                                                              ["s1", "s0"])
    assert bib.author[0].origin == "observed" and bib.publisher == []
    assert (bib.publication_place[0].value, bib.publication_place[0].origin) == ("Praha", "inferred")
    (change,) = out.resolved.changes
    assert change.field_path == "resolved.bibliography.publication_place[0]" and change.old_value == ["V Praze"]
    assert warnings(out, "ungrounded_value") and warnings(out, "competing_values")
    assert warnings(out, "competing_observed_values") == []  # cover caps = same title

    # Structure: TOC + heading merged, hierarchy and bounds derived, unresolved reference stays unresolved.
    nodes = {n.title.value: n for n in out.resolved.structure}
    assert list(nodes) == ["Úvod", "Kapitola II. Na horách", "Na vrcholu", "Kapitola III. Domů"]
    intro, ch2, sub, ch3 = nodes.values()
    assert (intro.start_scan_id, intro.end_scan_id, intro.toc_scan_ids) == ("s6", "s10", ["s3"])
    assert (ch2.start_scan_id, ch2.end_scan_id) == ("s11", None)
    assert (sub.parent_id, sub.start_scan_id, sub.title.origin) == (ch2.id, "s12", "observed")
    assert (ch3.start_scan_id, ch3.printed_page_reference) == (None, "40")
    assert warnings(out, "unresolved_toc_reference")[0].field_path == "resolved.structure[3]"
    assert warnings(out, "invalid_scan_reference") and warnings(out, "ungrounded_chapter")
    assert warnings(out, "toc_heading_mismatch") == []
    (issue,) = [w for w in out.resolved.warnings if w.detected_by == "llm"]
    assert issue.scan_ids == ["s0", "s1"]


def test_toc_reference_resolved_through_inferred_label_and_mismatch_flagged():
    answer = {
        "bibliography": [],
        "chapters": [
            {"title": "Úvod", "level": 1, "toc_scans": [4], "printed_page_reference": "1", "heading_scan": 7,
             "notes": None},
            # TOC says 6 (scan 12) but the LLM matched the heading on scan 13: kept, flagged.
            {"title": "Kapitola II. Na horách", "level": 1, "toc_scans": [4], "printed_page_reference": "6",
             "heading_scan": 13, "notes": None},
        ],
        "issues": [],
    }
    pages = list(PAGES)
    pages[3] = page("IV", "roman", "TableOfContents", toc=[("Úvod", "1"), ("Kapitola II. Na horách", "6"),
                                                            ("Část", "2")])
    answer["chapters"].append({"title": "Část", "level": 1, "toc_scans": [4], "printed_page_reference": "2",
                               "heading_scan": None, "notes": None})
    out = reconcile_book(make_client(FakeClient(completion(json.dumps(answer)))), make_book(pages))
    nodes = out.resolved.structure
    assert nodes[1].start_scan_id == "s12" and warnings(out, "toc_heading_mismatch")[0].scan_ids == ["s12", "s11"]
    assert nodes[2].start_scan_id == "s7"  # page 2 exists only as an inferred label
    assert warnings(out, "contradictory_order")  # "Část" (scan 8) listed after a chapter starting on scan 13


def test_too_long_input_fails_before_any_request():
    fake = FakeClient()
    pages = [page(heading="Nadpis " * 20) for _ in range(20)]
    with pytest.raises(ReconcileError, match="reconcile_max_chars=1000"):
        reconcile_book(make_client(fake, reconcile_max_chars=1000), make_book(pages))
    assert fake.requests == []
