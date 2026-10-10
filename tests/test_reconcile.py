import json

import pytest
from test_llm import FakeClient, completion
from test_observe import BLANK

from vllm_doc_processing.config import build_config
from vllm_doc_processing.llm import LLMClient
from vllm_doc_processing.models import AnnotatedBook, ScanObservation, ScanRecord, SourceInfo, dump_json
from vllm_doc_processing import reconcile
from vllm_doc_processing.prompts import PROMPT_VERSIONS
from vllm_doc_processing.reconcile import ReconcileError, reconcile_book


def page(number=None, system="arabic", page_type="normalPage", heading=None, level=1, toc=(), biblio=(),
         side=None, number_side=None):
    numbers = []
    if number is not None:
        label = str(number) if system == "arabic" else number
        value = number if system == "arabic" else {"III": 3, "V": 5}[number]
        numbers = [{"side": number_side, "raw": label, "normalized": label, "numeric_value": value,
                    "numeral_system": system, "confidence": 0.9, "notes": None}]
    return {
        **BLANK,
        "page_type": page_type,
        "side": side,
        "printed_numbers": numbers,
        "headings": [{"text": heading, "level": level, "side": None, "confidence": 0.9}] if heading else [],
        "toc_entries": [{"title": t, "printed_page_reference": r, "level": 1, "confidence": 0.9} for t, r in toc],
        "bibliographic_candidates": [{"field": f, "value": v, "confidence": 0.9, "notes": None} for f, v in biblio],
    }


# index: expected NDK label (scan position = index + 1)
PAGES = [
    page(page_type="frontCover", biblio=[("title", "CESTY PO ŠUMAVĚ")]),  # 0 [Ia]
    page(page_type="blank"),  # 1 [Ib]: corrected to FrontEndSheet by the LLM, so not counted
    page(page_type="titlePage", side="left", biblio=[("title", "Cesty po Šumavě"), ("author", "Karel Klostermann"),
                                                     ("publication_place", "V Praze")]),  # 2 [I]
    page(page_type="blank"),  # 3 [II]
    page("III", "roman", "tableOfContents",
         toc=[("Úvod", "1"), ("Kapitola II. Na horách", "6"), ("Kapitola III. Domů", "40")]),  # 4 III
    page(page_type="blank"),  # 5 [IV]
    page("V", "roman", "preface"),  # 6 V
    page(1, heading="KAPITOLA I. Úvod"),  # 7 1
    page(),  # 8 [2]
    page(3),  # 9 3
    page(31),  # 10 4: misread, corrected (NDK 1.1.2) and flagged
    page(5),  # 11 5
    page(page_type="illustration"),  # 12 [5a]: plate outside the count
    page(page_type="blank"),  # 13 [5b]
    page(6, heading="Kapitola II. Na horách"),  # 14 6
    page(heading="Na vrcholu", level=2),  # 15 unresolved: 6 -> 9 over one page (scan missing?)
    page(9),  # 16 9
    page(10),  # 17 10
    page(page_type="blank"),  # 18 [11]
    page(page_type="backCover"),  # 19 [12]: after the last printed number everything is counted (NDK 1.1.4 c)
]
EXPECTED = ["[Ia]", "[Ib]", "[I]", "[II]", "III", "[IV]", "V", "1", "[2]", "3", "4", "5", "[5a]", "[5b]", "6", None,
            "9", "10", "[11]", "[12]"]

ANSWER = {
    "scan_corrections": [
        {"scan": 2, "page_type": "frontEndSheet", "side": "keep", "leaf": "keep",
         "reason": "Inside of the front cover."},
        {"scan": 3, "page_type": None, "side": "right", "leaf": "keep",
         "reason": "First page of the book block is a right page."},
        {"scan": 99, "page_type": "blank", "side": "keep", "leaf": "keep", "reason": "Out of range."},
    ],
    "bibliography": [
        {"field": "title", "value": "Cesty po Šumavě", "source_scans": [1, 3], "notes": None},
        {"field": "title", "value": "Cesty", "source_scans": [3], "notes": None},
        {"field": "author", "value": "Karel Klostermann", "source_scans": [3], "notes": None},
        {"field": "publication_place", "value": "Praha", "source_scans": [3], "notes": "nominative form"},
        # Cites a scan with bibliographic candidates, but no observed publisher supports it.
        {"field": "publisher", "value": "Invented Press", "source_scans": [3], "notes": None},
    ],
    "chapters": [
        {"title": "Úvod", "level": 1, "toc_scans": [5], "printed_page_reference": "1", "heading_scan": 8,
         "notes": None},
        {"title": "Kapitola II. Na horách", "level": 1, "toc_scans": [5], "printed_page_reference": "6",
         "heading_scan": 15, "notes": None},
        {"title": "Na vrcholu", "level": 2, "toc_scans": [], "printed_page_reference": None, "heading_scan": 16,
         "notes": None},
        {"title": "Kapitola III. Domů", "level": 1, "toc_scans": [5], "printed_page_reference": "40",
         "heading_scan": None, "notes": None},
        {"title": "Doslov", "level": 1, "toc_scans": [], "printed_page_reference": None, "heading_scan": 99,
         "notes": None},
        # Invented title on a real TOC scan and reference; a title adding words to its TOC entry.
        {"title": "Závěr", "level": 1, "toc_scans": [5], "printed_page_reference": "40", "heading_scan": None,
         "notes": None},
        {"title": "Úvod do dějin", "level": 1, "toc_scans": [5], "printed_page_reference": "1", "heading_scan": None,
         "notes": None},
    ],
    "issues": [{"code": "title_variants", "message": "Cover title in capitals.", "scans": [1, 3]}],
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
    assert "- scans 5-7: roman III-V" in text and 'TOC entry level 1: "Kapitola III. Domů" -> "40"' in text
    assert "scan 9: normalPage (0.9); side unknown; no printed number; label [2]" in text  # every scan is listed
    assert [s.observation for s in out.scans] == [s.observation for s in book.scans] and book.resolved is None
    assert [c.stage for c in out.run.calls] == ["reconcile"] and out.run.postprocess_model == "text/model"
    assert out.run.prompt_versions["reconcile"] == PROMPT_VERSIONS["reconcile"]
    assert AnnotatedBook.model_validate_json(dump_json(out)) == out

    # Corrections from the LLM change the resolved view (logged), and the numbering follows them.
    scans = out.resolved.scans
    assert (scans[1].page_type.value, scans[1].page_type.origin) == ("frontEndSheet", "inferred")
    assert scans[2].side.value == "right" and scans[2].page_type.origin == "observed"
    assert [c.field_path for c in out.resolved.changes][:2] == ["resolved.scans[1].page_type", "resolved.scans[2].side"]
    assert warnings(out, "invalid_scan_reference")

    # NDK page labels: roman front matter and arabic body coexist, binding parts and plates are lettered,
    # a misread number inside an intact sequence is corrected and flagged, a gap stays unresolved.
    assert [s.page_number for s in scans] == EXPECTED
    assert [(lab.origin, lab.notes) for lab in scans[10].page_labels] == [("inferred", "printed '31'")]
    (conflict,) = warnings(out, "page_number_conflict")
    assert conflict.scan_ids == ["s9", "s10", "s11"] and "imply 4" in conflict.message
    assert warnings(out, "page_number_gap")[0].scan_ids == ["s14", "s16"]

    # Bibliography: grounded in observations, changes logged, invented and competing values not kept.
    bib = out.resolved.bibliography
    assert (bib.title.value, bib.title.origin, bib.title.source_scan_ids) == ("Cesty po Šumavě", "observed",
                                                                              ["s2", "s0"])
    assert bib.author[0].origin == "observed" and bib.publisher == []
    assert (bib.publication_place[0].value, bib.publication_place[0].origin) == ("Praha", "inferred")
    (change,) = [c for c in out.resolved.changes if c.field_path.startswith("resolved.bibliography")]
    assert change.field_path == "resolved.bibliography.publication_place[0]" and change.old_value == ["V Praze"]
    assert warnings(out, "ungrounded_value") and warnings(out, "competing_values")
    assert warnings(out, "competing_observed_values") == []  # cover caps = same title

    # Structure: TOC + heading merged, hierarchy and bounds derived, unresolved reference stays unresolved.
    nodes = {n.title.value: n for n in out.resolved.structure}
    assert list(nodes) == ["Úvod", "Kapitola II. Na horách", "Na vrcholu", "Kapitola III. Domů"]
    intro, ch2, sub, ch3 = nodes.values()
    assert (intro.start_scan_id, intro.end_scan_id, intro.toc_scan_ids) == ("s7", "s13", ["s4"])
    assert (ch2.start_scan_id, ch2.end_scan_id) == ("s14", None)
    assert (sub.parent_id, sub.start_scan_id, sub.title.origin) == (ch2.id, "s15", "observed")
    assert (ch3.start_scan_id, ch3.printed_page_reference) == (None, "40")
    assert warnings(out, "unresolved_toc_reference")[0].field_path == "resolved.structure[3]"
    assert warnings(out, "ungrounded_chapter") and warnings(out, "toc_heading_mismatch") == []
    assert warnings(out, "unmatched_toc_entry")[0].field_path == "resolved.structure[4]"
    assert "Úvod do dějin" in warnings(out, "ungrounded_title")[0].message
    (issue,) = [w for w in out.resolved.warnings if w.detected_by == "llm"]
    assert issue.scan_ids == ["s0", "s2"]


def test_toc_reference_resolved_through_computed_label_and_mismatch_flagged():
    pages = list(PAGES)
    pages[4] = page("III", "roman", "tableOfContents", toc=[("Úvod", "1"), ("Kapitola II. Na horách", "6"),
                                                             ("Část", "2")])
    pages[15] = page(heading="II. Na horách")
    answer = {
        "scan_corrections": [],
        "bibliography": [],
        "chapters": [
            {"title": "Úvod", "level": 1, "toc_scans": [5], "printed_page_reference": "1", "heading_scan": 8,
             "notes": None},
            # TOC says 6 (scan 15) but the LLM matched the (abbreviated) heading on scan 16: kept, flagged.
            {"title": "Kapitola II. Na horách", "level": 1, "toc_scans": [5], "printed_page_reference": "6",
             "heading_scan": 16, "notes": None},
            {"title": "Část", "level": 1, "toc_scans": [5], "printed_page_reference": "2", "heading_scan": None,
             "notes": None},
        ],
        "issues": [],
    }
    out = reconcile_book(make_client(FakeClient(completion(json.dumps(answer)))), make_book(pages))
    nodes = out.resolved.structure
    assert nodes[1].start_scan_id == "s15" and warnings(out, "toc_heading_mismatch")[0].scan_ids == ["s15", "s14"]
    assert nodes[2].start_scan_id == "s8"  # page [2] is computed, not printed
    assert warnings(out, "contradictory_order")  # "Část" (scan 9) listed after a chapter starting on scan 16


def ndk_labels(pages):
    book = make_book(pages)
    pagination = reconcile._paginate(book.scans, reconcile._observed_scans(book.scans))
    return [pagination.page_number(s.scan_index) for s in book.scans]


def test_ndk_spread_and_side_parity_decide_which_unnumbered_pages_are_counted():
    spread = {**page(5, number_side="right"), "side": "both"}
    pages = [
        page(3, side="right"),
        spread,  # left page unnumbered -> "[4],5"
        page(page_type="illustration", side="right"),  # plate recto: 6 would be on the wrong side -> lettered
        page(page_type="blank", side="left"),  # [6]
        page(side="right"),  # [7]
        page(8, side="left"),
    ]
    assert ndk_labels(pages) == ["3", "[4],5", "[5a]", "[6]", "[7]", "8"]


def test_observed_plate_leaf_is_lettered_even_where_parity_would_count_it():
    # Between 16 (left) and 18 (left) one of three pages is 17. Parity would count the plate recto; its observed
    # leaf kind says it is an inserted plate, so the following right page is [17] (NDK: 16, [16a], [16b], [17], 18).
    plate = {**page(page_type="illustration", side="right"), "leaf": "plate", "leaf_reason": "glossy paper, tipped in"}
    pages = [page(16, side="left"), plate, {**page(page_type="blank", side="left"), "leaf": "plate"},
             page(side="right"), page(18, side="left")]
    assert ndk_labels(pages) == ["16", "[16a]", "[16b]", "[17]", "18"]
    pages[3] = page(side="left")  # parity alone (no leaf evidence) would count the plate recto
    pages[1], pages[2] = page(page_type="illustration", side="right"), page(page_type="blank", side="left")
    assert ndk_labels(pages) == ["16", "[17]", "[17a]", "[17b]", "18"]


def test_leaf_follows_page_type_corrections_and_spread_pages():
    def spread(left, right, subpages, leaf=None):
        numbers = [n for n in (page(left, number_side="left"), page(right, number_side="right"))
                   for n in n["printed_numbers"]]
        return {**page(), "side": "both", "leaf": leaf, "printed_numbers": numbers, "subpages": subpages}

    pages = [
        {**page(page_type="frontCover"), "leaf": "binding"},
        {**page(page_type="frontEndSheet"), "leaf": "binding"},  # really the title page
        {**page(page_type="blank"), "leaf": "plate"},  # really a blank verso in the book block
        page(3),
        spread(4, 5, []),
        # The scan-level "plate" fits only the right page: the left text page stays in the count.
        spread(None, None, [{"side": "left", "page_type": "normalPage", "confidence": 0.9},
                            {"side": "right", "page_type": "illustration", "confidence": 0.9}], leaf="plate"),
        spread(None, 7, [{"side": "left", "page_type": "blank", "confidence": 0.9, "leaf": "plate"}]),
    ]
    answer = {
        "scan_corrections": [
            {"scan": 2, "page_type": "titlePage", "side": "keep", "leaf": "keep", "reason": "Full title."},
            {"scan": 3, "page_type": None, "side": "keep", "leaf": "book_block", "reason": "Verso of the title."},
        ],
        "bibliography": [], "chapters": [], "issues": [],
    }
    out = reconcile_book(make_client(FakeClient(completion(json.dumps(answer)))), make_book(pages))
    scans = out.resolved.scans
    assert [s.page_number for s in scans] == ["[1a]", "[1]", "[2]", "3", "4,5", "[6],[6a]", "[6b],7"]
    assert (scans[2].leaf.value, scans[2].leaf.origin) == ("book_block", "inferred")
    assert scans[1].leaf is None and scans[5].subpages[0].leaf is None and scans[5].subpages[1].leaf.value == "plate"
    assert [w.field_path for w in warnings(out, "leaf_type_conflict")] == ["resolved.scans[1].leaf",
                                                                          "resolved.scans[5].subpages[0].leaf"]
    assert {c.field_path for c in out.resolved.changes} >= {"resolved.scans[1].leaf", "resolved.scans[2].leaf"}


def test_reproduces_kramerius_labels_of_a_1902_monograph():
    # Structure of a local Kramerius record (Keltové a Němci či Slované?, 1902, 74 scans), with idealized
    # observations: front binding lettered, unprinted front pages counted, back matter counted to the end.
    pages = [page(page_type="frontCover", side=None), page(page_type="frontEndSheet"), page(page_type="titlePage"),
             *[page() for _ in range(4)], *[page(n) for n in range(6, 70)], page(page_type="blank"),
             page(page_type="backEndSheet"), page(page_type="backCover")]
    expected = ["[1a]", "[1b]", "[1]", "[2]", "[3]", "[4]", "[5]", *map(str, range(6, 70)), "[70]", "[71]", "[72]"]
    assert ndk_labels(pages) == expected


def test_ndk_unnumbered_volume_counts_from_one_and_letters_binding():
    pages = [page(page_type="frontCover"), page(page_type="titlePage"), page(page_type="blank"), page(),
             page(page_type="backCover")]
    assert ndk_labels(pages) == ["[1a]", "[1]", "[2]", "[3]", "[4]"]


def test_too_long_input_fails_before_any_request():
    fake = FakeClient()
    pages = [page(heading="Nadpis " * 20) for _ in range(20)]
    with pytest.raises(ReconcileError, match="reconcile_max_chars=1000"):
        reconcile_book(make_client(fake, reconcile_max_chars=1000), make_book(pages))
    assert fake.requests == []
