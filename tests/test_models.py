import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vllm_doc_processing.models import (
    AnnotatedBook,
    BiblioField,
    Bibliography,
    PageType,
    PrintedNumber,
    dump_json,
    load_json,
)

EXAMPLE = Path(__file__).parents[1] / "examples" / "annotated_book.example.json"


def sid(n: int) -> str:
    """Scan ID (page UUID) of the n-th scan in the example."""
    return f"5b1f0c2e-7d4a-4e8b-9c3f-{n:012d}"


def example() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_example_validates_and_round_trips_non_ascii():
    book = load_json(EXAMPLE.read_bytes())
    text = dump_json(book)
    assert "Cesty po Šumavě" in text  # stored as UTF-8, not \u escapes
    assert load_json(text) == book
    assert book.resolved.bibliography.title.value == "Cesty po Šumavě"


def test_spread_keeps_two_printed_numbers():
    spread = load_json(EXAMPLE.read_bytes()).scans[3].observation
    assert spread.side == "both"
    assert [(n.side, n.raw) for n in spread.printed_numbers] == [("left", "v"), ("right", "vi")]
    with pytest.raises(ValidationError):
        PrintedNumber(side="both", raw="1")


def test_bibliography_fields_match_biblio_vocabulary():
    assert set(Bibliography.model_fields) == {f.value for f in BiblioField}
    assert len(PageType) == 38 and PageType("TableOfContents") is PageType.TABLE_OF_CONTENTS


def test_unknown_scan_reference_rejected():
    data = example()
    data["resolved"]["structure"][0]["start_scan_id"] = "no-such-scan"
    with pytest.raises(ValidationError, match="unknown scan IDs"):
        AnnotatedBook.model_validate(data)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["scans"].reverse(), "contiguous scan_index"),
        (lambda d: d["scans"][1].update(scan_id=sid(1)), "duplicate scan_id"),
        (lambda d: d["source"].update(scan_count=4), "scan_count"),
        (lambda d: d["scans"][0].update(observation_call_id="nope"), "observation_call_id"),
        (lambda d: d["resolved"]["structure"][1].update(start_scan_id=sid(5), end_scan_id=sid(4)), "after end"),
        (lambda d: d["resolved"]["structure"][2].update(parent_id="node-4"), "listed before"),
        (lambda d: d["resolved"]["structure"][2].update(level=1), "level must exceed"),
        (lambda d: d["scans"][1]["observation"].update(subpages=[{"side": "left"}]), "side='both'"),
        (lambda d: d["scans"][0].update(extra_field=1), "extra"),
    ],
)
def test_invariants(mutate, message):
    data = example()
    mutate(data)
    with pytest.raises(ValidationError, match=message):
        AnnotatedBook.model_validate(data)


def test_unknowns_stay_null():
    book = AnnotatedBook.model_validate(
        {"book_id": "b", "source": {"scan_count": 1}, "scans": [{"scan_id": sid(1), "scan_index": 0, "filename": "a.jpg"}]}
    )
    assert book.scans[0].observation is None and book.resolved is None
    assert book.schema_version == "0.1"
