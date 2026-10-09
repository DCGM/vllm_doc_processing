import json

import pytest
from PIL import Image
from test_llm import FakeClient, completion

from vllm_doc_processing import cli
from vllm_doc_processing.config import build_config
from vllm_doc_processing.images import build_inventory
from vllm_doc_processing.llm import LLMClient
from vllm_doc_processing.models import PageType
from vllm_doc_processing.observe import observe_scan
from vllm_doc_processing.prompts import OBSERVE_PROMPT_NUMBER, OBSERVE_SYSTEM, PAGE_TYPE_DESCRIPTIONS, PROMPT_VERSIONS, observe_user_prompt

SPREAD = {
    "page_type": "TableOfContents",
    "page_type_confidence": 0.9,
    "side": "both",
    "side_confidence": 0.95,
    "subpages": [
        {"side": "left", "page_type": "NormalPage", "confidence": 0.8},
        {"side": "right", "page_type": "TableOfContents", "confidence": 0.9},
    ],
    "printed_numbers": [
        {"side": "left", "raw": "[xii]", "normalized": "XII", "numeric_value": 12, "numeral_system": "roman",
         "confidence": 0.7, "notes": None},
        {"side": "right", "raw": "13", "normalized": "13", "numeric_value": 13, "numeral_system": "arabic",
         "confidence": 0.9, "notes": None},
    ],
    "headings": [],
    "toc_entries": [{"title": "I. Na horách", "printed_page_reference": "17", "level": 1, "confidence": 0.9}],
    "bibliographic_candidates": [],
    "notes": None,
}
BLANK = {
    "page_type": "Blank", "page_type_confidence": 0.9, "side": "left", "side_confidence": 0.6, "subpages": [],
    "printed_numbers": [], "headings": [], "toc_entries": [], "bibliographic_candidates": [], "notes": None,
}


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    (book_dir / "order.txt").write_text("c\na\nb\n")
    for name in "abc":
        Image.new("L", (40, 60), 255).save(book_dir / f"{name}.png")
    return book_dir


def make_client(fake):
    return LLMClient(build_config({"provider": "openrouter", "model": "vendor/vlm"}, {}), client=fake, sleep=lambda s: None)


def test_prompt_covers_vocabulary_and_separates_scan_position():
    assert set(PAGE_TYPE_DESCRIPTIONS) == set(PageType)
    assert all(f"- {t.value}: " in OBSERVE_SYSTEM for t in PageType)
    user = observe_user_prompt(4, 120, "Last page number: 7")
    assert "Scan position: 5 of 120" in user and "not a page number" in user and "Last page number: 7" in user
    assert "Context" not in observe_user_prompt(0, 1)
    assert PROMPT_VERSIONS["observe"].startswith(f"{OBSERVE_PROMPT_NUMBER}-")


def test_observe_scan_sends_one_image_and_validates_spread(book):
    inventory = build_inventory(book, book / "order.txt")
    fake = FakeClient(completion(json.dumps(SPREAD)))
    result = observe_scan(make_client(fake), inventory, inventory.scans[1], context="Previous: TitlePage")

    request = fake.requests[0]
    system, user = request["messages"]
    assert system["content"] == OBSERVE_SYSTEM
    images = [part for part in user["content"] if part["type"] == "image_url"]
    assert len(images) == 1 and images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Scan position: 2 of 3" in user["content"][-1]["text"]
    assert "Previous: TitlePage" in user["content"][-1]["text"]
    assert request["response_format"]["json_schema"]["name"] == "ScanObservation"

    obs = result.value
    assert obs.side == "both" and [n.numeric_value for n in obs.printed_numbers] == [12, 13]
    assert obs.toc_entries[0].printed_page_reference == "17"
    assert result.call.scan_id == "a" and result.call.stage == "observe"


def test_invariant_violation_is_recorded_and_retried(book):
    """Subpages on a single page violate the observation contract: invalid_response, then retry."""
    inventory = build_inventory(book, book / "order.txt")
    bad = {**SPREAD, "side": "left"}
    fake = FakeClient(completion(json.dumps(bad)), completion(json.dumps(BLANK)))
    result = observe_scan(make_client(fake), inventory, inventory.scans[0])
    assert [c.status for c in result.calls] == ["invalid_response", "ok"]
    assert "subpages" in result.calls[0].error
    assert result.value.page_type == PageType.BLANK and result.value.printed_numbers == []


def test_observe_command_prints_json_lines_and_reports_failures(book, monkeypatch, capsys):
    fake = FakeClient(
        completion(json.dumps(BLANK)),
        *[completion("not json")] * 4,  # 1 attempt + 3 retries, all invalid
    )
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, client=fake, sleep=lambda s: None))
    argv = ["observe", "--input", str(book), "--provider", "openrouter", "--model", "vendor/vlm", "--scans", "b", "c"]
    assert cli.main(argv) == cli.EXIT_RUNTIME

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(line["scan_id"], line["scan_index"]) for line in lines] == [("c", 0), ("b", 2)]  # order-file order
    assert lines[0]["observation"]["page_type"] == "Blank" and lines[0]["error"] is None
    assert lines[1]["observation"] is None and "failed after 4 attempt(s)" in lines[1]["error"]
    assert [c["status"] for c in lines[1]["calls"]] == ["invalid_response"] * 4
    assert all(line["prompt_version"] == PROMPT_VERSIONS["observe"] for line in lines)


def test_observe_command_rejects_unknown_scan_ids(book, capsys):
    argv = ["observe", "--input", str(book), "--provider", "openrouter", "--model", "m", "--scans", "zzz"]
    assert cli.main(argv) == cli.EXIT_CONFIG
    assert "zzz" in capsys.readouterr().err
