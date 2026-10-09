import json
import logging

from PIL import Image
from test_llm import FakeClient, completion
from test_observe import BLANK, SPREAD

from vllm_doc_processing.config import build_config
from vllm_doc_processing.context import build_context
from vllm_doc_processing.images import build_inventory
from vllm_doc_processing.llm import LLMClient
from vllm_doc_processing.models import ScanObservation, ScanRecord, dump_json, load_json
from vllm_doc_processing.pipeline import observe_book

TITLE = {
    **BLANK,
    "page_type": "TitlePage",
    "side": "right",
    "bibliographic_candidates": [
        {"field": "title", "value": "Cesty po Šumavě", "confidence": 0.9, "notes": None},
        {"field": "author", "value": "Karel Klostermann", "confidence": 0.9, "notes": None},
    ],
}


def text_page(number, heading=None):
    return {
        **BLANK,
        "page_type": "NormalPage",
        "side": "left" if number % 2 == 0 else "right",
        "printed_numbers": [
            {"side": None, "raw": str(number), "normalized": str(number), "numeric_value": number,
             "numeral_system": "arabic", "confidence": 0.9, "notes": None},
        ],
        "headings": [{"text": heading, "level": 1, "side": None, "confidence": 0.9}] if heading else [],
    }


def scan(index, obs):
    observation = None if obs is None else ScanObservation.model_validate(obs)
    return ScanRecord(scan_id=f"s{index}", scan_index=index, filename=f"s{index}.png", observation=observation)


def test_context_summarizes_prior_observations():
    prior = [scan(0, TITLE), scan(1, SPREAD), scan(2, None), scan(3, text_page(13, "KAPITOLA I. Úvod"))]
    context = build_context(prior, recent_scans=2, max_chars=2000)
    assert '- title: "Cesty po Šumavě" (scan 1)' in context
    assert "scan 2: XII (left), 13 (right); scan 4: 13" in context  # latest printed numbers by scan position
    assert 'level 1 "KAPITOLA I. Úvod" (scan 4)' in context
    assert "Table of contents entries seen on scan 2 (1 entries)" in context
    assert "scan 3 not observed" in context and "page number 13 on scan 4 after 13 on scan 2" in context
    recent = context.split("Previous scans:\n")[1].splitlines()
    assert recent == ["- scan 3: not observed", '- scan 4: NormalPage; right; page 13; heading "KAPITOLA I. Úvod"']
    assert build_context([], recent_scans=2, max_chars=2000) is None


def test_context_is_bounded():
    many = {
        **TITLE,
        "bibliographic_candidates": [
            {"field": f, "value": f"{f} {'x' * 300} {i}", "confidence": None, "notes": None}
            for f in ("title", "author", "publisher", "series_name") for i in range(6)
        ],
        "headings": [{"text": "H" * 500, "level": None, "side": None, "confidence": None}] * 5,
    }
    prior = [scan(i, many) for i in range(200)]
    for max_chars in (200, 700, 2000):
        context = build_context(prior, recent_scans=50, max_chars=max_chars)
        assert len(context) <= max_chars
    long = build_context(prior, recent_scans=50, max_chars=100_000)
    assert long.count("\n- scan ") == 50  # recent scans capped by count
    assert max(len(line) for line in long.splitlines()) < 1000  # values and lists are capped
    assert "+3 more" in long


def test_mock_book_run_is_sequential_bounded_and_deterministic(tmp_path, caplog):
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    names = [f"uuid-{(i * 7) % 20:02d}" for i in range(20)]  # order file order != sorted order
    (book_dir / "order.txt").write_text("\n".join(names))
    for name in names:
        Image.new("L", (40, 60), 255).save(book_dir / f"{name}.png")
    answers = [completion(json.dumps(TITLE))]
    answers += [completion(json.dumps(text_page(i, "KAPITOLA II." if i == 9 else None))) for i in range(1, 5)]
    answers += [completion("not json")] * 4  # scan 6 fails after 1 attempt + 3 retries
    answers += [completion(json.dumps(text_page(i, "KAPITOLA II." if i == 9 else None))) for i in range(6, 20)]
    config = build_config({"provider": "openrouter", "model": "vendor/vlm", "context_max_chars": 600}, {})
    inventory = build_inventory(book_dir, book_dir / "order.txt")

    def run():
        fake = FakeClient(*answers)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="vllm_doc_processing.pipeline"):
            book = observe_book(LLMClient(config, client=fake, sleep=lambda s: None), inventory, book_dir / "order.txt")
        return book, fake.requests

    book, requests = run()
    assert [s.scan_id for s in book.scans] == names
    observed = [s.observation is not None and s.observation_call_id is not None for s in book.scans]
    assert observed == [i != 5 for i in range(20)]
    assert book.resolved is None and book.run.totals.requests == 23 and book.run.totals.failed_requests == 4
    assert load_json(dump_json(book)) == book  # schema-valid intermediate book

    texts = []
    for request in requests[:5] + requests[8:]:  # one request per scan, failed scan 6: last attempt
        content = request["messages"][1]["content"]
        assert sum(part["type"] == "image_url" for part in content) == 1  # never a prior image
        texts.append(content[-1]["text"])
    assert "Context" not in texts[0]
    assert '"Cesty po Šumavě" (scan 1)' in texts[1] and "- scan 1: TitlePage" in texts[1]
    assert "scan 6 not observed" in texts[6] and "- scan 6: not observed" in texts[6]
    assert '"KAPITOLA II." (scan 10)' in texts[19]
    contexts = [t.split("may contain errors):\n")[1].rsplit("\n\nAnnotate", 1)[0] for t in texts[1:]]
    assert max(map(len, contexts)) <= 600

    assert not any("base64" in r.getMessage() for r in caplog.records)
    log_lines = [r.getMessage() for r in caplog.records if r.name == "vllm_doc_processing.pipeline"]
    assert len(log_lines) == 20 and "scan 20/20" in log_lines[-1] and "status=ok" in log_lines[-1]
    assert "attempts=4 status=failed" in log_lines[5]

    book2, requests2 = run()
    assert [r["messages"] for r in requests2] == [r["messages"] for r in requests]
    assert [s.observation for s in book2.scans] == [s.observation for s in book.scans]
