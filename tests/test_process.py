import json

import openai
import pytest
from PIL import Image
from test_context import TITLE, text_page
from test_llm import FakeClient, completion, http_error

from vllm_doc_processing import llm
from vllm_doc_processing.checkpoint import Checkpoint
from vllm_doc_processing.cli import EXIT_CONFIG, EXIT_INTERRUPTED, EXIT_RUNTIME, main
from vllm_doc_processing.models import load_json

NAMES = [f"uuid-{(i * 3) % 10}" for i in range(10)]  # order file order != sorted order
OBSERVATIONS = [TITLE] + [text_page(i, "KAPITOLA I." if i == 1 else None) for i in range(1, 10)]
RECONCILED = {"scan_corrections": [], "bibliography": [], "chapters": [], "issues": []}


@pytest.fixture
def run(tmp_path, monkeypatch):
    """``run(*outcomes, *flags)`` executes ``vllm-doc process`` with a fake API; returns (exit code, requests)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    (book_dir / "order.txt").write_text("\n".join(NAMES))
    for i, name in enumerate(NAMES):
        Image.new("L", (40, 60), 200 + i).save(book_dir / f"{name}.png")

    def run_cli(outcomes, *flags):
        fake = FakeClient(*outcomes)
        monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: fake)
        argv = ["process", "--input", str(book_dir), "--output", str(tmp_path / "out" / "book.json"),
                "--provider", "openrouter", "--model", "vendor/vlm", *flags]
        return main(argv), fake.requests

    return run_cli


def answers(start=0, stop=10):
    return [completion(json.dumps(o)) for o in OBSERVATIONS[start:stop]]


def output(tmp_path):
    return load_json((tmp_path / "out" / "book.json").read_text())


def checkpoint(tmp_path):
    return Checkpoint.model_validate_json((tmp_path / "out" / "book.checkpoint.json").read_text())


def stages(requests):
    return ["observe" if r["messages"][1]["content"][0]["type"] == "image_url" else "reconcile" for r in requests]


def test_mock_cli_run_writes_valid_book_and_checkpoint(run, tmp_path):
    code, requests = run(answers() + [completion(json.dumps(RECONCILED))])
    assert code == 0 and stages(requests) == ["observe"] * 10 + ["reconcile"]
    book = output(tmp_path)
    assert [s.scan_id for s in book.scans] == NAMES and all(s.observation for s in book.scans)
    assert book.resolved is not None and [r.page_number for r in book.resolved.scans][1:4] == ["1", "2", "3"]
    assert book.run.finished_at and book.run.tool_version and book.run.totals.requests == 11 and not book.run.warnings
    saved = checkpoint(tmp_path).book  # observation stage only, with every call
    assert saved.resolved is None and saved.run.finished_at is None and len(saved.run.calls) == 11
    assert saved.run.postprocess_model is None and "reconcile" not in saved.run.prompt_versions

    # Without --resume or --fresh an existing checkpoint is never silently reused or overwritten.
    assert run([])[0] == EXIT_CONFIG
    assert run([], "--dry-run") == (0, [])


def test_interrupted_run_resumes_without_repeating_completed_scans(run, tmp_path):
    code, first = run(answers(0, 6) + [KeyboardInterrupt()])
    assert code == EXIT_INTERRUPTED and not (tmp_path / "out" / "book.json").exists()
    saved = checkpoint(tmp_path).book
    assert [s.observation is not None for s in saved.scans] == [True] * 6 + [False] * 4

    code, second = run(answers(6) + [completion(json.dumps(RECONCILED))], "--resume")
    assert code == 0 and stages(second) == ["observe"] * 4 + ["reconcile"]
    assert "- scan 6: normalPage" in second[0]["messages"][1]["content"][-1]["text"]  # context from the checkpoint
    book = output(tmp_path)
    assert all(s.observation for s in book.scans) and book.resolved is not None
    assert book.run.totals.requests == 11 and book.run.started_at == saved.run.started_at


def test_failed_scans_and_reconciliation_are_retried_on_resume(run, tmp_path):
    bad = http_error(openai.BadRequestError, 400, "image rejected")  # not retryable
    code, _ = run(answers(0, 3) + [bad] + answers(4) + [http_error(openai.AuthenticationError, 401, "no")])
    assert code == EXIT_RUNTIME and not (tmp_path / "out" / "book.json").exists()
    assert len(checkpoint(tmp_path).book.run.calls) == 11  # failed reconcile attempt recorded

    code, requests = run(answers(3, 4), "--resume", "--skip-postprocess")
    assert code == 0 and stages(requests) == ["observe"]
    book = output(tmp_path)
    assert all(s.observation for s in book.scans) and book.resolved is None
    assert book.run.warnings == ["reconciliation skipped (--skip-postprocess); resolved is null"]

    code, requests = run([completion(json.dumps(RECONCILED))], "--resume")
    assert code == 0 and stages(requests) == ["reconcile"]
    assert output(tmp_path).resolved is not None and output(tmp_path).run.totals.requests == 13


def test_changed_inputs_or_settings_invalidate_resume(run, tmp_path):
    assert run(answers(0, 4), "--max-pages", "4", "--skip-postprocess")[0] == 0
    book_dir = tmp_path / "book"

    for flags in (["--model", "other/vlm"], ["--max-pages", "3"]):
        code, requests = run([], "--resume", *flags)
        assert code == EXIT_CONFIG and requests == []

    Image.new("L", (40, 60), 0).save(book_dir / f"{NAMES[2]}.png")
    assert run([], "--resume", "--dry-run") == (EXIT_CONFIG, [])
    Image.new("L", (40, 60), 202).save(book_dir / f"{NAMES[2]}.png")  # restore

    # Reconciliation settings and a larger --max-pages do not invalidate the observations.
    code, requests = run(answers(4) + [completion(json.dumps(RECONCILED))], "--resume", "--postprocess-model", "text/llm")
    assert code == 0 and stages(requests) == ["observe"] * 6 + ["reconcile"]
    assert requests[-1]["model"] == "text/llm" and len(output(tmp_path).scans) == 10

    code, requests = run(answers() + [completion(json.dumps(RECONCILED))], "--fresh")
    assert code == 0 and len(requests) == 11 and output(tmp_path).run.totals.requests == 11
