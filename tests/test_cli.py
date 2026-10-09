import json
from pathlib import Path

import pytest
from PIL import Image

from vllm_doc_processing.cli import EXIT_CONFIG, main

EXAMPLE_CONFIG = Path(__file__).parents[1] / "examples" / "config.example.json"


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    (book_dir / "order.txt").write_text("a\nb\n")
    for name in ("a", "b"):
        Image.new("L", (40, 60), 255).save(book_dir / f"{name}.png")
    return book_dir


def args(book_dir, *extra):
    return ["process", "--input", str(book_dir), "--output", str(book_dir.parent / "out.json"), "--dry-run", *extra]


def dry_run(capsys, argv):
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_help_works_without_credentials(book, capsys):
    for argv in (["--help"], ["process", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 0
    assert "--dry-run" in capsys.readouterr().out


def test_precedence_defaults_then_file_then_cli(book, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"provider": "openai", "model": "file-model", "postprocess_model": "file-text"}))

    out = dry_run(capsys, args(book, "--config", str(config), "--model", "cli-model"))
    assert out["config"]["model"] == "cli-model"  # CLI beats file
    assert out["config"]["postprocess_model"] == "file-text"  # file beats default
    assert out["effective_base_url"] == "https://api.openai.com/v1"  # provider default
    assert out["api_key_env"] == "OPENAI_API_KEY"
    assert "sk-test" not in json.dumps(out)


def test_example_config_with_cli_override(book, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    out = dry_run(capsys, args(book, "--config", str(EXAMPLE_CONFIG), "--base-url", "http://localhost:8000/v1"))
    assert out["config"]["provider"] == "openrouter"
    assert out["effective_base_url"] == "http://localhost:8000/v1"
    assert out["effective_postprocess_model"] == out["config"]["model"]


@pytest.mark.parametrize(
    "file_values, extra, message",
    [
        ({}, ["--model", "m"], "provider"),  # required key missing
        ({"provider": "openrouter", "model": "m", "modle": "x"}, [], "modle"),  # typo rejected
        ({"provider": "openrouter", "model": "m", "api_key": "sk-secret"}, [], "environment"),
        ({"provider": "openrouter", "model": "m", "postprocess_model": {"x": "sk-secret"}}, [], "postprocess_model"),
    ],
)
def test_invalid_configuration_fails_without_leaking(book, tmp_path, capsys, file_values, extra, message):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(file_values))
    assert main(args(book, "--config", str(config), *extra)) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert message in err
    assert "sk-secret" not in err


def test_invalid_paths(book, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    base = ["--provider", "openrouter", "--model", "m"]
    assert main(args(book.parent / "missing", *base)) == EXIT_CONFIG
    assert main(args(book, "--order-file", str(book / "nope.txt"), *base)) == EXIT_CONFIG
    inside = ["process", "--input", str(book), "--output", str(book / "out.json"), "--dry-run", *base]
    assert main(inside) == EXIT_CONFIG
    err = capsys.readouterr().err
    assert "input directory not found" in err and "order file not found" in err and "must not be written into" in err


def test_provider_override_drops_file_base_url(book, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"provider": "openrouter", "model": "m", "base_url": "https://openrouter.ai/api/v1"}))
    out = dry_run(capsys, args(book, "--config", str(config), "--provider", "openai"))
    assert out["effective_base_url"] == "https://api.openai.com/v1"
    out = dry_run(capsys, args(book, "--config", str(config), "--provider", "openai", "--base-url", "http://x/v1"))
    assert out["effective_base_url"] == "http://x/v1"


def test_missing_api_key_warns_in_dry_run_but_fails_real_run(book, capsys):
    base = ["--provider", "openrouter", "--model", "m"]
    out = dry_run(capsys, args(book, *base))
    assert out["api_key_set"] is False
    real = ["process", "--input", str(book), "--output", str(book.parent / "out.json"), *base]
    assert main(real) == EXIT_CONFIG
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_dry_run_reports_inventory_and_max_pages(book, capsys):
    Image.new("L", (10, 10)).save(book / "stray.jpg")
    assert main(args(book, "--provider", "openrouter", "--model", "m", "--max-pages", "1")) == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert (out["listed_scans"], out["selected_scans"], out["config"]["max_pages"]) == (2, 1, 1)
    assert out["unlisted_images"] == ["stray.jpg"]
    assert "stray.jpg" in captured.err
    (book / "b.png").write_bytes(b"not an image")  # beyond --max-pages: not decoded
    dry_run(capsys, args(book, "--provider", "openrouter", "--model", "m", "--max-pages", "1"))
    assert main(args(book, "--provider", "openrouter", "--model", "m")) == EXIT_CONFIG
    assert "b.png: cannot decode image" in capsys.readouterr().err
