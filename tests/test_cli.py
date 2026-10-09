import json
from pathlib import Path

import pytest

from vllm_doc_processing.cli import EXIT_CONFIG, main

EXAMPLE_CONFIG = Path(__file__).parents[1] / "examples" / "config.example.json"


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    (book_dir / "order.txt").write_text("a\nb\n")
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
        ({"provider": "openrouter", "model": "m"}, [], "OPENROUTER_API_KEY"),  # no key in env
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
