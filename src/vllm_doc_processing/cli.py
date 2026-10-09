"""Command-line entry point: ``vllm-doc``."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import API_KEY_ENV, ConfigError, api_key, build_config, read_config_file, require_api_key
from .images import build_inventory

EXIT_CONFIG = 2
EXIT_RUNTIME = 1


def _tool_version() -> str:
    try:
        return version("vllm-doc-processing")
    except PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-doc", description="Annotate a digitized book with a hosted vision-language model.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_tool_version()}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser(
        "process",
        help="annotate one book (directory of scans + order file)",
        description="Annotate one book. Settings precedence: built-in defaults < --config file < command-line flags. "
        f"API keys are read only from the environment ({', '.join(f'{k}: {v}' for k, v in API_KEY_ENV.items())}).",
    )
    p.add_argument("--input", required=True, type=Path, metavar="BOOK_DIR", help="directory with the scans of one book")
    p.add_argument(
        "--order-file",
        type=Path,
        metavar="PATH",
        help="image names without extensions, one per line, in scan order (default: BOOK_DIR/order.txt)",
    )
    p.add_argument("--output", required=True, type=Path, metavar="PATH", help="annotated book JSON to write")
    p.add_argument("--config", type=Path, metavar="PATH", help="optional JSON configuration file")
    p.add_argument("--provider", choices=sorted(API_KEY_ENV), help="API provider (required here or in --config)")
    p.add_argument("--base-url", help="override the provider's default OpenAI-compatible base URL")
    p.add_argument("--model", help="vision model ID (required here or in --config)")
    p.add_argument("--postprocess-model", help="text model ID for reconciliation (default: --model)")
    p.add_argument("--max-pages", type=int, metavar="N", help="process only the first N scans of the order file")
    p.add_argument("--dry-run", action="store_true", help="validate configuration, paths and scan images, print the effective settings, make no API calls (a missing API key is only a warning)")
    return parser


def _check_paths(args: argparse.Namespace) -> Path:
    """Validate input/output locations; return the order file path."""
    book_dir: Path = args.input
    if not book_dir.is_dir():
        raise ConfigError(f"input directory not found: {book_dir}")
    order_file: Path = args.order_file or book_dir / "order.txt"
    if not order_file.is_file():
        raise ConfigError(f"order file not found: {order_file}")
    output: Path = args.output
    if output.is_dir():
        raise ConfigError(f"output path is a directory: {output}")
    if output.resolve().is_relative_to(book_dir.resolve()):
        raise ConfigError(f"output must not be written into the input directory: {output}")
    return order_file


def run_process(args: argparse.Namespace) -> int:
    file_values = read_config_file(args.config) if args.config else {}
    cli_values = {
        "provider": args.provider,
        "base_url": args.base_url,
        "model": args.model,
        "postprocess_model": args.postprocess_model,
        "max_pages": args.max_pages,
    }
    config = build_config(file_values, cli_values)
    order_file = _check_paths(args)
    if not args.dry_run:
        require_api_key(config)

    inventory = build_inventory(args.input, order_file, config.max_pages)
    if inventory.unlisted:
        shown = ", ".join(inventory.unlisted[:10]) + (" ..." if len(inventory.unlisted) > 10 else "")
        print(
            f"vllm-doc: warning: {len(inventory.unlisted)} image file(s) not in the order file are ignored: {shown}",
            file=sys.stderr,
        )

    if args.dry_run:
        key_set = api_key(config) is not None
        if not key_set:
            print(f"vllm-doc: warning: {config.api_key_env} is not set; a real run would fail", file=sys.stderr)
        summary = {
            "input": str(args.input),
            "order_file": str(order_file),
            "output": str(args.output),
            "config": config.model_dump(),
            "effective_base_url": config.effective_base_url,
            "effective_postprocess_model": config.effective_postprocess_model,
            "api_key_env": config.api_key_env,
            "api_key_set": key_set,
            "listed_scans": inventory.total_listed,
            "selected_scans": len(inventory.scans),
            "unlisted_images": inventory.unlisted,
        }
        print(json.dumps(summary, indent=2))
        return 0

    print("vllm-doc: error: processing is not implemented yet; use --dry-run", file=sys.stderr)
    return EXIT_RUNTIME


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_process(args)
    except ConfigError as exc:
        print(f"vllm-doc: error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

