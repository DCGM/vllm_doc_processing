"""Command-line entry point: ``vllm-doc``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import API_KEY_ENV, Config, ConfigError, api_key, build_config, read_config_file, require_api_key
from .images import Inventory, build_inventory
from .llm import LLMClient, LLMError
from .models import CallRecord, UsageTotals
from .observe import observe_scan
from .prompts import PROMPT_VERSIONS

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
        description="Annotate one book. " + _SETTINGS_HELP,
    )
    _add_input_args(p)
    p.add_argument("--output", required=True, type=Path, metavar="PATH", help="annotated book JSON to write")
    _add_config_args(p)
    p.add_argument("--postprocess-model", help="text model ID for reconciliation (default: --model)")
    p.add_argument("--max-pages", type=int, metavar="N", help="process only the first N scans of the order file")
    p.add_argument("--dry-run", action="store_true", help="validate configuration, paths and scan images, print the effective settings, make no API calls (a missing API key is only a warning)")

    o = sub.add_parser(
        "observe",
        help="annotate selected scans independently (no context) and print JSON lines; for prompt checks",
        description="Send each selected scan to the vision model without context from other scans and print "
        "one JSON line per scan (observation or error, and every API call attempt). Paid. " + _SETTINGS_HELP,
    )
    _add_input_args(o)
    _add_config_args(o)
    selection = o.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scans", nargs="+", metavar="ID", help="scan IDs (names from the order file)")
    selection.add_argument("--max-pages", type=int, metavar="N", help="the first N scans of the order file")
    return parser


_SETTINGS_HELP = (
    "Settings precedence: built-in defaults < --config file < command-line flags. "
    f"API keys are read only from the environment ({', '.join(f'{k}: {v}' for k, v in API_KEY_ENV.items())})."
)


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True, type=Path, metavar="BOOK_DIR", help="directory with the scans of one book")
    p.add_argument(
        "--order-file",
        type=Path,
        metavar="PATH",
        help="image names without extensions, one per line, in scan order (default: BOOK_DIR/order.txt)",
    )


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, metavar="PATH", help="optional JSON configuration file")
    p.add_argument("--provider", choices=sorted(API_KEY_ENV), help="API provider (required here or in --config)")
    p.add_argument("--base-url", help="override the provider's default OpenAI-compatible base URL")
    p.add_argument("--model", help="vision model ID (required here or in --config)")


def _load_config(args: argparse.Namespace, **cli_values: Any) -> Config:
    file_values = read_config_file(args.config) if args.config else {}
    cli_values = {"provider": args.provider, "base_url": args.base_url, "model": args.model, **cli_values}
    return build_config(file_values, cli_values)


def _order_file(args: argparse.Namespace) -> Path:
    if not args.input.is_dir():
        raise ConfigError(f"input directory not found: {args.input}")
    order_file: Path = args.order_file or args.input / "order.txt"
    if not order_file.is_file():
        raise ConfigError(f"order file not found: {order_file}")
    return order_file


def _check_paths(args: argparse.Namespace) -> Path:
    """Validate input/output locations; return the order file path."""
    book_dir: Path = args.input
    order_file = _order_file(args)
    output: Path = args.output
    if output.is_dir():
        raise ConfigError(f"output path is a directory: {output}")
    if output.resolve().is_relative_to(book_dir.resolve()):
        raise ConfigError(f"output must not be written into the input directory: {output}")
    return order_file


def _inventory(
    book_dir: Path, order_file: Path, max_pages: int | None, scan_ids: set[str] | None = None
) -> Inventory:
    inventory = build_inventory(book_dir, order_file, max_pages, scan_ids)
    if inventory.unlisted:
        shown = ", ".join(inventory.unlisted[:10]) + (" ..." if len(inventory.unlisted) > 10 else "")
        print(
            f"vllm-doc: warning: {len(inventory.unlisted)} image file(s) not in the order file are ignored: {shown}",
            file=sys.stderr,
        )
    return inventory


def run_process(args: argparse.Namespace) -> int:
    config = _load_config(args, postprocess_model=args.postprocess_model, max_pages=args.max_pages)
    order_file = _check_paths(args)
    if not args.dry_run:
        require_api_key(config)

    inventory = _inventory(args.input, order_file, config.max_pages)

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


def run_observe(args: argparse.Namespace) -> int:
    config = _load_config(args)
    order_file = _order_file(args)
    if args.scans:
        inventory = _inventory(args.input, order_file, None, scan_ids=set(args.scans))
    else:
        if args.max_pages < 1:
            raise ConfigError("--max-pages must be at least 1")
        inventory = _inventory(args.input, order_file, args.max_pages)
    scans = inventory.scans
    client = LLMClient(config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)

    calls: list[CallRecord] = []
    failed = 0
    for scan in scans:
        line: dict[str, Any] = {
            "scan_id": scan.scan_id,
            "scan_index": scan.scan_index,
            "filename": scan.filename,
            "model": config.model,
            "prompt_version": PROMPT_VERSIONS["observe"],
        }
        try:
            result = observe_scan(client, inventory, scan)
            line |= {"observation": result.value.model_dump(mode="json"), "error": None}
            scan_calls = result.calls
        except LLMError as exc:
            failed += 1
            line |= {"observation": None, "error": str(exc)}
            scan_calls = exc.calls
        calls += scan_calls
        line["calls"] = [c.model_dump(mode="json") for c in scan_calls]
        print(json.dumps(line, ensure_ascii=False), flush=True)

    totals = UsageTotals.from_calls(calls)
    print(f"vllm-doc: {len(scans) - failed}/{len(scans)} scan(s) observed; {totals.model_dump_json()}", file=sys.stderr)
    return EXIT_RUNTIME if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_observe(args) if args.command == "observe" else run_process(args)
    except ConfigError as exc:
        print(f"vllm-doc: error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

