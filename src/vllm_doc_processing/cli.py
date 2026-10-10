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
from .checkpoint import load_checkpoint, run_identity, write_atomic
from .images import Inventory, build_inventory, read_order_file
from .llm import LLMClient, LLMError
from .models import CallRecord, UsageTotals, dump_json
from .observe import observe_scan
from .pipeline import process_book
from .prompts import PROMPT_VERSIONS
from .reconcile import ReconcileError

EXIT_CONFIG = 2
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


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
    p.add_argument(
        "--checkpoint",
        type=Path,
        metavar="PATH",
        help="observation checkpoint, saved after every scan (default: OUTPUT with suffix .checkpoint.json)",
    )
    restart = p.add_mutually_exclusive_group()
    restart.add_argument(
        "--resume",
        action="store_true",
        help="continue from the checkpoint: observe only scans without an observation, then reconcile again; "
        "fails if images, order file or observation settings changed",
    )
    restart.add_argument("--fresh", action="store_true", help="discard an existing checkpoint and start over")
    p.add_argument(
        "--skip-postprocess", action="store_true", help="skip reconciliation; write observations only (resolved: null)"
    )
    p.add_argument("--dry-run", action="store_true", help="validate configuration, paths, scan images and (with --resume) the checkpoint, print the effective settings, make no API calls (a missing API key is only a warning)")

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


def _check_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Validate input/output/checkpoint locations; return the order file and checkpoint paths."""
    book_dir: Path = args.input
    order_file = _order_file(args)
    output: Path = args.output
    checkpoint: Path = args.checkpoint or output.with_name(output.stem + ".checkpoint.json")
    for what, path in (("output", output), ("checkpoint", checkpoint)):
        if path.is_dir():
            raise ConfigError(f"{what} path is a directory: {path}")
        if path.resolve().is_relative_to(book_dir.resolve()):
            raise ConfigError(f"{what} must not be written into the input directory: {path}")
    if checkpoint.resolve() == output.resolve():
        raise ConfigError("checkpoint and output must be different files")
    return order_file, checkpoint


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
    order_file, checkpoint_path = _check_paths(args)
    if not args.dry_run:
        require_api_key(config)

    inventory = _inventory(args.input, order_file, config.max_pages)
    identity = run_identity(config, inventory, read_order_file(order_file))
    resume_from = load_checkpoint(checkpoint_path, identity, inventory) if args.resume else None
    if checkpoint_path.exists() and not (args.resume or args.fresh):
        raise ConfigError(f"checkpoint {checkpoint_path} exists; pass --resume to continue it or --fresh to start over")

    if args.dry_run:
        key_set = api_key(config) is not None
        if not key_set:
            print(f"vllm-doc: warning: {config.api_key_env} is not set; a real run would fail", file=sys.stderr)
        summary = {
            "input": str(args.input),
            "order_file": str(order_file),
            "output": str(args.output),
            "checkpoint": str(checkpoint_path),
            "config": config.model_dump(),
            "effective_base_url": config.effective_base_url,
            "effective_postprocess_model": config.effective_postprocess_model,
            "api_key_env": config.api_key_env,
            "api_key_set": key_set,
            "listed_scans": inventory.total_listed,
            "selected_scans": len(inventory.scans),
            "unlisted_images": inventory.unlisted,
            "checkpoint_exists": checkpoint_path.exists(),
        }
        if resume_from:
            summary["checkpoint_observed_scans"] = sum(s.observation is not None for s in resume_from.book.scans)
            summary["checkpoint_totals"] = resume_from.book.run.totals.model_dump()
        print(json.dumps(summary, indent=2))
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    client = LLMClient(config)
    try:
        book = process_book(
            client,
            inventory,
            order_file,
            checkpoint_path,
            identity,
            resume_from=resume_from,
            skip_postprocess=args.skip_postprocess,
        )
    except (LLMError, ReconcileError) as exc:
        print(f"vllm-doc: error: {exc}", file=sys.stderr)
        print(f"vllm-doc: observations are kept in {checkpoint_path}; no output written; continue with --resume", file=sys.stderr)
        return EXIT_RUNTIME
    except KeyboardInterrupt:
        print(f"\nvllm-doc: interrupted; completed scans are kept in {checkpoint_path}; continue with --resume", file=sys.stderr)
        return EXIT_INTERRUPTED
    book.run.tool_version = _tool_version()
    write_atomic(args.output, dump_json(book))

    observed = sum(s.observation is not None for s in book.scans)
    print(
        f"vllm-doc: {observed}/{len(book.scans)} scan(s) observed, "
        f"{'reconciled' if book.resolved else 'not reconciled'}; wrote {args.output}; {book.run.totals.model_dump_json()}",
        file=sys.stderr,
    )
    for warning in book.run.warnings:
        print(f"vllm-doc: warning: {warning}", file=sys.stderr)
    if observed < len(book.scans):
        print("vllm-doc: retry the failed scans with --resume", file=sys.stderr)
        return EXIT_RUNTIME
    return 0


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

