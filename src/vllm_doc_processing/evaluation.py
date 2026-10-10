"""Deterministic scoring of predictions against gold annotations, and agreement with comparators.

Accuracy is computed only against human-verified gold labels (``gold.py``). MetaKat/Kramerius
imports are predictions: they are scored against gold like any system, and their agreement with
this tool is reported separately and never called accuracy. See docs/EVALUATION.md.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .gold import GoldBook, GoldValue
from .predictions import (
    SCAN_FIELDS,
    Incomparable,
    Prediction,
    PredictionError,
    normalize_label,
    normalize_number,
    normalize_text,
)

EVALUATOR_VERSION = "1"

SCORED = ("correct", "wrong_value", "missing_value", "spurious_value")
"""Outcomes in the accuracy denominator. ``missing_value``: reference has a value, prediction null;
``spurious_value``: reference expects null, prediction has a value."""

UNSCORED = (
    "no_prediction", "incomparable", "ambiguous_unscorable", "scan_missing", "hash_mismatch", "document_incompatible"
)
"""Eligible reference items that could not be scored (lower coverage)."""

GOLD_FIELD = {f: f.removesuffix("_exact").removesuffix("_normalized") for f in SCAN_FIELDS}


@dataclass(frozen=True)
class Expected:
    """Reference for one field: ``acceptable`` values (None inside = null accepted); None = not scorable."""

    status: str
    acceptable: tuple[Any, ...] | None


@dataclass
class RefScan:
    book_id: str
    scan_id: str
    scan_index: int
    image_sha256: str | None
    fields: dict[str, Expected]


def expected(gold: GoldValue) -> Expected:
    value = gold.model_dump(mode="json")  # enums as plain strings
    if gold.status == "verified":
        return Expected("verified", (value["value"],))
    if gold.status in ("absent", "not_applicable"):
        return Expected(gold.status, (None,))
    if gold.status == "ambiguous":
        return Expected("ambiguous", tuple(value["alternatives"]) or None)
    return Expected("not_reviewed", None)


def _canon(field: str, value: Any) -> Any:
    if value is None or isinstance(value, Incomparable):
        return value
    if value == []:  # an empty list of numbers/values means "none", like null
        return None
    if field == "printed_numbers_exact":
        return tuple(sorted(value))
    if field == "printed_numbers_normalized":
        return tuple(sorted(normalize_number(v) for v in value))
    if field == "page_number":
        return normalize_label(value)
    if field.startswith("bibliography."):
        return frozenset(normalize_text(v) for v in value)
    return value


def outcome(field: str, exp: Expected, predicted: Any) -> str | None:
    """None if the reference is not to be scored at all (not reviewed / no reference value)."""
    if exp.acceptable is None:
        return {"ambiguous": "ambiguous_unscorable", "incomparable": "incomparable"}.get(exp.status)
    if isinstance(predicted, Incomparable):
        return "incomparable"
    if _canon(field, predicted) in {_canon(field, a) for a in exp.acceptable}:
        return "correct"
    if predicted is None:
        return "missing_value"
    return "spurious_value" if all(a is None for a in exp.acceptable) else "wrong_value"


class FieldStats:
    def __init__(self) -> None:
        self.outcomes: Counter[str] = Counter()
        self.by_status: dict[str, Counter[str]] = {}

    def add(self, status: str, result: str | None) -> None:
        self.by_status.setdefault(status, Counter())[result or "not_scored"] += 1
        if result is not None:
            self.outcomes[result] += 1

    def summary(self) -> dict[str, Any]:
        scored = sum(self.outcomes[o] for o in SCORED)
        eligible = scored + sum(self.outcomes[o] for o in UNSCORED)
        return {
            "eligible": eligible,
            "scored": scored,
            "correct": self.outcomes["correct"],
            "accuracy": round(self.outcomes["correct"] / scored, 4) if scored else None,
            "coverage": round(scored / eligible, 4) if eligible else None,
            "outcomes": {o: self.outcomes[o] for o in SCORED + UNSCORED},
            "by_reference_status": {s: dict(sorted(c.items())) for s, c in sorted(self.by_status.items())},
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Incomparable):
        return {"incomparable": value.raw}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return value


def score_scans(refs: list[RefScan], pred: dict[str, Prediction], fields: tuple[str, ...]) -> tuple[dict, list, dict]:
    """``pred`` maps book_id of the reference to its prediction. Returns field stats, errors, scan summary."""
    stats = {f: FieldStats() for f in fields}
    errors: list[dict] = []
    scans = Counter()
    for ref in refs:
        book = pred.get(ref.book_id)
        ps = book.scans.get(ref.scan_id) if book else None
        if ps is None:
            scan_result = "scan_missing"
        elif ref.image_sha256 and ps.image_sha256 and ref.image_sha256 != ps.image_sha256:
            scan_result = "hash_mismatch"
        else:
            scan_result = None
            scans["hash_checked" if ref.image_sha256 and ps.image_sha256 else "hash_unchecked"] += 1
            if ref.scan_index != ps.scan_index:
                scans["scan_index_differs"] += 1
        scans[scan_result or "compared"] += 1
        for f in fields:
            exp = ref.fields.get(f)
            if exp is None:
                continue
            if exp.acceptable is None:
                stats[f].add(exp.status, outcome(f, exp, None))
                continue
            if scan_result:
                result = scan_result
            elif f not in ps.values:
                result = "no_prediction"
            else:
                result = outcome(f, exp, ps.values[f])
            stats[f].add(exp.status, result)
            if result not in ("correct", "no_prediction", "scan_missing"):
                errors.append(
                    {
                        "book_id": ref.book_id,
                        "scan_id": ref.scan_id,
                        "scan_index": ref.scan_index,
                        "field": f,
                        "outcome": result,
                        "reference_status": exp.status,
                        "reference": [_json_value(a) for a in exp.acceptable],
                        "predicted": _json_value(ps.values.get(f)) if ps and scan_result is None else None,
                    }
                )
    summaries = {f: s.summary() for f, s in stats.items() if s.by_status}  # fields the reference has
    return summaries, errors, dict(sorted(scans.items()))


def gold_refs(gold: list[GoldBook]) -> list[RefScan]:
    refs = []
    for book in gold:
        for s in book.scans:
            fields = {f: expected(getattr(s, GOLD_FIELD[f])) for f in SCAN_FIELDS}
            refs.append(RefScan(book.book_id, s.scan_id, s.scan_index, s.image_sha256, fields))
    return refs


def comparator_refs(comparator: Prediction, fields: tuple[str, ...]) -> list[RefScan]:
    """A comparator as reference for agreement: its non-null values; null = no reference value."""
    refs = []
    for s in comparator.scans.values():
        exp = {}
        for f in fields:
            v = s.values.get(f)
            if f in s.values:
                exp[f] = (
                    Expected("incomparable", None) if isinstance(v, Incomparable)
                    else Expected("no_reference_value", None) if v is None
                    else Expected("reference_value", (v,))
                )
        refs.append(RefScan(comparator.book_id, s.scan_id, s.scan_index, s.image_sha256, exp))
    return refs


def match_books(ref_books: dict[str, set[str]], preds: list[Prediction]) -> dict[str, Prediction]:
    """Assign predictions to reference books by shared scan IDs (book IDs differ between systems)."""
    out: dict[str, Prediction] = {}
    for book_id, scan_ids in sorted(ref_books.items()):
        hits = [p for p in preds if scan_ids & p.scans.keys()]
        if len(hits) > 1:
            raise PredictionError(
                f"several predictions of {hits[0].label} cover book {book_id}: {', '.join(p.path for p in hits)}"
            )
        if hits:
            out[book_id] = hits[0]
    return out


def score_bibliography(
    refs: dict[str, dict[str, Expected]], pred: dict[str, Prediction], checks: dict[str, dict[str, Any]]
) -> tuple[dict, list]:
    """Only predictions of the whole, unchanged book (``checks[book_id]['compatible']``) are scored."""
    stats: dict[str, FieldStats] = {}
    errors = []
    for book_id, fields in sorted(refs.items()):
        book = pred.get(book_id)
        for name, exp in sorted(fields.items()):
            f = f"bibliography.{name}"
            st = stats.setdefault(name, FieldStats())
            if exp.acceptable is None:
                st.add(exp.status, outcome(f, exp, None))
                continue
            if book is None:
                result = "scan_missing"
            elif book.bibliography is None or name not in book.bibliography:
                result = "no_prediction"
            elif not checks[book_id]["compatible"]:
                result = "document_incompatible"
            else:
                value = book.bibliography[name] or None
                result = outcome(f, exp, value)
                if result != "correct":
                    errors.append(
                        {
                            "book_id": book_id,
                            "field": f,
                            "outcome": result,
                            "reference_status": exp.status,
                            "reference": [_json_value(a) for a in exp.acceptable],
                            "predicted": value,
                        }
                    )
            st.add(exp.status, result)
    return {f: s.summary() for f, s in sorted(stats.items())}, errors


def score_structure(gold: GoldBook, pred: Prediction | None, check: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chapter list of a verified whole book: title matching, start scans, levels, TOC references.

    Scored only if the prediction covers the whole book with no image hash mismatch (``check``).
    """
    status = gold.structure.status
    if status not in ("verified", "absent", "not_applicable"):
        return None
    result: dict[str, Any] = {"book_id": gold.book_id, "reference_status": status}
    if pred is None or pred.structure is None:
        return result | {"scored": False, "reason": "scan_missing" if pred is None else "no_prediction"}
    if not check["compatible"]:
        return result | {"scored": False, "reason": "document_incompatible"}
    gold_ch = gold.structure.value or []
    pred_ch = pred.structure
    if not gold_ch:
        result["no_chapters_correct"] = not pred_ch
    used: set[int] = set()
    matches = []
    for g in gold_ch:
        key = normalize_text(g.title)
        j = next((j for j, p in enumerate(pred_ch) if j not in used and p.title and normalize_text(p.title) == key), None)
        if j is not None:
            used.add(j)
        matches.append((g, pred_ch[j] if j is not None else None))
    matched = [(g, p) for g, p in matches if p]
    starts = [(g, p) for g, p in matched if g.start_scan_id]
    refs = [(g, p) for g, p in matched if g.printed_page_reference]
    wrong_refs = [
        (g, p) for g, p in refs
        if normalize_number(g.printed_page_reference) != normalize_number(p.printed_page_reference or "")
    ]
    gold_starts = {g.start_scan_id for g in gold_ch if g.start_scan_id}
    pred_starts = {p.start_scan_id for p in pred_ch if p.start_scan_id}
    return result | {
        "scored": True,
        "hashes_verified": check["hashes_verified"],
        "gold_chapters": len(gold_ch),
        "predicted_chapters": len(pred_ch),
        "title_matches": len(matched),
        "title_recall": _ratio(len(matched), len(gold_ch)),
        "title_precision": _ratio(len(matched), len(pred_ch)),
        "start_scan_correct": sum(g.start_scan_id == p.start_scan_id for g, p in starts),
        "start_scan_scored": len(starts),
        "level_correct": sum(g.level == p.level for g, p in matched),
        "toc_reference_correct": len(refs) - len(wrong_refs),
        "toc_reference_scored": len(refs),
        "start_set_recall": _ratio(len(gold_starts & pred_starts), len(gold_starts)),
        "start_set_precision": _ratio(len(gold_starts & pred_starts), len(pred_starts)),
        "unmatched_gold_titles": [g.title for g, p in matches if p is None],
        "unmatched_predicted_titles": [p.title for j, p in enumerate(pred_ch) if j not in used],
        "wrong_starts": [
            {"title": g.title, "reference": g.start_scan_id, "predicted": p.start_scan_id}
            for g, p in starts
            if g.start_scan_id != p.start_scan_id
        ],
        "wrong_toc_references": [
            {"title": g.title, "reference": g.printed_page_reference, "predicted": p.printed_page_reference}
            for g, p in wrong_refs
        ],
    }


def _provided(preds: list[Prediction], fields: tuple[str, ...]) -> tuple[str, ...]:
    """Fields that the system outputs at this layer by contract (others are reported as not provided)."""
    return tuple(f for f in fields if any(f in p.fields for p in preds))


def document_check(
    book_id: str, scan_count: int, ref_hashes: dict[str, str | None], pred: Prediction | None
) -> dict[str, Any]:
    """Whether document-level results (bibliography, structure) of ``pred`` may be scored for this book.

    Requires a prediction of the whole book (as many scans as the book has, including every reference
    scan) and no image hash mismatch. ``hashes_verified`` is false when hashes are missing on either side.
    """
    if pred is None:
        return {"book_id": book_id, "compatible": False, "reason": "no prediction"}
    missing = sorted(set(ref_hashes) - pred.scans.keys())
    pairs = [(h, pred.scans[i].image_sha256) for i, h in ref_hashes.items() if i in pred.scans]
    mismatches = sum(bool(a and b and a != b) for a, b in pairs)
    if mismatches:
        reason = "image hash mismatch"
    elif missing or len(pred.scans) < scan_count:
        reason = "partial prediction"
    elif len(pred.scans) != scan_count:
        reason = "scan count differs"
    else:
        reason = None
    return {
        "book_id": book_id,
        "compatible": reason is None,
        "reason": reason,
        "book_scans": scan_count,
        "prediction_scans": len(pred.scans),
        "missing_reference_scans": len(missing),
        "hash_mismatches": mismatches,
        "hashes_verified": bool(pairs) and all(a and b for a, b in pairs) and not missing,
    }


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def _groups(preds: list[Prediction]) -> dict[tuple[str, str], list[Prediction]]:
    groups: dict[tuple[str, str], list[Prediction]] = {}
    for p in preds:
        groups.setdefault((p.system, p.layer), []).append(p)
    return dict(sorted(groups.items()))


def _incomparable(preds: list[Prediction]) -> dict[str, dict[str, int]]:
    seen: dict[str, Counter[str]] = {}
    for p in preds:
        for s in p.scans.values():
            for f, v in s.values.items():
                if isinstance(v, Incomparable):
                    seen.setdefault(f, Counter())[v.raw] += 1
    return {f: dict(sorted(c.items())) for f, c in sorted(seen.items())}


def _usage(preds: list[Prediction]) -> dict[str, Any] | None:
    totals = [p.provenance["totals"] for p in preds if "totals" in p.provenance]
    if not totals:
        return None
    costs = [t["cost_usd"] for t in totals if t["cost_usd"] is not None]
    return {
        "books": len(totals),
        "requests": sum(t["requests"] for t in totals),
        "failed_requests": sum(t["failed_requests"] for t in totals),
        "prompt_tokens": sum(t["prompt_tokens"] for t in totals),
        "completion_tokens": sum(t["completion_tokens"] for t in totals),
        "cost_usd": round(sum(costs), 6) if costs else None,
        "cost_complete": len(costs) == len(totals) and all(t["cost_complete"] for t in totals),
        "retry_attempts": sum(p.provenance["retry_attempts"] for p in preds),
        "unobserved_scans": sum(p.provenance["unobserved_scans"] for p in preds),
    }


def evaluate(
    gold: list[tuple[str, str, GoldBook]], systems: list[Prediction], comparators: list[Prediction]
) -> dict[str, Any]:
    """Build the report. ``gold`` holds (path, sha256, book) triples."""
    _unique_gold(gold)
    books = [b for _, _, b in gold]
    refs = gold_refs(books)
    ref_books = {b.book_id: {s.scan_id for s in b.scans} for b in books}
    biblio_refs = {b.book_id: {f.value: expected(v) for f, v in b.bibliography.items()} for b in books}

    accuracy = []
    if books:
        for (system, layer), preds in _groups(systems + comparators).items():
            matched = match_books(ref_books, preds)
            provided = _provided(preds, SCAN_FIELDS)
            fields, errors, scans = score_scans(refs, matched, provided)
            checks = {
                b.book_id: document_check(
                    b.book_id, b.scan_count, {s.scan_id: s.image_sha256 for s in b.scans}, matched.get(b.book_id)
                )
                for b in books
            }
            biblio, biblio_errors, structure = {}, [], []
            if any(p.bibliography is not None for p in preds):
                biblio, biblio_errors = score_bibliography(biblio_refs, matched, checks)
            if any(p.structure is not None for p in preds):
                structure = [
                    s for b in books if (s := score_structure(b, matched.get(b.book_id), checks[b.book_id]))
                ]
            accuracy.append(
                {
                    "system": system,
                    "layer": layer,
                    "role": preds[0].role,
                    "books_matched": sorted(matched),
                    "books_unmatched": sorted(ref_books.keys() - matched.keys()),
                    "fields_not_provided": [f for f in SCAN_FIELDS if f not in provided],
                    "scans": scans,
                    "documents": [checks[b] for b in sorted(matched)] if biblio or structure else [],
                    "fields": fields,
                    "bibliography": biblio,
                    "structure": structure,
                    "errors": sorted(errors, key=lambda e: (e["book_id"], e["scan_index"], e["field"]))
                    + biblio_errors,
                }
            )

    agreement = []
    agree_fields = ("page_type", "side", "leaf", "page_number")
    for (system, layer), preds in _groups(systems).items():
        for (comp, comp_layer), comps in _groups(comparators).items():
            refs_c = [r for c in comps for r in comparator_refs(c, agree_fields)]
            matched = match_books({c.book_id: set(c.scans) for c in comps}, preds)
            fields, errors, scans = score_scans(refs_c, matched, _provided(preds, agree_fields))
            biblio_c = {
                c.book_id: {f: Expected("reference_value", (v,)) for f, v in (c.bibliography or {}).items() if v}
                for c in comps
            }
            checks = {
                c.book_id: document_check(
                    c.book_id, len(c.scans), {s.scan_id: s.image_sha256 for s in c.scans.values()}, matched.get(c.book_id)
                )
                for c in comps
            }
            biblio, biblio_errors = score_bibliography(biblio_c, matched, checks)
            agreement.append(
                {
                    "system": system,
                    "layer": layer,
                    "comparator": comp,
                    "scans": scans,
                    "fields": {f: _as_agreement(s) for f, s in fields.items()},
                    "bibliography": {f: _as_agreement(s) for f, s in biblio.items()},
                    "disagreements": sorted(errors, key=lambda e: (e["book_id"], e["scan_index"], e["field"]))
                    + biblio_errors,
                }
            )

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "inputs": {
            "gold": [
                {"path": path, "sha256": sha, "book_id": b.book_id, "gold_version": b.gold_version,
                 "dataset_version": b.dataset_version, "scans": len(b.scans), "complete_book": b.complete}
                for path, sha, b in gold
            ],
            "predictions": [
                {"path": p.path, "sha256": p.sha256, "system": p.system, "layer": p.layer, "role": p.role,
                 "book_id": p.book_id, "scans": len(p.scans)}
                for p in systems + comparators
            ],
        },
        "gold_status_counts": _status_counts(books),
        "provenance": {
            system: {"files": [p.provenance for p in preds], "usage": _usage(preds)}
            for (system, layer), preds in _groups(systems).items()
            if layer == "observed"
        },
        "incomparable_values": {
            f"{s} ({layer})": v for (s, layer), ps in _groups(systems + comparators).items() if (v := _incomparable(ps))
        },
        "accuracy": accuracy,
        "agreement": agreement,
    }


def _as_agreement(summary: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in summary.items() if k not in ("accuracy", "correct", "by_reference_status")}
    return out | {"agreeing": summary["correct"], "agreement": summary["accuracy"]}


def _unique_gold(gold: list[tuple[str, str, GoldBook]]) -> None:
    ids = [b.book_id for _, _, b in gold]
    scan_ids = [s.scan_id for _, _, b in gold for s in b.scans]
    if len(ids) != len(set(ids)) or len(scan_ids) != len(set(scan_ids)):
        raise PredictionError("gold files must describe distinct books with distinct scan IDs")


def _status_counts(books: list[GoldBook]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for b in books:
        for s in b.scans:
            for f in dict.fromkeys(GOLD_FIELD.values()):
                counts.setdefault(f, Counter())[getattr(s, f).status] += 1
        for f, v in b.bibliography.items():
            counts.setdefault(f"bibliography.{f.value}", Counter())[v.status] += 1
        counts.setdefault("structure", Counter())[b.structure.status] += 1
    return {f: dict(sorted(c.items())) for f, c in counts.items()}


# --- Markdown -------------------------------------------------------------------

MD_MAX_ERRORS = 100


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f} %"


def _field_table(fields: dict[str, dict], key: str = "accuracy", correct: str = "correct") -> list[str]:
    head = f"| field | eligible | scored | {correct} | {key} | coverage | missing | spurious | wrong | incomparable | no prediction | unscorable | scan missing / hash mismatch |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for f, s in fields.items():
        o = s["outcomes"]
        lines.append(
            f"| {f} | {s['eligible']} | {s['scored']} | {s[correct]} | {_pct(s[key])} | {_pct(s['coverage'])} | "
            f"{o['missing_value']} | {o['spurious_value']} | {o['wrong_value']} | {o['incomparable']} | "
            f"{o['no_prediction']} | {o['ambiguous_unscorable']} | {o['scan_missing']} / {o['hash_mismatch']} |"
        )
    return lines


def _errors(items: list[dict], what: str) -> list[str]:
    if not items:
        return []
    lines = [f"<details><summary>{len(items)} {what}</summary>", "", "| book | scan | index | field | outcome | reference | predicted |", "|---|---|---|---|---|---|---|"]
    for e in items[:MD_MAX_ERRORS]:
        ref = e["reference_status"] + (f": {e['reference']}" if e["reference"] else "")
        lines.append(
            f"| {e['book_id']} | {e.get('scan_id', '')} | {e.get('scan_index', '')} | {e['field']} | {e['outcome']} | {ref} | {e['predicted']} |"
        )
    if len(items) > MD_MAX_ERRORS:
        lines.append(f"\n{len(items) - MD_MAX_ERRORS} more in the JSON report.")
    return lines + ["", "</details>", ""]


def to_markdown(report: dict[str, Any]) -> str:
    out = ["# Evaluation report", "", f"Evaluator version {report['evaluator_version']}.", "", "## Inputs", ""]
    out += ["| kind | path | sha256 | book | scans |", "|---|---|---|---|---|"]
    for g in report["inputs"]["gold"]:
        complete = ", complete book" if g["complete_book"] else ""
        out.append(f"| gold v{g['gold_version']} ({g['dataset_version'] or 'no dataset version'}{complete}) | {g['path']} | {g['sha256'][:12]} | {g['book_id']} | {g['scans']} |")
    for p in report["inputs"]["predictions"]:
        out.append(f"| {p['role']}: {p['system']} ({p['layer']}) | {p['path']} | {p['sha256'][:12]} | {p['book_id']} | {p['scans']} |")
    for system, prov in report["provenance"].items():
        out += ["", f"### Run provenance: {system}", ""]
        for f in prov["files"]:
            out.append(
                f"- `{f['book_id']}`: {f['provider']} vision `{f['vision_model']}`, postprocess `{f['postprocess_model']}`, "
                f"prompts {f['prompt_versions']}, schema {f['schema_version']}, image {f['image_policy']}, "
                f"context {f['context']}, OCR {f['ocr_mode']}, duration {f['duration_s']} s, "
                f"retries {f['retry_attempts']}, error rate {f['error_rate']}, unobserved scans {f['unobserved_scans']}"
            )
        if prov["usage"]:
            out.append(f"- usage: {prov['usage']}")
    if report["gold_status_counts"]:
        statuses = ("verified", "absent", "not_applicable", "ambiguous", "not_reviewed")
        out += ["", "## Gold labels by review status", "", "| field | " + " | ".join(statuses) + " |", "|---|" + "---|" * len(statuses)]
        for f, c in report["gold_status_counts"].items():
            out.append(f"| {f} | " + " | ".join(str(c.get(s, 0)) for s in statuses) + " |")
    if report["incomparable_values"]:
        out += ["", "## Values without an equivalent (never scored as equal)", ""]
        out += [f"- {k}: {v}" for k, v in report["incomparable_values"].items()]

    out += ["", "## Accuracy against gold annotations", ""]
    if not report["accuracy"]:
        out += ["No gold annotations given.", ""]
    for a in report["accuracy"]:
        out += [
            f"### {a['system']} ({a['layer']}, {a['role']})",
            "",
            f"Scans: {a['scans']}; books not covered: {a['books_unmatched'] or 'none'}; "
            f"fields not provided: {', '.join(a['fields_not_provided']) or 'none'}",
            "",
        ]
        for d in a["documents"]:
            state = "scored" if d["compatible"] else f"not scored ({d['reason']})"
            out.append(
                f"- document level of `{d['book_id']}`: {state}; scans {d['prediction_scans']}/{d['book_scans']}, "
                f"hash mismatches {d['hash_mismatches']}, hashes verified {d['hashes_verified']}"
            )
        out += [""] if a["documents"] else []
        out += _field_table(a["fields"])
        if a["bibliography"]:
            out += ["", "Bibliography (normalized value sets):", ""] + _field_table(a["bibliography"])
        for s in a["structure"]:
            out += ["", f"Structure of `{s['book_id']}`: " + ", ".join(f"{k} {v}" for k, v in s.items() if k not in ("book_id",) and not isinstance(v, list))]
            for k in ("unmatched_gold_titles", "unmatched_predicted_titles", "wrong_starts", "wrong_toc_references"):
                if s.get(k):
                    out.append(f"- {k}: {s[k]}")
        out += [""] + _errors(a["errors"], "error cases")

    out += ["## Agreement with comparator systems (not accuracy)", ""]
    if not report["agreement"]:
        out += ["No comparator predictions given.", ""]
    for a in report["agreement"]:
        out += [f"### {a['system']} ({a['layer']}) vs {a['comparator']}", "", f"Scans: {a['scans']}", ""]
        out += _field_table(a["fields"], "agreement", "agreeing")
        if a["bibliography"]:
            out += ["", "Bibliography:", ""] + _field_table(a["bibliography"], "agreement", "agreeing")
        out += [""] + _errors(a["disagreements"], "disagreements")
    return "\n".join(out).rstrip() + "\n"
