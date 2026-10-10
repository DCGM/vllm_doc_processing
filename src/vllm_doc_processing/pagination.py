"""Deterministic page labels in Czech NDK notation (Pravidla pro popis monografií 2.4, section 1.1).

Every page of every scan gets a label derived from the printed numbers observed on the scans:

* printed number, in sequence: as printed, canonical (``12``, ``XII``; ``IIII`` -> ``IV``);
* printed number that disagrees with both agreeing neighbours: replaced by the correct number
  (rule 1.1.2), flagged ``page_number_conflict``;
* counted page without a printed number: computed number in brackets (``[1]``, ``[13]``, ``[259]``);
* page outside the count (binding parts, inserted plates, loose leaves): the preceding number with a
  letter, in brackets (``[1a]`` before page 1, ``[26a]``, ``[26b]`` after page 26) (rule 1.1.4);
* a spread is labelled ``5,6`` / ``[4],5`` (rule 1.1.6) at scan level (``ResolvedScan.page_number``).

Which unnumbered pages are counted is decided from the arithmetic between printed numbers, the page
types and resolved leaf kinds (``UNCOUNTED_TYPES``, ``UNCOUNTED_LEAVES``: never counted) and the side parity of the
book (odd numbers on the right, unless the observed numbers say otherwise). If printed numbers jump
by more than the pages in between (scans missing or a number misread), the pages between stay
unlabelled. NDK practice brackets every number that is not printed; this module does the same.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .models import Leaf, LeafSide, PageLabel, PageType, PrintedNumber, ReconciliationWarning, ScanRecord, ScanSide

UNCOUNTED_TYPES = frozenset(
    {
        PageType.FRONT_JACKET,
        PageType.JACKET,
        PageType.FRONT_COVER,
        PageType.BACK_COVER,
        PageType.COVER,
        PageType.SPINE,
        PageType.EDGE,
        PageType.FRONT_END_SHEET,
        PageType.BACK_END_SHEET,
        PageType.FRONT_END_PAPER,
        PageType.BACK_END_PAPER,
        PageType.FLYLEAF,
        PageType.FRONTISPIECE,
    }
)
"""Never part of the page count: binding, jacket, loose leaves and the frontispiece (NDK 1.2.1)."""

UNCOUNTED_LEAVES = frozenset({"plate", "binding", "loose"})
"""Observed ``leaf`` kinds outside the page count: inserted plates, binding parts, loose sheets (NDK 1.1.4)."""

UNCOUNTED_COST = 10
PARITY_COST = 1


@dataclass
class _Page:
    k: int  # page ordinal over the whole book; a spread is two pages
    scan: int  # scan_index
    side: LeafSide | None
    page_type: PageType | None
    numbers: list[PrintedNumber]
    leaf: str | None = None
    kind: str = "free"  # printed | corrected | counted | lettered | observed | free (unresolved)
    value: int | None = None
    system: str | None = None
    sources: list[int] = field(default_factory=list)  # scan indexes the label is derived from
    note: str | None = None


@dataclass
class Pagination:
    labels: dict[int, list[PageLabel]]
    """Labels per scan_index, one per labelled page (left before right on a spread)."""
    unresolved: set[int] = field(default_factory=set)
    """scan_index of scans with at least one page left unlabelled."""
    runs: list[str] = field(default_factory=list)
    """Human-readable summary of the printed sequences, e.g. '- scans 7-20: arabic 1-14'."""
    warnings: list[ReconciliationWarning] = field(default_factory=list)

    def page_number(self, scan_index: int) -> str | None:
        """Scan-level NDK label; None if a page of the scan has no label."""
        labels = self.labels[scan_index]
        return ",".join(lab.label for lab in labels) if labels and scan_index not in self.unresolved else None


def paginate(
    scans: Sequence[ScanRecord],
    types: Sequence[PageType | None],
    sides: Sequence[ScanSide | None],
    subtypes: Sequence[dict[str, PageType | None]],
    leaves: Sequence[Leaf | None],
    subleaves: Sequence[dict[str, Leaf | None]],
) -> Pagination:
    """Label every page; ``types``/``sides``/``subtypes`` and ``leaves``/``subleaves`` are the (resolved)
    values per scan. A page of a spread takes its type from ``subtypes`` (else the scan's) and its leaf
    kind only from ``subleaves``."""
    odd_side = _odd_side(scans)
    pages = _pages(scans, types, sides, subtypes, leaves, subleaves, odd_side)
    pos = {s.scan_index: s.scan_index + 1 for s in scans}
    ids = {s.scan_index: s.scan_id for s in scans}
    result = Pagination({s.scan_index: [] for s in scans})

    def warn(code: str, message: str, scan_indexes: list[int]) -> None:
        result.warnings.append(
            ReconciliationWarning(
                code=code,
                message=message,
                detected_by="check",
                field_path=f"resolved.scans[{scan_indexes[len(scan_indexes) // 2]}].page_number",
                scan_ids=list(dict.fromkeys(ids[i] for i in scan_indexes)),
            )
        )

    anchors: list[_Page] = []
    for p in pages:
        if len(p.numbers) == 1 and p.numbers[0].numeric_value is not None and p.numbers[0].numeral_system in (
            "arabic",
            "roman",
        ):
            p.kind, p.value, p.system, p.sources = "printed", p.numbers[0].numeric_value, p.numbers[0].numeral_system, [p.scan]
            anchors.append(p)
        elif p.numbers:
            p.kind = "observed"  # several numbers or no numeric value: kept as observed, not used for counting

    # Rule 1.1.2: a wrong number inside an intact sequence is replaced by the correct one.
    for a, b, c in zip(anchors, anchors[1:], anchors[2:], strict=False):
        if a.system == b.system == c.system and a.value - a.k == c.value - c.k != b.value - b.k:
            expected = b.k + a.value - a.k
            if expected < 1:
                continue
            warn(
                "page_number_conflict",
                f"scan {pos[b.scan]} shows page number {_fmt(b.value, b.system)}, but the numbers on scan "
                f"{pos[a.scan]} ({_fmt(a.value, a.system)}) and scan {pos[c.scan]} ({_fmt(c.value, c.system)}) "
                f"imply {_fmt(expected, b.system)}; labelled {_fmt(expected, b.system)} (NDK 1.1.2)",
                [a.scan, b.scan, c.scan],
            )
            b.kind, b.note, b.sources = "corrected", f"printed {b.numbers[0].raw!r}", [a.scan, b.scan, c.scan]
            b.value = expected

    if not anchors:  # unnumbered volume: count from 1 (NDK 1.1.4 d); leading binding parts are lettered
        free = [p for p in pages if p.kind == "free"]
        start = next((i for i, p in enumerate(free) if not _uncounted(p)), len(free))
        for p in free[:start]:
            p.kind = "lettered"
        _count_forward(free[start:], 0, "arabic", [])
    else:
        first = anchors[0]
        _count_back([p for p in pages[: first.k] if p.kind == "free"], first.value, first.system, [first.scan])
        run = [first]
        for a, b in zip(anchors, anchors[1:], strict=False):
            free = [p for p in pages[a.k + 1 : b.k] if p.kind == "free"]
            missing = b.value - a.value - 1
            where = f"from {_fmt(a.value, a.system)} (scan {pos[a.scan]}) to {_fmt(b.value, b.system)} (scan {pos[b.scan]})"
            if a.system == b.system and 0 <= missing <= len(free):
                _fill_between(free, a.value, b.value, a.system, odd_side, [a.scan, b.scan])
                run.append(b)
                continue
            result.runs.append(_run_line(run, pos))
            run = [b]
            if a.system == b.system and missing > len(free):
                warn(
                    "page_number_gap",
                    f"page numbers jump {where} with {len(free)} unnumbered page(s) in between: scans may be "
                    "missing or a number misread; pages in between left unlabelled",
                    [a.scan, b.scan],
                )
                continue
            if a.system == b.system:
                warn(
                    "page_number_sequence_break",
                    f"page numbers do not increase {where}; labelled as printed (NDK: 'nekonzistence v paginaci')",
                    [a.scan, b.scan],
                )
            _count_back(free, b.value, b.system, [b.scan])  # a new sequence starts at b
        result.runs.append(_run_line(run, pos))
        last = anchors[-1]
        _count_forward([p for p in pages[last.k + 1 :] if p.kind == "free"], last.value, last.system, [last.scan])

    result.unresolved = _labels(pages, result.labels, ids, anchors[0].system if anchors else "arabic")
    return result


def _uncounted(p: _Page) -> bool:
    return p.page_type in UNCOUNTED_TYPES or p.leaf in UNCOUNTED_LEAVES


def _odd_side(scans: Sequence[ScanRecord]) -> LeafSide:
    """Side carrying odd page numbers, by majority of observed numbers with a known side (default right)."""
    votes = 0
    for s in scans:
        obs = s.observation
        for n in obs.printed_numbers if obs else []:
            side = n.side or (obs.side if obs.side in ("left", "right") else None)
            if n.numeric_value is not None and side is not None and n.numeral_system in ("arabic", "roman"):
                votes += 1 if (n.numeric_value % 2 == 1) == (side == "right") else -1
    return "left" if votes < 0 else "right"


def _pages(scans, types, sides, subtypes, leaves, subleaves, odd_side: LeafSide) -> list[_Page]:
    pages: list[_Page] = []
    for s, page_type, side, sub, leaf, subleaf in zip(scans, types, sides, subtypes, leaves, subleaves, strict=True):
        numbers = list(s.observation.printed_numbers) if s.observation else []
        if side != "both":
            side = side if side in ("left", "right") else None
            pages.append(_Page(len(pages), s.scan_index, side, page_type, numbers, leaf))
            continue
        for page_side in ("left", "right"):
            mine = [n for n in numbers if (n.side or _side_by_parity(n, odd_side)) == page_side]
            pages.append(_Page(len(pages), s.scan_index, page_side, sub.get(page_side) or page_type, mine,
                               subleaf.get(page_side)))
    return pages


def _side_by_parity(n: PrintedNumber, odd_side: LeafSide) -> LeafSide:
    if n.numeric_value is None:
        return "left"
    even_side: LeafSide = "left" if odd_side == "right" else "right"
    return odd_side if n.numeric_value % 2 else even_side


def _count_back(free: list[_Page], value: int, system: str, sources: list[int]) -> None:
    """Pages before a printed ``value`` (book start, or a new sequence): the last countable pages get
    ``[value-1]``, ``[value-2]``... down to 1; the others are lettered (NDK 1.1.4)."""
    countable = [p for p in free if not _uncounted(p)]
    n = min(value - 1, len(countable))
    for i, p in enumerate(countable[len(countable) - n :] if n else []):
        p.kind, p.value, p.system, p.sources = "counted", value - n + i, system, sources
    for p in free:
        if p.kind == "free":
            p.kind, p.sources = "lettered", sources


def _count_forward(free: list[_Page], value: int, system: str, sources: list[int]) -> None:
    """Pages after the last printed ``value`` continue the count in brackets, binding parts included
    (NDK 1.1.4 c: ``256, 257, 258, [259], [260], [261]``)."""
    for p in free:
        value += 1
        p.kind, p.value, p.system, p.sources = "counted", value, system, sources


def _fill_between(free: list[_Page], a: int, b: int, system: str, odd_side: LeafSide, sources: list[int]) -> None:
    """Assign ``a+1 .. b-1`` to some of the unnumbered pages between printed ``a`` and ``b`` and letter the
    rest, at minimum cost: counting a binding part, insert or plate costs ``UNCOUNTED_COST``, a number on the wrong
    side ``PARITY_COST``. Ties put lettered pages first (``16, [16a], [16b], [17], 18``)."""
    n, m = len(free), b - a - 1

    def cost(p: _Page, value: int) -> int:
        expected = odd_side if value % 2 else ("left" if odd_side == "right" else "right")
        return UNCOUNTED_COST * (_uncounted(p)) + PARITY_COST * (p.side not in (None, expected))

    # best[i][r]: minimal cost of pages i.. with the last r numbers (b-r .. b-1) still to assign
    inf = float("inf")
    best = [[inf] * (m + 1) for _ in range(n + 1)]
    best[n][0] = 0
    for i in range(n - 1, -1, -1):
        for r in range(min(m, n - i) + 1):
            letter = best[i + 1][r]
            count = cost(free[i], b - r) + best[i + 1][r - 1] if r else inf
            best[i][r] = min(letter, count)
    r = m
    for i, p in enumerate(free):
        count = cost(p, b - r) + best[i + 1][r - 1] if r else inf
        if r and count < best[i + 1][r]:
            p.kind, p.value, p.system = "counted", b - r, system
            r -= 1
        else:
            p.kind = "lettered"
        p.sources = sources


def _labels(pages: list[_Page], out: dict[int, list[PageLabel]], ids: dict[int, str], system: str) -> set[int]:
    """Append the labels to ``out``; return the scans with an unlabelled page."""
    unresolved: set[int] = set()
    base, letters = _fmt(1, system), 0  # pages before page 1 are lettered [1a], [1b], ... (NDK 1.1.4 a)
    for p in pages:
        sources = list(dict.fromkeys(ids[i] for i in p.sources))
        if p.kind in ("printed", "corrected", "counted"):
            base, letters = _fmt(p.value, p.system), 0
            label = base if p.kind != "counted" else f"[{base}]"
            out[p.scan].append(
                PageLabel(
                    side=p.side,
                    label=label,
                    numeric_value=p.value,
                    numeral_system=p.system,
                    origin="observed" if p.kind == "printed" else "inferred",
                    source_scan_ids=sources,
                    confidence=p.numbers[0].confidence if p.kind == "printed" else None,
                    notes=p.note,
                )
            )
        elif p.kind == "lettered":
            letters += 1
            out[p.scan].append(
                PageLabel(side=p.side, label=f"[{base}{_letter(letters)}]", origin="inferred", source_scan_ids=sources)
            )
        elif p.kind == "observed":
            for n in p.numbers:
                out[p.scan].append(
                    PageLabel(
                        side=p.side,
                        label=n.normalized or n.raw,
                        numeric_value=n.numeric_value,
                        numeral_system=n.numeral_system,
                        source_scan_ids=[ids[p.scan]],
                        confidence=n.confidence,
                    )
                )
        else:
            unresolved.add(p.scan)
    return unresolved


def _letter(i: int) -> str:
    """1 -> a, 26 -> z, 27 -> aa (plain letters only: NDK skips 'ch' and accented letters)."""
    s = ""
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(ord("a") + r) + s
    return s


def _run_line(run: list[_Page], pos: dict[int, int]) -> str:
    first, last = run[0], run[-1]
    a, b = _fmt(first.value, first.system), _fmt(last.value, last.system)
    if first is last:
        return f"- scan {pos[first.scan]}: {first.system} {a}"
    return f"- scans {pos[first.scan]}-{pos[last.scan]}: {first.system} {a}-{b}"


_ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
          (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def to_roman(value: int) -> str:
    out = ""
    for v, s in _ROMAN:
        while value >= v:
            out, value = out + s, value - v
    return out


def roman_value(text: str) -> int | None:
    """Value of a canonical roman numeral (any case); None otherwise."""
    text, value, i = text.upper(), 0, 0
    for v, s in _ROMAN:
        while text.startswith(s, i):
            value, i = value + v, i + len(s)
    return value if text and i == len(text) and to_roman(value) == text else None


def _fmt(value: int, system: str) -> str:
    return to_roman(value) if system == "roman" and value > 0 else str(value)
