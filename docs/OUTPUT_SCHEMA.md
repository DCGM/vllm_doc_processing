# Proposed JSON output schema (design contract for issue #1)

**Status: proposed; implementation #1 finalizes exact Pydantic field definitions.**
One run processes **one book** from one directory of **ordered scans**. Output is a single versioned JSON object, not an export of MetaKat's existing `MetakatIO`. Uncertain fields are null; do not substitute guesses.

## Sketch (illustrative; not yet a final JSON Schema)
```json
{
  "schema_version": "0.1",
  "book_id": "book-001",
  "source": {"input_directory": "<local path, optional>", "scan_count": 2},
  "bibliography": {
    "title": {"value": "Example", "source_scan_ids": ["scan-0001"], "origin": "observed"},
    "subtitle": null,
    "authors": [{"value": "A. Writer", "role": "author", "source_scan_ids": ["scan-0001"], "origin": "observed"}],
    "publisher": null,
    "publication_place": null,
    "publication_date": null,
    "edition": null,
    "part_name": null,
    "part_number": null,
    "series_name": null,
    "series_number": null
  },
  "scans": [
    {
      "scan_id": "scan-0001",
      "scan_index": 0,
      "filename": "0001.tif",
      "image_sha256": "<hex-digest>",
      "observation": {
        "page_type": "TitlePage",
        "side": "right",
        "printed_numbers": [],
        "bibliographic_candidates": [],
        "headings": [],
        "toc_entries": []
      }
    },
    {
      "scan_id": "scan-0002",
      "scan_index": 1,
      "filename": "0002.jpg",
      "image_sha256": "<hex-digest>",
      "observation": {
        "page_type": "TableOfContents",
        "side": "both",
        "printed_numbers": [
          {"side": "left", "raw": "iv", "normalized": "IV", "numeric_value": 4, "numeral_system": "roman", "source_scan_id": "scan-0002"},
          {"side": "right", "raw": "v", "normalized": "V", "numeric_value": 5, "numeral_system": "roman", "source_scan_id": "scan-0002"}
        ],
        "bibliographic_candidates": [],
        "headings": [],
        "toc_entries": [{"title": "Introduction", "printed_page_reference": "1", "source_scan_id": "scan-0002"}]
      }
    }
  ],
  "structure": [
    {
      "id": "chapter-1",
      "type": "chapter",
      "title": "Introduction",
      "parent_id": null,
      "start_scan_id": null,
      "end_scan_id": null,
      "toc_source_scan_ids": ["scan-0002"]
    }
  ],
  "reconciliation": {"warnings": [], "changes": [], "unresolved_questions": []},
  "run": {"provider": "openrouter", "vision_model": "<configured model>", "postprocess_model": "<configured model>", "calls": [], "totals": {}}
}
```

The minimal sketch omits some optional fields for clarity: confidence, per-subpage types, published work/volume identifiers, specific contributor roles, cost, request status, image dimensions, raw-vs-reconciled alternatives, and the full MetaKat vocabulary. Issue #1 must define those accurately before implementing the annotator.

## Target coverage relative to MetaKat
Reference: https://github.com/DCGM/MetaKat/blob/main/metakat/schemas/base_objects.py

### Bibliography
Book-specific subset of `MetakatTitle` and `MetakatVolume`: title/subTitle, partNumber/partName, edition, placeTerm, publisher, dateIssued, manufacturePublisher/manufacturePlaceTerm, author/illustrator/photographer/translator/editor, seriesName/seriesNumber; optionally identifiers/language in extension fields. Preserve multiple distinct contributors and publishers and include source scan IDs. Do not assume publisher or printed publication date from image filenames or surrounding context.

### Page type taxonomy
Use MetaKat's `PageType` values for comparability:
`Abstract, Advertisement, Appendix, BackCover, BackEndPaper, BackEndSheet, Bibliography, Blank, CalibrationTable, Cover, CustomInclude, Dedication, Edge, Errata, FlyLeaf, FragmentsOfBookbinding, FrontCover, FrontEndPaper, FrontEndSheet, FrontJacket, Frontispiece, Illustration, Impressum, Imprimatur, Index, Jacket, ListOfIllustrations, ListOfMaps, ListOfTables, Map, NormalPage, Obituary, Preface, SheetMusic, Spine, Table, TableOfContents, TitlePage`.
Unknown = null (not `NormalPage`).

### Physical side and spreads
MetaKat `PageSideType` is `left | right | single_page`. This experiment intentionally uses `left | right | both | null` for **scans**, because one image may contain two facing book pages. Mapping to MetaKat requires special handling: `both` is a spread and cannot be represented as one MetaKat side value. Each printed-number observation and optional subpage classification should use `side: left | right | null`. `left` or `right` does not necessarily imply an odd/even number. Unknown is null.

### Printed pagination
Always separate `scan_index` from the number printed on the paper. Preserve `raw` (e.g. `[12]`, `xii`), `normalized`, parsed `numeric_value` and `numeral_system: arabic | roman | unknown`, plus side within scan, evidence scan ID, confidence/notes. Zero, repeat, unnumbered and reset sequences are valid observations. Absent printed number is **not** proof of a missing physical scan. Infer probable sequence only in a separate reconciliation field with an origin flag.

### Logical structure
Nested `chapter` / `section` elements carry ID, parent ID, title and optional subtitle/part number, TOC source scans, observed printed target-page references, resolved start/end scan IDs and associated evidence. Permit unresolved destinations. A TOC entry is evidence, not automatic proof that the destination was captured.

## Provenance and reconciliation contract
Every assertion should support: `value`, `origin: observed | inferred | catalogued` (catalogued reserved for future external metadata), `source_scan_ids`, optional `confidence` and notes. Self-rated VLM confidence must not be treated as a calibrated probability. Initial observations remain unchanged; postprocessing returns consolidated fields, conflict warnings and an audit trail (`field_path`, old/new value, reason, source scan IDs). This document sketches the presentation; #1 must choose the exact normalized Pydantic implementation and update this page accordingly.

## JSON invariants
- `schema_version` required; all input scans preserved and sorted, indices start at zero, IDs unique.
- No duplicated/missing scan IDs and no references to unknown scan IDs.
- `both` permits two independent pagination observations; an individual printed-number observation never uses `both`.
- Absent or illegible content uses null/empty observations, not ungrounded numeric filling.
- Chapter range references must exist and be ordered; unresolved boundaries permitted.
- Model/provider/call records preserve the difference between failed, retried, escalated and reconciled requests.
