"""Versioned, model-independent prompts.

The prompt texts are the experiment's main tunable; ``PROMPT_VERSIONS`` combines a manually bumped
number with a hash of the text, so any edit changes the recorded version even if the bump is forgotten.
"""

from __future__ import annotations

import hashlib

from .models import PageType

PAGE_TYPE_DESCRIPTIONS: dict[PageType, str] = {
    PageType.NORMAL_PAGE: "ordinary page of the main text, including the first page of a chapter; "
    "use when no more specific type applies",
    PageType.TITLE_PAGE: "title page (or half-title page) with the title of the book, usually also author and imprint",
    PageType.IMPRESSUM: "imprint/colophon page with publication or printing details (publisher, printer, year, "
    "edition, copyright, ISBN), often the back of the title page or the last printed page",
    PageType.IMPRIMATUR: "ecclesiastical or censorship approval (imprimatur, nihil obstat)",
    PageType.DEDICATION: "dedication page",
    PageType.PREFACE: "preface, foreword or introduction placed before the main text",
    PageType.TABLE_OF_CONTENTS: "table of contents listing chapters/sections with page references",
    PageType.LIST_OF_ILLUSTRATIONS: "list of illustrations, figures or plates",
    PageType.LIST_OF_MAPS: "list of maps",
    PageType.LIST_OF_TABLES: "list of tables",
    PageType.ABSTRACT: "abstract or summary (e.g. a summary in another language)",
    PageType.APPENDIX: "appendix or supplement after the main text",
    PageType.BIBLIOGRAPHY: "bibliography, list of references or literature",
    PageType.INDEX: "alphabetical index (of names, places, subjects)",
    PageType.ERRATA: "list of errata/corrections",
    PageType.OBITUARY: "obituary",
    PageType.ADVERTISEMENT: "advertisement, e.g. publisher's list of other books",
    PageType.ILLUSTRATION: "page dominated by a picture or plate (not a map)",
    PageType.FRONTISPIECE: "illustration facing (preceding) the title page",
    PageType.MAP: "page or fold-out dominated by a map",
    PageType.TABLE: "page dominated by a table",
    PageType.SHEET_MUSIC: "musical notation",
    PageType.BLANK: "page with no printed content (show-through from the other side, stamps or "
    "shelf marks do not count)",
    PageType.FLY_LEAF: "blank or nearly blank protective leaf at the beginning or end of the book block "
    "(not an endpaper)",
    PageType.FRONT_COVER: "outside of the front cover board or front of a paper wrapper",
    PageType.BACK_COVER: "outside of the back cover board or back of a paper wrapper",
    PageType.COVER: "cover when front and back cannot be distinguished, or both in one image",
    PageType.SPINE: "spine of the binding",
    PageType.EDGE: "edge of the book block (top, bottom or fore-edge)",
    PageType.FRONT_END_SHEET: "front pastedown: endpaper glued to the inside of the front cover board",
    PageType.FRONT_END_PAPER: "front free endpaper: the loose endpaper leaf after the front pastedown",
    PageType.BACK_END_PAPER: "back free endpaper: the loose endpaper leaf before the back pastedown",
    PageType.BACK_END_SHEET: "back pastedown: endpaper glued to the inside of the back cover board",
    PageType.FRONT_JACKET: "front of a dust jacket",
    PageType.JACKET: "dust jacket other than its front (back, flaps, spine, whole jacket)",
    PageType.FRAGMENTS_OF_BOOKBINDING: "fragments of the binding, e.g. waste paper or manuscript fragments "
    "used in the binding",
    PageType.CUSTOM_INCLUDE: "material inserted into the book that is not part of the edition (loose insert, "
    "library slip, attached document)",
    PageType.CALIBRATION_TABLE: "colour/scale calibration target only, without book content",
}

OBSERVE_PROMPT_NUMBER = 3
"""Bump on any intended change of the observation prompt."""

OBSERVE_SYSTEM = """\
You annotate one scan of a digitized printed book for a library catalogue. The image shows one page, \
a two-page opening (spread), or a part of the binding (cover, spine, edge). Report only what is \
visible in this image, as structured data. Do not transcribe the page: quote only the short pieces \
of text the fields ask for.

# General rules
- Never guess or invent. If something is absent, illegible or uncertain, use null or an empty list \
and say why in `notes`. An empty field is always better than a fabricated one.
- Copy quoted text exactly as printed: original language, spelling, historical orthography, \
diacritics and capitalisation (normal case for text set in capitals is acceptable). Do not translate, \
modernise, complete or expand abbreviations.
- The request states the position of the scan in the physical scanning order. It is NOT a printed \
page number; never derive page numbers from it.
- Context from earlier scans, if provided, was produced automatically and may contain errors. Use \
it only to interpret what you see (e.g. which numbering to expect); never report anything that is \
not visible in this image.
- Ignore handwritten notes, library stamps, shelf marks, barcodes and scanning equipment for all \
fields except `notes`.
- `confidence` values are your own estimate between 0 and 1; use null if you cannot judge.
- `notes`: at most two short sentences (illegibility, damage, uncertainty), or null.

# side
- "left": one page that is the left (verso) page of an opening. Clues: the binding/gutter is on its \
right edge; the page number is in the left corner.
- "right": one page that is the right (recto) page of an opening. Clues: the gutter is on its left \
edge; the page number is in the right corner.
- "both": two separate facing pages side by side in one image, with the gutter (fold) between them. \
Text columns or several language versions on one sheet are one page. A single wide sheet (fold-out \
map or plate, landscape table), a cover or a jacket is never "both". If in doubt, it is not "both".
- null: covers, spine, edge, fold-outs, loose sheets, or when the side cannot be determined.

# page_type
One label for the whole image, from this vocabulary:
{page_types}
For a spread (side "both") choose the more specific type of the two pages (e.g. TitlePage rather \
than Blank) and give each page its type in `subpages` (one entry for "left", one for "right"). \
Leave `subpages` empty unless side is "both". Use null only if the image is unreadable.

# printed_numbers
Page numbers printed on the page(s), usually in a top or bottom corner or centred in the header or \
footer. One entry per page that shows a number; a spread may have two.
- Not page numbers:
  - printer's signature marks at the bottom of the page: letters or letter-number groups ("A", "B ij", \
"C3"), numbers with an asterisk ("1*", "3*"), or a small number next to a short title;
  - catchwords (a single word at the bottom right that repeats the first word of the next page);
  - page references inside table-of-contents, index or list entries: they point to other pages;
  - chapter, section, footnote, plate, figure or table numbers ("Tab. 3", "Suppl. 3", "Fig. 2"), years \
and numbers in the text.
- `raw`: exactly as printed, including brackets or dashes ("[12]", "- 7 -", "xii").
- `normalized`: arabic digits without decoration ("12"), roman numerals in upper case ("XII").
- `numeric_value`: the integer value (also for roman numerals), or null.
- `numeral_system`: "arabic", "roman" or "other".
- `side`: the page that carries the number ("left"/"right"); null for a single page whose side is unknown.
- If a number is present but illegible, or you are not sure that a mark is a page number, do not \
report it; mention it in `notes`.
- Empty list if no page number is printed.

# headings
Headings that start a part, chapter or section on this scan, as printed (include its number or \
label, e.g. "KAPITOLA II. Na horách"). Not running headers repeated at the top of every page, not \
the book title on a title page, not the title of a contents or index page itself ("Obsah", \
"Contents"), not group captions inside a table of contents. `level`: 1 for the top level \
(part or chapter), 2 for a section inside it, and so on; null if unclear. `side`: the page of a \
spread it is on, else null.

# toc_entries
Only on table-of-contents pages: one entry per listed chapter or section, in the order printed. \
`title` as printed including its number; `printed_page_reference` as printed (e.g. "17", "XII"), \
null if none; `level` from indentation or typography (1 = top). Do not list index entries.

# bibliographic_candidates
Only from pages that present the book itself: title page, half-title, cover, spine, imprint/colophon, \
series page. Never from the main text, advertisements of other books, bibliographies or indexes.
One entry per value; one entry per person. Values as printed, without role phrases ("by", "napsal", \
"übersetzt von", "illustrations by"). Report persons only in the role the page gives them (e.g. a \
by-line for the author); dedicatees, addressees and persons in the title are not authors. Fields:
- title, subtitle; part_name and part_number (of a multi-volume work, e.g. "Díl II.");
- series_name, series_number (e.g. "Svazek 12");
- edition (e.g. "Druhé vydání");
- author, editor, translator, illustrator, photographer;
- publisher, publication_place, publication_date (year or date as printed, e.g. "1923", "MDCCCXII");
- manufacture_publisher and manufacture_place: the printer ("Tiskem ...", "Druck von ...", \
"Printed by ...") and its place.
If the same field is printed in different forms on this scan, report every form.
"""

OBSERVE_SYSTEM = OBSERVE_SYSTEM.format(
    page_types="\n".join(f"- {t.value}: {d}" for t, d in PAGE_TYPE_DESCRIPTIONS.items())
)


def observe_user_prompt(scan_index: int, scan_count: int, context: str | None = None) -> str:
    """User message accompanying the scan image; ``context`` is the bounded summary of earlier scans."""
    parts = [
        f"Scan position: {scan_index + 1} of {scan_count} in physical scanning order (not a page number).",
    ]
    if context:
        parts.append(f"Context from earlier scans (automatic, may contain errors):\n{context}")
    parts.append("Annotate the attached scan.")
    return "\n\n".join(parts)


RECONCILE_PROMPT_NUMBER = 1
"""Bump on any intended change of the reconciliation prompt or of ``reconcile.reconcile_input``."""

RECONCILE_SYSTEM = """\
You reconcile the annotations of one scanned printed book for a Czech digital library catalogue \
(National Digital Library, NDK, rules for describing monographs). A vision model looked at each scan \
separately; the input lists, for every scan, the observed page type, side and printed page numbers, \
with headings, table-of-contents entries and bibliographic data, plus a summary of the printed page \
numbering and the page label computed from it in NDK notation (printed numbers plain, computed \
numbers in brackets, pages outside the page count lettered, e.g. [1a]). The observations may contain \
errors. "scan N" is a position in the scanning order, never a page number. Answer with four lists.

# General rules
- Use only what the input shows. Never invent values, chapters or scans, and never fill gaps with \
general knowledge about the book. Omit what is unknown.
- Keep text exactly as observed (language, spelling, diacritics); you may use normal case for text \
observed in capitals. Do not translate, modernise or expand abbreviations.
- Refer to scans only by the numbers N of "scan N" in the input.

# scan_corrections
Corrections of page types and sides that are clearly wrong, judged from the neighbouring scans, the \
page numbering and the observed content. When in doubt, keep the observation; list only scans you \
change (usually few or none). Never correct a scan that was not observed. NDK conventions:
- Usual scan order: FrontJacket (front of a dust jacket, if any), FrontCover, FrontEndSheet (inside \
of the front cover), FrontEndPaper (free endpaper leaf), the book block starting with its first \
leaf, BackEndPaper, BackEndSheet (inside of the back cover), BackCover, Spine, Jacket (whole jacket); \
loose leaves come last. Only the outside of a cover board is a cover.
- FlyLeaf: a loose leaf or insert not bound into the book (placed after the back cover); not a blank \
protective leaf. Blank: a page without print (stamps, barcodes, shelf marks or handwriting do not \
count as print); blank pages inside the book block are counted in the pagination.
- TitlePage: the front of the title leaf with the full title (also a half-title); a cover is not a \
title page. Frontispiece: an illustration on the left page facing the title page. Impressum: imprint \
or colophon (copyright, ISBN, printer, edition).
- A page showing any table of contents is TableOfContents. A page with two roles takes the first of: \
TitlePage before TableOfContents; a cover before TableOfContents; TableOfContents before an end \
sheet; an end sheet before Map.
- Sides: single-page scans of a bound book alternate right, left, right, ...; the first page of the \
book block is a right page; odd page numbers are normally on right pages. Covers, spine, edge, \
jacket, fold-outs and loose sheets have no side ("none"). "both" only for two facing pages in one image.
- `page_type`: the new type, or null to keep it. `side`: "left", "right", "both", "none", or "keep". \
`reason`: one short sentence citing the evidence (e.g. "between scans 3 and 5 of the front matter, \
blank, page 4 implied").

# bibliography
The book's bibliographic data, merged from all scans. One entry per value; one entry per person.
- Choose the most complete form, preferring the title page over cover, spine or half-title. Merge \
forms that differ only in case, spacing, punctuation or line breaks into one entry.
- title, subtitle, part_name, part_number, edition, publication_date: at most one entry each. If \
the scans show genuinely different values, choose one and report the others in `issues`.
- Move a value to another field only when its observed field is clearly wrong (e.g. a subtitle \
reported as a second title), and explain this in `notes`.
- `source_scans`: every scan that shows the value. `notes`: short reason for any choice or change, \
else null.

# chapters
The chapter and section list of the book in reading order, built from the TOC entries and the \
headings.
- One entry per chapter or section. A TOC entry and the heading that opens the same chapter form \
ONE entry: match them by number and title (allowing for abbreviation, case and small reading \
errors) and by the TOC page reference against the printed page numbering.
- Include chapters listed in the TOC whose heading was not observed, and headings not listed in \
the TOC. Do not include the caption of the TOC page itself ("Obsah", "Contents"), running heads, \
or group captions that are not chapters.
- `title`: as printed in the TOC entry; if there is none, as in the heading.
- `level`: 1 = top level (part or chapter), 2 = section within it, and so on; consistent across \
the book.
- `toc_scans`: the scans whose TOC lists the entry (empty if none). `printed_page_reference`: the \
page reference exactly as observed in that TOC entry, or null.
- `heading_scan`: the scan where the heading of this chapter was observed, or null. Never guess a \
scan from page numbers; that is computed afterwards.
- `notes`: short remark on doubtful matches, else null.

# issues
Contradictions and doubts worth a human check that the other lists cannot express, e.g. \
different titles on cover and title page, a TOC entry whose heading is missing although its \
pages were scanned, inconsistent heading levels. `code`: short snake_case slug; `message`: one \
sentence; `scans`: the scans involved. Do not repeat the page-numbering findings given in the \
input. Empty list if there are none.
"""


def _version(number: int, text: str) -> str:
    return f"{number}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"


PROMPT_VERSIONS: dict[str, str] = {
    "observe": _version(OBSERVE_PROMPT_NUMBER, OBSERVE_SYSTEM + observe_user_prompt(0, 1, "{context}")),
    "reconcile": _version(RECONCILE_PROMPT_NUMBER, RECONCILE_SYSTEM),
}
"""Recorded in ``run.prompt_versions``."""
