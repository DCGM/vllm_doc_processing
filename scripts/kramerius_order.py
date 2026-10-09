"""Rebuild physical scan order of downloaded Kramerius 7 documents (development helper, stdlib only).

For each ``<doc-uuid>.ids`` file given, walks the document tree in the Kramerius search index
(children by ``own_parent.pid``, sorted by ``rels_ext_index.sort``, recursing into non-page nodes) and

* rewrites ``<doc-uuid>.ids`` with the page UUIDs in physical order (the old file is kept as ``.ids.orig``);
* writes ``<doc-uuid>.kramerius.json``: document metadata and, per page, the library's page number and
  page type (reference annotations, not ground truth checked by us).

Fails if the downloaded IDs and the Kramerius pages differ.

    python scripts/kramerius_order.py data/test/*.ids [--base-url https://kramerius.lib.cas.cz]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DOC_FIELDS = "pid,model,root.pid,root.title,title.search,date.str,authors,languages.facet,count_page"
PAGE_FIELDS = "pid,model,page.number,page.type,page.index,rels_ext_index.sort"


def search(base: str, query: str, fields: str, sort: str | None = None) -> list[dict]:
    docs, start = [], 0
    while True:
        params = {"q": query, "fl": fields, "rows": 500, "start": start, "wt": "json"}
        if sort:
            params["sort"] = sort
        url = f"{base}/search/api/client/v7.0/search?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            body = json.load(response)["response"]
        docs += body["docs"]
        start += len(body["docs"])
        if not body["docs"] or start >= body["numFound"]:
            return docs
        time.sleep(0.2)


def pages_in_order(base: str, pid: str) -> list[dict]:
    """Depth-first walk of the structural tree below ``pid``; returns page records in order."""
    out = []
    children = search(base, f'own_parent.pid:"{pid}"', PAGE_FIELDS, "rels_ext_index.sort asc")
    for child in children:
        if child["model"] == "page":
            out.append(child)
        else:
            out += pages_in_order(base, child["pid"])
    return out


def process(base: str, ids_file: Path) -> None:
    doc_uuid = ids_file.name.removesuffix(".ids")
    doc_pid = f"uuid:{doc_uuid}"
    meta = search(base, f'pid:"{doc_pid}"', DOC_FIELDS)
    if not meta:
        raise SystemExit(f"{doc_pid}: not found in {base}")
    pages = pages_in_order(base, doc_pid)
    ordered = [p["pid"].removeprefix("uuid:") for p in pages]
    downloaded = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    if sorted(ordered) != sorted(downloaded):
        missing, extra = set(downloaded) - set(ordered), set(ordered) - set(downloaded)
        raise SystemExit(f"{doc_pid}: page sets differ; not in Kramerius: {len(missing)}, not downloaded: {len(extra)}")

    backup = ids_file.with_name(ids_file.name + ".orig")
    if not backup.exists():
        backup.write_text(ids_file.read_text())
    ids_file.write_text("".join(f"{i}\n" for i in ordered))
    record = {
        "source": base,
        "document": meta[0],
        "pages": [
            {
                "scan_index": i,
                "scan_id": pid,
                "page_number": p.get("page.number"),
                "page_type": p.get("page.type"),
            }
            for i, (pid, p) in enumerate(zip(ordered, pages, strict=True))
        ],
    }
    out = ids_file.with_name(f"{doc_uuid}.kramerius.json")
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    print(f"{doc_uuid}: {meta[0].get('model')}, {len(ordered)} pages, {meta[0].get('title.search', '')[:70]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids_files", nargs="+", type=Path)
    parser.add_argument("--base-url", default="https://kramerius.lib.cas.cz", help="Kramerius 7 instance (default: KNAV)")
    args = parser.parse_args()
    for ids_file in args.ids_files:
        process(args.base_url.rstrip("/"), ids_file)


if __name__ == "__main__":
    sys.exit(main())
