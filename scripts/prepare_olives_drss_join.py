"""Join official PRIME DRSS visit labels to the frozen patient-disjoint split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def column_index(col: str) -> int:
    n = 0
    for char in col:
        n = n * 26 + ord(char) - ord("A") + 1
    return n


def workbook_rows(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as book:
        shared = ["".join(t.text or "" for t in item.iter(NS + "t"))
                  for item in ET.fromstring(book.read("xl/sharedStrings.xml")).iter(NS + "si")]
        for row in ET.fromstring(book.read("xl/worksheets/sheet5.xml")).iter(NS + "row"):
            cells = {}
            for cell in row.iter(NS + "c"):
                value = cell.find(NS + "v")
                if value is None or value.text is None:
                    continue
                col = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                cells[col] = shared[int(value.text)] if cell.attrib.get("t") == "s" else value.text
            yield cells


def run(labels_zip: Path, split_csv: Path, out_csv: Path, out_audit: Path):
    with zipfile.ZipFile(labels_zip) as labels:
        workbook = labels.read("OLIVES_Dataset_Labels/full_labels/OCT-DR.xlsx")
    rows = list(workbook_rows(workbook))
    visit_columns = {}
    for column, heading in rows[1].items():
        if heading != "DRSS Level":
            continue
        anchor = max((c for c in rows[0] if column_index(c) <= column_index(column)), key=column_index)
        visit_name = rows[0][anchor]
        visit_columns[column] = "W0" if visit_name == "Screen" else "W" + visit_name.split()[-1]
    if len(visit_columns) != 21:
        raise ValueError("Unexpected DRSS workbook visit layout")
    scores = {}
    for row in rows[2:]:
        patient, eye = row.get("A"), row.get("D")
        if not patient or eye not in {"OD", "OS"}:
            continue
        for column, visit in visit_columns.items():
            value = row.get(column, "")
            if value.isdigit():
                key = (patient, eye, visit)
                if key in scores and scores[key] != int(value):
                    raise ValueError(f"Conflicting DRSS: {key}")
                scores[key] = int(value)
    with split_csv.open(newline="", encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))
    joined = []
    missing = Counter()
    for row in split_rows:
        if row["trial"] != "Prime_FULL":
            continue
        parts = Path(row["source_image_path"]).parts
        if len(parts) != 6 or parts[1] != "Prime_FULL":
            raise ValueError(f"Unexpected PRIME image path: {row['source_image_path']}")
        patient, visit, eye = parts[2:5]
        key = patient, eye, visit
        if key not in scores:
            missing[(row["split"], visit)] += 1
            continue
        joined.append({"source_image_path": row["source_image_path"],
                       "patient_key": row["patient_key"], "split": row["split"],
                       "visit_key": "/".join(key), "DRSS": scores[key]})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    counts = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in joined if r["split"] == split]
        visits = {r["visit_key"]: r["DRSS"] for r in subset}
        counts[split] = {"scans": len(subset), "visits": len(visits),
                         "patients": len({r["patient_key"] for r in subset}),
                         "visit_level_counts": dict(sorted(Counter(map(str, visits.values())).items()))}
    audit = {"status": "official_prime_drss_visit_join", "source_labels_sha256": digest(labels_zip),
             "frozen_split_sha256": digest(split_csv), "join_csv_sha256": digest(out_csv),
             "join_key": "PRIME path patient code + eye + visit; Screen maps to W0",
             "drss_sheet": "OCT-DR.xlsx/DRSS", "counts": counts,
             "unmatched_prime_scans": {f"{s}:{v}": n for (s, v), n in sorted(missing.items())},
             "limitation": "DRSS is a visit-level clinical grade, shared by the OCT B-scans of that eye and visit."}
    out_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels-zip", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-audit", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.labels_zip, a.split_csv, a.out_csv, a.out_audit), indent=2))
