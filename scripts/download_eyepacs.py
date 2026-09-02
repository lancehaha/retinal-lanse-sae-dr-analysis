from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.manifest import sha256_path


COMPETITION = "diabetic-retinopathy-detection"


def assert_inside(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Refusing path outside external root: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Restartable official EyePACS competition downloader.")
    parser.add_argument("--manifest", default="research/eyepacs_kaggle_manifest.json")
    parser.add_argument("--external-root", default="/Volumes/Elements/Retina-LanSE")
    parser.add_argument("--output", default="/Volumes/Elements/Retina-LanSE/raw_archives/eyepacs")
    parser.add_argument("--kaggle", default="/Users/hgh54913/Library/Python/3.9/bin/kaggle")
    parser.add_argument("--file", action="append", dest="selected_files")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    root, output = Path(args.external_root), Path(args.output)
    assert_inside(output, root)
    if not root.is_dir():
        parser.error(f"external root is not mounted: {root}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expected = {item["name"]: int(item["expected_bytes"]) for item in manifest["files"]}
    if args.full and args.selected_files:
        parser.error("choose either --full or one or more --file arguments")
    selected = list(expected) if args.full else (args.selected_files or ["sample.zip", "trainLabels.csv.zip"])
    unknown = sorted(set(selected) - set(expected))
    if unknown:
        parser.error(f"files absent from locked manifest: {unknown}")

    observed = []
    for name in selected:
        target = output / name
        if not target.exists() or target.stat().st_size != expected[name]:
            command = [args.kaggle, "competitions", "download", "-c", COMPETITION, "-f", name, "-p", str(output)]
            subprocess.run(command, check=True)
        actual_bytes = target.stat().st_size
        if actual_bytes != expected[name]:
            raise RuntimeError(f"size mismatch for {name}: expected {expected[name]}, observed {actual_bytes}")
        observed.append({
            "name": name,
            "expected_bytes": expected[name],
            "observed_bytes": actual_bytes,
            "sha256": sha256_path(target) if args.verify_existing or target.exists() else None,
        })

    evidence = {
        "schema_version": "1.0",
        "competition": COMPETITION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "external_root": str(root.resolve()),
        "files": observed,
    }
    evidence_path = output / "download_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
