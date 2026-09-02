from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retina_lanse.data.manifest import sha256_path


REPO_ID = "YukunZhou/RETFound_mae_natureCFP"
FILENAME = "RETFound_mae_natureCFP.pth"
PINNED_REVISION = "556830f78214f0e8da35af965292ac5a3180ac47"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned official gated RETFound CFP checkpoint.")
    parser.add_argument("--external-root", default="/Volumes/Elements/Retina-LanSE")
    parser.add_argument("--output", default="/Volumes/Elements/Retina-LanSE/model_weights/retfound_cfp")
    args = parser.parse_args()

    root, output = Path(args.external_root).resolve(), Path(args.output).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing model output outside external root: {output}") from exc
    if not root.is_dir():
        parser.error(f"external root is not mounted: {root}")
    output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(root / "model_weights" / "huggingface"))
    # Import only after HF_HOME is fixed so token/cache discovery cannot fall
    # back to the system disk.
    from huggingface_hub import hf_hub_download

    checkpoint = Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        revision=PINNED_REVISION,
        local_dir=output,
        token=True,
    ))
    evidence = {
        "schema_version": "1.0",
        "repo_id": REPO_ID,
        "filename": FILENAME,
        "revision": PINNED_REVISION,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "sha256": sha256_path(checkpoint),
        "license": "CC BY-NC 4.0",
    }
    evidence_path = output / "checkpoint_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
