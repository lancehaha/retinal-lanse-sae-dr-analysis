#!/bin/bash
#SBATCH --job-name=retinavlm-octave-archive-audit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:30:00
#SBATCH --output=/home/h/heshi/retinavlm-scratch/external_octave_2025/octave-acquire-%j.out

set -euo pipefail
ROOT="$HOME/retinavlm-scratch/external_octave_2025"
ARCHIVE="$ROOT/OCTAVE.zip"
URL='https://zenodo.org/records/14580071/files/OCTAVE.zip?download=1'
mkdir -p "$ROOT"
test ! -e "$ROOT/archive_audit_complete.json"
curl --fail --location --retry 4 --retry-delay 10 --connect-timeout 30 \
  --continue-at - --output "$ARCHIVE" "$URL"
printf '%s  %s\n' 'c731a32afba29a5f6e541ba463e5fc3e' "$ARCHIVE" | md5sum --check --status
sha256sum "$ARCHIVE" > "$ROOT/OCTAVE.zip.sha256"
unzip -Z -1 "$ARCHIVE" > "$ROOT/OCTAVE_archive_members.txt"
"$HOME/retinavlm-scratch/venv/bin/python" - "$ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1])
archive = root / 'OCTAVE.zip'
members = (root / 'OCTAVE_archive_members.txt').read_text().splitlines()
if not members:
    raise ValueError('Archive has no members')
for member in members:
    path = PurePosixPath(member)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe archive member: {member}')
archive_digest = hashlib.sha256()
with archive.open('rb') as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        archive_digest.update(chunk)
receipt = {
    'status': 'archive_verified_not_extracted',
    'source': 'https://zenodo.org/records/14580071',
    'archive_bytes': archive.stat().st_size,
    'archive_sha256': archive_digest.hexdigest(),
    'expected_md5': 'c731a32afba29a5f6e541ba463e5fc3e',
    'member_count': len(members),
    'member_list_sha256': hashlib.sha256(
        (root / 'OCTAVE_archive_members.txt').read_bytes()).hexdigest(),
    'extracted': False,
    'olives_data_touched': False,
    'claim_boundary': 'External posterior-hyaloid anatomy data; not an OLIVES B2/B3 map gate or an equivalent binary label',
}
(root / 'archive_audit_complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
PY
