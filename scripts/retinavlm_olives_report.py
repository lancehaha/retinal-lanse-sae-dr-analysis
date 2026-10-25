"""Frozen, label-bounded OCT report targets and literal claim extraction."""

from __future__ import annotations

import re


FINDING_NAMES = {
    "B1": "intraretinal hyperreflective foci",
    "B2": "partially attached vitreous face",
    "B3": "fully attached vitreous face",
    "B4": "intraretinal fluid",
    "B5": "diffuse retinal thickening or macular edema",
    "B6": "vitreous debris",
    "SRF": "subretinal fluid",
}
REPORT_QUESTION = "Describe the verified retinal OCT findings briefly."
CLAIM_PATTERN = {
    key: re.compile(r"(?<!\w)" + re.escape(name) + r"\s*:\s*(present|absent)\b",
                    flags=re.IGNORECASE)
    for key, name in FINDING_NAMES.items()
}


def report_target(row: dict[str, str]) -> str:
    """Use only verified 0/1 labels; omit unknown findings completely."""
    parts = [f"{name}: {'present' if row[key] == '1' else 'absent'}"
             for key, name in FINDING_NAMES.items() if row.get(key) in {"0", "1"}]
    if not parts:
        raise ValueError("No verified finding labels for report target")
    return " " + "; ".join(parts) + "."


def parse_report_claims(text: str) -> dict[str, int | None]:
    """Read only the frozen literal format; ambiguous claims remain unknown."""
    if not isinstance(text, str):
        raise TypeError("Report must be text")
    claims = {}
    for key, pattern in CLAIM_PATTERN.items():
        values = {1 if match.group(1).lower() == "present" else 0
                  for match in pattern.finditer(text)}
        claims[key] = values.pop() if len(values) == 1 else None
    return claims


def report_claim_counts(row: dict[str, str], claims: dict[str, int | None]) -> dict[str, dict[str, int]]:
    """Return per-finding miss/unsupported counts with explicit denominators."""
    result = {}
    for key in FINDING_NAMES:
        label = row.get(key)
        if label not in {"0", "1"}:
            continue
        y = int(label)
        claim = claims.get(key)
        result[key] = {
            "positive_label": y,
            "negative_label": 1 - y,
            "supported_present": int(y == 1 and claim == 1),
            "unsupported_present": int(y == 0 and claim == 1),
            "miss": int(y == 1 and claim != 1),
            "ambiguous_or_missing": int(claim is None),
        }
    return result
