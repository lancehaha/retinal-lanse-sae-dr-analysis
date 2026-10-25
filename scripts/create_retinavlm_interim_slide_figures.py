"""Export interim RetinaVLM figures from verified tabular artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"released": "#5577AA", "adapted_byol": "#D58C42",
          "A": "#5577AA", "B": "#5B9B78", "C": "#D58C42"}
FINDINGS = ["B1", "B2", "B3", "B4", "B5", "B6", "SRF"]
LABELS = ["Hyperreflective\nfoci", "Partial vitreous\nattachment",
          "Full vitreous\nattachment", "Intraretinal\nfluid",
          "Retinal thickening\n/ edema", "Vitreous\ndebris", "Subretinal\nfluid"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight",
                facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def plot_reference_f1(rows: list[dict[str, str]], out: Path) -> None:
    selected = {(row["variant"], row["finding"]): float(row["F1"])
                for row in rows if row["split"] == "test"}
    if set(selected) != {(arm, finding) for arm in COLORS if arm in
                        {"released", "adapted_byol"} for finding in FINDINGS}:
        raise ValueError("Expected exactly fourteen singleton test F1 scores")
    x = np.arange(len(FINDINGS))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for offset, arm, name in [(-0.19, "released", "Released RetinaVLM"),
                              (0.19, "adapted_byol", "BYOL encoder + aligned adapter")]:
        values = [selected[arm, finding] for finding in FINDINGS]
        bars = ax.bar(x + offset, values, width=0.36, color=COLORS[arm],
                      label=name, zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x, [f"{finding}\n{label}" for finding, label in
                      zip(FINDINGS, LABELS, strict=True)], fontsize=9)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("F1 on 1,274 OCT scans from 13 test patients")
    ax.set_title("Exploratory complete-VLM reference: finding F1", weight="bold")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    fig.text(0.08, 0.01,
             "Macro-F1: 0.559 released; 0.574 adapted. Paired-patient difference CI: "
             "[-0.028, +0.045]. Earlier test reuse; not an A/B/C result.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(bottom=0.28, top=0.88)
    save(fig, out / "reference_finding_f1")


def plot_map_gate(rows: list[dict[str, str]], out: Path) -> None:
    layers = [8, 16, 24, 30]
    lookup = {(row["finding"], int(row["layer"])): row for row in rows}
    if set(lookup) != {(finding, layer) for finding in FINDINGS for layer in layers}:
        raise ValueError("Expected complete seven-finding by four-layer gate")
    passed = np.array([[lookup[finding, layer]["validated_layer"].lower() == "true"
                        for layer in layers] for finding in FINDINGS], dtype=int)
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    from matplotlib.colors import ListedColormap

    ax.imshow(passed, cmap=ListedColormap(["#E6E8EB", "#5B9B78"]),
              vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [str(layer) for layer in layers])
    ax.set_yticks(range(7), [f"{finding}  {label.replace(chr(10), ' ')}"
                             for finding, label in zip(FINDINGS, LABELS, strict=True)])
    ax.set_xlabel("Llama decoder layer")
    ax.set_title("J-Lens gate: five validated finding-layer maps", weight="bold")
    for i in range(7):
        for j in range(4):
            ax.text(j, i, "PASS" if passed[i, j] else "—", ha="center",
                    va="center", weight="bold" if passed[i, j] else "normal",
                    color="white" if passed[i, j] else "#777777", fontsize=10)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(0.08, 0.01,
             "Validation gate combines statistics, image controls, Jacobian check, "
             "and answer influence. Passing does not localize a lesion.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.37, bottom=0.14, top=0.89)
    save(fig, out / "jlens_map_gate")


def rolling_median(values: np.ndarray, window: int = 100) -> np.ndarray:
    return np.array([np.median(values[max(0, i - window + 1):i + 1])
                     for i in range(len(values))])


def plot_training(rows: list[dict[str, str]], out: Path) -> None:
    by_arm: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    if set(by_arm) != {"A", "B", "C"} or any(
            len(block) != 3000 or [int(row["step"]) for row in block]
            != list(range(1, 3001)) for block in by_arm.values()):
        raise ValueError("Expected 3,000 sequential training rows per arm")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharex=True)
    for ax, (field, label) in zip(
            axes, [("answer_loss", "Answer loss"),
                   ("report_loss", "Report loss")], strict=True):
        for arm in ("A", "B", "C"):
            y = np.array([float(row[field]) for row in by_arm[arm]])
            ax.plot(range(1, 3001), rolling_median(y),
                    color=COLORS[arm], label=arm, linewidth=1.6)
        ax.set_title(label)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("100-step rolling median")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.grid(alpha=0.25)
    axes[0].legend(title="Arm", frameon=False)
    fig.suptitle("Matched A/B/C training diagnostics through 3,000 steps",
                 weight="bold")
    fig.text(0.08, 0.015,
             "All three arms saw the same sampled cases. Loss curves do not measure "
             "held-out finding accuracy or report quality.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(bottom=0.18, top=0.86, wspace=0.26)
    save(fig, out / "abc_training_losses")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = {
        "reference_f1": args.root / "f1_results/adapter_alignment_singleton"
                        "/report_tables/finding_f1_table.csv",
        "map_gate": args.root / "jlens_maps_singleton/report_tables"
                    "/map_gate_table_final_singleton.csv",
        "training_curve": args.root / "abc_training/training_curves_step3000"
                          "/training_curve.csv",
    }
    plot_reference_f1(read_csv(inputs["reference_f1"]), args.out)
    plot_map_gate(read_csv(inputs["map_gate"]), args.out)
    plot_training(read_csv(inputs["training_curve"]), args.out)
    figures = sorted(args.out.glob("*.png")) + sorted(args.out.glob("*.svg"))
    manifest = {
        "status": "interim_retinavlm_slide_figures_rendered",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matplotlib_version": matplotlib.__version__,
        "code": {"path": str(Path(__file__).resolve()),
                 "sha256": digest(Path(__file__).resolve())},
        "source_tables": {name: {"path": str(path), "sha256": digest(path)}
                          for name, path in inputs.items()},
        "figures": {path.name: {"path": str(path), "sha256": digest(path)}
                    for path in figures},
        "claim_boundary": "Exploratory reference and map gate; training losses are diagnostics; no A/B/C outcome result",
    }
    (args.out / "provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote 3 PNG and 3 SVG figures to {args.out}")


if __name__ == "__main__":
    main()
