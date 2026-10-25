"""Durable per-batch records for matched RetinaVLM training runs.

Each JSON line is one complete optimizer step with nested sample/layer scores.
On checkpoint resume, keep replayed later rows in an attempt archive and
truncate the canonical log to the restored checkpoint step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class BatchRecorder:
    def __init__(self, directory: Path, *, run_id: str, resume_step: int):
        if not run_id or resume_step < 0:
            raise ValueError("Invalid run ID or resume step")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "train_batches.jsonl"
        self.run_id = run_id
        rows = []
        if self.path.exists():
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                complete, tail = raw.rsplit(b"\n", 1) if b"\n" in raw else (b"", raw)
                archive = directory / "incomplete_last_batch.bin"
                if archive.exists():
                    raise FileExistsError("Incomplete-batch archive already exists")
                archive.write_bytes(tail)
                temporary = directory / "train_batches.jsonl.repaired"
                temporary.write_bytes(complete + (b"\n" if complete else b""))
                os.replace(temporary, self.path)
            with self.path.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    row = json.loads(line)
                    if row.get("run_id") != run_id or row.get("global_step") != line_number:
                        raise ValueError(f"Nonsequential batch record at line {line_number}")
                    rows.append(row)
        if len(rows) < resume_step:
            raise ValueError("Checkpoint is ahead of durable batch records")
        if len(rows) > resume_step:
            archive = directory / f"replayed_after_step_{resume_step:08d}.jsonl"
            if archive.exists():
                raise FileExistsError("Resume attempt archive already exists")
            with archive.open("w", encoding="utf-8") as file:
                for row in rows[resume_step:]:
                    file.write(json.dumps(row, separators=(",", ":")) + "\n")
                file.flush()
                os.fsync(file.fileno())
            temporary = directory / "train_batches.jsonl.writing"
            with temporary.open("w", encoding="utf-8") as file:
                for row in rows[:resume_step]:
                    file.write(json.dumps(row, separators=(",", ":")) + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        self.next_step = resume_step + 1
        self.file = self.path.open("a", encoding="utf-8")

    def append(self, row: dict) -> None:
        required = {"run_id", "arm", "seed", "global_step", "checkpoint_version",
                    "map_version", "patient_count", "image_count", "verified_positive_targets",
                    "verified_negative_targets", "unknown_targets", "learning_rate",
                    "total_loss", "answer_loss", "report_loss", "auxiliary_loss",
                    "elapsed_seconds", "sample_layer_scores"}
        if required - row.keys():
            raise ValueError(f"Batch record missing keys: {sorted(required - row.keys())}")
        if row["run_id"] != self.run_id or row["global_step"] != self.next_step:
            raise ValueError("Batch record ID or step is out of sequence")
        if not isinstance(row["sample_layer_scores"], list):
            raise ValueError("Sample/layer scores must be a list")
        if (row["verified_positive_targets"] < 0
                or row["verified_negative_targets"] < 0
                or row["unknown_targets"] < 0
                or row["verified_positive_targets"]
                   + row["verified_negative_targets"] + row["unknown_targets"] != 7):
            raise ValueError("Batch target counts must cover all seven findings")
        keys = [(item["sample_key"], item["finding"], item["layer"])
                for item in row["sample_layer_scores"]]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate sample/finding/layer map row")
        encoded = json.dumps(row, separators=(",", ":"), allow_nan=False)
        self.file.write(encoded + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.next_step += 1

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()
