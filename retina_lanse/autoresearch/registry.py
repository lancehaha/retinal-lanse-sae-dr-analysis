from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
  id INTEGER PRIMARY KEY, encoder_name TEXT NOT NULL, encoder_checkpoint_hash TEXT NOT NULL,
  preprocessing_version TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(encoder_name, encoder_checkpoint_hash, preprocessing_version)
);
CREATE TABLE IF NOT EXISTS sae_runs (
  id INTEGER PRIMARY KEY, model_version_id INTEGER NOT NULL, checkpoint_hash TEXT NOT NULL UNIQUE,
  split_hash TEXT NOT NULL, configuration_json TEXT NOT NULL, random_seed INTEGER NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(model_version_id) REFERENCES model_versions(id)
);
CREATE TABLE IF NOT EXISTS neurons (id INTEGER PRIMARY KEY, sae_run_id INTEGER NOT NULL, neuron_index INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'unreviewed', UNIQUE(sae_run_id, neuron_index));
CREATE TABLE IF NOT EXISTS hypotheses (id INTEGER PRIMARY KEY, neuron_id INTEGER NOT NULL, description TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY, hypothesis_id INTEGER, experiment_type TEXT NOT NULL, status TEXT NOT NULL, payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY, experiment_id INTEGER, metric_name TEXT NOT NULL, value REAL NOT NULL, split TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence_samples (id INTEGER PRIMARY KEY, neuron_id INTEGER NOT NULL, image_id TEXT NOT NULL, activation REAL NOT NULL, evidence_role TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feature_cards (id INTEGER PRIMARY KEY, neuron_id INTEGER NOT NULL, card_json TEXT NOT NULL, evidence_status TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS clinician_reviews (id INTEGER PRIMARY KEY, feature_card_id INTEGER NOT NULL, reviewer_pseudonym TEXT NOT NULL, decision TEXT NOT NULL, review_json TEXT NOT NULL);
"""


def initialize_registry(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target) as connection:
        connection.executescript(SCHEMA)

