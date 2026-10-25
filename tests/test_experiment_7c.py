from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.run_sae_feature_view_transformers import (
    MODEL_DIM,
    SELECTED_FEATURES,
    SAEFeatureSetTransformer,
    apply_activation_scaling,
    fit_activation_scaling,
    load_passed_cards,
    selected_activation_matrix,
)


class Experiment7CTests(unittest.TestCase):
    def test_passed_card_contract_uses_last_hypothesis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cards.jsonl"
            rows = []
            for feature_id in range(SELECTED_FEATURES):
                rows.append({
                    "feature_id": feature_id,
                    "final_status": "revised_and_passed" if feature_id == 0 else "passed_initial",
                    "hypothesis_versions": [
                        {"version": 1, "hypothesis": f"initial {feature_id}"},
                        *([{"version": 2, "hypothesis": "final zero"}] if feature_id == 0 else []),
                    ],
                })
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            ids, texts, audit = load_passed_cards(path)
            self.assertEqual(len(ids), SELECTED_FEATURES)
            self.assertEqual(texts[0], "final zero")
            self.assertEqual(audit[0]["final_status"], "revised_and_passed")

    def test_selected_matrix_removes_noncard_features(self):
        ids = np.full((2, 1024), -1, dtype=np.int64)
        values = np.zeros((2, 1024), dtype=np.float32)
        ids[0, :3], values[0, :3] = [100, 50, 999], [2.0, 1.0, 4.0]
        ids[1, :2], values[1, :2] = [50, 100], [3.0, 0.5]
        selected = np.concatenate([np.asarray([50, 100]), np.arange(1000, 1249)]).astype(np.int64)
        matrix = selected_activation_matrix(ids, values, selected)
        self.assertEqual(matrix.shape, (2, SELECTED_FEATURES))
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0)
        self.assertAlmostEqual(float(matrix[0, 1]), 2.0)
        self.assertEqual(int(np.count_nonzero(matrix)), 4)

    def test_activation_scaling_is_finite_and_masks_inactive(self):
        matrix = np.zeros((3, SELECTED_FEATURES), dtype=np.float32)
        matrix[0, :2] = [1.0, 2.0]
        matrix[1, 1:3] = [3.0, 4.0]
        matrix[2, 2:4] = [5.0, 6.0]
        scaling = fit_activation_scaling(matrix[:2])
        features = apply_activation_scaling(matrix, scaling)
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(features[matrix == 0] == 0))

    def test_transformer_is_permutation_and_padding_invariant(self):
        rng = np.random.default_rng(4106)
        bank = rng.normal(size=(SELECTED_FEATURES, MODEL_DIM)).astype(np.float32)
        model = SAEFeatureSetTransformer("card_transformer", 1, bank).eval()
        ids = torch.full((2, SELECTED_FEATURES), -1, dtype=torch.int64)
        activations = torch.zeros((2, SELECTED_FEATURES, 2), dtype=torch.float32)
        ids[:, :24] = torch.arange(24)
        activations[:, :24] = torch.randn(2, 24, 2)
        order = torch.randperm(SELECTED_FEATURES)
        with torch.inference_mode():
            original = model(ids, activations)
            reordered = model(ids[:, order], activations[:, order])
            changed = activations.clone()
            changed[ids < 0] = 999.0
            padded = model(ids, changed)
        self.assertTrue(torch.allclose(original, reordered, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(original, padded, atol=1e-5, rtol=1e-5))
        self.assertFalse(any("position" in name or "rank" in name or "dense" in name for name, _ in model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
