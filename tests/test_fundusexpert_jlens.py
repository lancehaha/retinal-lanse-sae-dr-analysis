import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from fundusexpert_jlens.build_ddr import location_from_mask
from fundusexpert_jlens.evaluate import paired_bootstrap, score
from fundusexpert_jlens.lens import auxiliary_loss, estimate_direction, fit_calibrator
from fundusexpert_jlens.protocol import assign_splits, audit_manifest, target_report, verified_label


class ProtocolTests(unittest.TestCase):
    def test_mask_location_uses_image_coordinates(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[:2, :2] = True
        self.assertEqual(location_from_mask(mask), "image-relative upper-left")
        self.assertIsNone(location_from_mask(np.zeros((8, 8), dtype=bool)))

    def test_empty_mask_is_not_negative_without_exhaustive_annotation(self):
        row = {"hard_exudates": "absent", "exhaustive_hard_exudate_annotation": False}
        self.assertIsNone(verified_label(row))
        self.assertIn('"uncertain"', target_report(row))

    def test_patient_and_duplicate_links_are_transitive(self):
        with tempfile.TemporaryDirectory() as folder:
            img = Path(folder) / "image.jpg"
            img.touch()
            rows = [{"image_id": "a", "patient_id": "p", "duplicate_group": None},
                    {"image_id": "b", "patient_id": "p", "duplicate_group": "d"},
                    {"image_id": "c", "patient_id": None, "duplicate_group": "d"}]
            for row in rows:
                row.update(image_path=str(img), hard_exudates="unknown", source="DDR")
            assign_splits(rows)
            self.assertEqual(len({r["split"] for r in rows}), 1)
            self.assertEqual(audit_manifest(rows)["images"], 3)

    def test_reject_unverified_negative(self):
        with tempfile.TemporaryDirectory() as folder:
            img = Path(folder) / "image.jpg"
            img.touch()
            with self.assertRaises(ValueError):
                audit_manifest([{"image_id": "a", "image_path": str(img), "split": "test",
                                 "hard_exudates": "absent", "source": "DDR"}])


class LossTests(unittest.TestCase):
    def test_jacobian_uses_final_prompt_position(self):
        class TinyModel(torch.nn.Module):
            def forward(self, input_ids, **unused):
                hidden = torch.nn.functional.one_hot(input_ids, num_classes=3).float().requires_grad_()
                logits = hidden @ torch.tensor([[0., 0.], [0., 2.], [0., 0.]])
                return SimpleNamespace(hidden_states=[hidden], logits=logits)

        example = {"pixel_values": torch.zeros(1), "input_ids": torch.tensor([[0, 1]]),
                   "attention_mask": torch.ones(1, 2), "image_flags": torch.ones(1, 1),
                   "pre_answer_position": 1}
        direction = estimate_direction(TinyModel(), [example], layer=0, token_id=1)
        self.assertTrue(torch.allclose(direction, torch.tensor([0., 1., 0.])))

    def test_future_readout_backpropagates_to_pre_answer_state(self):
        class TinyCausalModel(torch.nn.Module):
            def forward(self, input_ids, **unused):
                hidden = torch.nn.functional.one_hot(input_ids, num_classes=3).float().requires_grad_()
                logits = hidden.cumsum(dim=1) @ torch.tensor([[0., 0.], [0., 2.], [0., 0.]])
                return SimpleNamespace(hidden_states=[hidden], logits=logits)

        example = {"pixel_values": torch.zeros(1), "input_ids": torch.tensor([[0, 1, 0]]),
                   "attention_mask": torch.ones(1, 3), "image_flags": torch.ones(1, 1),
                   "pre_answer_position": 1, "readout_position": 2}
        direction = estimate_direction(TinyCausalModel(), [example], layer=0, token_id=1)
        self.assertTrue(torch.allclose(direction, torch.tensor([0., 1., 0.])))

    def test_calibrator_and_unknown_label_mask(self):
        slope, intercept = fit_calibrator(torch.tensor([-1., -0.5, 0.5, 1.]), torch.tensor([0, 0, 1, 1]), steps=30)
        self.assertGreater(slope, 0)
        h = torch.tensor([[0.5, 0.5], [0., 1.]], requires_grad=True)
        loss = auxiliary_loss(h, torch.tensor([[1., 0.]]), torch.tensor([1, -1]), slope, intercept)
        loss.backward()
        self.assertNotEqual(float(h.grad[0].abs().sum()), 0)
        self.assertEqual(float(h.grad[1].abs().sum()), 0)


class EvaluationTests(unittest.TestCase):
    def test_paired_claim_metric(self):
        common = {"grade": "unknown", "grade_pred": "unknown", "location_truth": None,
                  "location_pred": None, "valid": True, "report": ""}
        a = [dict(common, image_id="n", patient="p", label=0, finding="present"),
             dict(common, image_id="p", patient="q", label=1, finding="present")]
        b = [dict(common, image_id="n", patient="p", label=0, finding="absent"),
             dict(common, image_id="p", patient="q", label=1, finding="present")]
        self.assertEqual(score(a)["unsupported_claims"], 1)
        self.assertEqual(paired_bootstrap(a, b, "unsupported_claims_per_image", n=20)["difference"], 0.5)


if __name__ == "__main__":
    unittest.main()
