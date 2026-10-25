from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from retina_lanse.autoresearch.registry import initialize_registry
from retina_lanse.data.preprocessing import normalize_retfound_cfp, preprocess_retfound_cfp
from retina_lanse.data.embedding_shards import load_embedding_run
from retina_lanse.data.manifest import parse_eyepacs_identity, sha256_path, stable_patient_split
from retina_lanse.data.splits import assert_no_patient_leakage, patient_level_splits
from retina_lanse.encoders import DeterministicSmokeEncoder, RETFoundUnavailableError, load_retfound_cfp
from retina_lanse.evaluation.metrics import (
    evaluate_embeddings,
    evaluate_embeddings_batched,
    evaluate_reconstruction,
)
from retina_lanse.evaluation.statistics import paired_patient_bootstrap_difference
from retina_lanse.sae.model import TopKSAE
from retina_lanse.sae.trainer import train_sae_batched
from retina_lanse.utils import sha256_file, stable_hash, verify_file_hash_bundle
from scripts.audit_near_duplicate_leakage import lsh_bands
from scripts.remediate_near_duplicate_manifest import remediate_records
from scripts.run_vlm_ablation_diagnostics import bootstrap_ci, ece, metric_bundle
from scripts.run_vlm_featurecard_ablation import select_context_cards
from scripts.score_clinician_reviews import fleiss_kappa, pairwise_exact_agreement
from scripts.check_stage6_external_gate import check_stage6_gate
from scripts.prepare_stage6_image_manifest import build_records, load_labels
from scripts.run_feature_card_dr_prediction import (
    bh_adjust,
    card_matrix,
    evidence_logits,
    fit_evidence_tables,
    parse_batch_probability_json,
    parse_probability_json,
    softmax,
)
from scripts.run_supervised_identity_rank_fusion import (
    DICTIONARY_SIZE,
    IdentityRankTransformer,
    apply_scaling,
    fit_scaling,
    validate_trace,
)


class PipelineTests(unittest.TestCase):
    def test_experiment6_card_matrix_preserves_feature_identity(self):
        feature_ids = torch.full((2, 1024), -1, dtype=torch.int32).numpy()
        values = torch.zeros((2, 1024), dtype=torch.float32).numpy()
        feature_ids[0, :3] = [9, 3, 7]
        values[0, :3] = [1.2, 0.8, 0.4]
        feature_ids[1, :2] = [3, 9]
        values[1, :2] = [1.1, 0.2]
        cards = torch.arange(256, dtype=torch.int64).numpy()
        matrix = card_matrix(feature_ids, values, cards)
        self.assertEqual(matrix.shape, (2, 256))
        self.assertAlmostEqual(float(matrix[0, 9]), 1.2)
        self.assertAlmostEqual(float(matrix[1, 3]), 1.1)
        self.assertEqual(int((matrix > 0).sum()), 5)

    def test_experiment6_evidence_probabilities_are_complete(self):
        matrix = torch.zeros((20, 256), dtype=torch.float32).numpy()
        grades = torch.tensor([0, 1, 2, 3, 4] * 4).numpy()
        for row in range(20):
            matrix[row, grades[row]] = 0.5 + grades[row]
        tables = fit_evidence_tables(matrix, grades)
        probabilities = softmax(evidence_logits(matrix, tables, torch.ones(256, dtype=torch.bool).numpy()))
        self.assertEqual(probabilities.shape, (20, 5))
        self.assertTrue(torch.allclose(torch.from_numpy(probabilities.sum(axis=1)), torch.ones(20, dtype=torch.float64)))
        parsed = parse_probability_json('{"grade_probabilities":[1,2,3,4,5]}')
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(sum(parsed["grade_probabilities"]), 1.0)
        self.assertIsNone(parse_probability_json('{"grade_probabilities":[1,2]}'))
        batch = parse_batch_probability_json(
            '{"predictions":[{"case_id":0,"grade_probabilities":[1,0,0,0,0]},'
            '{"case_id":1,"grade_probabilities":[0,0,1,0,0]}]}',
            2,
        )
        self.assertEqual(sorted(batch), [0, 1])
        self.assertIsNone(parse_batch_probability_json('{"predictions":[]}', 2))
        adjusted = bh_adjust(torch.tensor([0.01, 0.03, 0.20]).numpy())
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))

    def test_experiment7_trace_and_padding_contract(self):
        feature_ids = torch.full((2, 1024), -1, dtype=torch.int64).numpy()
        values = torch.zeros((2, 1024), dtype=torch.float32).numpy()
        feature_ids[:, :3] = [1, 2, 3]
        values[:, :3] = [3.0, 2.0, 1.0]
        validate_trace(feature_ids, values, 2)
        dense = torch.randn(2, 1024).numpy()
        scaling = fit_scaling(dense, feature_ids, values)
        dense_scaled, values_scaled = apply_scaling(dense, feature_ids, values, scaling)
        model = IdentityRankTransformer(1, use_identity=True, use_position=True).eval()
        ids_tensor = torch.from_numpy(feature_ids)
        value_tensor = torch.from_numpy(values_scaled)
        with torch.inference_mode():
            first = model(torch.from_numpy(dense_scaled), ids_tensor, value_tensor)
            changed = value_tensor.clone()
            changed[ids_tensor < 0] = 999.0
            second = model(torch.from_numpy(dense_scaled), ids_tensor, changed)
        self.assertEqual(model.feature_identity_embedding.num_embeddings, DICTIONARY_SIZE + 1)
        self.assertTrue(torch.allclose(first, second, atol=1e-5))

    def test_stage6_manifest_builder_matches_images_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (12, 8), color=(20, 30, 40)).save(image_root / "IDRiD_001.jpg")
            labels_path = root / "labels.csv"
            labels_path.write_text("image,grade\nIDRiD_001.jpg,2\n", encoding="utf-8")
            labels = load_labels(labels_path)
            records = build_records("idrid", root, image_root, labels)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["split"], "external")
            self.assertEqual(records[0]["referable_dr"], 1)
            self.assertEqual(len(records[0]["sha256"]), 64)

    def test_stage6_gate_fails_closed_without_approved_external_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = check_stage6_gate(
                dataset="idrid",
                external_root=root,
                dataset_root=root / "datasets" / "idrid",
                approval_path=root / "datasets" / "idrid" / "access_approval.json",
                manifest_path=root / "datasets" / "idrid" / "manifest.jsonl",
                release_manifest_path=root / "experiments" / "release.json",
                clinician_status_path=root / "experiments" / "clinician.json",
            )
            self.assertEqual(report["status"], "blocked_external_access")
            self.assertFalse(report["inference_performed"])
            self.assertTrue(any("approved dataset directory" in reason for reason in report["reasons"]))

    def test_stage6_gate_accepts_complete_provenance_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "idrid"
            dataset.mkdir(parents=True)
            image = dataset / "sample.png"
            image.write_bytes(b"not-a-real-image-but-a-verified-fixture")
            digest = sha256_path(image)
            (dataset / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "relative_path": "sample.png",
                        "patient_id": "p1",
                        "label": 1,
                        "sha256": digest,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (dataset / "access_approval.json").write_text(
                json.dumps(
                    {
                        "approved": True,
                        "source_url": "https://official.example/idrid",
                        "license": "research terms",
                        "agreement_reference": "approval-1",
                        "archive_sha256": "a" * 64,
                        "access_date": "2026-09-05",
                        "labels_verified": True,
                        "patient_grouping_verified": True,
                        "preprocessing_recorded": True,
                    }
                ),
                encoding="utf-8",
            )
            experiments = root / "experiments"
            experiments.mkdir()
            release = experiments / "release.json"
            release.write_text(
                json.dumps(
                    {
                        "all_heavy_outputs_on_external_drive": True,
                        "locked_internal_test_accessed": True,
                        "paper_claim_boundary": "frozen exploratory evidence",
                    }
                ),
                encoding="utf-8",
            )
            clinician = experiments / "clinician.json"
            clinician.write_text(
                json.dumps(
                    {
                        "status": "complete_clinician_reviews",
                        "complete_rows": 192,
                        "expected_rows": 192,
                    }
                ),
                encoding="utf-8",
            )
            report = check_stage6_gate(
                dataset="idrid",
                external_root=root,
                dataset_root=dataset,
                approval_path=dataset / "access_approval.json",
                manifest_path=dataset / "manifest.jsonl",
                release_manifest_path=release,
                clinician_status_path=clinician,
            )
            self.assertEqual(report["status"], "ready_for_one_time_run")
            self.assertEqual(report["evidence"]["manifest"]["rows"], 1)
    def test_patient_splits_are_deterministic_and_disjoint(self):
        patients = [f"p{i}" for i in range(100)]
        first = patient_level_splits(patients, (0.7, 0.2, 0.1), 7)
        second = patient_level_splits(reversed(patients), (0.7, 0.2, 0.1), 7)
        self.assertEqual(first, second)
        records = [{"patient_id": p, "split": split} for p, split in first.items()]
        assert_no_patient_leakage(records)

    def test_topk_budget(self):
        model = TopKSAE(16, 64, 7)
        z = model.encode(torch.randn(10, 16))
        self.assertTrue(torch.all((z > 0).sum(dim=1) <= 7))

    def test_batched_metrics_match_materialized_metrics(self):
        torch.manual_seed(3)
        model = TopKSAE(16, 64, 7)
        embeddings = torch.randn(23, 16)
        expected = evaluate_embeddings(model, embeddings)
        observed = evaluate_embeddings_batched(model, embeddings, batch_size=5, device="cpu")
        self.assertEqual(expected["max_active_features"], observed["max_active_features"])
        for key in expected:
            if key != "max_active_features":
                self.assertAlmostEqual(float(expected[key]), float(observed[key]), places=5)

    def test_reconstruction_metrics_are_exact_for_identity(self):
        embeddings = torch.randn(9, 6)
        metrics = evaluate_reconstruction(embeddings, embeddings.clone())
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["explained_variance"], 1.0)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0, places=6)

    def test_frozen_analysis_bundle_rejects_post_freeze_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.py"
            path.write_text("value = 1\n", encoding="utf-8")
            file_hashes = {str(path.resolve()): sha256_file(path)}
            bundle = stable_hash(
                [
                    {"path": frozen_path, "sha256": digest}
                    for frozen_path, digest in sorted(file_hashes.items())
                ]
            )
            self.assertEqual(verify_file_hash_bundle(file_hashes, bundle), (True, "verified"))
            path.write_text("value = 2\n", encoding="utf-8")
            verified, reason = verify_file_hash_bundle(file_hashes, bundle)
            self.assertFalse(verified)
            self.assertIn("changed after authorization", reason)

    def test_patient_bootstrap_is_paired_clustered_and_deterministic(self):
        left = [2.0, 4.0, 8.0, 10.0]
        right = [1.0, 3.0, 5.0, 7.0]
        patients = ["a", "a", "b", "b"]
        first = paired_patient_bootstrap_difference(
            left, right, patients, resamples=200, seed=11
        )
        second = paired_patient_bootstrap_difference(
            left, right, patients, resamples=200, seed=11
        )
        self.assertEqual(first, second)
        self.assertEqual(first["patients"], 2)
        self.assertEqual(first["images"], 4)
        self.assertAlmostEqual(first["difference"], 2.0)

    def test_lsh_band_partition_guarantees_candidate_for_twelve_bit_changes(self):
        reference = bytes(32)
        changed_value = 0
        # Put one changed bit in each of twelve different 13-band partitions.
        widths = [20] * 8 + [19] * 5
        offset = 0
        for width in widths[:12]:
            changed_value |= 1 << (254 - offset)
            offset += width
        changed = (changed_value << 1).to_bytes(32, "big")
        left = lsh_bands(reference)
        right = lsh_bands(changed)
        self.assertTrue(any(a == b for a, b in zip(left, right)))

    def test_near_duplicate_remediation_excludes_whole_lower_priority_patient(self):
        records = [
            {"patient_id": "train-a", "split": "train", "relative_path": "a-left.jpeg"},
            {"patient_id": "train-a", "split": "train", "relative_path": "a-right.jpeg"},
            {"patient_id": "val-b", "split": "validation", "relative_path": "b-left.jpeg"},
            {"patient_id": "val-b", "split": "validation", "relative_path": "b-right.jpeg"},
            {"patient_id": "test-c", "split": "internal_test", "relative_path": "c.jpeg"},
        ]
        audit = {
            "status": "fail",
            "pass": False,
            "confirmed_cross_split_near_duplicates": 1,
            "confirmed": [
                {
                    "left_patient": "train-a",
                    "left_split": "train",
                    "right_patient": "val-b",
                    "right_split": "validation",
                }
            ],
        }
        retained, report = remediate_records(records, audit)
        self.assertEqual([row["patient_id"] for row in retained], ["train-a", "train-a", "test-c"])
        self.assertEqual(report["excluded_patients"], 1)
        self.assertEqual(report["excluded_images"], 2)
        self.assertTrue(report["training_records_unchanged"])
        self.assertEqual(report["patients_reassigned"], 0)

    def test_embedding_run_loader_verifies_completed_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "embeddings_000000_000001.pt"
            torch.save(
                {
                    "checkpoint_hash": "checkpoint",
                    "representation": "final_mean_patch",
                    "relative_paths": ["train/1_left.jpeg", "train/1_right.jpeg"],
                    "patient_ids": ["1", "1"],
                    "splits": ["train", "train"],
                    "embeddings": torch.arange(8, dtype=torch.float32).reshape(2, 4),
                },
                shard,
            )
            evidence = {
                "status": "complete",
                "failures": 0,
                "images": 2,
                "checkpoint_hash": "checkpoint",
                "representation": "final_mean_patch",
                "shards": [
                    {
                        "path": str(shard),
                        "images": 2,
                        "bytes": shard.stat().st_size,
                        "sha256": sha256_path(shard),
                    }
                ],
            }
            (root / "extraction_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
            run = load_embedding_run(root, expected_split="train")
            self.assertEqual(tuple(run.embeddings.shape), (2, 4))
            self.assertEqual(run.patient_ids, ["1", "1"])

    def test_embedding_run_loader_rejects_nonfinite_tensor_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "embeddings_000000_000000.pt"
            torch.save(
                {
                    "checkpoint_hash": "checkpoint",
                    "representation": "final_mean_patch",
                    "relative_paths": ["train/1_left.jpeg"],
                    "patient_ids": ["1"],
                    "splits": ["train"],
                    "embeddings": torch.tensor([[float("nan"), 1.0]]),
                },
                shard,
            )
            evidence = {
                "status": "complete",
                "failures": 0,
                "images": 1,
                "checkpoint_hash": "checkpoint",
                "representation": "final_mean_patch",
                "shards": [
                    {
                        "path": str(shard),
                        "images": 1,
                        "bytes": shard.stat().st_size,
                        "sha256": sha256_path(shard),
                    }
                ],
            }
            (root / "extraction_evidence.json").write_text(
                json.dumps(evidence), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                load_embedding_run(root, expected_split="train")

    def test_publication_trainer_preserves_topk_budget(self):
        torch.manual_seed(4)
        model = TopKSAE(8, 24, 3)
        train = torch.randn(32, 8)
        validation = torch.randn(12, 8)
        history, _ = train_sae_batched(
            model,
            train,
            validation,
            epochs=2,
            batch_size=8,
            evaluation_batch_size=6,
            learning_rate=1e-3,
            weight_decay=1e-5,
            seed=4106,
            device="cpu",
            resample_dead=True,
            resample_warmup_epochs=1,
            resample_interval_epochs=1,
            resample_calibration_size=16,
        )
        self.assertEqual(len(history), 2)
        self.assertLessEqual(history[-1]["validation_max_active_features"], 3)

    def test_smoke_encoder_is_frozen_and_deterministic(self):
        model = DeterministicSmokeEncoder(16, 4, 9)
        x = torch.rand(2, 3, 32, 32)
        self.assertTrue(torch.equal(model(x), model(x)))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_retfound_preprocessing_is_finite_and_channel_standardized(self):
        image = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32)
        normalized = normalize_retfound_cfp(image)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.allclose(normalized.mean(dim=(-2, -1)), torch.zeros(3), atol=1e-5))
        self.assertTrue(torch.allclose(normalized.std(dim=(-2, -1), correction=0), torch.ones(3), atol=1e-5))

    def test_retfound_preprocessing_handles_constant_black_image(self):
        output = preprocess_retfound_cfp(Image.new("RGB", (64, 64), color=0))
        self.assertEqual(tuple(output.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(output).all())

    def test_retfound_loader_fails_closed_without_checkpoint(self):
        with self.assertRaises(RETFoundUnavailableError):
            load_retfound_cfp("/definitely/missing/checkpoint.pth", "/definitely/missing/repo")

    def test_eyepacs_identity_and_patient_split(self):
        self.assertEqual(parse_eyepacs_identity("123_left.jpeg"), ("123", "left"))
        left = stable_patient_split("123", seed=17, train_fraction=0.7, validation_fraction=0.2)
        right = stable_patient_split("123", seed=17, train_fraction=0.7, validation_fraction=0.2)
        self.assertEqual(left, right)

    def test_vlm_diagnostics_are_deterministic_and_patient_clustered(self):
        targets = torch.tensor([0, 1, 0, 1]).numpy()
        scores = torch.tensor([0.05, 0.95, 0.05, 0.95]).numpy()
        self.assertAlmostEqual(ece(targets, scores), 0.05, places=6)
        bundle = metric_bundle(targets, scores)
        self.assertAlmostEqual(bundle["auroc"], 1.0)
        first = bootstrap_ci(targets, scores, ["a", "a", "b", "b"], seed=5, resamples=100)
        second = bootstrap_ci(targets, scores, ["a", "a", "b", "b"], seed=5, resamples=100)
        self.assertEqual(first, second)
        self.assertEqual(first["patients"], 2)

    def test_random_feature_cards_match_count_and_are_reproducible(self):
        cards = {1: {"hypothesis": "a"}, 2: {"hypothesis": "b"}, 3: {"hypothesis": "c"}, 4: {"hypothesis": "d"}}
        active = select_context_cards([1, 2, 99], cards, "active", 4106, "train/x.jpeg")
        random_first = select_context_cards([1, 2, 99], cards, "random", 4106, "train/x.jpeg")
        random_second = select_context_cards([1, 2, 99], cards, "random", 4106, "train/x.jpeg")
        self.assertEqual(len(active), len(random_first))
        self.assertEqual(random_first, random_second)
        self.assertTrue(all(card["feature_id"] not in {1, 2} for card in random_first))

    def test_clinician_agreement_statistics(self):
        ratings = [[0, 0, 1], [2, 2, 2], [3, 3, 3]]
        self.assertAlmostEqual(pairwise_exact_agreement(ratings), 7 / 9)
        self.assertIsNotNone(fleiss_kappa(ratings))

    def test_registry_schema_initializes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.sqlite"
            initialize_registry(path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
