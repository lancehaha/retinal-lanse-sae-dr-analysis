# Label-agnostic SAE feature atlas run

Status: complete for computational feature indexing; bounded local language-grounding pilots complete.

The run indexed all 71,134 EyePACS training embeddings with the frozen RETFound final-mean-patch representation and the fixed 16,384-feature Top-K-32 SAE. Feature prominence was defined before clinical comparison as descending global activation mass, subject to at least 50 activations. This produced 6,763 eligible features and 512 feature cards, each containing 32 high-activation image paths and activation values. DR labels were attached only as separate audit fields; labels did not affect feature ranking or discovery.

The run recorded 2,276,288 nonzero activation entries across 35,567 patients. Of the indexed images, 28,130 had EyePACS clinical labels and 5,527 were referable-DR positive (grade >= 2). Most feature cards remain `hypothesis_status=pending_llm_or_clinician_review`; a prespecified 16-card subset has local VLM hypotheses in `/Volumes/Elements/Retina-LanSE/experiments/feature_grounding_qwen25vl_expanded16/`, with no claim of clinical semantic validity.

Machine-readable outputs are on the external drive:

- `/Volumes/Elements/Retina-LanSE/experiments/feature_atlas_autoresearch/feature_atlas.jsonl`
- `/Volumes/Elements/Retina-LanSE/experiments/feature_atlas_autoresearch/feature_atlas_summary.json`

The next stage is blinded ophthalmologist review of the 64-image packet at `/Volumes/Elements/Retina-LanSE/experiments/clinician_review_packet_qwen25vl_expanded16/`. Reviewers must score hypothesis match, specificity, and nuisance independently before any feature card is frozen or used for a locked-test evaluation.
