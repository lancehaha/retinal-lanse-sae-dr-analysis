# Experiment 5C results: ten-round extended feature rescue

Experiment date: 2026-09-11; record verified and updated: 2026-09-12  
Primary artifact: `/Volumes/Elements/Retina-LanSE/experiments/experiment_5c_extended_feature_rescue_10rounds_20260911/extended_feature_rescue_10rounds_summary.json`

## Outcome

All five requested cards completed ten hypothesis-generation and validation
rounds. All 50 generation responses were parseable, and every card met the
predeclared stability rule of at least 8/10 passing validation rounds.

| Feature ID | Passing rounds | Mean high rate | Mean low rate | Mean contrast | Status |
|---:|---:|---:|---:|---:|---|
| 9180 | 9/10 | 0.800 | 0.175 | 0.625 | Stable after 10 rounds |
| 3322 | 10/10 | 0.975 | 0.022 | 0.953 | Stable after 10 rounds |
| 7388 | 10/10 | 0.925 | 0.066 | 0.859 | Stable after 10 rounds |
| 15184 | 9/10 | 0.800 | 0.175 | 0.625 | Stable after 10 rounds |
| 12428 | 9/10 | 0.850 | 0.131 | 0.719 | Stable after 10 rounds |

The repeated Qwen hypotheses were approximately:

- 9180: more pronounced retinal blood vessels in high-activation images.
- 3322 and 7388: more pronounced central retinal artery / vessel dilation.
- 15184 and 12428: more pronounced retinal hemorrhages.

Two exact cross-card hypothesis groups were detected: 3322/7388 and
15184/12428. This is an audit finding, not evidence that the cards are
biologically identical. The prompts were isolated, so no card could copy
another card's hypothesis during generation.

## Provenance and integrity

- Per-round records:
  `/Volumes/Elements/Retina-LanSE/experiments/experiment_5c_extended_feature_rescue_10rounds_20260911/extended_feature_rescue_10rounds.jsonl`.
- Cards requested/processed: 5/5.
- Rounds and generations: 50/50 complete and parseable.
- Labels in prompts: false.
- Cross-card hypothesis context: false.
- Stable cards: 5; not stable: 0.
- Checkpoint SHA-256:
  `03a7e71e63eb82d26239b0b625ddb54b9b2068fc7794df790a895e1a74bb2cc7`.
- Normalization SHA-256:
  `6349d0599e31c31afda32b19d2951d4c194eb4b4818c3efe5fbf308a0c87ca06`.
- Feature-card SHA-256:
  `b276e2b7455a5efd479b46b4798102242cf20d702fc71bfc74ec53f1f6f6c77a`.
- Heavy outputs are stored on `/Volumes/Elements/Retina-LanSE`.
- Per-card JSONL SHA-256: `a105485ba9dc788f62b1df03d6e0d234a12f9ffd486439b52c51452315c717a5`.
- Summary JSON SHA-256: `fefd3baa43ce3493923b68fbc9b5163201d83eb02e837dec1d8608bdb08c71d7`.
- Runner SHA-256: `8c3e9ec888641f54f0f3ec476a0372325a8b7c339c96042cb2c8fa647fecf8af`.

## Interpretation

The extended search rescued all five cards under the operational stability
criterion, showing that the original two-round failures were not stable under
this longer same-model search. This does not make the hypotheses clinically
validated: repeated model-generated hypotheses and judgments can still be
circular, and two pairs produced identical descriptions. Independent
ophthalmologist review and lesion-grounded validation remain necessary.
