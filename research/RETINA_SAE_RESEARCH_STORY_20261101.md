# Retinal SAE research: question, design, and findings

## The question

We wanted to know whether a sparse autoencoder (SAE) could make retinal-image diabetic-retinopathy (DR) prediction more accurate, more efficient with limited labels, and easier to explain. The images carry **DR grades 0–4**. The main binary task is **referable DR, grade ≥2**. These labels do not directly diagnose whether a person has diabetes.

The intended pipeline was: fundus image → frozen RETFound representation → label-free SAE features → supervised DR prediction, with an AutoResearch process that proposes and checks visual meanings for prominent features → evidence-based explanation. The key scientific question was whether the SAE adds useful information beyond the original RETFound representation, and whether feature explanations correspond to visible, reliable retinal evidence.

## What each component did

| Component | Role |
|---|---|
| RETFound | Frozen image encoder producing a 1,024-value dense vector, or 14×14 patch tokens in the local pilot. |
| SAE | Trained without DR labels to reconstruct those representations using a limited number of active dictionary features. Global and later patch/tile SAEs were separate models. |
| Prediction head | Logistic regression, MLP, or Transformer trained with EyePACS DR labels. The frozen encoder and SAE were not jointly trained for DR. |
| Feature cards / AutoResearch | Samples images or patches that strongly activate a feature, proposes a visual hypothesis with Qwen, checks it on fresh high- and low-activation examples, and audits training-only grade associations. |
| Qwen prediction / report | Separate exploratory attempt to use images, patches, cards, and grade-association evidence to predict grades or explain a frozen score. Qwen was not the numerical classifier. |

## Design progression

We began with patient-separated EyePACS cohorts and a clean 102-image IDRiD external grading test. The global SAE was trained on 71,134 label-free EyePACS training embeddings. Downstream supervised comparisons used the same frozen representation and matched patient splits. The main internal measure was AUROC: 0.5 is chance ranking and 1.0 is perfect ranking; five-grade quadratic weighted kappa (QWK) was a secondary measure of ordinal agreement. External IDRiD results were kept separate because its image distribution and cohort are different.

| Phase | Question and method | Main observed result |
|---|---|---|
| Dense and compact SAE fusion (Experiments 1–2) | Does a 64-feature SAE code add to the dense vector? | EyePACS test AUROC: dense **0.8397**, SAE-only **0.7286**, fusion **0.8388**. No fusion gain. |
| Label efficiency and external test (3) | Does fusion help with fewer labels or transfer to IDRiD? | At 10% labels: dense **0.8203**, fusion **0.8200**. At 100% labels on IDRiD: **0.8067** versus **0.8087**; the small difference was not established as a gain. |
| SAE activation capacity (4) | Would genuinely allowing 32 through 1,024 active features help? | Reconstruction explained variance rose from **0.9112** to **0.9901**, but K=1,024 fusion AUROC was **0.8341** against a matched dense **0.8408**. Better reconstruction did not yield better DR ranking. |
| Active values only (5) | What if we append the strongest 1,024 nonzero magnitudes, without feature IDs? | Dense **0.8402**, active-value-only **0.5506**, fusion **0.8396** AUROC. Magnitudes alone lost useful feature identity. |
| Feature-card evidence (6) | Can verified association summaries or LLM card text directly predict DR? | Dense **0.8402**; direct statistical card rule **0.6875**; learned identity-preserving card probe **0.7954**; Qwen with real descriptions **0.5407**. Real descriptions did not beat shuffled descriptions (**0.5557**). |
| Supervised identity/rank fusion (7) | Can a Transformer learn from dense values, feature IDs, strengths, and ranks? | Internal AUROC **0.8450** versus dense **0.8402**; gain was too small to establish primary binary success. Five-grade QWK improved **0.4311 → 0.4962** internally, but transfer was not established. |
| Patch SAE and grounded report pilot (8–9) | Can local patch features support a spatial Transformer and lesion-grounded explanation? | On 162 pilot images from 81 patients, dense MLP AUROC **0.794**, full patch model **0.780**. The patch card and lesion-grounding gates did not validate feature meanings. |
| Image, patch, and card Qwen pilot (10) | Does showing Qwen the real image and active patches/cards help? | On the same 162-image pilot, dense MLP **0.794**; image-only Qwen **0.483**; image+patches **0.392**; image+patches+real cards **0.447**; image+cards without patches **0.442**. These were exploratory and below the dense baseline. |
| Higher-resolution tile grounding (11) | Do smaller retinal tiles yield lesion-correlated SAE features? | Label-free tile SAE reconstruction passed its quality gate, but **0** features met the prespecified lesion-grounding criteria. Downstream grounded reports were therefore not supported. |

The AUROCs above come from different model heads and cohorts where indicated. Compare arms within each phase; the pilot numbers are not directly interchangeable with the larger EyePACS test.

## What AutoResearch actually did

The feature-analysis loop was intended to go beyond selecting features that appear only in DR images. A feature may occur in both normal and DR images yet still contribute useful evidence in combination. We therefore sampled high- and low-activation examples, asked Qwen to hypothesize what a feature might visually represent, checked the hypothesis on unseen contrastive examples, and measured grade association across training images. Failed or incomplete judgments remained explicit. A later retry tested five troublesome cards with further bounded hypothesis rounds. These checks measure consistency of the same model's descriptions; they do **not** verify lesion identity. Independent specialist annotation and lesion masks are needed for that stronger claim.

## Interpretation

The dense RETFound representation remained the reliable DR-prediction baseline. SAE features carried some grade information, especially when their identities were preserved, but the current global fusion, direct card rule, and Qwen prompts did not show a robust binary-prediction improvement. The supervised Transformer gave a promising internal five-grade QWK signal, while the patch and tile programs exposed a larger bottleneck: feature meanings and retinal lesion grounding were not established. Consequently, the present evidence supports a research program in representation analysis and explainability, not a validated lesion-grounded diagnostic system or clinical deployment claim.

## Next testable direction

Separate **prediction** from **explanation validity**. For prediction, compare a frozen dense baseline against a prespecified card-plus-grade-statistics and lesion-knowledge prompt on the same patient-disjoint cohort, with real, shuffled, and no-card controls. For explanations, require retinal-specialist or mask-grounded validation of cited local features and a deletion/control test showing that cited evidence changes the frozen model's score more than random active evidence. Freeze prompts, thresholds, and success criteria before opening a new holdout.

## Source records

The numerical values are transcribed from the project result records under `research/` for Experiments 2–7 and the external-drive JSON results for Experiments 8, 10B, 10C, and 11. This document summarizes the experimental story; the per-experiment protocols and result files retain the detailed cohort definitions, confidence intervals, seeds, and failure audits.
