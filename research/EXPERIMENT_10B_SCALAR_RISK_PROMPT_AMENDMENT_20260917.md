# Experiment 10B: Scalar-Risk Qwen Prompt Amendment

Experiment 10's five-image gate showed that Qwen2.5-VL:3B can process the retinal image and patch contact sheet, but card-context prompts caused it to copy the five-probability JSON placeholders as all zeros. Those invalid distributions prevented AUROC calculation. The original outputs remain preserved and are not combined with this amended experiment.

Experiment 10B freezes the same Experiment 8 162-image/81-patient pilot holdout, 34 failed-but-parsed Patch-SAE cards, train-only association summaries, eight-event selection, dense MLP benchmark, prompt arms, model, temperature, shuffle seed, and 2,000 patient-cluster bootstrap analysis. The only change is Qwen's response contract:

```json
{"predicted_grade": 0, "referable_risk_0_to_100": 0, "visible_findings": [], "uncertainty": "low", "image_quality": "adequate"}
```

The prompt names the fields but does not contain numerical placeholder values. `referable_risk_0_to_100 / 100` is the primary AUROC score. `predicted_grade` supplies the secondary five-grade QWK, macro-F1, and ordinal MAE. A five-image, four-arm gate must have complete valid JSON before the full run starts. This remains an exploratory, previously accessed pilot—not a clinical validation or a claim of validated lesion semantics.
