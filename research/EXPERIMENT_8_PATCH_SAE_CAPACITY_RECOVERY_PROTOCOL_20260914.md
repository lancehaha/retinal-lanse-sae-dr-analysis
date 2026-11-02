# Experiment 8A: Patch-SAE capacity recovery

## Motivation

The original Experiment 8 patch SAE used Top-K 16 per 14 x 14 RETFound patch. It completed label-free training but failed its reconstruction gate: validation explained variance was 0.7874, below the predeclared 0.85 minimum. Dead-feature fraction (0.0486) and exact Top-K compliance passed, so the bottleneck is representational capacity rather than collapse.

## Prespecified recovery sequence

The original K=16 result remains immutable. Train the same SAE configuration with Top-K 32. If and only if K=32 still fails the explained-variance gate, train K=64. Do not train K=1024 per patch: that would create up to 200,704 active entries per image and defeat the local sparse-token motivation.

Everything besides Top-K remains fixed: 1,024-dimensional patch input, 16,384 dictionary features, seed 4106, same 1,600/200 patient split, train-only normalization, AdamW, learning rate 3e-4, weight decay 1e-5, batch size 1,024, 20 epochs, decoder-column normalization, deterministic dead-feature resampling, and no labels in SAE training.

## Gate and downstream use

For each capacity, require explained variance >=0.85, dead-feature fraction <=0.25, and max active features <= requested Top-K. Select the first capacity passing all gates. Only that capacity may enter the frozen downstream Transformer and report stages. If K=64 fails, record the capacity recovery as failed and do not run a clinical prediction experiment under this protocol.
