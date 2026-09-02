# Structured research topic

## Title

Retina-LanSE: Spatially Grounded Sparse Feature Discovery in Retinal Foundation Models

## Keywords

retinal foundation model; sparse autoencoder; mechanistic interpretability; diabetic retinopathy; lesion localization; external validation; RETFound

## TL;DR

Train global and tile-level Top-K sparse autoencoders on frozen RETFound-CFP representations, then test whether individual sparse features reproducibly associate with and spatially localize predefined diabetic-retinopathy lesions across independent datasets, devices, perturbations, and random seeds.

## Abstract-level scope

The study audits representations rather than building a diagnostic device. Sparse features are learned without lesion labels on a large fundus corpus. Only after model selection and freezing are predefined lesion labels and masks used for hypothesis testing. Evidence combines image-level association, pixel/tile localization, lesion-versus-control occlusion, cross-dataset transfer, confounder tests, and blinded ophthalmologist review.

## Research question

Can sparse autoencoders decompose frozen RETFound-CFP embeddings into stable features that correspond to clinically recognizable DR lesions and localize those lesions more faithfully than matched sparse and attribution baselines?

## Primary hypothesis

At a matched activation budget, a dual-scale Retina-LanSE will provide better reconstruction fidelity than simple sparse baselines and will yield a preregistered subset of features with reproducible lesion association, localization, and intervention evidence on external datasets.

## Success criterion

Success requires all three evidence layers: representation fidelity, external lesion association/localization after multiplicity correction, and blinded expert agreement. Reconstruction alone is insufficient.

## Failure condition

The central claim fails if lesion-associated features do not generalize across external datasets/seeds, if localization is no better than matched spatial controls, or if activations are explained primarily by camera, image quality, borders, or optic-disc brightness.

## Intended venue bar

Medical-imaging or trustworthy-ML workshop/conference paper first. A clinical journal claim requires institutional data governance and prospective or substantially broader external validation.

