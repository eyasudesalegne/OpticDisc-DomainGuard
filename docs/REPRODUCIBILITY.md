# Reproducibility and provenance

## Why this repository is consolidated

The final manuscript was produced from a staged experimental workflow rather than a single immutable notebook. The recovered May 16 cross-domain run reports completion of metadata auditing, preprocessing audit panels, ResNet-50 embedding extraction/cache reuse, handcrafted feature extraction/cache reuse, feature assembly, grouped internal validation, pairwise source-only/reverse transfer, leave-one-domain-out generalization, CORAL/domain-standardization ablation, domain-classifier auditing, few-shot adaptation/calibration, domain-risk gating, anatomical plausibility scanning, and manuscript-ready tables/figures.

A subsequent G1020 annotation-aware phase reports disc-aware crop generation, ResNet-50 embedding extraction, source-only transfer, leave-one-domain-out validation, PCA-CORAL, few-shot adaptation, pairwise domain classification, risk gating, vCDR support where available, and export of tables/figures. Later manuscript revisions add modern frozen-backbone sensitivity and end-to-end fine-tuning experiments.

Accordingly, this GitHub repository is a cleaned implementation of the **final scientific workflow**. It does not claim that a newly written notebook is a byte-for-byte copy of an unrecovered historical Colab file.

## Reproduction levels

### Level A — protocol reproduction

Reproduce the scientific design from raw datasets: grouped HYGD source validation, source-only threshold selection, fixed ACRIMA/G1020 transfer, domain-separability auditing, adaptation experiments, selective prediction, and anatomical review.

### Level B — numerical reproduction

Requires the exact adapted manifests, preprocessing choices, cached representations, package versions, seeds, and model settings used by the historical runs. Historical Drive artifacts should be retained separately from this public repository where dataset/provider restrictions apply.

### Level C — archival notebook reproduction

Requires the exact original `.ipynb`. Drive search did not expose an identifiable Colab file under the DomainGuard/OpticDisc names, while generated run reports clearly document the staged workflow. Therefore this repository does not mislabel reconstructed code as the untouched original notebook.

## Zero-shot versus adaptation

This distinction is mandatory:

- **Fixed external transfer:** target labels are used only after predictions are generated, for evaluation.
- **Few-shot adaptation:** a declared labelled target support set may be used for adaptation/calibration; evaluation must remain separate.

Never merge these regimes in one headline result.

## Historical output paths

The recovered cross-domain report referenced outputs including:

```text
results/metrics/cross_domain_internal_group_cv_metrics.csv
results/metrics/cross_domain_source_only_pairwise_metrics.csv
results/metrics/cross_domain_leave_one_domain_out_metrics.csv
results/metrics/cross_domain_harmonization_ablation_metrics.csv
results/metrics/cross_domain_domain_classifier_audit.csv
results/metrics/cross_domain_fewshot_adaptation_metrics.csv
results/metrics/cross_domain_domain_risk_gate_metrics.csv
```

These filenames are useful provenance anchors but are not committed here unless their redistribution is appropriate and their values have been verified against the submitted manuscript.
