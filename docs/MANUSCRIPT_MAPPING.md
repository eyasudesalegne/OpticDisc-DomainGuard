# Manuscript-to-code mapping

This document maps the manuscript's eight-module OpticDisc-DomainGuard design to the cleaned repository implementation.

| Manuscript module | Repository location | Scientific boundary |
|---|---|---|
| Optic-disc-aware preprocessing / features | `configs/default.yaml`, forthcoming preprocessing/feature modules | Anatomical branches must be explicitly identified |
| Grouped source validation | `scripts/03_internal_validation.py` | HYGD grouping prevents case leakage |
| Fixed external transfer | `scripts/04_external_transfer.py` | No target fitting, calibration, threshold tuning, or model selection |
| Dataset-identity audit | `scripts/05_domain_identity_audit.py`, `src/domainguard/domain_audit.py` | Measures retained domain-origin signal |
| Threshold transfer | `src/domainguard/metrics.py` and external-transfer stage | Threshold selected on source only |
| Harmonization / few-shot adaptation | `scripts/07_fewshot_adaptation.py`, `src/domainguard/adaptation.py` | Must be reported separately from zero-shot transfer |
| Domain-risk gating | `scripts/08_risk_gating.py`, `src/domainguard/risk_gating.py` | Reports coverage-performance trade-off |
| Modern-backbone sensitivity | configuration + dedicated extension | Tests whether transfer failure persists with stronger representations |

## Manuscript anchor results

The final manuscript reports the best ResNet-50/logistic-regression source pipeline at AUROC **0.986** under grouped HYGD validation and **0.639 / 0.486** under fixed transfer to ACRIMA / G1020. These values are manuscript reference points, not hard-coded expected outputs. Reproduction scripts should calculate metrics from predictions rather than force agreement.

## Dataset inventory

- HYGD: 747 images; 548 GON+, 199 GON−; 288 case IDs.
- ACRIMA: 703 images; 394 GON+, 309 GON−.
- G1020: 1,020 images; 296 GON+, 724 GON−.

## Interpretation

The framework is an audit, not a model leaderboard. A stronger internal AUROC does not override failed external transfer, unstable operating thresholds, high domain separability, poor calibration, or anatomically implausible attribution.
