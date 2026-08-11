# OpticDisc-DomainGuard

**A Cross-Domain Reliability Audit for Glaucoma Classification from Fundus Images**

Research code and reproducibility scaffold accompanying the manuscript by **Eyasu Desalegne Beyene** and **Niyazi Kılıç** (Istanbul University–Cerrahpaşa).

## Overview

OpticDisc-DomainGuard is a reliability-auditing framework for fundus-based glaucoma classification. Rather than treating strong internal discrimination as evidence of deployment reliability, it explicitly separates **source-domain learning** from **fixed external transfer** and audits domain shift from complementary statistical, representation-space, calibration, adaptation, selective-prediction, and anatomical perspectives.

The manuscript uses **HYGD** as the source domain and **ACRIMA** and **G1020** as external target domains. The reported dataset inventory is 747 HYGD images (288 case IDs), 703 ACRIMA images, and 1,020 G1020 images. The best reported ResNet-50/logistic-regression source pipeline reaches AUROC 0.986 under grouped HYGD validation, but only 0.639 on ACRIMA and 0.486 on G1020 under fixed external transfer. This gap motivates the DomainGuard audit.

## Audit modules

The repository follows the eight linked modules described in the manuscript:

1. Optic-disc-aware preprocessing and feature construction
2. Grouped source-domain validation
3. Fixed external transfer without target-domain refitting, recalibration, threshold optimization, or model selection
4. Dataset-identity and embedding-space domain-separability auditing
5. Source-selected threshold-transfer auditing
6. Harmonization and few-shot target adaptation
7. Domain-risk gating / selective prediction
8. Modern-backbone sensitivity analysis

## Repository structure

```text
OpticDisc-DomainGuard/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── docs/
│   ├── REPRODUCIBILITY.md
│   ├── MANUSCRIPT_MAPPING.md
│   └── LIMITATIONS_AND_RISKS.md
├── notebooks/
│   └── README.md
├── src/
│   └── domainguard/
│       ├── __init__.py
│       ├── config.py
│       ├── metrics.py
│       ├── domain_audit.py
│       ├── adaptation.py
│       └── risk_gating.py
└── scripts/
    ├── 03_internal_validation.py
    ├── 04_external_transfer.py
    ├── 05_domain_identity_audit.py
    ├── 07_fewshot_adaptation.py
    └── 08_risk_gating.py
```

## Experimental contract

The central scientific constraint is strict separation of zero-shot external transfer from target adaptation. In fixed external validation, target labels are evaluation-only. They must not influence feature scaling, probability calibration, operating-threshold selection, classifier fitting, or model selection. Experiments that deliberately use labelled target support are identified separately as **few-shot adaptation**.

## Historical Colab workflow

The original Drive experiment was developed in staged Colab runs under:

```text
/content/drive/MyDrive/glaucoma_q1_hygd_experiment/
```

The May 16 cross-domain run generated metadata audits, ResNet-50 embeddings, handcrafted/hybrid features, grouped internal validation, source-only pairwise transfer, leave-one-domain-out evaluation, CORAL/domain-standardization experiments, dataset-identity auditing, few-shot adaptation, domain-risk gating, and manuscript-ready tables/figures. A subsequent G1020 annotation-aware phase added disc-aware crops, PCA-CORAL, vCDR support where available, and anatomical checks. Later manuscript experiments added modern frozen backbones and end-to-end fine-tuning.

This repository is therefore organized as a **clean reproducibility implementation of the manuscript workflow**, rather than presenting one historical one-cell Colab run as if it alone generated every final manuscript result.

## Data

Raw datasets are **not redistributed** here. Obtain HYGD, ACRIMA, and G1020 from their original providers and follow `data/README.md` to construct local manifests. Do not commit raw medical images, downloaded archives, patient identifiers, checkpoints, or local Drive paths.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Copy the configuration and adapt dataset paths locally:

```bash
cp configs/default.yaml configs/local.yaml
```

Then run the audit stages from `scripts/`. The scripts intentionally require explicit manifests and do not silently download or reinterpret datasets.

## Reproducibility status

The repository distinguishes three categories:

- **Manuscript-specified:** directly described in the manuscript/workflow records.
- **Recovered historical workflow:** supported by the experiment reports written by the original Colab runs.
- **Clean implementation:** engineering added here to make the workflow readable and reusable; it must not be mistaken for an untouched archival copy of the original notebook.

See `docs/REPRODUCIBILITY.md` and `docs/MANUSCRIPT_MAPPING.md`.

## Intended use

This is a retrospective research and reliability-audit framework. It is **not a clinical diagnostic device**, does not establish prospective deployment readiness, and should not be used for patient-care decisions.

## Citation

If you use this repository, please cite the associated manuscript. Bibliographic publication metadata can be updated in `CITATION.cff` once the article is published.

## License

Code is released under the MIT License. Dataset licenses and terms remain those of the original dataset providers.
