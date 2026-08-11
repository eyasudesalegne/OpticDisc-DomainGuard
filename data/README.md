# Data preparation

Raw medical images are not distributed in this repository.

## Domains used in the manuscript

| Domain | Images | GON+ | GON- | Grouping used in study |
|---|---:|---:|---:|---|
| HYGD | 747 | 548 | 199 | 288 case IDs; grouped source validation |
| ACRIMA | 703 | 394 | 309 | image-level external evaluation in adapted metadata |
| G1020 | 1,020 | 296 | 724 | image-level external evaluation; anatomical audit support |

## Manifest

Create `data/manifests/domainguard_manifest.csv` with at least:

```text
path,label,domain,group_id
/path/to/image.jpg,1,HYGD,case_001
/path/to/image.jpg,0,ACRIMA,acrima_0001
/path/to/image.jpg,1,G1020,g1020_0001
```

`label` is binary (0/1). `domain` must be one of `HYGD`, `ACRIMA`, or `G1020` for manuscript reproduction. `group_id` must preserve HYGD case identity so images from one case cannot leak across source folds.

## Strict external-transfer rule

For zero-shot external evaluation, ACRIMA/G1020 labels are evaluation-only. Do not use target data to fit scalers, tune hyperparameters, calibrate probabilities, select thresholds, refit the classifier, or select models. Labelled target support is permitted only in experiments explicitly reported as few-shot adaptation.

## Historical Drive layout

The recovered experiment reports reference:

```text
/content/drive/MyDrive/glaucoma_q1_hygd_experiment/
```

and a G1020 annotation-aware phase under:

```text
/content/drive/MyDrive/glaucoma_q1_hygd_experiment/cross_domain_phase2_g1020_annotation/
```

The historical phase reported detection of a G1020/ORIGA annotation table and vCDR values for a subset, but no reliable bounding-box schema in that scan. Do not treat that automatic bbox attempt as validated anatomy.
