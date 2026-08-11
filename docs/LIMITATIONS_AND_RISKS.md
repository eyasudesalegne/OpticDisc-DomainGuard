# Limitations and risks

1. **Retrospective public-dataset evaluation.** Results do not establish prospective clinical performance or deployment readiness.
2. **Domain shift is multifactorial.** Camera characteristics, field of view, illumination, population, label protocols, preprocessing, and dataset curation can all contribute.
3. **Dataset identity is diagnostic evidence, not causal attribution.** High domain-classification accuracy shows retained origin information but does not by itself identify the causal source of transfer failure.
4. **Few-shot results are not zero-shot results.** Once labelled target examples are used, the experiment is adaptation and must be reported separately.
5. **Saliency is not causal explanation.** Grad-CAM or related attribution should be treated as an anatomical audit, not proof that a model reasons clinically.
6. **G1020 annotation recovery requires care.** The historical automatic scan reported vCDR support for a subset but did not validate a reliable bbox schema. Any mask/bbox-derived experiment should verify the provider annotation format independently.
7. **Image-level external evaluation.** In the adapted metadata used for the manuscript, verified patient identifiers were not available for ACRIMA; external analyses therefore do not imply patient-level independent sampling.
8. **Reconstruction provenance.** The public repository is a clean implementation based on the manuscript and recovered experiment reports. It is not represented as an untouched archival copy of every historical Colab cell.

## Clinical-use boundary

OpticDisc-DomainGuard is research software for reliability auditing. It must not be used to diagnose glaucoma, triage patients, or make treatment decisions.
