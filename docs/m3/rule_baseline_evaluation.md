# M3 Rule Baseline Evaluation Protocol

## Status

This document defines the evaluation procedure for the experimental rule-based pair compatibility baseline. It does not report completed evaluation results or approve the current scoring weights.

## 1. Purpose

Compare the interpretable rule baseline with the logistic-regression compatibility model on the same eligible evaluation examples.

The rule baseline provides a transparent reference that can be inspected alongside the learned model.

## 2. Fair Comparison Requirements

- Use the same eligible item pairs and evaluation split for both models.
- Preserve the official split identities and keep the test split locked until the project owners authorize its use.
- Fit any learned model using training data only.
- Do not use unavailable annotations as negative labels.
- Record excluded pairs, their exclusion reasons, and the final number of evaluated pairs.
- Record dataset, feature-cache, model, and configuration versions.
- Do not compare results calculated on different populations as if they were directly comparable.

## 3. Score Interpretation

The current rule baseline returns an experimental heuristic score in the range [0, 1]. It is not a calibrated probability.

The current weights, CLIP score mapping, color-distance transformation, and behavior when color information is unavailable are provisional. They require review by the responsible owners before being adopted as shared project decisions.

The draft `PairwiseScore` field named `probability` must not be populated with the heuristic score as though it were a probability unless the owners explicitly approve that contract decision.

## 4. Evaluation Metrics

Select metrics according to the availability and meaning of the evaluation labels.

Where valid binary pair-compatibility labels are available, consider ROC-AUC and average precision for comparing score discrimination. Report the eligible sample count and class distribution alongside these metrics.

Only calculate metrics supported by the evaluation data and label definitions. Record the exact metric definitions and positive-class meaning. Do not report unmeasured results or introduce an unapproved acceptance threshold.

If ranking-level evaluation is required, document its relevance labels, candidate-generation policy, and approved ranking metrics separately. Pairwise metrics alone do not establish outfit-ranking quality.

## 5. Missing Information and Exclusions

Record missing features separately from negative compatibility labels. For each model, document how unavailable feature values are handled.

For the rule baseline, the current implementation uses the CLIP-derived score when color information is unavailable. This behavior is experimental and must be reviewed before contract freeze.

Document invalid pairs, unresolved item IDs, missing annotations, and other exclusions. Apply the same eligibility policy to both models.

## 6. Results Record

For each evaluated model, record:

- Model name and version.
- Dataset and evaluation split identifiers.
- Number of eligible pairs and number excluded, with reasons.
- Positive and negative label counts, where applicable.
- Metric names, definitions, and measured values.
- Feature-cache and configuration versions.
- Any limitations that affect interpretation.

Do not fill in metric values until the evaluation has actually been run. Do not describe the rule baseline as better than the learned model without comparable results.

## 7. Contract and Approval Status

The M3-to-M4 contract remains a draft until the responsible owners agree on pair-score semantics, score range, OCS aggregation, structural validity, Top-K behavior, tie-breaking, missing-item handling, and provenance.

The current rule weights and missing-color behavior are experimental. Record the relevant owner decisions before treating them as approved project requirements.

## 8. Current Completion Status

The rule-baseline implementation and its unit and integration tests have been completed locally. Comparative evaluation against logistic regression, measured performance results, and final contract approval remain pending.