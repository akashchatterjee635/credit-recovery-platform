# Model card: credit-risk model

## Intended use

This model estimates default risk for research and shadow-mode evaluation. It is
not an approval engine and must not be used as the sole basis for lending,
pricing, adverse action, or borrower eligibility.

## Evaluation contract

Every release must report overall and subgroup sample size, prevalence, ROC-AUC,
PR-AUC, Brier score, expected calibration error, reliability curves, TPR, FPR,
precision, and recall at the registered operating threshold. Applicant-level 95%
bootstrap confidence intervals and the bootstrap seed are required. Groups below
the configured minimum size carry an uncertainty warning and must not be ranked.

## Stability

Release evaluation covers random seeds, train/calibration splits, thresholds,
small feature perturbations, missing features, income/debt shocks, retraining,
constraint updates, and distribution drift. Governance thresholds are declared
before final evaluation in `configs/governance_thresholds.yaml`.

## Known limitations and unacceptable uses

Historical data can encode past inequity and dataset shift. Calibration does not
prove causal validity. Synthetic temporal data cannot support predictive claims.
Do not expose protected attributes to the production scoring feature set merely
because they are used for fairness auditing. Do not deploy when artifact lineage,
subgroup uncertainty, or temporal availability cannot be verified.
