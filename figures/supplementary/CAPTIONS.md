# Supplementary figure captions

Supplementary Figure SF1. Out-of-fold classification curves. Receiver operating characteristic and precision-recall curves for the three-seed fine-tuned ensembles and the seven archived single-run baselines, pooled over fixed out-of-fold predictions. These are not untouched independent tests, and no threshold was chosen from these curves.

Supplementary Figure SF2. Probability reliability on the out-of-fold cohorts. Ten fixed probability bins; bins with fewer than 20 records are not drawn. Brier scores and all confusion counts are in Supplementary Table S5. This is a descriptive check, not a recalibration.

Supplementary Figure SF3. Validation trajectories of all 54 activity fits. Each line is one candidate and seed. Colors show the learning rate, with the values in each panel legend (for LegNet, the maximum rate of its OneCycle schedule). The two near-constant high-learning-rate DNABERT-2 runs are included. Stopping rules differ by model, so the curves do not show exhaustive convergence.

Supplementary Figure SF4. Candidate grids and validation-only choices. Means and sample SD across three seeds; dashed vertical lines mark the selected recipes. The wide DNABERT-2 spread at the highest learning rate includes the near-constant fits, and no fit was removed. Choices at the edge of a grid are shown.

Supplementary Figure SF5. Recorded fitting time. Sum over the nine candidate and seed fits per model and cell. Run time depends on hardware and implementation and excludes the cost of pretraining. The same number of candidates does not mean equal GPU time, steps or operations.

Supplementary Figure SF6. Prediction-interval coverage of previously frozen models. Marginal nominal 90% intervals under a disjoint calibration and audit design. These are exploratory results for previously frozen models, not a coverage guarantee for the selected models. Subgroup coverage and counts are in Supplementary Table S13.

Supplementary Figure SF7. Near-constant fits and the predefined flag. The predefined undefined-Pearson flag marks no fit. A later descriptive screen (prediction SD below 1e-4) finds two near-constant fits, one per cell, for DNABERT-2 at learning rate 5e-5 with seed 20260932. All 54 fits and the three-seed selection are unchanged, and no cause is inferred.

Supplementary Figure SF8. Exploratory diagnostics with previously frozen models. A: mean signed error (prediction minus measured activity) of retained test-chromosome elements in low, middle and high measured-activity bands; high-activity elements are underpredicted by both convolutional models in both cells. B: difference in MAE between elements removed by the overlap screen and retained elements, with 95% bootstrap intervals over 1 kb genomic groups. These analyses use previously frozen models and populations and are exploratory (Supplementary Tables S11 and S14).
