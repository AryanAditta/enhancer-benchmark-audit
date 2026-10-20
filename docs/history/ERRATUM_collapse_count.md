# Erratum and clarification: two practically collapsed DNABERT2 fits

**Reviewed: 14 October 2026.** This corrects the interpretation of the unqualified “0/54 collapsed” sentence in the original Phase 38 findings. The sealed archive, original findings/summary, frozen training source, run metadata, selection lock and numerical results are preserved.

## Correct reporting

**Two of the 54 fits showed nearly constant validation predictions (practical collapse).** Both were DNABERT2 at LR 5e-5, seed 20260932, one in each cell. Separately, **zero fits triggered the frozen undefined-Pearson flag**. These are different statements; report both rather than silently changing the definition of a failed fit.

| Cell | Best validation Pearson | Best-checkpoint prediction SD | Selected epoch | Epochs run |
| --- | --- | --- | --- | --- |
| HepG2 | 0.016084040 | 7.52709406e-06 | 1 | 8 |
| K562 | 0.028554879 | 3.53277963e-06 | 5 | 10 |

The epoch logs show extremely small prediction dispersion from epoch 1 throughout both runs, validation correlations close to zero, and normalized training loss around 1. This supports “no useful validation sequence discrimination in the recorded trajectories”; it does not establish a mechanistic root cause. The other 52 best checkpoints have prediction SDs between 0.225469 and 0.539390 in raw activity units.

## Frozen rule versus descriptive diagnosis

`analysis_tools/mpra_fairness_training.py::safe_r` returns undefined Pearson when prediction SD is below 1e-8 (or predictions are non-finite, or targets have zero SD). Job metadata sets `collapsed` from undefined best-validation Pearson. Both identified fits have finite Pearson and SD above 1e-8, so their saved `collapsed:false` values are consistent with that frozen implementation.

For this reporting audit only, a **post hoc descriptive screen, best-checkpoint SD < 1e-4**, identifies exactly these two fits. This screen was not a prospective selection rule, is not applied to exclude runs, and does not change the learning-rate search or any test result. None of the 36 CNN fits meets this descriptive screen. See `diagnostics/all_54_fit_collapse_diagnostics.csv` in the review handoff, with the two original per-fit metadata files and epoch logs alongside it.

## Effect on validation selection

All three predeclared seeds remain in each candidate mean, including the nearly constant fit. Both cells correctly retain LR 2e-5 under the frozen all-three-seed policy. The following two-seed numbers are explanatory arithmetic only, not replacement results or a proposed exclusion policy.

| Cell | 2e-5: all 3 seeds | 5e-5: all 3 seeds | 5e-5: other 2 seeds only |
| --- | --- | --- | --- |
| HepG2 | 0.639262081 | 0.434886507 | 0.644287741 |
| K562 | 0.677098327 | 0.457076987 | 0.671338042 |

In HepG2, the other two high-LR seeds average 0.644288, above the all-three-seed 2e-5 mean 0.639262 (and the corresponding two-seed 2e-5 mean 0.639530). Thus the unstable seed materially affects the selected LR. In K562 the other two high-LR seeds average 0.671338, still below the all-three-seed 2e-5 mean 0.677098 (and the corresponding two-seed mean 0.679092).

## What remains unchanged

All 54 jobs, the six locked recipes, all 72 checkpoints, all 60 evaluation sets, paired intervals and figures remain unchanged. This is a reporting addendum, not a revised training study. Do not relabel the two finite-score candidates as retrospectively ineligible: the protocol makes a candidate ineligible if any seed has **undefined** Pearson, and these two seeds did not.

Suggested manuscript wording: “All seeds were retained in validation selection. Although none triggered the frozen undefined-Pearson flag, two DNABERT2 fits at LR 5e-5 produced nearly constant predictions and are reported as practical collapses. The unstable seed affected LR selection in HepG2. The diagnostic did not alter selection or evaluation.”
