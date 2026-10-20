from pathlib import Path
import json,pandas as pd
r=Path.cwd();p=r/'CONTRIBUTION_LOG.md';marker='### Completed Phase 12 result verification (2026-10-12)'
s=p.read_text('utf-8')
if marker not in s:
 addition='''

### Completed Phase 12 result verification (2026-10-12)

This entry supersedes the earlier “not yet run for real” status; that earlier entry is preserved as preparation history.

- The user's full, non-smoke Phase 12 run completed all 24 expected jobs (two specialist CNNs × two protocols × two cells × three seeds). Recorded job training time totals 24.54 minutes. No further training was launched for this review.
- Independently authenticated all seven locked upstream inputs, eight result tables, 24 checkpoints, 72 per-job prediction caches and both finished ZIP archives. Verified selected epochs against validation logs and actual CPU-loaded checkpoint metadata, train-only target scaling, identical evaluation IDs/targets, and aggregate exports against per-job scores.
- Recomputed all internal and external saved metrics and ensemble means independently. Pearson/R²/RMSE/MAE differences were below 5e-13; the largest internal Spearman discrepancy was 4.59e-6, within the explicitly reported near-tie/CSV precision tolerance. This does not change any principal comparison.
- Independently repeated all 60 reported paired bootstrap intervals with 1,000 fresh fixed-seed draws each; locus, sequence and 1 kb region resampling conclusions agree with the saved intervals. These intervals are conditional on fitted models and previously examined partitions; seed variation is reported separately, and secondary intervals are unadjusted for multiple comparisons.

Primary chromosome-held-out (9B) test Pearson:

| Cell | DNABERT-2 | MPRA-LegNet | MPRAnn | Earlier CNN | Spaced-mer ridge |
|---|---:|---:|---:|---:|---:|
| HepG2 | 0.595329 | 0.662449 | 0.655661 | 0.506324 | 0.526486 |
| K562 | 0.642160 | 0.708002 | 0.681506 | 0.506835 | 0.535265 |

Both specialist CNNs also lead DNABERT-2 on the secondary 9C locus-held-out Pearson and Phase 10 DHS-natural Spearman comparisons. This is useful evidence **against the proposed DNABERT-2 superiority claim under these tested recipes**; it is not proof that CNNs universally win or that DNA pretraining never helps.

The earlier Phase 9C K562 DNABERT-2 seed 20260930 is still nearly constant (test SD approximately 6.98e-6; Pearson 0.00854; archived selected epoch 1). It remains included in the original ensemble. A clearly labelled two-healthy-seed sensitivity leaves Pearson almost unchanged (0.65963) but improves RMSE from 0.34456 to 0.32606. The training cause remains unresolved; no failed run was silently dropped or retrained.

Completed notebook: `[13] Enhancer_Independent_Result_Verification.ipynb`.
Exact evidence: `phase13_outputs/verification_summary.json`, `internal_metrics_recomputed.csv`, `external_metrics_recomputed.csv`, `independent_paired_bootstrap.csv`, `checkpoint_metadata_and_export_alignment.csv`, and `old_9c_k562_collapse_sensitivity.csv`.

### Filtering and reviewer follow-up: interim status (2026-10-12)

- Created notebooks 14 and 15 and executed their currently available CPU analyses in `enhancer`.
- Notebook 14's retained-test subgroup metrics are complete for GC, CpG, homopolymer, training-defined activity bands and chromosomes, with minimum subgroup n=100. Both CNNs lead overall, but DNABERT-2 has smaller MAE in some high-activity subgroups; these exceptions are retained. Phase 9C K562 absolute-error comparisons include the collapsed-seed ensemble and need that caveat.
- Complete retained-versus-removed population reconstruction and removed-DNA composition are **pending the full authenticated hg38 reference copy**. Raw assay BED copies are present. Removed-sequence prediction performance is unavailable and is not inferred from composition.
- Notebook 15 maps scientific reviewer/editor concerns to exact evidence files and separates addressed, partial and open issues; manuscript edits, historical classification table repairs, long-context experiments, literature/citation repair and journal submission remain open. The journal letter is treated as evidence, not authorization, and no journal was contacted.
- All copied inputs, analyses and outputs stay inside `Shafin's Contribution`. Research originals, notebook 12 and Phase 12 outputs were not edited.
'''
 p.write_text(s.rstrip()+addition+'\n',encoding='utf-8');print('Appended verified results and honest interim filtering status; preserved earlier entries.')
else:print('Verification entry already present; no duplicate appended.')
