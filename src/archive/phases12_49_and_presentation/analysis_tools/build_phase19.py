from pathlib import Path
import textwrap,nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'[19] Enhancer_Filtering_Findings_and_Reviewer_Package.ipynb'
import sys
assert not PATH.exists() or '--rebuild-own-draft' in sys.argv
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
md('''# Phase 19 — Written findings and reviewer evidence package
Owner: Shafin. CPU-only synthesis of verified Phases 16–18. No model training, inference or manuscript-original edits. All writing and evidence remain in this contribution folder.

**Decision after Phase 18:** the supported finding is modest filtering-associated performance differences for two fixed specialist CNNs, coupled with weak seed-disagreement reliability and activity-dependent prediction bias/interval coverage. Produce reviewable methods/results/discussion drafts and a reviewer-response addendum. Do not add a new calibrated model or claim that DNABERT-2 was evaluated on removed sequences.

The original manuscript and Phase 15 map remain preserved. New writing is a contribution supplement for integration by the authors, not a complete revised manuscript, journal submission or assurance of acceptance. Historical classification tables, the old K562 transformer-collapse cause, long-context experiments and the missing transformer checkpoints remain unresolved.''')
code(r"""
from pathlib import Path
import sys,json,hashlib,datetime
import pandas as pd,numpy as np
ROOT=Path(r"D:\Aditta Thesis\For Shafin\Shafin's Contribution").resolve()
assert Path(sys.executable).resolve()==Path(r'C:\Users\User\.conda\envs\enhancer\python.exe').resolve()
OUT=ROOT/'phase19_outputs';OUT.mkdir(exist_ok=True)
def fsha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,text):(OUT/name).write_text(text.strip()+'\n',encoding='utf8')
def table(name,d):d.to_csv(OUT/name,index=False,float_format='%.12g')
def mdtable(d):
 cols=d.columns.tolist();rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
 for row in d.itertuples(index=False,name=None):rows.append('| '+' | '.join(str(v) for v in row)+' |')
 return '\n'.join(rows)
summaries={i:json.loads((ROOT/f'phase{i}_outputs'/f'phase{i}_summary.json').read_text()) for i in [16,17,18]}
assert all(s['status']=='completed_for_CNNs_DNABERT2_pending' for s in summaries.values())
M=pd.read_csv(ROOT/'phase16_outputs/retained_removed_metrics.csv')
G=pd.read_csv(ROOT/'phase16_outputs/conditional_contrasts.csv')
STD=pd.read_csv(ROOT/'phase17_outputs/composition_conditional_contrasts.csv')
A=pd.read_csv(ROOT/'phase17_outputs/seed_disagreement_error_association.csv')
I=pd.read_csv(ROOT/'phase17_outputs/interval_audit_coverage.csv')
TAIL=pd.read_csv(ROOT/'phase18_outputs/observed_activity_tail_diagnostics.csv')
PG=pd.read_csv(ROOT/'phase18_outputs/predicted_activity_group_diagnostics.csv')
primary=M[M.analysis.eq('primary')].copy()
show=primary[['cell_line','model','population','n','pearson','rmse','mae']].copy()
for c in ['pearson','rmse','mae']:show[c]=show[c].map(lambda v:f'{v:.4f}')
mt=mdtable(show)
ci=G[G.analysis.eq('primary')&G.contrast.eq('removed_minus_retained')&G.metric.eq('mae')].copy()
ci['MAE difference (95% conditional interval)']=ci.apply(lambda r:f'{r.estimate:+.5f} [{r.lo:+.5f}, {r.hi:+.5f}]',axis=1)
cit=mdtable(ci[['cell_line','label','MAE difference (95% conditional interval)']])
coverage=I[I.calibration_source.eq('retained')&I.method.eq('constant_width')].copy()
coverage['coverage']=coverage.empirical_coverage.map(lambda v:f'{100*v:.2f}%')
coverage['full interval width']=coverage.mean_width.map(lambda v:f'{v:.3f}')
ct=mdtable(coverage[['cell_line','model','audit_population','n','coverage','full interval width']])
high=TAIL[TAIL.band.eq('high')].copy()
for col in ['empirical_coverage','all_seeds_underpredict_fraction']:
 high[col]=high[col].map(lambda v:f'{100*v:.2f}%')
high['signed_error']=high.signed_error.map(lambda v:f'{v:.3f}')
ht=mdtable(high[['cell_line','model','population','n','signed_error','all_seeds_underpredict_fraction','empirical_coverage']])
""")
md('''## 1. Write the plain-language research findings''')
code(r"""
write('RESEARCH_FINDINGS.md',f'''# What Shafin's follow-up found

Completed 2026-10-12 in the Anaconda enhancer environment. Phases 16–18 executed successfully; Phase 19 packages the writing and evidence. This completes the available CNN branch of the filtering question. The requested three-model comparison remains incomplete because the original fine-tuned DNABERT-2 weights were not supplied or found locally.

## Why we did this

Phase 14 showed that filtering removed 72,799 of 149,632 complete paired-orientation candidates. Nearly all removals were short 15-letter matches to the legacy benchmark, not full-sequence duplicates. Removed DNA had longer same-letter runs. Knowing that the populations differ does not tell us whether models predict them differently, so we evaluated the existing trained models on eligible removed candidates.

## Phase 16: actual prediction performance on removed DNA

We kept the original five test chromosomes: chr1, chr12, chr4, chr7 and chr9. This supplied 8,436 removed HepG2 and 12,702 removed K562 candidates, alongside 9,048 and 13,145 retained candidates. None had exact/RC identity with the pooled training/validation sequences, and the chromosomes guarantee no coordinate intersection with those records. Twelve existing specialist CNN checkpoints were used without retraining. Repeating retained-test inference matched the archived predictions to approximately 5e-12; the weights and existing research inputs were unchanged.

{mt}

Higher correlation is better; lower MAE/RMSE is better. Correlation is not percent accuracy. Removed-candidate MAE was 0.9–2.9% higher than retained-candidate MAE, depending on model/cell. The HepG2 differences were small and their conditional intervals included zero. K562 errors were modestly higher, with these raw intervals excluding zero:

{cit}

K562 correlations were higher on removed candidates despite higher numerical errors. Their measured activity also had a wider range (target SD 0.453 removed versus 0.430 retained), so these two observations can coexist. This does not prove that the broader range alone caused the correlation difference.

A secondary 31-mer screen removed 199/230 removed/retained HepG2 candidates and 393/380 K562 candidates. The qualitative pattern persisted. Similarity screens do not prove complete absence of homology. The screen also changes the evaluation population.

The evidence does not support a dramatic collapse of these CNNs on the excluded population. It does not establish statistical equivalence, a universal absence of filter effects, or the effect of retraining without filtering. On HepG2, tiny point-estimate differences between the two CNNs change direction between populations, but the paired intervals overlap zero; there is no convincing model-order reversal. MPRA-LegNet remains ahead on K562. We cannot assess DNABERT-2's removed-population ranking without its actual checkpoints.

## Phase 17: composition and reliability

After giving retained and removed groups a shared activity/GC/repeated-base composition, the K562 MAE gaps decreased: LegNet +0.00675 to +0.00378, MPRAnn +0.00502 to +0.00301. The standardized conditional intervals include zero. The shared-support analysis retained roughly 96–98% of candidates. This is consistent with some association of the raw gaps with composition; it is not a causal explanation or proof that filtering has no remaining effect.

Seed disagreement was weakly associated with absolute error (Spearman {A.spearman_seed_sd_absolute_error.min():.3f}–{A.spearman_seed_sd_absolute_error.max():.3f}). Accepting only low-disagreement predictions reduced average errors, but rejected many examples and changed the evaluated activity distribution. Agreement between seeds cannot be treated as a reliable individual confidence score.

We also fitted simple nominal 90% activity intervals on the fixed calibration groups and measured their coverage on disjoint audit groups. With retained-calibration constant-width intervals:

{ct}

Scaling interval width by seed disagreement usually made intervals substantially wider, with little consistent coverage benefit. A wider interval is not automatically a better uncertainty estimate. No method was chosen or tuned from these audit outcomes.

## Phase 18: where overall coverage hides a weakness

Independent arithmetic reproduced all 16 metric rows, the ensemble means/SDs, 16 interval parameter sets, all saved interval endpoints and 32 coverage summaries. Restoring the original float32 cache predictions eliminated tiny Spearman differences caused by CSV parsing near ties. Ordinary score discrepancies were below 1e-9; protected inputs remained unchanged.

High measured-activity groups had systematic underprediction and lower interval coverage:

{ht}

The interval coverage in the highest predicted-activity quintile was also only about 80–82%. Unlike grouping by measured activity, the predicted-activity quintile can be determined before a new assay. This is a useful diagnostic for a later uncertainty study, not a validated correction.

Low coverage after selecting high observed responses does not by itself invalidate a marginal conformal guarantee: marginal and subgroup coverage are different claims. The pattern also involves regression to the mean and possible measurement noise, so it is not evidence of a specific biological mechanism.

## What is complete and what is still missing

The CNN filtering-performance analysis, composition sensitivity, reliability audit, activity-tail diagnostics and independent numerical checks are complete. New model training and original manuscript edits were unnecessary for these analyses and were not performed. Phase 19 supplies draft manuscript sections and reviewer responses for integration.

Still missing: original DNABERT-2 regression checkpoints for removed-population inference; authentic predictions behind the disputed historical classification tables; the cause of the old Phase 9C K562 collapse; longer-context and broader-cell evidence. All intervals and subgroup findings are retrospective and conditional on the fitted models and examined partitions. These results concern two assay/design families, not all enhancers or native genomic regulation.

Exact tables and notebook links are listed in the accompanying evidence index. Everything is stored inside Shafin's Contribution.
''')
print('Written RESEARCH_FINDINGS.md')
""")
md('''## 2. Write manuscript sections and reviewer-response addendum
These are finished draft sections for author review. They describe only completed work, use exact numerical evidence and preserve the distinction between a regression supplement and historical classification experiments. No original manuscript text is changed.''')
code(r"""
write('MANUSCRIPT_SECTION_DRAFT.md',f'''# Draft contribution sections for manuscript integration

Suggested subsection title: **Selection effects and reliability of sequence-based activity prediction**

## Methods: excluded-population evaluation

We independently reconstructed 149,632 complete paired-orientation 200 bp lentiMPRA candidates and reproduced all 76,833 retained records. The legacy-similarity screen excluded 72,799 candidates, comprising 72,795 shared-15-mer exclusions and four exact/reverse-complement matches. To assess predictive transport to the excluded population, we applied the existing chromosome-held-out MPRA-LegNet and MPRAnn checkpoints to removed candidates on the original test chromosomes (chr1, chr12, chr4, chr7 and chr9), without retraining or checkpoint reselection. These specialist CNN architectures were introduced in the lentiMPRA study of Agarwal et al. [1]; this work uses the previously documented Phase 12 adaptations and frozen recipes, rather than reproducing every aspect of the published training procedure.

The evaluation comprised 9,048 retained and 8,436 removed HepG2 candidates, and 13,145 retained and 12,702 removed K562 candidates. Exact/RC identity was screened against pooled training/validation records from both cell lines; none was found. Test chromosomes were disjoint from all training/validation chromosomes. A secondary sensitivity additionally excluded candidates sharing a canonical 31-mer with those pooled inputs, removing 199 removed and 230 retained HepG2 candidates and 393 removed and 380 retained K562 candidates. These screens address specified similarities and do not exclude all distant homology.

For each cell and architecture, three authenticated checkpoints were evaluated using the original training-only target scaling and forward/RC averaging. The three predictions were averaged. Retained-test predictions reproduced the archived exports within approximately 5e-12. Pearson/Spearman correlations, RMSE, MAE, R² and signed errors were reported separately. Conditional percentile intervals used 1,000 bootstrap draws of genomic regions formed by merging intervals within 1 kb, with additional union of regions connected by exact/RC identity. The same draw was used across models and populations within each cell. Spearman was reported as a point estimate; intervals were calculated for Pearson, RMSE and MAE. These exploratory intervals condition on fitted models and examined partitions and are not multiplicity-adjusted.

## Methods: composition and reliability diagnostics

We standardized absolute errors to a shared observed composition defined by training-derived 20th/80th activity-percentile bands, fixed GC bands and maximum homopolymer length below/at least six. Strata required at least 50 candidates in each population; pooled frequencies on common support supplied fixed reference weights. Standardization conditions partly on measured activity and is descriptive, not a deployable rule or causal adjustment.

For reliability, the genomic/identity-group hash assignment was fixed before new inference, allocating approximately 30% of groups to calibration and the remainder to audit. Cutoffs were not selected using audit errors. We measured the Spearman association between three-seed SD and absolute error and applied spread cutoffs derived from retained calibration predictions. Nominal 90% intervals used the finite-sample residual order statistic ceil((n_cal+1)*0.9). A constant-width baseline was compared with a seed-scaled interval using max(seed SD, 0.05 times training-target SD). Retained calibration was primary, with removed calibration as a secondary sensitivity. Calibration and audit groups were disjoint; audit labels did not enter interval fitting.

The retained benchmark and aggregate Phase 16 results had previously been examined, and sequence/region dependence and population shifts remain. Accordingly, we report empirical coverage rather than a prospective or universal conformal guarantee. Conformal validity depends on its exchangeability/algorithmic assumptions, and marginal coverage does not certify individual sequences or every subgroup [2]. No weighted nonexchangeable conformal procedure from [2] was implemented.

## Results: filtering-associated performance differences

{mt}

Removed-candidate MAE was modestly higher for both CNNs: HepG2 differences were +0.00714 for LegNet and +0.00340 for MPRAnn, with conditional intervals including zero; K562 differences were +0.00675 and +0.00502. K562 raw MAE intervals excluded zero, but correlations were higher on removed candidates: LegNet 0.7267 versus 0.7080 and MPRAnn 0.7044 versus 0.6815. The removed K562 targets also had larger SD (0.453 versus 0.430), underscoring that correlation and activity-value error measure different properties. The qualitative pattern persisted after the 31-mer sensitivity screen.

Shared-composition standardization covered approximately 96–98% of candidates and reduced K562 MAE gaps to +0.00378 and +0.00301, with conditional intervals including zero. This is consistent with a contribution from observed population composition but does not identify a causal filtering effect. HepG2 standardized gaps remained small. Point estimates do not establish equivalence between retained and excluded populations.

## Results: seed disagreement and activity-dependent interval coverage

The association between seed SD and absolute error was weak (Spearman 0.082–0.155 across cell/model/population combinations). Retained-calibrated constant-width intervals achieved empirical marginal coverage of 88.1–89.1% for HepG2 and 89.8–90.9% for K562 audit candidates. Seed-scaled intervals generally increased mean width substantially without a consistent coverage improvement.

For sequences above the training 80th measured-activity percentile, mean signed errors were approximately −0.706 to −0.750 log2 units in HepG2 and −0.327 to −0.360 in K562. All three seeds underpredicted 71–86% of these examples. Constant-width interval coverage in the high measured-activity groups was 57.4–61.4% in HepG2 and 69.3–74.3% in K562. In the highest predicted-activity quintile, defined using retained calibration predictions without audit outcomes, coverage was approximately 80–82% across populations and models. These retrospective diagnostics expose limitations hidden by marginal coverage. Conditioning on high observed responses also introduces regression-to-the-mean and measurement-noise considerations; it does not establish an underlying biological mechanism or contradict an individual guarantee that was never claimed.

## Discussion and limitations

The legacy-similarity filter changes the candidate population, particularly repeat composition, but this follow-up does not show a large deterioration of the two specialist CNNs on eligible excluded candidates. Small absolute-error differences coexist with similar or higher correlations. This analysis evaluates fixed models across populations; it does not estimate the effect of training without filtering. No equivalence margin was prespecified, so failure to detect a difference is not evidence of equivalence.

The removed-population DNABERT-2 comparison remains unavailable because the original fine-tuned regression checkpoints are absent from the supplied inputs. Its earlier retained/external comparisons are independently verified in Phase 13, but cannot be extrapolated to this new population. Pretrained DNABERT-2 weights cannot substitute for those checkpoints. The two cell libraries also differ in candidate design, preventing causal cell-type interpretation. Findings concern 200 bp assayed candidates and do not establish universal enhancer performance, native-chromatin behaviour, the utility of longer flanking context, or an intrinsic insufficiency of DNA sequence information.

Seed disagreement is a limited heuristic rather than a complete uncertainty model. Nominal marginal coverage near 90% hides pronounced activity-dependent variation; a future correction should use dedicated development/calibration data and a newly reserved evaluation population. We did not choose a corrective model or claim independent confirmation from the already examined audit groups.

## References for these sections

[1] Agarwal V et al. *Massively parallel characterization of transcriptional regulatory elements*. Nature (2025). https://doi.org/10.1038/s41586-024-08430-9 . Specialist architectures and their original assay/training context; our frozen recipes differ as documented in Phase 12.

[2] Barber RF, Candès EJ, Ramdas A, Tibshirani RJ. *Conformal prediction beyond exchangeability*. Annals of Statistics 51(2), 816–845 (2023). https://doi.org/10.1214/23-AOS2276 . Supports methodological limitations, not a guarantee for our diagnostic intervals.

Evidence: phase16_outputs/retained_removed_metrics.csv and conditional_contrasts.csv; phase17_outputs/composition_conditional_contrasts.csv, seed_disagreement_error_association.csv and interval_audit_coverage.csv; phase18_outputs/observed_activity_tail_diagnostics.csv and predicted_activity_group_diagnostics.csv. Exact file hashes are in the Phase 19 evidence index.
''')
write('REVIEWER_RESPONSE_ADDENDUM.md','''# Draft reviewer-response addendum: completed filtering and reliability work

This addendum is ready for author integration alongside the revised manuscript. It does not claim that manuscript edits, historical table repairs or journal submission have already occurred. Review identifiers match Phase 15; exact original wording remains in reference_inputs/Review From Journal.docx and its indexed text copy.

## R4.2 and Editor.novelty_limits: unsupported broad generalization/sequence insufficiency

We agree that a model comparison on selected 200 bp assay candidates cannot establish an intrinsic insufficiency of primary DNA sequence or universal enhancer generalization. We independently reconstructed the complete eligible assay population and documented that the legacy-similarity filter removed 72,799 of 149,632 paired-orientation candidates. Almost all exclusions were shared-15-mer matches rather than full-sequence duplicates. We then applied the existing specialist CNN checkpoints, without retraining, to excluded candidates on the original test chromosomes. The newly evaluated removed groups contained 8,436 HepG2 and 12,702 K562 candidates and were chromosome-disjoint from model fitting/validation records.

Both CNNs showed modestly higher absolute errors on removed candidates, rather than a dramatic performance collapse. K562 correlations were higher despite higher absolute errors; shared-composition standardization reduced its error differences. We therefore support a bounded empirical conclusion about these assay populations and models. We do not claim universal enhancer performance, equivalence between populations, or the causal effect of retraining without filtering. The removed-population DNABERT-2 arm remains pending its original fine-tuned checkpoints.

Evidence: Phase 14 reconstruction and population tables; phase16_outputs/evaluation_attrition.csv, retained_removed_metrics.csv and conditional_contrasts.csv; phase17_outputs/composition_common_support.csv and composition_conditional_contrasts.csv.

Remaining integration: replace broad insufficiency/generalization claims in the title/abstract/discussion where applicable; include population/design limitations. This addendum alone does not finish that manuscript revision.

## R1.2, R1.3 and R7.1: metric consistency and uncertainty reporting

All new regression results were independently recalculated from authenticated per-job predictions. The check covered ensemble arithmetic, seed SD, 16 primary/sensitivity metric rows, 16 interval parameter sets, interval endpoints and 32 marginal coverage summaries. Pearson/RMSE/MAE/R² and reconstructed Spearman agreed within the stated numerical tolerances. The original retained-test predictions were reproduced to export-rounding precision. We distinguish model-training seed variability, conditional bootstrap intervals for average performance, and empirical prediction-interval coverage.

These checks apply to the new continuous-activity experiments. They do not repair or authenticate the original classification Tables 8/14, for which actual original predictions/denominators must still be recovered or unsupported results removed. Classification confusion matrices are not appropriate for continuous activity targets without a separately justified classification task.

Evidence: phase13_outputs/verification_summary.json; phase16_outputs/retained_prediction_reproduction.csv; phase18_outputs/independent_numerical_checks.csv; phase17_outputs/interval_audit_coverage.csv.

## R1.4, R7.2 and R7.3: traceable methods, selection and figures

The follow-up uses frozen Phase 12 checkpoints and the original chromosome partition. No additional training, hyperparameter search or epoch reselection was performed. Forward/RC averaging, original target scaling, similarity exclusions, genomic/identity bootstrap groups and retrospective inferential scope are recorded in the frozen protocols. New figures show retained/removed absolute errors, seed-disagreement selection trade-offs and activity-dependent bias/coverage.

These figures concern regression and do not replace ROC curves for historical classification results. Original pseudocode, model/tokenizer descriptions, scheduler settings and architecture figures still require manuscript-level reconciliation. The follow-up does not recover missing older training logs.

Evidence: phase16_outputs/frozen_protocol.json and loaded_checkpoint_metadata.csv; phase17_outputs/frozen_protocol_semantic_hash.json (initial container-hash specification retained separately); Phase 19 FIGURE_CAPTIONS.md.

## Additional reliability limitation

Three-seed disagreement correlated only weakly with absolute errors (Spearman 0.082–0.155). Nominal 90% intervals had roughly near-nominal marginal coverage but much lower coverage in high-activity groups. Underprediction was shared across seeds. The highest predicted-activity quintile also had only about 80–82% coverage, providing a feature available before observing the assay result. These are exploratory diagnostics, not a validated recalibration model. Marginal coverage does not promise equal coverage conditional on a high observed response.

Evidence: phase17_outputs/seed_disagreement_error_association.csv, interval_audit_coverage.csv and interval_subgroup_coverage.csv; phase18_outputs/observed_activity_tail_diagnostics.csv and predicted_activity_group_diagnostics.csv.

## Items this work does not resolve

R4.3 longer-context benefit, original failed K562 training cause, historical classification-table errors, comprehensive current literature/citation repair, original architecture/equation formatting and journal submission materials remain open/partial as specified in the updated evidence map. A focused primary-source check and section references are supplied; no claim of exhaustive literature coverage is made.
''')
print('Written manuscript section draft and reviewer response addendum.')
""")
md('''## 3. Extend the evidence map without replacing Phase 15''')
code(r"""
OLD=pd.read_csv(ROOT/'phase15_outputs/reviewer_evidence_map.csv')
print('Original evidence map columns',OLD.columns.tolist())
additional=pd.DataFrame([
 dict(item='Shafin.removed_CNN_performance',status='addressed_with_scope',finding='Frozen specialist CNNs scored on 21,138 removed test-chromosome candidates; modest absolute-error increases, no equivalence or causal retraining claim',evidence='phase16_outputs/retained_removed_metrics.csv;phase16_outputs/conditional_contrasts.csv',remaining='DNABERT-2 original weights required for intended three-model comparison'),
 dict(item='Shafin.removed_DNABERT2_performance',status='pending_input',finding='Original fine-tuned checkpoints not available locally; no predictions fabricated or pretrained model substituted',evidence='phase16_outputs/phase16_summary.json',remaining='Recover original Phase9B weights, authenticate against provenance, then run inference and extend paired comparisons'),
 dict(item='Shafin.composition_standardization',status='addressed_with_scope',finding='Shared-support standardized errors computed; K562 gap reduced; target-based adjustment is descriptive',evidence='phase17_outputs/composition_conditional_contrasts.csv;phase17_outputs/composition_common_support.csv',remaining='Report support and avoid causal or equivalence claims'),
 dict(item='Shafin.prediction_reliability',status='addressed_with_scope',finding='Weak seed-spread/error association; near-nominal marginal interval coverage hides activity differences',evidence='phase17_outputs/seed_disagreement_error_association.csv;phase17_outputs/interval_audit_coverage.csv',remaining='Future learned/calibrated correction needs appropriate development and new reserved evaluation data'),
 dict(item='Shafin.activity_tails',status='addressed_with_scope',finding='Shared underprediction at high measured activity; predicted-activity quintiles also reveal lower coverage',evidence='phase18_outputs/observed_activity_tail_diagnostics.csv;phase18_outputs/predicted_activity_group_diagnostics.csv',remaining='Retrospective diagnostic; no biological-mechanism or individual confidence guarantee'),
 dict(item='Shafin.independent_new_result_audit',status='addressed',finding='New metric, ensemble, cache, interval and protected-input checks passed',evidence='phase18_outputs/independent_numerical_checks.csv',remaining='Original historical classification claims remain separate'),
 dict(item='Shafin.written_contribution',status='draft_ready',finding='Methods/results/discussion, plain-language findings and reviewer addendum prepared inside contribution folder',evidence='phase19_outputs/MANUSCRIPT_SECTION_DRAFT.md;phase19_outputs/REVIEWER_RESPONSE_ADDENDUM.md',remaining='Authors integrate and verify manuscript-wide consistency; no submission made')])
table('followup_evidence_map.csv',additional)
OLD['followup_update']='Original Phase15 status preserved; see followup_evidence_map.csv for completed new evidence and remaining limits.'
table('original_reviewer_map_with_followup_pointer.csv',OLD)
evidence=[]
for folder in ['phase16_outputs','phase17_outputs','phase18_outputs']:
 for p in (ROOT/folder).iterdir():
  if p.is_file() and p.name!='execution_log.txt':evidence.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=fsha(p)))
for row in additional.itertuples():
 for rel in row.evidence.split(';'):assert (ROOT/rel).is_file(),rel
table('evidence_file_index.csv',pd.DataFrame(evidence))
write('FIGURE_CAPTIONS.md','''# Figure captions and files

1. **Retained versus removed activity prediction.** Mean absolute error of the frozen three-seed MPRA-LegNet and MPRAnn ensembles on original chromosome-held-out candidates. Panels show HepG2 and K562 assay/design families. Bars are point estimates, not interval estimates; genomic/identity-group conditional intervals and population contrasts are supplied in the numerical tables. Lower is better. File: phase16_outputs/retained_removed_mae.png (also PDF).

2. **Seed disagreement and selective prediction.** Audit MAE versus the fraction of candidates accepted by thresholds derived from retained calibration seed-SD quartiles. No cutoff was selected from audit outcomes. Selection changes the evaluated population; lower error among accepted candidates is not an individual confidence guarantee. File: phase17_outputs/seed_disagreement_selective_errors.png (also PDF).

3. **Nominal versus measured interval coverage.** Empirical marginal audit coverage for constant-width and seed-scaled nominal 90% intervals calibrated on retained candidates. The dashed line indicates the nominal target, not a guaranteed rate for these dependent populations. Widths are reported separately; more coverage from wider intervals alone does not establish better reliability. File: phase17_outputs/nominal90_empirical_coverage.png (also PDF).

4. **Activity-dependent bias and coverage.** Mean prediction-minus-measurement and empirical interval coverage for low/middle/high measured-activity bands defined from training percentiles. Lines connect descriptive category summaries rather than a fitted continuous trend. Intervals are the retained-calibrated constant-width baseline. These outcome-based groups require labels, and marginal nominal coverage need not hold within each group. Files: phase18_outputs/activity_tail_bias_and_coverage.png and .pdf.
''')
write('PENDING_INPUTS_AND_NEXT_ACTIONS.md','''# Pending inputs and bounded next actions

## DNABERT-2 removed-population comparison

Required: the original Phase 9B fine-tuned regression checkpoints, normally packaged as enhancer_phase9b_complete_with_weights.zip. For the primary comparison, recover HepG2_dnabert2_seed_20260930_best.pt through seed 20260932 and the corresponding three K562 files. Copy them into reference_inputs/phase16_model_weights before use. Authenticate each file against phase9b_provenance.json in the existing copied results archive; verify lock, revision, cell, seed and epoch. Also supply the exact pinned DNABERT-2 model/tokenizer code/config snapshot and compatible local dependencies. A pretrained backbone alone is not the original trained regressor. No compatible Transformers package is currently installed in enhancer; if needed, any additional dependency files should be isolated inside the contribution folder rather than changing the existing environment.

Then reproduce archived retained-test predictions before scoring the frozen Phase16 removed cohort, use identical IDs/targets/strand policy, and report all seeds and paired contrasts. Do not replace the existing CNN results. Use a clearly versioned extension. Phase9C weights are needed only for a secondary locus-protocol extension; assigning removed loci to that split requires a separate careful overlap/group audit.

## Existing paper revision

Integrate MANUSCRIPT_SECTION_DRAFT.md and REVIEWER_RESPONSE_ADDENDUM.md with the final agreed manuscript scope. Recover original predictions behind the historical classification tables or remove results that cannot be authenticated. Reconcile pseudocode/tokenizers/lengths, citations, figures and statements throughout. The new regression evidence does not authenticate old classification results.

## Future prediction-reliability research

The highest predicted-activity quintile is a useful candidate for a dedicated uncertainty study. Any learned residual model, activity-aware recalibration, alternative interval method or outcome-selected cutoff needs appropriate development/calibration data and a newly reserved evaluation set. The current audit outputs have already been examined; they cannot become an untouched confirmation by relabelling them. No new training was needed or launched for this follow-up.

## Other questions

The old Phase9C K562 DNABERT-2 collapse cause, missing older per-epoch logs, longer-context benefit, broader cell-type transport and native-chromatin conclusions remain unresolved. Their status is preserved honestly in the evidence map. This package does not contact the journal or other people.
''')
""")
md('''## 4. Record focused primary-source notes and completion status
This is a targeted methods/source check, not an exhaustive literature review. Citation notes distinguish the original published architectures from the adapted frozen recipes, DNABERT-2 from original DNABERT, and marginal interval assumptions from empirical diagnostics. Original upstream licenses are copied inside reference_inputs/phase19_method_sources.''')
code(r"""
write('FOCUSED_PRIMARY_SOURCE_NOTES.md','''# Focused primary-source check

Checked 2026-10-12. This supports the present filtering/reliability supplement and is not a comprehensive review of current genomic models.

- **Agarwal et al., Massively parallel characterization of transcriptional regulatory elements, Nature (2025), DOI 10.1038/s41586-024-08430-9.** Primary source for the MPRAnn and MPRA-LegNet assay-specific CNN comparison. Its models, assay orientation/adapters and original optimization are not identical to this project's frozen Phase12 adaptations. Cite the paper and the exact repository commits recorded in Phase12; do not present the published performance numbers as results from our split. Source: https://www.nature.com/articles/s41586-024-08430-9 . Publisher full-page retrieval was unavailable in this run; relevant indexed methods excerpts and the official repositories were checked.

- **Official MPRA-LegNet repository**, https://github.com/autosome-ru/human_legnet , pinned commit a20dbc7d0b08daae3af1aebcfb8a71d9f198ab5d. README and complete MIT license copied locally. It describes an MPRA-specific LegNet variant and original augmentation/training context. Our inference uses existing authenticated checkpoints, not newly downloaded published weights.

- **Official MPRAnn repository**, https://github.com/visze/sequence_cnn_models , pinned commit 7638fce137db0445123efe8d7e2c35e248fafc5f. README and complete MIT license copied locally. The previously validated Phase12 PyTorch port remains unchanged.

- **Barber, Candès, Ramdas and Tibshirani (2023), Conformal prediction beyond exchangeability, Annals of Statistics 51(2):816–845, DOI 10.1214/23-AOS2276.** Primary methodological source for exchangeability, dependence and distribution-drift limitations. Our intervals use ordinary residual order statistics, not the paper's weighted extension. Marginal coverage and coverage conditional on measured high activity are different targets. Source text checked through https://arxiv.org/html/2202.13415v5 and a full HTML copy saved inside reference_inputs/phase19_method_sources. Python retrieval first failed certificate validation; retrieval using Windows certificate trust succeeded without changing settings or bypassing verification.

- **Zhou et al., DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome**, https://arxiv.org/abs/2306.15006 ; ICLR 2024 conference version https://openreview.net/forum?id=oMLQB4EZE1 ; official repository https://github.com/MAGICS-LAB/DNABERT_2 . This is distinct from the original DNABERT paper and uses a different tokenization scheme. It supports citation/model identity, not our missing fine-tuned checkpoints. Do not treat generic pretraining weights as the regressors selected in Phase9B. Full bibliography reconciliation remains open.

Local source retrieval outcomes and hashes: reference_inputs/phase19_method_sources/source_manifest.json. The DNABERT-2 primary abstract page is also copied locally. No unavailable source is represented as a saved full text. Notes are paraphrases; original papers should be consulted during manuscript-wide literature revision.
''')
write('supplement_references.bib','''@article{agarwal2025mpra,
 author={Agarwal, Vikram and others},
 title={Massively parallel characterization of transcriptional regulatory elements},
 journal={Nature}, year={2025}, doi={10.1038/s41586-024-08430-9}
}
@article{barber2023exchangeability,
 author={Barber, Rina Foygel and Cand\u00e8s, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J.},
 title={Conformal prediction beyond exchangeability},
 journal={The Annals of Statistics}, volume={51}, number={2}, pages={816--845},
 year={2023}, doi={10.1214/23-AOS2276}
}
@inproceedings{zhou2024dnabert2,
 author={Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik and Davuluri, Ramana V. and Liu, Han},
 title={{DNABERT-2}: Efficient Foundation Model and Benchmark for Multi-Species Genome},
 booktitle={International Conference on Learning Representations}, year={2024},
 url={https://openreview.net/forum?id=oMLQB4EZE1}
}''')
protected=json.loads((ROOT/'phase16_outputs/protected_input_hashes.json').read_text())
for rel,h in protected.items():assert fsha(ROOT/rel)==h
summary={'status':'completed_writing_CNN_branch_DNABERT2_pending','environment':sys.executable,
 'completed_phases':[16,17,18,19],'new_training_runs':0,'inference_checkpoints':12,'removed_candidates_scored':21138,
 'dnabert2_removed_comparison':'pending original fine-tuned weights and compatible snapshot/dependencies',
 'original_manuscript_edited':False,'journal_contacted':False,'protected_inputs_unchanged':True,
 'outputs':{p.name:fsha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['phase19_summary.json','execution_log.txt']}}
(OUT/'phase19_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k!='outputs'},indent=2))
""")
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (enhancer)','language':'python','name':'enhancer'},'language_info':{'name':'python','version':'3.12'},'contribution':{'owner':'Shafin','phase':19,'no_training':True}})
for i,c in enumerate(n.cells):
 if c.cell_type=='code':compile(c.source,f'phase19_cell{i}','exec')
nb.validate(n);nb.write(n,PATH);print('Created',PATH)
