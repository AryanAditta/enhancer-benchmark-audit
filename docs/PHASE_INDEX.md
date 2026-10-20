# Original notebook and source history

This index preserves the project history alongside the six current reproduction notebooks. The six notebooks at the top of `notebooks/` remain the starting point for current manuscript results.

The handoff sequence below was supplied by Shafin Ahmed on 20 October 2026. It records work stages and does not assert a final manuscript author order or sole authorship of every line.

| Work stage | Main notebooks | Archive |
|---|---:|---|
| Aditta: initial DNABERT work | 4 | `notebooks/original_phases/01_legacy_dnabert_experiments/` |
| Arshee: subsequent work | 3 | `notebooks/original_phases/02_legacy_longformer_experiments/` |
| Aditta: Phases 1–11C3 | 17 | `notebooks/original_phases/03_phases01_11c3_audit_and_models/` |
| Shafin Ahmed: Phases 12–49 | 38 | `notebooks/original_phases/04_phases12_49_validation_and_reporting/` |

There are **62 main original notebooks plus five distinct saved/recovery versions**, and **83 distinct supporting code files**. All **145 original notebook paths and 129 code paths** are accounted for in [ORIGINAL_SOURCE_INVENTORY.json](ORIGINAL_SOURCE_INVENTORY.json). Byte-identical backup copies share an archive destination; differing versions are retained.

## How to interpret the archive

Archived files are exact copies, including saved outputs, errors and dated pending-status messages. Their presence records the work performed; it is not a fresh validation of their results or proof that every historical branch supports the current manuscript. Superseded figures, two-epoch recipes, historical shards, exploratory filtering diagnostics and failed attempts remain labelled here.

Archived notebooks have not been rerun as complete historical workflows for this upload. The new raw-preparation adapter executes authenticated preparation cells from Phases 1, 9A and 10 and reuses the Phase 14 read-only genome reader; frozen training helpers also support the documented fresh-inference and completed-job authentication checks. Original notebook files, machine paths, dependencies, Kaggle settings and input/output expectations remain unchanged. Full historical execution requires the original study layout and authenticated artifact bundles. Current routes and validation limits are documented in [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md).

The [collapse-count erratum](history/ERRATUM_collapse_count.md) accompanies Phase 38. Its relative handoff-diagnostic paths refer to the original review bundle. The repository also supplies the full fit logs, metadata and validation-prediction records used by the current reporting route.

## Aditta and Arshee: original model notebooks

| Work stage | Original notebook | Historical role |
|---|---|---|
| Aditta | [DNABERT - Ind Test (Base, BiLSTM).ipynb](../notebooks/original_phases/01_legacy_dnabert_experiments/DNABERT%20-%20Ind%20Test%20%28Base%2C%20BiLSTM%29.ipynb) | Independent-test DNABERT architecture experiments; historical evaluation, preceding the overlap audit. |
| Aditta | [DNABERT- Ind Test (1).ipynb](../notebooks/original_phases/01_legacy_dnabert_experiments/DNABERT-%20Ind%20Test%20%281%29.ipynb) | Independent-test DNABERT architecture experiments; historical evaluation, preceding the overlap audit. |
| Aditta | [DNABERT-10 Fold.ipynb](../notebooks/original_phases/01_legacy_dnabert_experiments/DNABERT-10%20Fold.ipynb) | Ten-fold DNABERT architecture experiments; historical evaluation, preceding the overlap audit. |
| Aditta | [DNABERT-5.ipynb](../notebooks/original_phases/01_legacy_dnabert_experiments/DNABERT-5.ipynb) | Five-fold DNABERT architecture experiments; historical evaluation, preceding the overlap audit. |
| Arshee | [10 fold final.ipynb](../notebooks/original_phases/02_legacy_longformer_experiments/10%20fold%20final.ipynb) | Ten-fold Longformer architecture experiments; historical 200-bp setting, preceding the overlap audit. |
| Arshee | [5 fold final.ipynb](../notebooks/original_phases/02_legacy_longformer_experiments/5%20fold%20final.ipynb) | Five-fold Longformer architecture experiments; historical 200-bp setting, preceding the overlap audit. |
| Arshee | [ind-test.ipynb](../notebooks/original_phases/02_legacy_longformer_experiments/ind-test.ipynb) | Independent-test Longformer architecture experiments; historical 200-bp setting, preceding the overlap audit. |

## Aditta: Phases 1–11C3

| Phase | Original notebook / purpose | Current entry notebook(s) | Historical/current relationship |
|---|---|---|---|
| 1 | [Original identity and split audit](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B1%5D%20Enhancer_Kaggle_Phase1_Audit_and_Splits.ipynb) | 01 | Foundational audit; preserve the original definitions and frozen groups. |
| 2 | [GENA correction and grouped classification](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B2%5D%20Enhancer_Kaggle_Phase2_GENA_Correction.ipynb) | 02 | Historical one-seed comparison; the two transformer arms were later rerun in Phases 30–33. |
| 3 | [Composition and robustness diagnostics](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B3%5D%20Enhancer_Kaggle_Phase3_Robustness.ipynb) | 01, 02, 05 | Historical sensitivity analyses; retain their original comparison scope. |
| 4 | [External MPRA dataset construction](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B4%5D%20Enhancer_Kaggle_Phase4_External_MPRA_Dataset.ipynb) | 03, 04 | Early activity-quintile dataset; distinct from full continuous activity in Phase 9A. |
| 5 | [Legacy-trained model transfer](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B5%5D%20Enhancer_Kaggle_Phase5_External_Transfer.ipynb) | 04 | Earlier binary transfer experiment; separate from current quantitative transfer. |
| 6 | [MPRA-specific activity classification](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B6%5D%20Enhancer_Kaggle_Phase6_MPRA_Adaptation.ipynb) | 03, 04 | Historical top/bottom-quintile classification; do not substitute for continuous regression. |
| 7 | [Chromosome and seed robustness](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B7%5D%20Enhancer_Kaggle_Phase7_Chromosome_Robustness.ipynb) | 03, 04 | Historical binary-activity stress test with its own partitions. |
| 8 | [Earlier benchmark synthesis](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B8%5D%20Enhancer_Kaggle_Phase8_Main_Benchmark_Synthesis.ipynb) | 01, 02, 06 | Earlier tables/figures, before the consistent three-seed transformer reruns. |
| 9 | [Continuous-activity population and frozen splits](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B9%5D%20Enhancer_Kaggle_Phase9A_Full_Activity_and_Splits.ipynb) | 03 | Foundational 9A data/split provenance for chromosome 9B and locus 9C branches. |
| 9B | [Chromosome-held-out regression](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B9B%5D%20Enhancer_Kaggle_Phase9B_Chromosome_Regression.ipynb) | 03, 04 | Historical short-duration regression; preserved separately from the later 54-fit training study. |
| 9C | [Locus-held-out regression](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B9C%5D%20Enhancer_Kaggle_Phase9C_Locus_Regression.ipynb) | 03, 04 | Separate historical locus split, including weak/collapsed runs; not the current primary 9B experiment. |
| 10 | [Frozen external MPRA transfer](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B10%5D%20Enhancer_Kaggle_Phase10_Independent_MPRA_Transfer.ipynb) | 04 | Earlier transfer study and external-cohort provenance; later selected arms evaluated in Phase 36. |
| 11A | [Historical test-overlap impact](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B11A%5D%20Enhancer_Kaggle_Phase11A_Historical_Overlap_Impact.ipynb) | 01, 05 | Earlier leakage-impact analysis; later consolidated in Phase 40. |
| 11B | [Earlier statistics and calibration](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B11B%5D%20Enhancer_Kaggle_Phase11B_Statistics_and_Calibration.ipynb) | 02, 05 | Historical reporting; the current headline family is reproduced from Phase 41 records. |
| 11C | [Original transformer seed-robustness shard](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B11C%5D%20Enhancer_Kaggle_Phase11C_Transformer_Seed_Robustness.ipynb) | 02 | Historical shard; its implementation/completion was audited in Phase 30 and replaced by the consistent Phase 31 study. |
| 11C2 | [Additional historical seed shard](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B11C2%5D%20Enhancer_Kaggle_Phase11C_Transformer_Seed_Robustness.ipynb) | 02 | Keep as a separate supplied shard; do not infer full completion or equivalence from its filename. |
| 11C3 | [Additional historical seed shard](../notebooks/original_phases/03_phases01_11c3_audit_and_models/%5B11C3%5D%20Enhancer_Kaggle_Phase11C_Transformer_Seed_Robustness.ipynb) | 02 | Keep as a separate supplied shard; current consistent results come from Phases 31–33. |

## Shafin Ahmed: Phases 12–49

| Phase | Original notebook / purpose | Current entry notebook(s) | Historical/current relationship |
|---|---|---|---|
| 12 | [Specialist MPRA CNN baselines](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B12%5D%20Enhancer_Kaggle_Phase12_Strong_Baselines.ipynb) | 03, 04 | Original specialist-CNN study on 9B/9C; later 9B recipe comparison is Phases 34–38. |
| 13 | [Independent verification of earlier results](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B13%5D%20Enhancer_Independent_Result_Verification.ipynb) | 03, 04, 05 | Checks the Phase 12/historical regression artifacts within their original scope. |
| 14 | [Filtering and subgroup composition](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B14%5D%20Enhancer_Filtering_and_Subgroup_Analysis.ipynb) | 01, 03 | Exploratory retained/removed-population analysis; historical context, not a new confirmatory test. |
| 15 | [Reviewer-evidence mapping](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B15%5D%20Enhancer_Reviewer_Evidence_Map.ipynb) | 06 | Historical technical evidence map; saved open/partial statuses describe that date. |
| 16 | [Filtered-out population inference](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B16%5D%20Enhancer_Filtered_Out_Performance.ipynb) | 03, 04 | Historical checkpoint inference on excluded candidates; no retraining. |
| 17 | [Composition and reliability diagnostics](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B17%5D%20Enhancer_Filtering_Reliability_and_Composition.ipynb) | 03, 05 | Exploratory population standardization, seed disagreement and interval diagnostics. |
| 18 | [Activity tails and independent diagnostic audit](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B18%5D%20Enhancer_Activity_Tails_and_Independent_Audit.ipynb) | 03, 05 | Exploratory reliability/tail analyses following Phases 16–17. |
| 19 | [Written filtering findings and evidence package](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B19%5D%20Enhancer_Filtering_Findings_and_Reviewer_Package.ipynb) | 06 | Historical synthesis; pending items are dated rather than current repository status. |
| 20 | [Scope and claim freeze](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B20%5D%20Enhancer_Scope_and_Claim_Freeze.ipynb) | 01–06 | Earlier scope snapshot; later experiments and manuscript revisions remain separately identified. |
| 21 | [Historical classification verification](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B21%5D%20Enhancer_Historical_Classification_Audit.ipynb) | 02 | Authentication of older single-run results and submitted-table dispositions. |
| 22 | [Recovered checkpoint prediction reproduction](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B22%5D%20Enhancer_Checkpoint_and_Prediction_Reproduction.ipynb) | 03, 04 | Earlier regression and removed-population checkpoint reproduction; requires original artifacts. |
| 23 | [Statistical reporting ledger](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B23%5D%20Enhancer_Statistical_Reporting_Freeze.ipynb) | 05 | Earlier conditional-uncertainty and subgroup ledger; retain original definitions. |
| 24 | [Filtering/threshold reproducibility](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B24%5D%20Enhancer_Dataset_Filtering_and_Threshold_Reproducibility.ipynb) | 01, 03 | Original sequence-level filtering checks require separately acquired DNA inputs. |
| 25 | [Earlier publication tables](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B25%5D%20Enhancer_Publication_Tables.ipynb) | 06 | Presentation snapshot superseded by later tables and manuscript revisions. |
| 26 | [Earlier publication figures](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B26%5D%20Enhancer_Publication_Figures_and_Diagrams.ipynb) | 06 | Presentation snapshot; current figure exports and sources remain separately available. |
| 27 | [Literature and contribution assessment](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B27%5D%20Enhancer_Literature_and_Contribution_Assessment.ipynb) | 06 | Dated targeted literature/source review, before later citation corrections. |
| 28 | [Earlier portable package](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B28%5D%20Enhancer_Portable_Reproducibility_Package.ipynb) | 01–06 | Historical full artifact-package builder; large artifact ZIPs are not embedded in GitHub. |
| 29 | [Earlier pre-writing readiness review](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B29%5D%20Enhancer_Final_PreWriting_Readiness_Review.ipynb) | 01–06 | Dated readiness gate, not proof that every later manuscript revision was already complete. |
| 30 | [Historical transformer compatibility audit](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B30%5D%20Enhancer_Phase11C_Compatibility_Audit.ipynb) | 02 | Prospective lock/reuse decision for the consistent three-seed classification study. |
| 31 | [Consistent three-seed classification training](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B31%5D%20Enhancer_Phase11C_Consistent_Three_Seed_Training.ipynb) | 02 | Current transformer classification training provenance: 60 fixed model/task/seed/fold fits. |
| 32 | [Classification checkpoint and metric verification](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B32%5D%20Enhancer_Phase11C_Independent_Verification.ipynb) | 02, 05 | Independent verification of the consistent classification study. |
| 33 | [Classification findings and evidence update](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B33%5D%20Enhancer_Phase11C_Findings_and_Evidence_Update.ipynb) | 02, 06 | Three-seed results; keep original single-run baselines visibly distinct. |
| 34 | [Prospective MPRA training protocol](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B34%5D%20Enhancer_MPRA_Training_Fairness_Protocol.ipynb) | 03 | Locks the validation-only candidate grid, seeds, recipes and stopping rules. |
| 35 | [MPRA training adequacy and limited search](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B35%5D%20Enhancer_MPRA_Training_Adequacy_and_Search.ipynb) | 03 | Current training provenance: all 54 fits and own-trajectory two-epoch checkpoints. |
| 36 | [Validation recipe selection and evaluation](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B36%5D%20Enhancer_MPRA_Validation_Selection_and_Evaluation.ipynb) | 03, 04 | Locks six recipe choices before scoring internal/external sets; previously examined cohorts remain disclosed. |
| 37 | [MPRA checkpoint/prediction verification](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B37%5D%20Enhancer_MPRA_Fairness_Independent_Verification.ipynb) | 03, 04, 05 | Authenticates 54 jobs, 72 checkpoints and 60 evaluation sets within the declared scope. |
| 38 | [MPRA findings and reporting](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B38%5D%20Enhancer_MPRA_Fairness_Findings_and_Reviewer_Update.ipynb) | 03, 04, 06 | Read the collapse-count erratum: two practical near-constant fits versus zero frozen undefined-Pearson flags. |
| 39 | [Final evidence authority freeze](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B39%5D%20Enhancer_Final_Evidence_Freeze.ipynb) | 01–06 | Evidence/claim provenance at the synthesis freeze; later editorial versions remain separate. |
| 40 | [Historical leakage-impact synthesis](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B40%5D%20Enhancer_Historical_Leakage_Impact.ipynb) | 01, 05 | Supports the duplicate/overlap results and descriptive historical test comparisons. |
| 41 | [Final statistical family](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B41%5D%20Enhancer_Final_Statistics.ipynb) | 05 | Source of the 14 headline conditional comparisons; current reproduction reruns the declared draws. |
| 42 | [Final numerical tables](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B42%5D%20Enhancer_Final_Tables.ipynb) | 06 | Numerical-table synthesis; later Word table formatting is archived with the presentation scripts. |
| 43 | [Final figure synthesis](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B43%5D%20Enhancer_Final_Figures.ipynb) | 06 | Historical plot assembly; current manuscript figure revisions have their own drawing sources. |
| 44 | [Supplement assembly](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B44%5D%20Enhancer_Supplement_Builder.ipynb) | 06 | Historical supplement builder; current exported tables/figures remain the publication assets. |
| 45 | [Claim/evidence audit](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B45%5D%20Enhancer_Claim_Evidence_Audit.ipynb) | 01–06 | Dated claim ledger and checks, not a new experimental study. |
| 46 | [Earlier manuscript rebuild](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B46%5D%20Enhancer_Manuscript_Rebuild.ipynb) | 06 | Earlier manuscript draft builder; current supervisor draft lives in revision52 locally. |
| 47 | [Earlier public reproducibility package](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B47%5D%20Enhancer_Public_Reproducibility_Package.ipynb) | 01–06 | Previous package snapshot; current six entry notebooks are the public reporting route. |
| 48 | [Sharing/submission package assembly](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B48%5D%20Enhancer_Submission_Package.ipynb) | 06 | Historical local packaging; does not establish an external submission or acceptance. |
| 49 | [Manuscript feedback corrections](../notebooks/original_phases/04_phases12_49_validation_and_reporting/%5B49%5D%20Enhancer_Manuscript_Feedback_Corrections.ipynb) | 02, 06 | Editorial/citation/figure corrections; later revision52 presentation scripts are also archived. |

## Saved versions and failed attempts

| Source context | Archived notebook | Status |
|---|---|---|
| `Shafin's Contribution/.ipynb_checkpoints/[12] Enhancer_Kaggle_Phase12_Strong_Baselines-checkpoint.ipynb` | [checkpoint_2c03994a8d_[12] Enhancer_Kaggle_Phase12_Strong_Baselines-checkpoint.ipynb](../notebooks/original_phases/05_saved_versions/checkpoint_2c03994a8d_%5B12%5D%20Enhancer_Kaggle_Phase12_Strong_Baselines-checkpoint.ipynb) | Saved checkpoint snapshot; distinguish from the main executed notebook. |
| `Shafin's Contribution/.ipynb_checkpoints/[13] Enhancer_Independent_Result_Verification-checkpoint.ipynb` | [checkpoint_01373f0b54_[13] Enhancer_Independent_Result_Verification-checkpoint.ipynb](../notebooks/original_phases/05_saved_versions/checkpoint_01373f0b54_%5B13%5D%20Enhancer_Independent_Result_Verification-checkpoint.ipynb) | Saved checkpoint snapshot; distinguish from the main executed notebook. |
| `Shafin's Contribution/phase35_outputs/recovery_20261013T231246Z/[35] Enhancer_MPRA_Training_Adequacy_and_Search.ipynb` | [recovery_57b9131ad1_[35] Enhancer_MPRA_Training_Adequacy_and_Search.ipynb](../notebooks/original_phases/05_saved_versions/recovery_57b9131ad1_%5B35%5D%20Enhancer_MPRA_Training_Adequacy_and_Search.ipynb) | Preserved interrupted/failed attempt; not a replacement for the completed main phase. |
| `Shafin's Contribution/phase46_outputs/recovery/first_render_attempt_failed.ipynb` | [recovery_04f6356f9e_first_render_attempt_failed.ipynb](../notebooks/original_phases/05_saved_versions/recovery_04f6356f9e_first_render_attempt_failed.ipynb) | Preserved interrupted/failed attempt; not a replacement for the completed main phase. |
| `Shafin's Contribution/revision49_outputs/recovery/label_adapter_failed_notebook.ipynb` | [recovery_4ba1320e56_label_adapter_failed_notebook.ipynb](../notebooks/original_phases/05_saved_versions/recovery_4ba1320e56_label_adapter_failed_notebook.ipynb) | Preserved interrupted/failed attempt; not a replacement for the completed main phase. |

## Supporting scripts and restoration

The [supporting code archive](../src/archive/README.md) includes analysis, verification, training, resume tests, figure drawing, manuscript building, table/caption formatting and package assembly. Unique source versions are retained with their original hashes. Imported library copies remain dependencies or upstream model snapshots, not separately claimed project contributions.

To reconstruct every inventoried path into a **new empty directory**, without executing it:

```bash
python -B -X utf8 scripts/restore_historical_sources.py /path/to/new-empty-source-archive
```

The restoration command checks hashes, writes only to its new destination, and restores every duplicate path as well as the distinct versions. Large input/model artifacts remain a separate deposit requirement.
