"""Build and validate a small, portable results-review handoff; no training/inference."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlsplit
from html.parser import HTMLParser
import csv, hashlib, html, io, json, re, shutil, statistics, zipfile

ROOT = Path(__file__).resolve().parents[1]
SHARE = ROOT.parent / 'new_results'
AUDIT = ROOT / 'delivery_review_outputs' / '20261014'
BACKUP = AUDIT / 'before_review' / 'new_results'
P05 = '05_Training_fairness_study_Phases34-38'
P04 = '04_Classification_seed_study_Phases30-33'
P03 = '03_Paper_tables_and_figures_Phases20-29'
P06 = '06_Verification_and_provenance'
PROV = {}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def put(rel, text, source='', note='Authored review guide; no new experiment.'):
    p = SHARE / rel; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + '\n', encoding='utf-8')
    PROV[rel] = {'source_file': source, 'note': note}
    return p

def putcsv(rel, data, source='', note='Derived diagnostic; no change to frozen results.'):
    p = SHARE / rel; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    PROV[rel] = {'source_file': source, 'note': note}

def copy(source, rel):
    src = ROOT / source; assert src.is_file(), source
    dst = SHARE / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    PROV[rel] = {'source_file': source, 'note': 'Byte-identical copy of full-study source.'}

def link(path): return quote(path, safe='/#')

def mtable(header, data):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |'] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in data])

def main():
    assert ROOT.name == "Shafin's Contribution" and SHARE.is_dir() and BACKUP.is_dir()
    initial = json.loads((AUDIT / 'initial_inventory.json').read_text(encoding='utf-8'))['files']
    original = {r['file']: r for r in initial}
    path_map = {}
    for r in initial:
        for src in r['source_matches']: path_map[src] = r['file']
        PROV[r['file']] = {'source_file': r['source_matches'][0] if r['source_matches'] else '', 'note': 'Existing handoff file; source authenticated by hash.'}
    renamed = '02_Filtering_and_reviewer_followup_Phases13-19/phase13_verification_summary.json'
    assert sha(SHARE / renamed) == sha(ROOT / 'phase13_outputs/verification_summary.json')
    PROV[renamed] = {'source_file': 'phase13_outputs/verification_summary.json', 'note': 'Renamed byte-identical source copy.'}
    path_map['phase13_outputs/verification_summary.json'] = renamed

    frozen = json.loads((AUDIT / 'protected_files_before.json').read_text(encoding='utf-8'))
    for rel, h in frozen.items(): assert sha(ROOT / rel) == h, rel
    assert frozen['analysis_tools/mpra_fairness_training.py'] == '83a1ae9a9e03361e4fe517044ee7b3cc76f580db04878e4a12f1f7873c60d311'
    jobs = rows(ROOT / 'phase35_outputs/training_job_summary.csv'); assert len(jobs) == 54
    small = [r for r in jobs if float(r['validation_prediction_sd']) < 1e-4]
    assert {r['job'] for r in small} == {'HepG2__dnabert2__lr2__seed20260932', 'K562__dnabert2__lr2__seed20260932'}
    assert sum(r['collapsed'].lower() == 'true' for r in jobs) == 0
    diagnostics = []
    for r in jobs:
        logs = rows(ROOT / 'phase35_outputs/training_logs' / (r['job'] + '.csv'))
        diagnostics.append({'job': r['job'], 'cell': r['cell_line'], 'model': r['model'], 'learning_rate': r['learning_rate'], 'seed': r['seed'], 'selected_epoch': r['selected_epoch'], 'epochs_run': r['epochs_run'], 'validation_pearson': r['validation_pearson'], 'validation_prediction_sd': r['validation_prediction_sd'], 'protocol_undefined_pearson_collapsed_flag': r['collapsed'], 'posthoc_descriptive_sd_below_1e_minus4': r in small, 'min_epoch_prediction_sd': min(float(x['validation_prediction_sd']) for x in logs), 'max_epoch_prediction_sd': max(float(x['validation_prediction_sd']) for x in logs), 'max_absolute_epoch_validation_pearson': max(abs(float(x['validation_pearson'])) for x in logs)})
    putcsv(P05 + '/diagnostics/all_54_fit_collapse_diagnostics.csv', diagnostics, 'phase35_outputs/training_job_summary.csv; phase35_outputs/training_logs/*.csv')
    for r in small:
        for source in [f"phase35_outputs/jobs/{r['job']}.json", f"phase35_outputs/training_logs/{r['job']}.csv"]:
            copy(source, P05 + '/diagnostics/' + Path(source).name)
    mean3 = {}; other2 = {}
    for cell in ['HepG2', 'K562']:
        for grid in [1, 2]:
            a = [r for r in jobs if r['cell_line'] == cell and r['model'] == 'dnabert2' and int(r['grid_index']) == grid]
            mean3[cell, grid] = statistics.mean(float(r['validation_pearson']) for r in a)
            other2[cell, grid] = statistics.mean(float(r['validation_pearson']) for r in a if r['seed'] != '20260932')
    assert mean3['HepG2', 1] > mean3['HepG2', 2] and mean3['K562', 1] > mean3['K562', 2]
    erratum = '''# Erratum and clarification: two practically collapsed DNABERT2 fits

**Reviewed: 14 October 2026.** This corrects the interpretation of the unqualified “0/54 collapsed” sentence in the original Phase 38 findings. The sealed archive, original findings/summary, frozen training source, run metadata, selection lock and numerical results are preserved.

## Correct reporting

**Two of the 54 fits showed nearly constant validation predictions (practical collapse).** Both were DNABERT2 at LR 5e-5, seed 20260932, one in each cell. Separately, **zero fits triggered the frozen undefined-Pearson flag**. These are different statements; report both rather than silently changing the definition of a failed fit.

'''
    erratum += mtable(['Cell', 'Best validation Pearson', 'Best-checkpoint prediction SD', 'Selected epoch', 'Epochs run'], [[r['cell_line'], f"{float(r['validation_pearson']):.9f}", f"{float(r['validation_prediction_sd']):.9g}", r['selected_epoch'], r['epochs_run']] for r in small])
    erratum += '''

The epoch logs show extremely small prediction dispersion from epoch 1 throughout both runs, validation correlations close to zero, and normalized training loss around 1. This supports “no useful validation sequence discrimination in the recorded trajectories”; it does not establish a mechanistic root cause. The other 52 best checkpoints have prediction SDs between 0.225469 and 0.539390 in raw activity units.

## Frozen rule versus descriptive diagnosis

`analysis_tools/mpra_fairness_training.py::safe_r` returns undefined Pearson when prediction SD is below 1e-8 (or predictions are non-finite, or targets have zero SD). Job metadata sets `collapsed` from undefined best-validation Pearson. Both identified fits have finite Pearson and SD above 1e-8, so their saved `collapsed:false` values are consistent with that frozen implementation.

For this reporting audit only, a **post hoc descriptive screen, best-checkpoint SD < 1e-4**, identifies exactly these two fits. This screen was not a prospective selection rule, is not applied to exclude runs, and does not change the learning-rate search or any test result. None of the 36 CNN fits meets this descriptive screen. See `diagnostics/all_54_fit_collapse_diagnostics.csv` in the review handoff, with the two original per-fit metadata files and epoch logs alongside it.

## Effect on validation selection

All three predeclared seeds remain in each candidate mean, including the nearly constant fit. Both cells correctly retain LR 2e-5 under the frozen all-three-seed policy. The following two-seed numbers are explanatory arithmetic only, not replacement results or a proposed exclusion policy.

'''
    erratum += mtable(['Cell', '2e-5: all 3 seeds', '5e-5: all 3 seeds', '5e-5: other 2 seeds only'], [[c, f'{mean3[c,1]:.9f}', f'{mean3[c,2]:.9f}', f'{other2[c,2]:.9f}'] for c in ['HepG2', 'K562']])
    erratum += '''

In HepG2, the other two high-LR seeds average 0.644288, above the all-three-seed 2e-5 mean 0.639262 (and the corresponding two-seed 2e-5 mean 0.639530). Thus the unstable seed materially affects the selected LR. In K562 the other two high-LR seeds average 0.671338, still below the all-three-seed 2e-5 mean 0.677098 (and the corresponding two-seed mean 0.679092).

## What remains unchanged

All 54 jobs, the six locked recipes, all 72 checkpoints, all 60 evaluation sets, paired intervals and figures remain unchanged. This is a reporting addendum, not a revised training study. Do not relabel the two finite-score candidates as retrospectively ineligible: the protocol makes a candidate ineligible if any seed has **undefined** Pearson, and these two seeds did not.

Suggested manuscript wording: “All seeds were retained in validation selection. Although none triggered the frozen undefined-Pearson flag, two DNABERT2 fits at LR 5e-5 produced nearly constant predictions and are reported as practical collapses. The unstable seed affected LR selection in HepG2. The diagnostic did not alter selection or evaluation.”
'''
    canonical_erratum = ROOT / 'phase38_outputs/ERRATUM_collapse_count.md'
    canonical_erratum.write_text(erratum, encoding='utf-8')
    put(P05 + '/ERRATUM_collapse_count.md', erratum, 'phase38_outputs/ERRATUM_collapse_count.md', 'Reviewed reporting addendum; formal flags and practical nearconstant fits distinguished.')
    logpath = ROOT / 'CONTRIBUTION_LOG.md'; logtext = logpath.read_text(encoding='utf-8')
    marker = '### Portable results handoff and collapse-definition clarification (2026-10-14)'
    if marker not in logtext:
        logtext += '\n\n' + marker + '''

At the user's request, audited the existing new_results review folder and the user-added erratum. Preserved the user's original handoff, erratum and contribution log under delivery_review_outputs/20261014/before_review. Confirmed two practically nearconstant DNABERT2 fits at LR5e-5/seed20260932 from saved validation metadata and every epoch log. The frozen rule produced zero undefined-Pearson flags; the two practical collapses are a distinct post hoc descriptive diagnosis, not changed eligibility or excluded seeds. The preceding correction entry is retained, with this clarification controlling its interpretation. Canonical ERRATUM_collapse_count.md now states both counts and the all-three-seed selection policy explicitly. No training, inference, recipe, interval, checkpoint or test result was changed.

Prepared a small portable review handoff with a current evidence guide, corrected new-run two-epoch-prefix labels, conditional classification comparisons without an equivalence claim, historical-document notices, local figure/table indexes, small verification records, provenance and integrity manifests. Earlier Phase12–33 evidence, frozen Phase34–38 files and the sealed 17.9GB archive remain preserved. This is results delivery and reporting clarification; manuscript integration, final proofreading and submission remain separate. No external message or journal contact was sent.
'''
        logpath.write_text(logtext, encoding='utf-8')
    copy('CONTRIBUTION_LOG.md', 'CONTRIBUTION_LOG.md')
    report = (ROOT / 'phase38_outputs/MPRA_FAIRNESS_FINDINGS.md').read_text(encoding='utf-8')
    old = 'Collapsed validation fits across the full study: 0/54.'
    assert old in report
    report = report.replace(old, 'Frozen undefined-Pearson flags: 0/54. Practically nearconstant fits: 2/54 (post hoc descriptive diagnosis; both DNABERT2 at LR 5e-5, seed 20260932).')
    report = '> **Reviewed handoff copy.** The original sealed findings are preserved. This copy clarifies the collapse sentence; all numerical model comparisons remain unchanged. Read the [erratum](ERRATUM_collapse_count.md) before citing failure counts.\n\n' + report
    put(P05 + '/MPRA_FAIRNESS_FINDINGS.md', report, 'phase38_outputs/MPRA_FAIRNESS_FINDINGS.md', 'Derived review copy: collapse wording clarified; original sealed report unchanged.')

    proofs = [(13,'verification_summary.json'),(29,'summary.json'),(31,'summary.json'),(32,'summary.json'),(33,'FINAL_VALIDATION.json'),(33,'visual_QA.json'),(34,'summary.json'),(35,'summary.json'),(36,'summary.json'),(37,'summary.json'),(38,'FINAL_VALIDATION.json'),(38,'visual_QA.json')]
    for n, name in proofs: copy(f'phase{n}_outputs/{name}', f'{P06}/phase{n}_{name}')
    copy('phase36_outputs/selection_lock.json', P05 + '/selection_lock.json')
    copy('phase35_outputs/run_lock.json', P05 + '/run_lock.json')
    copy('phase33_outputs/figure_index.csv', P04 + '/figure_index.csv')
    for rel, source in [(P03+'/table_index.csv','phase25_outputs/table_index.csv'),(P03+'/figure_index.csv','phase26_outputs/figure_index.csv'),(P04+'/figure_index.csv','phase33_outputs/figure_index.csv'),(P05+'/figure_index.csv','phase38_outputs/figure_index.csv')]:
        data = rows(ROOT / source); kept=[]
        for r in data:
            oldfile = r['file']; sub = 'figures' if '/figures/' in oldfile else 'tables'
            local = sub + '/' + Path(oldfile).name
            target = SHARE / Path(rel).parent / local
            if not target.is_file():
                assert oldfile.endswith('.svg'), (rel, oldfile)
                continue
            assert sha(target) == r['sha256']
            r['file'] = local; r['full_study_file'] = oldfile; kept.append(r)
        putcsv(rel, kept, source, 'Portable index: local paths; only formats actually supplied are listed; source paths retained in a separate column.')
    ev = rows(ROOT / 'phase38_outputs/reviewer_evidence_update.csv')
    for r in ev:
        r['full_study_evidence_reference'] = r['evidence']
        r['evidence'] = {'Aditta1_training_fairness':'tables/ensemble_metrics.csv; tables/conditional_paired_intervals.csv','DNABERT2_two_epoch_duration':'tables/conditional_paired_intervals.csv; training_protocol.json','old9C_collapsed_seed':'../03_Paper_tables_and_figures_Phases20-29/tables/S12_collapsed_seed_sensitivity.csv'}[r['concern']]
    ev.append({'concern':'high_lr_DNABERT2_practical_collapse','status':'reporting_addendum_verified_without_reselection','evidence':'ERRATUM_collapse_count.md; diagnostics/all_54_fit_collapse_diagnostics.csv','remaining':'Two descriptive nearconstant fits; zero frozen undefined-Pearson flags. All seeds remain in selection. Root cause not established.','full_study_evidence_reference':'phase35_outputs/jobs/*dnabert2__lr2__seed20260932.json; phase35_outputs/training_logs/*dnabert2__lr2__seed20260932.csv'})
    putcsv(P05 + '/reviewer_evidence_update.csv', ev, 'phase38_outputs/reviewer_evidence_update.csv', 'Review-only update: portable evidence paths and explicit nearconstant-fit diagnostic.')

    for r in initial:
        rel = r['file']; folder = rel.split('/')[0]
        if not rel.endswith('.md') or folder[:2] not in {'01','02','03','04'}: continue
        text = (BACKUP / rel).read_text(encoding='utf-8')
        msg = 'Historical evidence snapshot. Later classification results are in folder 04 and later MPRA training-adequacy results are in folder 05; older readiness statements and model rankings are not the current all-study conclusion.'
        if folder == P04: msg = 'Classification results are current for the declared three-seed rerun. Statements in this older document that MPRA fairness remains open predate the completed Phases34–38 follow-up; use folder 05 for that question.'
        text = '> **Reading notice:** ' + msg + ' See the [current evidence map](../CURRENT_EVIDENCE_MAP.md).\n\n' + text
        put(rel, text, PROV[rel]['source_file'], 'Historical source text preserved after an added reading notice; local links made portable.')
    repro = (ROOT / 'phase38_outputs/README_REPRODUCTION.md').read_text(encoding='utf-8')
    put(P05+'/README_REPRODUCTION.md', '> **Full-study instructions, retained for context.** This small handoff does not contain model weights, row-level predictions, training code or executable notebooks. Use it for review, not to rerun the study. The instructions below apply to the separate full reproducibility archive.\n\n'+repro, 'phase38_outputs/README_REPRODUCTION.md', 'Review notice added to distinguish presentation files from full reproduction package.')
    descriptions = {
        '01_Strong_baselines_Phase12':'Historical primary/secondary regression comparison under the earlier recipes. Use these tables for historical comparisons; current MPRA duration controls and selected-model results are in folder 05.',
        '02_Filtering_and_reviewer_followup_Phases13-19':'Filtering, retained/removed populations and reviewer follow-up. Older manuscript sections and responses are drafts. Historical 9C collapse sensitivity is separate from the two new 9B practical collapses.',
        P03:'Historical bounded-scope pre-writing package: seven main tables, twenty supplementary tables and sixteen figures. T3/S1/S2/S19/S20 and classification graphics are superseded for the repeated fine-tuned model comparison by folder 04; current MPRA results are in folder 05. Completion of a bounded checklist is not completion of all original scientific questions or the manuscript.',
        P04:'Current fine-tuned classification seed study: 60 consistent local fits, three declared seeds and five fixed dependent folds. Ensemble AUROC difference intervals include zero; this does not prove equivalence. Read CLASSIFICATION_FINDINGS.md and retain its effective-seed and previously examined-cohort limits.',
        P05:'Current MPRA training-adequacy study: 54 fits, 72 checkpoints and 60 evaluation sets. All validation/test/external saved predictions reproduced exactly. Read the erratum and reviewed findings together. The same-run two-epoch control is not the original historical fit. Equal candidate counts are not equal compute or exhaustive tuning.',
        P06:'Small copies of full-study verification records. Each describes checks performed in the complete research folder at the recorded time. Their referenced large ZIPs and weights are intentionally not in this presentation handoff. The current share manifest authenticates the files actually supplied here.'}
    for folder, desc in descriptions.items():
        listing=[]
        for p in sorted((SHARE/folder).rglob('*')):
            if not p.is_file() or p.suffix=='.html' or p.name=='README.md':continue
            local=p.relative_to(SHARE/folder).as_posix()
            listing.append(f'- [{local}]({link(local)})')
        put(folder+'/README.md', '# '+folder+'\n\n'+desc+'\n\n[Start here](../00_START_HERE.md) · [Current evidence map](../CURRENT_EVIDENCE_MAP.md)\n\nThis folder is for evidence review. Original numerical tables and figure assets are retained; the source index records copies and presentation changes.\n\n## Included files\n\n'+'\n'.join(listing))

    metrics = rows(ROOT/'phase37_outputs/ensemble_metrics.csv'); ci = rows(ROOT/'phase37_outputs/conditional_paired_intervals.csv')
    cm = rows(ROOT/'phase32_outputs/ensemble_metrics.csv'); cc = rows(ROOT/'phase32_outputs/conditional_paired_ensemble_intervals.csv')
    def metric(cell, arm, cohort='internal'): return next(r for r in metrics if r['cell']==cell and r['arm']==arm and r['cohort']==cohort)
    def cint(cell, contrast): return next(r for r in ci if r['cell']==cell and r['cohort']=='internal' and r['metric']=='pearson' and r['contrast']==contrast)
    arms=[('DNABERT2: own new-run two-epoch prefix','dnabert2_fixedlr_early2'),('DNABERT2: longer, fixed/selected LR 2e-5','dnabert2_selected'),('MPRA-LegNet: validation-selected LR','legnet_selected'),('MPRAnn: validation-selected LR','mprann_selected')]
    primary=mtable(['Three-seed ensemble, 9B test Pearson','HepG2 (n=9,048)','K562 (n=13,145)'], [[label]+[f"{float(metric(c,arm)['pearson']):.6f}" for c in ['HepG2','K562']] for label,arm in arms])
    comparisons=mtable(['Contrast A − B','HepG2: difference [conditional 95% CI]','K562: difference [conditional 95% CI]'],[[label]+[f"{float(cint(c,k)['estimate_a_minus_b']):+.6f} [{float(cint(c,k)['lower_95']):+.6f}, {float(cint(c,k)['upper_95']):+.6f}]" for c in ['HepG2','K562']] for label,k in [('DB2 longer − own two-epoch prefix','duration_fixed_lr'),('Selected DB2 − selected LegNet','selected_vs_legnet'),('Selected DB2 − selected MPRAnn','selected_vs_mprann')]])
    classdata=[]
    for stage in ['1','2']:
        a=next(r for r in cm if r['stage']==stage and r['model']=='finetuned_dnabert2');b=next(r for r in cm if r['stage']==stage and r['model']=='finetuned_dnabert_legacy');r=next(r for r in cc if r['stage']==stage and r['metric']=='auroc')
        classdata.append([f'Stage {stage} (n={int(a["n"]):,})',f'{float(a["auroc"]):.6f}',f'{float(b["auroc"]):.6f}',f'{float(r["estimate"]):+.6f} [{float(r["lower_95"]):+.6f}, {float(r["upper_95"]):+.6f}]'])
    classes=mtable(['Three-seed pooled OOF ensemble','DNABERT2 AUROC','Original DNABERT AUROC','Difference [conditional 95% CI]'],classdata)
    costs=mtable(['All nine candidate/seed fits per cell/model','DNABERT2: hours','LegNet: minutes','MPRAnn: minutes'],[[c]+[f'{sum(float(r["elapsed_seconds"]) for r in jobs if r["cell_line"]==c and r["model"]==m)/unit:.2f}' for m,unit in [('dnabert2',3600),('legnet',60),('mprann',60)]] for c in ['HepG2','K562']])
    guide=f'''# Results for Aditta — start here

**Reviewed: 14 October 2026.** Open **00_OPEN_RESULTS.html** in a browser for a readable local version. Keep this entire folder together; all presentation links are relative.

This is a compact **results-review handoff**, not the full executable reproducibility package. It includes findings, source tables, figures, frozen plans and small verification records. Large weights, row-level predictions, input data, runtime caches, training code/notebooks and sealed multi-GB archives remain in Shafin's full research folder.

## What is finished, and what remains

The agreed computational evidence work is complete, including the 60-fit classification rerun and 54-fit MPRA follow-up, independent prediction verification, figures and full-study archive validation. **The manuscript is not yet integrated, proofread or submission-ready.** Scientific limits are documented, not all experimentally resolved. The earlier ten-priority checklist was completed for a bounded scope; it does not establish that every original requested experiment or scientific concern was solved.

## Read these first

1. [Current evidence map](CURRENT_EVIDENCE_MAP.md) — which results supersede older copies.
2. [MPRA erratum](P05/ERRATUM_collapse_count.md) and [reviewed MPRA findings](P05/MPRA_FAIRNESS_FINDINGS.md).
3. [Classification findings](P04/CLASSIFICATION_FINDINGS.md).
4. [Filtering findings](02_Filtering_and_reviewer_followup_Phases13-19/RESEARCH_FINDINGS.md).

## Important reporting correction

**Two DNABERT2 fits produced nearly constant validation predictions: 2/54 fits overall, both at LR 5e-5 and seed 20260932.** The frozen implementation reported zero *undefined-Pearson* flags; that is a different count. The erratum and all-54-fit diagnostic distinguish them. Both finite-score seeds remain in the locked all-three-seed LR selection. No score, test prediction, interval, checkpoint, recipe or figure was changed. The high-LR instability affected LR selection in HepG2. It is separate from the old historical 9C collapsed-seed question.

## Current MPRA activity results

The primary duration control compares each new DNABERT2 run's own validation-best checkpoint within its first two epochs against its longer trajectory at LR 2e-5. **These first-two-epoch scores are not the historical Phase 9B/12 fits.** LR selection used only mean validation Pearson across all three seeds before new test/external inference.

{primary}

{comparisons}

Longer training materially improved DNABERT2 under this recipe, supporting the specific two-epoch adequacy concern. LegNet retained the lead in both cells. DNABERT2 exceeded MPRAnn in K562; the HepG2 difference interval includes zero. This is a comparison of fitted recipes under a limited three-LR search, not a universal architecture ranking.

All displayed intervals are conditional on fitted models and previously examined cohorts, with unadjusted comparisons. MPRA internal resampling uses locus groups; three seeds do not make the intervals estimates of training/search uncertainty.

![Current MPRA test performance: black squares show ensembles; dots show seeds](P05/figures/A2_primary_test_performance.png)

[All six MPRA figures](P05/figure_contact_sheet.png) · [MPRA source tables](P05/README.md) · [Selected-recipe flags](P05/selected_recipe_adequacy_flags.csv)

### External transfer and training cost

On the 8,399-sequence DHS-natural cohort in each cell, selected LegNet had the highest ensemble Spearman (0.551907 HepG2; 0.557878 K562). Selected DNABERT2 scored 0.508442 and 0.526993. Longer DNABERT2 training improved external rank correlation but increased MAE/RMSE in both cells; all external ensemble R² values were negative. Ranking does not establish calibrated cross-assay activity prediction.

{costs}

These are recorded elapsed fit times, including in-fit validation/checkpointing. They are **not** GPU-only FLOPs, total project time, or pretraining cost. Matching candidate/seed counts does not match computation.

## Current classification results

{classes}

The ensemble AUROC difference intervals include zero in both stages; **this does not demonstrate equivalence or a statistical tie**. Mean-fold metrics, seed SD and pooled OOF ensemble metrics are separate quantities. Five fixed folds overlap in training data; the effective-seed schedule reuses some initializations across different folds. These limitations remain in the classification findings.

[Classification figures](P04/figure_contact_sheet.png) · [Classification tables and scope](P04/README.md)

## Folder guide

'''.replace('P05/',P05+'/').replace('P04/',P04+'/')
    for folder, desc in descriptions.items(): guide+=f'- [{folder}]({link(folder+"/README.md")}) — {desc}\n'
    guide+='''
## Limits that still need to appear in the paper

- Previously examined test/OOF/external cohorts; no fresh confirmatory evaluation is claimed.
- Three seeds, limited LR grids, conditional/unadjusted intervals, and different model computation/pretraining.
- Six of 18 selected fits have budget flags; two HepG2 DNABERT2 fits reached the 30-epoch cap. LegNet selected its upper LR boundary and MPRAnn its lower boundary in both cells. Convergence and exhaustive tuning are not established.
- Two new practically collapsed high-LR DNABERT2 fits remain visible and included in selection. Their mechanistic cause is not established.
- The historical 9C collapse cause and missing older epoch logs remain unresolved. No new long-context, native-locus or broader-cell experiment was supplied.
- The target is the assayed 200 bp reporter activity, not endogenous regulatory necessity or a universal limit of DNA sequence information.

## Integrity and the separate full archive

[SOURCE_FILE_INDEX.csv](SOURCE_FILE_INDEX.csv) maps each handoff file to its source or review derivation. [HANDOFF_MANIFEST.csv](HANDOFF_MANIFEST.csv) records file hashes. [HANDOFF_VALIDATION.json](HANDOFF_VALIDATION.json) records portable-link, index, table and provenance checks. The contribution log includes the reporting addendum and this delivery review.

Full-study archives remain separate: Phase 33 classification ZIP (~25.6 GB), Phase 38 MPRA ZIP (~17.9 GB), and earlier snapshots. The Phase 38 sealed archive retains the original collapse wording and the contribution log as it stood at sealing; the later erratum/log addendum accompany it separately. The supplied historical validation JSONs describe that sealed state, not equality with the subsequently amended live log. The presentation ZIP is a new, small review package and is not a replacement for those reproducibility archives.
'''
    put('00_START_HERE.md',guide)
    current=f'''# Current evidence map

Read this map before using an older table or readiness paragraph as a current conclusion.

| Question | Use now | Historical/context material |
| --- | --- | --- |
| MPRA duration adequacy and selected-model comparison | [Folder 05 findings]({P05}/MPRA_FAIRNESS_FINDINGS.md), tables/ensemble_metrics.csv, conditional_paired_intervals.csv | Folder 01 and folder 03 regression rankings describe earlier recipes. |
| New high-LR nearconstant fits | [Reviewed erratum]({P05}/ERRATUM_collapse_count.md), diagnostics/all_54_fit_collapse_diagnostics.csv | The sealed report's 0/54 means only the frozen undefined-Pearson flag. Do not silently redefine eligibility. |
| Fine-tuned DNABERT2 vs original DNABERT classification | [Folder 04 findings]({P04}/CLASSIFICATION_FINDINGS.md), repeated-study tables and figures | Folder 03 T3/S1/S2/S19/S20 and older classification figures remain historical; do not silently pool versions. |
| Filtering and retained/removed populations | [Folder 02 findings](02_Filtering_and_reviewer_followup_Phases13-19/RESEARCH_FINDINGS.md), folder 03 filtering tables | These are population/reliability analyses, not the new duration experiment. |
| Historical 9C nearly constant seed | Folder 03 tables/S12_collapsed_seed_sensitivity.csv | This historical run is distinct from both new 9B practical collapses; its cause is unresolved. |
| Ten-priority readiness checklist | Folder 03 checklist plus this current map | Completed for bounded supported scope, not proof all requested experiments or final manuscript tasks are finished. |
| Literature and biological scope | Folder 03 literature review and folder 02 references | Local full-study-only source-note links are labelled as such. No new literature search was performed in this handoff audit. |
| Integrity and reproduction | Folder 06 verification records and handoff manifests | This folder is a results review, not a runnable copy of the 111 GiB research workspace. |

## Writing actions still open

Integrate the new classification and MPRA evidence into Methods/Results/Discussion, replace superseded figure/table claims, include both practical collapse and formal-flag counts, retain budget/grid/external-calibration limits, then complete language/equation/layout checks and author/submission requirements. The drafts in folder 02 have not been rewritten into a finished manuscript by this handoff review.
'''
    put('CURRENT_EVIDENCE_MAP.md',current)

    # Rewrite actual local Markdown links; retain missing full-study references as explicit text.
    missing_full=[]
    pattern=re.compile(r'(!?)\[([^\]]*)\]\((<[^>]*>|[^)]*)\)')
    for p in sorted(SHARE.rglob('*.md')):
        rel=p.relative_to(SHARE).as_posix();text=p.read_text(encoding='utf-8')
        def portable(m):
            mark,label,target=m.groups();target=target.strip().strip('<>');parts=urlsplit(target)
            if parts.scheme in {'http','https','mailto'} or target.startswith('#'):return m.group(0)
            decoded=unquote(target).replace('\\','/');rootprefix=ROOT.as_posix()+'/'
            actual=None
            if p.parent==SHARE and decoded in {'SOURCE_FILE_INDEX.csv','HANDOFF_MANIFEST.csv','HANDOFF_VALIDATION.json'}:return m.group(0)
            if decoded.lower().startswith(rootprefix.lower()): actual=path_map.get(decoded[len(rootprefix):])
            elif (p.parent/decoded.split('#')[0]).is_file():return m.group(0)
            elif decoded in path_map:actual=path_map[decoded]
            if actual:
                import os
                local=os.path.relpath(SHARE/actual,p.parent).replace('\\','/')
                return f'{mark}[{label}]({link(local)})'
            missing_full.append({'file':rel,'reference':target})
            return f'{label} (full-study-only reference: `{decoded}`)'
        text=pattern.sub(portable,text);p.write_text(text,encoding='utf-8')
    (AUDIT/'full_study_only_references.json').write_text(json.dumps(missing_full,indent=2)+'\n',encoding='utf-8')

    from markdown_it import MarkdownIt
    md=MarkdownIt('commonmark',{'html':False}).enable('table')
    html_targets={p:p.with_suffix('.html') for p in SHARE.rglob('*.md')}
    html_targets[SHARE/'00_START_HERE.md']=SHARE/'00_OPEN_RESULTS.html'
    css='''body{margin:0;background:#f3f6fa;color:#162334;font:16px/1.65 system-ui,Segoe UI,sans-serif}main{max-width:1060px;margin:30px auto;background:white;padding:34px 42px;border:1px solid #d8e1ec;border-radius:12px}nav{background:#123350;padding:15px 24px;color:white}nav a{color:white;margin-right:24px}h1,h2,h3{line-height:1.3;color:#143e5a}h2{margin-top:2em;border-bottom:1px solid #dce5ee;padding-bottom:.3em}a{color:#005b9a;overflow-wrap:anywhere}img{max-width:100%;height:auto;border:1px solid #dce5ee}table{display:block;overflow-x:auto;border-collapse:collapse;font-size:14px;margin:18px 0}td,th{border:1px solid #ced9e5;padding:9px 12px;text-align:left}th{background:#eaf1f8}blockquote{border-left:4px solid #c98618;background:#fff7e8;padding:12px 18px;margin:18px 0}code{overflow-wrap:anywhere;background:#edf2f7;padding:1px 4px}pre{overflow:auto;background:#edf2f7;padding:16px}footer{font-size:13px;color:#546477;margin-top:40px}@media(max-width:700px){main{margin:0;padding:20px 16px;border-radius:0}nav{padding:12px 16px}h1{font-size:26px}}'''
    for p,target in html_targets.items():
        body=md.render(p.read_text(encoding='utf-8'))
        def html_link(m):
            raw=html.unescape(m.group(1));path=urlsplit(raw).path
            if urlsplit(raw).scheme:return m.group(0)
            dest=(target.parent/unquote(path)).resolve()
            if dest in html_targets:
                import os
                return 'href="'+html.escape(quote(os.path.relpath(html_targets[dest],target.parent).replace('\\','/'),safe='/'))+'"'
            return m.group(0)
        body=re.sub(r'href="([^"]+)"',html_link,body)
        import os
        home=quote(os.path.relpath(SHARE/'00_OPEN_RESULTS.html',target.parent).replace('\\','/'),safe='/')
        mapping=quote(os.path.relpath(SHARE/'CURRENT_EVIDENCE_MAP.html',target.parent).replace('\\','/'),safe='/')
        page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(p.stem)+' — Aditta results review</title><style>'+css+'</style></head><body><nav><a href="'+home+'">Results home</a><a href="'+mapping+'">Current evidence map</a></nav><main>'+body+'<footer>Portable results review · 14 October 2026 · No training or inference is run by this page.</footer></main></body></html>'
        put(target.relative_to(SHARE).as_posix(),page,PROV.get(p.relative_to(SHARE).as_posix(),{}).get('source_file',''),'Offline HTML rendering of the corresponding review Markdown; no external scripts/assets.')

    # Keep proof of the original archive separate from the updated live log and erratum.
    final=json.loads((ROOT/'phase38_outputs/FINAL_VALIDATION.json').read_text(encoding='utf-8'))
    archive=ROOT/'phase38_outputs'/final['archive'];archive_hash=sha(archive)
    assert archive_hash==final['archive_sha256'] and archive.stat().st_size==final['archive_bytes']
    postseal={'reviewed_utc':datetime.now(timezone.utc).isoformat(),'original_archive_sha256_verified_unchanged':archive_hash,'original_archive_bytes':archive.stat().st_size,'live_contribution_log_sha256':sha(ROOT/'CONTRIBUTION_LOG.md'),'current_erratum_sha256':sha(canonical_erratum),'sealed_archive_contains_later_erratum':False,'log_equality_statement_in_original_validation_is_historical_at_seal_time':True,'frozen_training_and_selection_unchanged':True,'new_inference_or_training_performed':False,'undefined_Pearson_flags_under_frozen_rule':0,'descriptively_nearconstant_fits':2,'posthoc_descriptive_sd_screen':1e-4}
    (ROOT/'phase38_outputs/POSTSEAL_ADDENDUM_STATUS.json').write_text(json.dumps(postseal,indent=2)+'\n',encoding='utf-8')
    copy('phase38_outputs/POSTSEAL_ADDENDUM_STATUS.json',P06+'/POSTSEAL_ADDENDUM_STATUS.json')
    for rel,h in frozen.items():assert sha(ROOT/rel)==h,rel

    # Numeric tables/figure assets must retain their original source bytes.
    exact=0
    for r in initial:
        rel=r['file'];p=SHARE/rel
        if '/tables/' in rel or p.suffix.lower() in {'.png','.pdf','.bib'}:
            assert sha(p)==r['sha256'],rel;exact+=1
    for rel in [P03+'/table_index.csv',P03+'/figure_index.csv',P04+'/figure_index.csv',P05+'/figure_index.csv']:
        for r in rows(SHARE/rel):assert (SHARE/Path(rel).parent/r['file']).is_file() and sha(SHARE/Path(rel).parent/r['file'])==r['sha256']
    # Build an exhaustive, portable file list before the manifest/validation seal.
    provenance=[]
    for p in sorted(SHARE.rglob('*')):
        if not p.is_file() or p.name in {'SOURCE_FILE_INDEX.csv','HANDOFF_MANIFEST.csv','HANDOFF_VALIDATION.json'}:continue
        rel=p.relative_to(SHARE).as_posix();meta=PROV.get(rel,{})
        src=meta.get('source_file','');source_hash=sha(ROOT/src) if src and (ROOT/src).is_file() else ''
        provenance.append({'file':rel,'bytes':p.stat().st_size,'sha256':sha(p),'full_study_source_file':src,'full_study_source_sha256':source_hash,'same_bytes_as_source':bool(source_hash and sha(p)==source_hash),'review_note':meta.get('note','Preserved existing handoff file.')})
    putcsv('SOURCE_FILE_INDEX.csv',provenance,note='Source/copy/derivation map for all supplied evidence files.')
    manifest=[]
    for p in sorted(SHARE.rglob('*')):
        if p.is_file() and p.name not in {'HANDOFF_MANIFEST.csv','HANDOFF_VALIDATION.json'}:manifest.append({'file':p.relative_to(SHARE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    putcsv('HANDOFF_MANIFEST.csv',manifest,note='SHA256 integrity manifest; excludes itself and the later validation record.')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if (tag=='a' and k=='href') or (tag=='img' and k=='src'):self.links.append(v)
    broken=[];count=0
    for p in SHARE.rglob('*.html'):
        parser=Links();parser.feed(p.read_text(encoding='utf-8'))
        for href in parser.links:
            u=urlsplit(href)
            if u.scheme in {'http','https','mailto','data'} or not u.path:continue
            count+=1;dest=(p.parent/unquote(u.path)).resolve()
            if not dest.is_relative_to(SHARE.resolve()) or not dest.is_file():broken.append({'file':p.relative_to(SHARE).as_posix(),'link':href})
    assert not broken,broken
    for p in SHARE.rglob('*.json'):json.loads(p.read_text(encoding='utf-8-sig'))
    from PIL import Image
    for p in SHARE.rglob('*.png'):
        with Image.open(p) as im:im.verify()
    pdfs=list(SHARE.rglob('*.pdf'))
    for p in pdfs:assert p.read_bytes().startswith(b'%PDF-'),p
    manifest_records=rows(SHARE/'HANDOFF_MANIFEST.csv')
    for r in manifest_records:assert sha(SHARE/r['file'])==r['sha256'] and (SHARE/r['file']).stat().st_size==int(r['bytes'])
    validation={'status':'passed','checked_utc':datetime.now(timezone.utc).isoformat(),'scope':'Portable results presentation and reporting addendum, not new training/inference.','initial_files':len(initial),'original_table_and_figure_assets_preserved':exact,'portable_local_links_checked':count,'broken_local_links':broken,'manifest_entries':len(manifest_records),'figure_png_count':len(list(SHARE.rglob('figures/*.png'))),'figure_pdf_count':len(pdfs),'frozen_training_protocol_selection_and_sealed_outputs_unchanged':True,'original_full_archive_sha256':archive_hash,'formal_undefined_Pearson_flags':0,'practically_nearconstant_fits':2,'posthoc_descriptive_screen_sd':1e-4,'numerical_performance_results_unchanged':True,'results_handoff_is_full_executable_reproducibility_package':False,'manuscript_finished':False,'historical_notices_and_current_evidence_map_added':True,'manifest_exclusions':['HANDOFF_MANIFEST.csv','HANDOFF_VALIDATION.json'],'visual_HTML_review':'pending_manual_browser_check'}
    put('HANDOFF_VALIDATION.json',json.dumps(validation,indent=2),note='Post-manifest validation record; excluded from manifest by design.')
    (AUDIT/'review_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed_pending_visual_HTML_check','share_files':sum(p.is_file() for p in SHARE.rglob('*')),'share_bytes':sum(p.stat().st_size for p in SHARE.rglob('*') if p.is_file()),'local_links_checked':count,'original_assets_preserved':exact,'archive_hash_unchanged':archive_hash,'nearconstant_fit_count':len(small)},indent=2))

if __name__=='__main__':main()
