from pathlib import Path
import json,pandas as pd,nbformat,hashlib
r=Path.cwd();a=json.loads((r/'phase13_outputs/verification_summary.json').read_text());b=json.loads((r/'phase14_outputs/filtering_analysis_summary.json').read_text());c=json.loads((r/'phase15_outputs/reviewer_map_summary.json').read_text())
assert a['status']==b['status']==c['status']=='completed' and c['missing_evidence_rows']==0
print('Reviewer map:',c)
p=r/'CONTRIBUTION_LOG.md';text=p.read_text('utf-8');marker='### Completed filtering analysis and reviewer evidence map (2026-10-12)'
if marker not in text:
 addition='''

### Completed filtering analysis and reviewer evidence map (2026-10-12)

This entry closes the interim raw-reference pending status above. The full hg38 reference and both ENCODE assay files are now copied under `reference_inputs/raw_sources` and match the original Phase 9A MD5 **and** SHA256 checksums.

- Independently reconstructed **149,632** complete paired-orientation assay candidates from raw data. Reproduced all **76,833** retained record IDs, DNA sequences, sequence hashes and activities exactly.
- The screen removed **72,799 candidates (48.65%)**: **72,795** for sharing at least one 15-mer with the legacy benchmark, and **4** for an exact or reverse-complement match. There were no non-ACGT exclusions or within-cell duplicate exclusions in this complete-pair population.

| Cell/design family | Complete assay pairs | Retained | Removed | Removed fraction |
|---|---:|---:|---:|---:|
| HepG2 / DNasePeakNoPromoter | 64,190 | 33,324 | 30,866 | 48.09% |
| K562 / peak | 85,442 | 43,509 | 41,933 | 49.08% |

The selection effect is specific rather than a wholesale activity shift. Retained mean GC is only about **0.67 percentage points higher** in each cell. Mean assayed activity shifts are small: retained minus removed is **+0.00935** log2 units for HepG2 and **−0.03856** for K562 (standardized differences +0.013 and −0.087). Removed DNA has longer homopolymers: mean maximum same-base run **5.34 / 5.40**, versus **4.74 / 4.75** retained; removed 95th percentiles are **9 / 10 bases**, versus **6 / 6** retained. The cell libraries use different design families, so these are not causal cell-type comparisons.

Completed descriptive retained-test subgroup checks for DNABERT-2 and both specialist CNNs, using fixed GC/homopolymer bins and training-only CpG/activity thresholds. Counts, Pearson, Spearman, R², RMSE, MAE and signed errors are saved; groups below n=100 are explicitly suppressed. CNNs lead overall, but some high-activity subgroups favour DNABERT-2 in MAE. Activity bands require measured labels and are diagnostics, not deployment-time routing rules. No subgroup results were used to tune models.

**Scope:** filtering-population differences do not establish model accuracy on removed sequences. No predictions on that removed population are available. The paper should describe results on retained, filtered 200 bp assay candidates and avoid treating this as a genome-wide or all-enhancer benchmark.

Additional completed verification in notebook 13:

- Reconstructed pre-export prediction vectors to show the tiny aggregate-CSV Spearman discrepancies are caused by rounding near ties; scores from authenticated original vectors match the saved calculation below 1e-12.
- Authenticated 15 parent result files and checked all 32 older validation-choice scores. Earlier per-epoch maxima cannot be independently verified because those original training logs are unavailable; the archived K562 collapse cause remains open.
- Independently confirmed that the 29,330 frozen external records have no exact/RC identity or shared canonical 31-mer with Phase 9A. Reproduced **11,215,289** distinct reference 31-mers, and confirmed no natural interval overlap. These screens do not exclude all homology. The 8,399 DHS-natural sequences form **7,581** regions under the 1 kb grouping rule.

**Created and fully executed in `enhancer`, with no notebook errors:**

1. `[13] Enhancer_Independent_Result_Verification.ipynb` → `phase13_outputs`.
2. `[14] Enhancer_Filtering_and_Subgroup_Analysis.ipynb` → `phase14_outputs`.
3. `[15] Enhancer_Reviewer_Evidence_Map.ipynb` → `phase15_outputs`.

The reviewer evidence map contains exact review paragraph references, file hashes and suggested narrower claims. All referenced evidence files exist. Scientific/manuscript concerns remain labelled partial/open where appropriate; completed technical verification and filtering analysis are distinguished from manuscript repair or editorial acceptance. The original manuscript was not changed and the journal was not contacted.

**Unresolved / future work:** original collapsed-seed training cause and unavailable older epoch logs; accuracy on removed sequences; longer-context and broader-cell experiments; historical classification-table recovery, literature/citation repairs, figures, revised manuscript and point-by-point response. None of those experiments or manuscript edits were silently performed.

All new work, copied inputs, intermediate files and logs remain inside `Shafin's Contribution`. Research originals, notebook 12 and Phase 12 outputs remain untouched. No packages were installed, system settings changed, or additional models trained.
'''
 p.write_text(text.rstrip()+addition+'\n',encoding='utf-8')
rows=[]
for number in [13,14,15]:
 path=next(r.glob(f'[[]{number}]*.ipynb'));n=nbformat.read(path,as_version=4);nbformat.validate(n)
 for i,cell in enumerate(n.cells):
  if cell.cell_type=='code':
   compile(cell.source,f'{path.name}:{i}','exec');assert cell.execution_count is not None
   assert not any(o.output_type=='error' for o in cell.outputs)
 rows.append({'notebook':path.name,'code_cells':sum(c.cell_type=='code' for c in n.cells),'status':'syntax, schema and full CPU execution passed','sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
report={'status':'completed','environment':r'C:\Users\User\.conda\envs\enhancer\python.exe','notebooks':rows,'phase13':a,'phase14_status':b['status'],'phase15':c,'contribution_log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'no_additional_training':True}
(r/'analysis_tools/final_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('Final log appended. All three notebooks pass syntax, notebook schema and complete CPU execution.')
